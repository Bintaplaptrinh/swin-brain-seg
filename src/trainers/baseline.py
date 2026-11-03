"""Train compact nnU-Net + BiRef reconstruction on BRISC. Run: python -m src.trainers.baseline."""

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import csv
from datetime import datetime
import gc
import json
import math
import os
from pathlib import Path
import platform
import random
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.flop_counter import FlopCounterMode
from tqdm import tqdm

from src.models.factory import (ATTENTION_DEFAULTS, DEPTHWISE_DEFAULTS, BIREF_DEFAULTS, EMA_SKIP_DEFAULTS, add_model_arguments,
                                    attention_options, depthwise_options, biref_options, ema_skip_options, build_model)
from src.models.budget import MAX_PARAMETERS, enforce_parameter_budget
from src.datasets.brisc import (BRISCDataset, discover, geometry, load_mask, representative_subset,
                   restore_split, split_manifest, stratified_split)
from src.utils.metrics import binary_metrics, json_safe, summarize
from .glioma_policy import AsyncGliomaPolicy, LEGACY_POLICY_DEFAULTS, policy_split as split_policy_reward


from src.core.paths import (ROOT, DATA_ROOT, configure_run_paths, checkpoint_path, history_path,
                            previous_history, create_run_dirs)
MODEL_FIELDS = ("variant", "groups", "num_basis", "size", "deep_supervision", "ema_decay_max", "smoke_test", *ATTENTION_DEFAULTS, *DEPTHWISE_DEFAULTS, *BIREF_DEFAULTS, *EMA_SKIP_DEFAULTS)
TRAIN_FIELDS = ("epochs", "batch_size", "workers", "seed", "lr", "weight_decay", "grad_clip", "amp", "nsd_tolerance", "val_fraction", "selection_metric", "augmentation", "boundary_weight", *LEGACY_POLICY_DEFAULTS)
GLIOMA_MIOU_TARGET = 0.8


def checkpoint_setting(config, name):
    if name in LEGACY_POLICY_DEFAULTS and name not in config:
        return LEGACY_POLICY_DEFAULTS[name]
    if name in EMA_SKIP_DEFAULTS and name not in config and config["variant"] != "biref-ema":
        return EMA_SKIP_DEFAULTS[name]
    if name in BIREF_DEFAULTS and name not in config and not str(config["variant"]).startswith("biref"):
        return BIREF_DEFAULTS[name]
    if name in {"augmentation", "boundary_weight"} and name not in config:
        return {"augmentation": "legacy", "boundary_weight": .1}[name]
    if name in DEPTHWISE_DEFAULTS and name not in config:
        return ({"dw_kernel_size": 3, "dw_dilation": 1}[name]
                if config["variant"] == "depthwise" else DEPTHWISE_DEFAULTS[name])
    if name == "selection_metric" and name not in config:
        return "weighted-miou"
    # Baseline checkpoints predate these options and have no attention modules.
    if name in ATTENTION_DEFAULTS and name not in config and not config["variant"].startswith("kan-two-way"):
        return ATTENTION_DEFAULTS[name]
    return config[name]


def validation_score(metrics, selection_metric):
    if selection_metric == "glioma-miou":
        score = metrics["per_class"]["Glioma"]["mIoU"]
    elif selection_metric == "weighted-miou":
        score = metrics["Weighted_mIoU"]
    else:
        raise ValueError(f"Unknown selection metric: {selection_metric}")
    if score is None or not math.isfinite(score):
        raise ValueError(f"Validation {selection_metric} must be a finite score")
    return score


class WeightEMA:
    """Warmup counts completed optimizer steps, including across resumes."""
    def __init__(self, model, decay_max=0.9995):
        if not 0.999 <= decay_max <= 0.9995:
            raise ValueError("EMA decay_max must be in [0.999, 0.9995]")
        self.model = deepcopy(model).eval().requires_grad_(False)
        self.decay_max, self.num_updates = decay_max, 0
        # DNA exposes shared convolutions under multiple state-dict names.
        # New experiments update each tensor once; preserve legacy resumes.
        self.unique_updates = getattr(model, "ema_unique_updates", False)

    @property
    def decay(self):
        return min(self.decay_max, (1 + self.num_updates) / (10 + self.num_updates))

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        alpha = self.decay
        if self.unique_updates:
            current = dict(model.named_parameters()) | dict(model.named_buffers())
            averages = dict(self.model.named_parameters()) | dict(self.model.named_buffers())
        else:
            current, averages = model.state_dict(), self.model.state_dict()
        for name, average in averages.items():
            value = current[name].detach()
            if average.is_floating_point():
                average.lerp_(value, 1 - alpha)
            else:
                average.copy_(value)

    def state_dict(self):
        return {"model": self.model.state_dict(), "num_updates": self.num_updates, "decay_max": self.decay_max,
                "unique_updates": self.unique_updates}

    def load_state_dict(self, state):
        if state.get("unique_updates", False) != self.unique_updates:
            raise ValueError("EMA tensor-update protocol differs from checkpoint")
        if state["decay_max"] != self.decay_max:
            raise ValueError("EMA decay ceiling differs from the checkpoint")
        self.model.load_state_dict(state["model"])
        self.num_updates = state["num_updates"]


def boundary_targets(target, output_size):
    """Mask-derived contour with invalid neighborhoods excluded at every scale."""
    valid = (target != -1)[:, None].float()
    foreground = (target == 1)[:, None].float()
    outer = F.max_pool2d(foreground, 3, 1, 1)
    inner = -F.max_pool2d(-foreground, 3, 1, 1)
    contour = (outer - inner).clamp(0, 1)
    # Exclude the image perimeter and letterbox borders from contour supervision.
    valid = -F.max_pool2d(-F.pad(valid, (1, 1, 1, 1), value=0), 3, 1)
    contour = F.adaptive_max_pool2d(contour * valid, output_size)
    valid = 1 - F.adaptive_max_pool2d(1 - valid, output_size)
    return contour, valid


def boundary_loss(predictions, target):
    losses = []
    for prediction in predictions:
        logits = prediction.float()
        truth, valid = boundary_targets(target, logits.shape[-2:])
        positive = (truth * valid).sum((1, 2, 3), keepdim=True)
        negative = ((1 - truth) * valid).sum((1, 2, 3), keepdim=True)
        pos_weight = (negative / positive.clamp_min(1)).clamp(1, 20)
        weights = (1 + (pos_weight - 1) * truth) * valid
        bce = (F.binary_cross_entropy_with_logits(logits, truth, reduction="none") * weights).sum() / weights.sum().clamp_min(1)
        probability = logits.sigmoid() * valid
        dice = (2 * (probability * truth).sum() + 1e-5) / (probability.sum() + (truth * valid).sum() + 1e-5)
        losses.append(bce + 1 - dice)
    return sum(losses) / len(losses)


def segmentation_loss(outputs, target, boundary_weight=.1):
    """CE + per-image foreground Dice; padded pixels are ignored at all scales."""
    boundaries = outputs.get("boundary", []) if isinstance(outputs, dict) else []
    if isinstance(outputs, dict):
        outputs = outputs["segmentation"]
    outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    weights = [2.0 ** -i for i in range(len(outputs))]
    total = outputs[0].new_zeros((), dtype=torch.float32)
    for weight, logits in zip(weights, outputs):
        labels = F.interpolate(target[:, None].float(), size=logits.shape[-2:], mode="nearest")[:, 0].long()
        valid = labels != -1
        if not valid.any():
            raise ValueError("No valid pixels at a deep-supervision scale; increase --size")
        foreground = logits.float().softmax(1)[:, 1] * valid
        truth = (labels == 1).float()
        dice = (2 * (foreground * truth).sum((1, 2)) + 1e-5) / (
            foreground.sum((1, 2)) + truth.sum((1, 2)) + 1e-5
        )
        total = total + weight * (F.cross_entropy(logits.float(), labels, ignore_index=-1) + 1 - dice.mean())
    result = total / sum(weights)
    return result + boundary_weight * boundary_loss(boundaries, target) if boundaries and boundary_weight > 0 else result


def write_json(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def atomic_save(path, state):
    """Close the written file, then tolerate transient Windows rename locks.

    Keep the old checkpoint until replacement succeeds. A unique temporary
    file avoids colliding with files left by a failed save. If the destination
    stays locked, retain the completed new checkpoint for manual recovery.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError as error:
            if attempt == 9:
                raise PermissionError(
                    f"Could not replace checkpoint {path} after 10 attempts. "
                    f"The existing file was kept; the completed new checkpoint is at {temporary}."
                ) from error
            time.sleep(min(0.1 * 2**attempt, 1.0))


def prepare_resume(args, checkpoint):
    """Preserve the actual earlier best; never select a future/stale checkpoint."""
    history = previous_history(args, checkpoint)
    if history.exists():
        for line in history.read_text(encoding="utf-8").splitlines():
            if json.loads(line)["epoch"] > checkpoint["epoch"]:
                raise ValueError("Run history is newer than --resume; use --output with a new folder")
    if checkpoint["epoch"] == checkpoint["best_epoch"]:
        best = checkpoint
    else:
        source = args.resume.parent / "best.pt"
        if not source.exists():
            raise FileNotFoundError("Resume needs the best.pt companion to preserve earlier model selection")
        best = torch.load(source, map_location="cpu", weights_only=True)
        if (best["epoch"] != checkpoint["best_epoch"] or best["best_score"] != checkpoint["best_score"]
                or best["splits"] != checkpoint["splits"]
                or best.get("policy_split") != checkpoint.get("policy_split")
                or any(checkpoint_setting(best["config"], key) != checkpoint_setting(checkpoint["config"], key)
                       for key in MODEL_FIELDS + TRAIN_FIELDS)):
            raise ValueError("The best.pt companion does not match the resumed training state")
    destination = history_path(args)
    if history.exists() and history != destination:
        destination.write_bytes(history.read_bytes())
    atomic_save(checkpoint_path(args, "best.pt"), best)
    if "validation" in best:
        write_json(args.output / "best_validation.json", best["validation"])


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def loader(samples, args, generator, training=False, policy=None, sampling_weight=None):
    multiplier = getattr(args, "glioma_sampling_weight", 1.) if sampling_weight is None else sampling_weight
    pituitary_multiplier = getattr(args, "pituitary_sampling_weight", 1.)
    sampler = None
    if training and (multiplier != 1 or pituitary_multiplier != 1):
        class_weights = {"Glioma": multiplier, "Pituitary": pituitary_multiplier}
        weights = [class_weights.get(s.tumor, 1.) for s in samples]
        sampler = WeightedRandomSampler(weights, len(samples), replacement=True, generator=generator)
    return DataLoader(
        BRISCDataset(samples, args.size, augment=training, augmentation=getattr(args, "augmentation", "legacy"),
                     glioma_policy=policy or getattr(args, "glioma_policy", "standard")),
        batch_size=args.batch_size, shuffle=training and sampler is None, sampler=sampler, num_workers=args.workers,
        pin_memory=args.device == "cuda", worker_init_fn=seed_worker,
        generator=generator, persistent_workers=False,
    )


@contextmanager
def inference_model(model):
    was_training = model.training
    supervision = model.decoder.deep_supervision
    model.eval()
    model.decoder.deep_supervision = False
    try:
        yield model
    finally:
        model.decoder.deep_supervision = supervision
        model.train(was_training)


def autocast(args):
    return torch.autocast(args.device, dtype=torch.float16, enabled=args.amp and args.device == "cuda")


def image_to_device(images, args):
    memory_format = torch.channels_last if getattr(args, "channels_last", False) else torch.contiguous_format
    return images.to(args.device, non_blocking=True, memory_format=memory_format)


def check_vram_budget(args):
    if args.device != "cuda":
        return
    limit = getattr(args, "max_vram_gb", 10) * 1e9
    used = torch.cuda.max_memory_reserved()
    if used > limit:
        raise RuntimeError(f"Peak reserved VRAM {used / 1e9:.2f} GB exceeds the {limit / 1e9:g} GB budget. "
                           "Reduce --batch-size or --size and use a fresh run directory.")
    if used > .9 * limit and not getattr(args, "_vram_warning_issued", False):
        print(f"WARNING: peak reserved VRAM is {used / 1e9:.2f} GB, close to the {limit / 1e9:g} GB limit.", flush=True)
        args._vram_warning_issued = True


def train_epoch(model, ema, batches, optimizer, scaler, args, epoch):
    model.train()
    loss_sum, seen, updated = 0.0, 0, 0
    progress = tqdm(batches, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
    for images, targets, _ in progress:
        images = image_to_device(images, args)
        targets = targets.to(args.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(args):
            loss = segmentation_loss(model(images), targets, getattr(args, "boundary_weight", .1))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # GradScaler reduces its scale if it skipped an overflowed update.
        if scaler.get_scale() >= previous_scale:
            ema.update(model)
            updated += 1
        check_vram_budget(args)
        count = images.shape[0]
        loss_sum += float(loss.detach()) * count
        seen += count
        progress.set_postfix(loss=f"{loss_sum / seen:.4f}", ema=f"{ema.decay:.6f}")
    if not updated:
        raise FloatingPointError("No optimizer update succeeded in this epoch")
    return loss_sum / seen


@torch.inference_mode()
def evaluate(model, batches, args, description):
    rows = []
    with inference_model(model):
        for images, _, indices in tqdm(batches, desc=description, dynamic_ncols=True):
            with autocast(args):
                logits = model(image_to_device(images, args))
            check_vram_budget(args)
            for output, index in zip(logits, indices.tolist()):
                sample = batches.dataset.samples[index]
                target = load_mask(sample.mask)
                height, width = target.shape
                rh, rw, top, left = geometry(height, width, args.size)
                # Restore continuous logits before argmax, never resize a binary prediction.
                cropped = output[:, top:top + rh, left:left + rw]
                restored = F.interpolate(cropped[None].float(), size=(height, width), mode="bilinear", align_corners=False)
                prediction = restored.argmax(1)[0].cpu().numpy().astype(bool)
                rows.append({"filename": sample.image.name, "tumor": sample.tumor, "plane": sample.plane,
                             **binary_metrics(prediction, target, args.nsd_tolerance)})
    return summarize(rows), rows


def synchronize(args):
    if args.device == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def profile_efficiency(model, args, training_peak):
    """Forward-only batch-1 timing, with training allocations released by caller."""
    tensor = torch.randn(1, 1, args.size, args.size, device=args.device)
    tensor = image_to_device(tensor, args)
    with inference_model(model):
        with autocast(args), FlopCounterMode(display=False) as counter:
            model(tensor)
        flops = counter.get_total_flops()
        operators = {str(op): count for op, count in counter.get_flop_counts()["Global"].items()}
        for _ in range(args.benchmark_warmup):
            with autocast(args):
                model(tensor)
        synchronize(args)
        if args.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()
        else:
            baseline = None
        timings = []
        for _ in range(args.benchmark_steps):
            synchronize(args)
            start = time.perf_counter()
            with autocast(args):
                model(tensor)
            synchronize(args)
            timings.append(1000 * (time.perf_counter() - start))
        memory = {
            "training_peak_allocated_MiB": training_peak.get("allocated"),
            "training_peak_reserved_MiB": training_peak.get("reserved"),
            "inference_baseline_allocated_MiB": baseline / 2**20 if baseline is not None else None,
            "inference_peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else None,
            "inference_peak_reserved_MiB": torch.cuda.max_memory_reserved() / 2**20 if args.device == "cuda" else None,
        }
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "parameters_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "GFLOPs": flops / 1e9,
        "flop_convention": "PyTorch FlopCounterMode supported operators; multiply-add = 2 FLOPs",
        "flop_limitation": "Partial operator count: excludes unregistered operations such as normalization and, in legacy KAN variants, exp/tanh. Not total arithmetic cost.",
        "counted_flops_by_operator": operators,
        "inference_ms_per_image": float(np.median(timings)),
        "inference_mean_ms_per_image": float(np.mean(timings)),
        "inference_p95_ms_per_image": float(np.percentile(timings, 95)),
        "input_shape": [1, 1, args.size, args.size], "batch_size": 1,
        "amp": args.amp and args.device == "cuda", "warmup": args.benchmark_warmup,
        "timed_iterations": args.benchmark_steps, "device": args.device,
        "device_name": torch.cuda.get_device_name() if args.device == "cuda" else platform.processor(),
        "timing_scope": "Forward only; synchronized; excludes loading, preprocessing, native-size restoration and metric calculation",
        "vram_scope": "PyTorch allocator on this device; training includes student, EMA and optimizer; excludes CUDA context and other processes",
        **memory,
    }


def rng_state(generator):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    npstate = state["numpy"]
    np.random.set_state((npstate[0], np.asarray(npstate[1], dtype=np.uint32), *npstate[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def display(metrics, label):
    def fmt(value):
        return f"{value:.5f}" if isinstance(value, (int, float)) else str(value)
    print(f"\n{label}", flush=True)
    for name, values in metrics["per_class"].items():
        print(f"  mIoU {name}: {fmt(values['mIoU'])} (n={values['images']})")
    print(f"  Weighted mIoU: {fmt(metrics['Weighted_mIoU'])}")
    overall = metrics["overall"]
    for key in ("Dice", "HD95_px", "ASSD_px", "NSD", "Recall", "Precision", "Specificity"):
        print(f"  {key}: {fmt(overall[key])}")
    for plane, values in metrics["per_plane"].items():
        print(f"  {plane}: mIoU={fmt(values['mIoU'])}, Dice={fmt(values['Dice'])} (n={values['images']})")
    print(f"  Empty predictions: {overall['empty_predictions']}; infinite distances: {overall['infinite_distance_images']}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, help="Run metadata folder; defaults to outputs/runs/<timestamp>_<variant>")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=12.0)
    parser.add_argument("--max-vram-gb", type=float, default=10, help="Peak PyTorch reserved-memory budget in decimal GB (at most 10)")
    parser.add_argument("--augmentation", choices=("legacy", "robust", "glioma"), default=None)
    parser.add_argument("--glioma-rl", action=argparse.BooleanOptionalAction, default=None,
                        help="Asynchronous CPU policy-gradient controller; default on for biref-ema")
    parser.add_argument("--glioma-policy", choices=("mild", "standard", "strong"), default="standard")
    parser.add_argument("--glioma-sampling-weight", type=float, default=1.)
    parser.add_argument("--policy-reward-fraction", type=float, default=None,
                        help="Reserve this fraction of training glioma for policy rewards; default .1 for new experiment, 0 for legacy")
    parser.add_argument("--rl-learning-rate", type=float, default=.1)
    parser.add_argument("--rl-exploration", type=float, default=.2)
    parser.add_argument("--rl-warmup", type=int, default=3)
    parser.add_argument("--boundary-weight", type=float, default=.1, help="Auxiliary mask-boundary loss weight for biref")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--ema-decay-max", type=float, default=0.9995)
    parser.add_argument("--nsd-tolerance", type=float, default=2.0, help="Contour tolerance in ORIGINAL-image pixels")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--selection-metric", choices=("glioma-miou", "weighted-miou"),
                        help="Best validation checkpoint criterion; defaults to glioma-miou for biref/depthwise, weighted-miou otherwise")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deep-supervision", action=argparse.BooleanOptionalAction, default=True)
    add_model_arguments(parser)
    parser.add_argument("--benchmark-warmup", type=int, default=10)
    parser.add_argument("--benchmark-steps", type=int, default=50)
    parser.add_argument("--resume", type=Path, help="Trusted checkpoint; restores optimizer, EMA, scheduler, splits and RNG")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate EMA in --resume on the saved test split")
    parser.add_argument("--smoke-test", action="store_true", help="One tiny epoch on representative subsets; NOT an experiment")
    args = parser.parse_args(argv)
    args.dw_dilation = depthwise_options(args)["dw_dilation"]
    if args.glioma_rl is None:
        args.glioma_rl = args.variant == "biref-ema"
    if args.policy_reward_fraction is None:
        args.policy_reward_fraction = .1 if args.variant == "biref-ema" or args.glioma_rl else 0.
    if args.augmentation is None:
        args.augmentation = "glioma" if args.variant == "biref-ema" or args.glioma_rl else ("robust" if args.variant == "biref" else "legacy")
    if args.selection_metric is None:
        args.selection_metric = "glioma-miou" if args.variant in {"biref-ema", "biref", "depthwise"} else "weighted-miou"
    if args.smoke_test:
        args.epochs, args.batch_size, args.size, args.workers = 1, 2, 64, 0
        args.benchmark_warmup, args.benchmark_steps = 1, 3
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.cudnn_benchmark is None:
        args.cudnn_benchmark = args.variant in {"biref-ema", "biref", "depthwise"} and args.device == "cuda"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; select --device cpu or install a CUDA-enabled PyTorch")
    if args.size < 64 or args.size % 32:
        parser.error("--size must be >=64 and divisible by 32")
    for name in ("epochs", "batch_size", "benchmark_warmup", "benchmark_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.workers < 0 or args.num_basis < 2 or args.seed < 0:
        parser.error("workers and seed must be nonnegative; num-basis must be >=2")
    if args.variant.startswith("kan-two-way"):
        if len(set(args.attention_stages)) != len(args.attention_stages) or not set(args.attention_stages) <= {4, 5, 6}:
            parser.error("--attention-stages must be a subset of 4 5 6 without duplicates")
        if min(args.attention_dim, args.attention_heads, args.attention_tokens, args.attention_depth) < 1:
            parser.error("Attention width, heads, tokens and depth must be positive")
        if args.attention_dim % 4 or args.attention_dim % (2 * args.attention_heads):
            parser.error("--attention-dim must be divisible by 4 and by twice --attention-heads")
        if not math.isfinite(args.attention_dropout) or not 0 <= args.attention_dropout < 1:
            parser.error("--attention-dropout must be finite and in [0,1)")
    if args.variant in {"biref-ema", "biref", "depthwise"} and (args.dw_kernel_size < 1 or args.dw_kernel_size % 2 == 0 or args.dw_dilation < 1):
        parser.error("--dw-kernel-size must be positive and odd; --dw-dilation must be positive")
    if not math.isfinite(args.max_vram_gb) or not 0 < args.max_vram_gb <= 10:
        parser.error("--max-vram-gb must be finite, positive and at most 10")
    if not math.isfinite(args.boundary_weight) or args.boundary_weight < 0:
        parser.error("--boundary-weight must be finite and nonnegative")
    if (not all(math.isfinite(v) for v in (args.policy_reward_fraction, args.rl_learning_rate, args.rl_exploration, args.glioma_sampling_weight))
            or not 0 <= args.policy_reward_fraction < 1 or args.rl_learning_rate <= 0
            or not 0 <= args.rl_exploration < 1 or args.rl_warmup < 0 or args.glioma_sampling_weight <= 0):
        parser.error("Invalid policy reward fraction, learning rate, exploration, warmup or sampling weight")
    if args.glioma_rl and (args.augmentation != "glioma" or args.policy_reward_fraction <= 0):
        parser.error("--glioma-rl requires --augmentation glioma and a positive --policy-reward-fraction")
    if args.variant == "biref-ema" and (len(set(args.ema_skip_stages)) != len(args.ema_skip_stages)
                                       or not set(args.ema_skip_stages) <= {1, 2, 3, 4, 5} or args.ema_skip_groups < 1):
        parser.error("EMA skip stages must be a subset of 1..5 without duplicates; groups must be positive")
    if not 0 < args.val_fraction < 1 or not 0.999 <= args.ema_decay_max <= 0.9995:
        parser.error("val-fraction must be in (0,1); ema-decay-max must be in [0.999,0.9995]")
    if any(not math.isfinite(v) for v in (args.lr, args.weight_decay, args.grad_clip, args.nsd_tolerance)):
        parser.error("Numeric hyperparameters must be finite")
    if args.lr <= 0 or args.weight_decay < 0 or args.grad_clip <= 0 or args.nsd_tolerance < 0:
        parser.error("lr and grad-clip must be positive; weight-decay and nsd-tolerance must be nonnegative")
    if args.evaluate_only and not args.resume:
        parser.error("--evaluate-only requires --resume")
    args.data_root = args.data_root.resolve()
    if args.resume:
        args.resume = args.resume.resolve()
    configure_run_paths(args, args.variant)
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    random.seed(args.seed)
    np.random.seed(args.seed % 2**32)
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    generator = torch.Generator().manual_seed(args.seed)
    # Loading is restricted to tensors and primitive containers written by this script.
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if checkpoint:
        immutable = MODEL_FIELDS
        if not args.evaluate_only:
            immutable += TRAIN_FIELDS
        for name in immutable:
            expected = checkpoint_setting(checkpoint["config"], name)
            if getattr(args, name) != expected:
                raise ValueError(f"--{name.replace('_', '-')} must match checkpoint ({expected})")
    if args.output.exists() and any(args.output.iterdir()):
        same_run = args.resume and args.resume.parent in (args.output, args.checkpoint_dir)
        if not same_run or args.evaluate_only:
            raise FileExistsError(f"Use a new/empty output folder or the resumed checkpoint's folder: {args.output}")
    create_run_dirs(args)
    train_pool, test_pool = discover(args.data_root, "train"), discover(args.data_root, "test")
    if checkpoint:
        train, val, test = restore_split(train_pool, test_pool, checkpoint["splits"])
    else:
        train, val = stratified_split(train_pool, args.val_fraction, args.seed)
        test = test_pool
    splits = split_manifest(train, val, test)
    policy_assignments, reward_samples = None, []
    if args.policy_reward_fraction > 0:
        if checkpoint and "policy_split" not in checkpoint:
            raise ValueError("Checkpoint is missing the reserved policy split")
        train, reward_samples, policy_assignments = split_policy_reward(
            train, args.policy_reward_fraction, args.seed, checkpoint["policy_split"] if checkpoint else None)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["glioma_miou_target"] = GLIOMA_MIOU_TARGET
    config["parameter_limit"] = MAX_PARAMETERS
    config["ema_tensor_updates"] = "unique" if args.variant.startswith("biref") else "legacy_state_entries"
    if policy_assignments is not None and not (args.output / "policy_split.json").exists():
        write_json(args.output / "policy_split.json", policy_assignments)
    if not (args.output / "config.json").exists():
        write_json(args.output / "config.json", {**config, "torch_version": str(torch.__version__), "python_version": platform.python_version(),
                   "ema_formula": "min(alpha_max, (1+t)/(10+t)); t = completed successful optimizer steps, first t=1",
                   "split_unit": "slice; patient identifiers unavailable"})
    if not (args.output / "splits.json").exists():
        write_json(args.output / "splits.json", splits)
    if args.smoke_test:
        train, val, test = (representative_subset(samples, limit) for samples, limit in ((train, 2), (val, 9), (test, 9)))
        if reward_samples:
            reward_samples = representative_subset(reward_samples, 3)
        print("SMOKE TEST: reduced data and resolution; scores are NOT experiment results.", flush=True)
    print(f"Device: {args.device}; output: {args.output}\nSlices: train={len(train)}, val={len(val)}, test={len(test)}", flush=True)
    for label, samples in (("train", train), ("val", val), ("test", test)):
        print(f"  {label}: {dict(Counter(s.tumor for s in samples))}")
    if reward_samples:
        print(f"  Policy reward holdout: {len(reward_samples)} glioma slices, excluded from segmentation training.", flush=True)
    memory_format = torch.channels_last if args.channels_last else torch.contiguous_format
    model = build_model(args.variant, args.groups, args.num_basis, **attention_options(args),
                        **depthwise_options(args), **biref_options(args), **ema_skip_options(args))
    parameters = (sum(p.numel() for p in model.parameters()) if args.evaluate_only else enforce_parameter_budget(model))
    model = model.to(device=args.device, memory_format=memory_format)
    print(f"Architecture: {args.variant}; parameters: {parameters:,} / {MAX_PARAMETERS:,}; VRAM budget: {args.max_vram_gb:g} GB", flush=True)
    if args.variant in {"biref-ema", "biref", "depthwise"}:
        print(f"  nnU-Net ResEnc + depthwise convolution at E4/E5/E6; kernel={args.dw_kernel_size}x{args.dw_kernel_size}, "
              f"dilation=({args.dw_dilation}, {args.dw_dilation}); skip connections; channels_last={args.channels_last}.", flush=True)
    if args.variant.startswith("biref"):
        print(f"  BiRef adaptation: context={args.biref_context}, detail={args.biref_detail}, boundary={args.biref_boundary}; "
              f"augmentation={args.augmentation}; boundary weight={args.boundary_weight:g}", flush=True)
    if args.variant == "biref-ema":
        print(f"  Efficient multi-scale attention skips: {args.ema_skip_stages}; groups={args.ema_skip_groups}.", flush=True)
    if getattr(model, "context_refiners", None):
        print(f"  Two-way context: stages={args.attention_stages}, dim={args.attention_dim}, "
              f"heads={args.attention_heads}, tokens={args.attention_tokens}, depth={args.attention_depth}", flush=True)
    print(f"  Checkpoint selection: validation {args.selection_metric}; glioma mIoU target >= {GLIOMA_MIOU_TARGET:.2f}", flush=True)
    model.decoder.deep_supervision = args.deep_supervision
    ema = WeightEMA(model, args.ema_decay_max)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.device == "cuda")
    start_epoch, best_score, best_epoch = 1, -math.inf, 0
    training_peak = {"allocated": None, "reserved": None}
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        best_score, best_epoch = checkpoint["best_score"], checkpoint["best_epoch"]
        training_peak = checkpoint["training_peak"]
        start_epoch = checkpoint["epoch"] + 1
        if not args.evaluate_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            restore_rng(checkpoint["rng"], generator)
    train_batches = loader(train, args, generator, training=True)
    val_batches = loader(val, args, torch.Generator().manual_seed(args.seed + 1))
    controller, reward_batches, policy_reward_score = None, None, None
    if args.glioma_rl and not args.evaluate_only:
        controller = AsyncGliomaPolicy(args.seed, args.rl_learning_rate, args.rl_exploration, args.rl_warmup)
        reward_batches = loader(reward_samples, args, torch.Generator().manual_seed(args.seed + 7))
        if checkpoint:
            controller.load_state_dict(checkpoint["glioma_controller"])
            policy_reward_score = checkpoint["policy_reward_score"]
        else:
            initial_reward, _ = evaluate(ema.model, reward_batches, args, "Initial policy reward EMA")
            policy_reward_score = initial_reward["per_class"]["Glioma"]["mIoU"]
    history_file = history_path(args)
    if not args.evaluate_only:
        if checkpoint:
            prepare_resume(args, checkpoint)
        for epoch in range(start_epoch, args.epochs + 1):
            policy_action, policy_log = None, None
            if controller is not None:
                policy_action = controller.choose(epoch)
                train_batches = loader(train, args, generator, training=True, policy=policy_action["augmentation"],
                                       sampling_weight=policy_action["glioma_sampling_weight"])
                print(f"  Glioma policy: {policy_action['augmentation']}; sampling weight={policy_action['glioma_sampling_weight']:g}", flush=True)
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            learning_rate = optimizer.param_groups[0]["lr"]
            train_loss = train_epoch(model, ema, train_batches, optimizer, scaler, args, epoch)
            synchronize(args)
            train_seconds = time.perf_counter() - start
            if controller is not None:
                reward_metrics, _ = evaluate(ema.model, reward_batches, args, "Policy reward EMA")
                current_reward_score = reward_metrics["per_class"]["Glioma"]["mIoU"]
                reward = controller.observe(policy_action, 100 * (current_reward_score - policy_reward_score),
                                            learn=not policy_action["warmup"])
                policy_reward_score = current_reward_score
                policy_log = {"action": policy_action, "reward": reward, "reward_mIoU": current_reward_score}
                # CPU policy learning runs while model-selection validation is evaluated.
            if args.device == "cuda":
                for key, amount in (("allocated", torch.cuda.max_memory_allocated()), ("reserved", torch.cuda.max_memory_reserved())):
                    training_peak[key] = max(training_peak[key] or 0, amount / 2**20)
            validation, _ = evaluate(ema.model, val_batches, args, "Validate EMA")
            display(validation, f"Epoch {epoch}: loss={train_loss:.5f}; EMA alpha={ema.decay:.6f}")
            score = validation_score(validation, args.selection_metric)
            improved = score > best_score
            if improved:
                best_score, best_epoch = score, epoch
            scheduler.step()
            state = {"epoch": epoch, "model": model.state_dict(), "ema": ema.state_dict(),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "best_score": best_score, "best_epoch": best_epoch, "config": config, "splits": splits,
                     "rng": rng_state(generator), "training_peak": training_peak, "validation": validation}
            if policy_assignments is not None:
                state["policy_split"] = policy_assignments
            if controller is not None:
                state["glioma_controller"] = controller.state_dict()
                state["policy_reward_score"] = policy_reward_score
                policy_log["preferences"] = state["glioma_controller"]["preferences"]
            atomic_save(checkpoint_path(args, "last.pt"), state)
            if improved:
                atomic_save(checkpoint_path(args, "best.pt"), state)
                write_json(args.output / "best_validation.json", validation)
            with history_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(json_safe({"epoch": epoch, "train_loss": train_loss, "lr": learning_rate,
                              "ema_alpha": ema.decay, "optimizer_steps": ema.num_updates, "train_seconds": train_seconds,
                              "epoch_seconds": time.perf_counter() - start, "validation": validation,
                              "glioma_policy": policy_log}), allow_nan=False) + "\n")
            glioma_miou = validation["per_class"]["Glioma"]["mIoU"]
            print(f"  Best validation {args.selection_metric}={best_score:.5f} at epoch {best_epoch}; checkpoint saved.", flush=True)
            print(f"  Glioma mIoU target >= {GLIOMA_MIOU_TARGET:.2f}: "
                  f"{'MET' if glioma_miou >= GLIOMA_MIOU_TARGET else 'not met'}"
                  f"{' (smoke test only)' if args.smoke_test else ''}", flush=True)
            del state
        if controller is not None:
            controller.close()
        selected_path = checkpoint_path(args, "best.pt")
        selected = torch.load(selected_path, map_location="cpu", weights_only=True)
        ema.load_state_dict(selected["ema"])
        selected_epoch = selected["epoch"]
        del selected
    else:
        selected_path, selected_epoch = args.resume, checkpoint["epoch"]
    # Release student weights, optimizer tensors and gradients before measuring inference VRAM.
    del model, optimizer, scheduler, scaler, checkpoint, train_batches, val_batches
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    test_batches = loader(test, args, torch.Generator().manual_seed(args.seed + 2))
    metrics, rows = evaluate(ema.model, test_batches, args, "Test EMA")
    efficiency = profile_efficiency(ema.model, args, training_peak)
    report = {
        "checkpoint": str(selected_path), "selected_epoch": selected_epoch,
        "selection": f"validation {args.selection_metric}" if not args.evaluate_only else "explicit --resume checkpoint",
        "split": "test", "smoke_test": args.smoke_test,
        "glioma_objective": {"target_mIoU": GLIOMA_MIOU_TARGET,
                             "observed_mIoU": metrics["per_class"]["Glioma"]["mIoU"],
                             "target_met": metrics["per_class"]["Glioma"]["mIoU"] >= GLIOMA_MIOU_TARGET,
                             "accuracy_experiment": not args.smoke_test},
        "evaluation_config": config,
        "metric_protocol": {
            "task": "binary foreground segmentation grouped by image-level tumor type",
            "overlap": "mIoU and Dice are per-image means; Weighted_mIoU uses tumor image counts as weights",
            "rates": "Recall/Precision/Specificity pool native-resolution pixel confusion counts; zero denominators are null",
            "coordinates": "original image size after removing letterbox padding and bilinear logit restoration",
            "distance_units": "native-image pixels, not mm; no physical spacing supplied",
            "HD95": "max of two directed contour-length-weighted 95th percentiles (surface-distance)",
            "ASSD": "pooled symmetric contour-length-weighted average distance",
            "NSD": "contour-length-weighted surface Dice", "NSD_tolerance_px": args.nsd_tolerance,
            "empty_masks": "one empty: distances Infinity, NSD 0; both empty: IoU/Dice 1, boundary metrics null",
            "aggregation": "boundary scores are per-image means; Infinity failures remain in headline means",
            "patient_grouping": "unavailable; training/validation split is slice-level, stratified by tumor and plane",
        },
        **metrics, "efficiency": efficiency,
    }
    # Evaluation-only writes separate files to preserve the original training report.
    prefix = "evaluation" if args.evaluate_only else "test"
    write_json(args.output / f"{prefix}_metrics.json", report)
    write_rows(args.output / f"{prefix}_per_image.csv", rows)
    display(metrics, f"Test results (EMA epoch {selected_epoch})")
    print(f"  Parameters: {efficiency['parameters']:,}; GFLOPs (supported ops): {efficiency['GFLOPs']:.4f}")
    print(f"  Inference: {efficiency['inference_ms_per_image']:.3f} ms/image (median, batch 1)")
    print(f"  Training peak VRAM: {efficiency['training_peak_allocated_MiB']} MiB; inference peak: {efficiency['inference_peak_allocated_MiB']} MiB")
    print(f"Reports saved in {args.output}", flush=True)


if __name__ == "__main__":
    main()
