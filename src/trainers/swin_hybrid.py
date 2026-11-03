"""Supervised large Swin/nnU-Net experiment with delayed EMA and early stopping."""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import gc
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from src.core.registry import build_model
from src.datasets.brisc import discover, representative_subset, restore_split, split_manifest, stratified_split
from .hybrid_loss import dynamic_loss, dynamic_weights
from src.utils.metrics import json_safe
from .baseline import (ROOT, WeightEMA, atomic_save, check_vram_budget, display, evaluate,
                    image_to_device, loader, profile_efficiency, restore_rng, rng_state, write_json, write_rows)


from src.core.paths import (DATA_ROOT, TOKENIZER_CHECKPOINT, configure_run_paths, checkpoint_path,
                            history_path, previous_history, create_run_dirs)

PROTOCOL = "swin_hybrid_512_unfrozen_v2"


class EarlyStopping:
    def __init__(self, start=100, patience=50, min_delta=1e-4):
        self.start, self.patience, self.min_delta = start, patience, min_delta
        self.anchor, self.bad_epochs = -math.inf, 0

    def update(self, score, epoch):
        if score > self.anchor + self.min_delta:
            self.anchor, self.bad_epochs = score, 0
        elif epoch > self.start:
            self.bad_epochs += 1
        return self.exhausted(epoch)

    def exhausted(self, epoch):
        return epoch > self.start and self.bad_epochs >= self.patience


def ema_for_epoch(model, ema, epoch, start, decay):
    return WeightEMA(model, decay) if ema is None and epoch >= start else ema


def recover_history(args, checkpoint):
    history = history_path(args)
    source = previous_history(args, checkpoint)
    rows, damaged_tail = [], False
    if source.exists():
        lines = source.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines)-1:
                    raise ValueError("Run history contains corruption before its final line")
                damaged_tail = True
                break
            if row["epoch"] > checkpoint["epoch"]:
                raise ValueError("History is newer than resumed checkpoint")
            if rows and row["epoch"] <= rows[-1]["epoch"]:
                raise ValueError("Run history epochs are not strictly increasing")
            rows.append(row)
        if damaged_tail:
            backup = args.output/f"history.interrupted.{time.time_ns()}.jsonl"
            backup.write_bytes(source.read_bytes())
    row = checkpoint.get("history_row")
    if row is not None and (not rows or rows[-1]["epoch"] < checkpoint["epoch"]):
        rows.append(row)
    if source.exists() or row is not None:
        temporary = history.with_suffix(".recovery.tmp")
        temporary.write_text("".join(json.dumps(json_safe(r), allow_nan=False)+"\n" for r in rows), encoding="utf-8")
        temporary.replace(history)


def recover_best(args, checkpoint):
    recover_history(args, checkpoint)
    if checkpoint["epoch"] == checkpoint["best_epoch"]:
        best = checkpoint
    else:
        best = torch.load(args.resume.parent/"best.pt", map_location="cpu", weights_only=True)
        if (best["epoch"] != checkpoint["best_epoch"] or best["best_score"] != checkpoint["best_score"]
                or best["splits"] != checkpoint["splits"] or best["tokenizer_sha256"] != checkpoint["tokenizer_sha256"]):
            raise ValueError("Best companion does not match resumed hybrid run")
    atomic_save(checkpoint_path(args, "best.pt"), best)
    write_json(args.output/"best_validation.json", best["validation"])


def write_status(output, **values):
    content = {"pid": os.getpid(), "updated_at": datetime.now().astimezone().isoformat(), **values}
    target = Path(output) / "status.json"
    temporary = target.with_suffix(".status.tmp")
    temporary.write_text(json.dumps(json_safe(content), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    for attempt in range(4):
        try:
            temporary.replace(target)
            return
        except PermissionError:
            time.sleep(.1 * (attempt + 1))


def make_model(args):
    return build_model("swin_hybrid", tokenizer_checkpoint=args.tokenizer_checkpoint, depthwise_kernel=3,
                       freeze_tokenizer=False, tokenizer_checkpointing=True).to(args.device)


def make_optimizer(model, args):
    swin = list(model.tokenizer.parameters())
    if not swin or not all(p.requires_grad for p in swin):
        raise ValueError("This protocol must fine-tune the entire Swin encoder")
    swin_ids = {id(p) for p in swin}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in swin_ids]
    return torch.optim.AdamW([
        {"params": backbone, "lr": args.lr, "name": "nnunet"},
        {"params": swin, "lr": args.lr*args.swin_lr_scale, "name": "swin"},
    ], weight_decay=args.weight_decay)


def make_scheduler(optimizer, epochs):
    # A shared cosine multiplier preserves Swin LR = 0.3 * backbone LR,
    # including the 1%-of-initial-LR floor for each parameter group.
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: .01 + .99*(1+math.cos(math.pi*min(epoch, epochs)/epochs))/2)


def train_one_epoch(model, ema, batches, optimizer, scaler, args, epoch, heartbeat=True):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation = args.effective_batch // args.batch_size
    loss_sum, seen, group_seen, successful = 0., 0, 0, 0
    component_sums = Counter()
    start = time.perf_counter()
    for index, (images, labels, _) in enumerate(batches):
        images = image_to_device(images, args)
        labels = labels.to(args.device, non_blocking=True)
        with torch.autocast(args.device, dtype=torch.float16, enabled=args.amp and args.device == "cuda"):
            loss, components = dynamic_loss(model(images), labels, epoch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite hybrid loss at epoch {epoch}, batch {index}")
        count = len(images)
        scaler.scale(loss * count / args.effective_batch).backward()
        group_seen += count
        if (index + 1) % accumulation == 0 or index + 1 == len(batches):
            scaler.unscale_(optimizer)
            if group_seen != args.effective_batch:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(args.effective_batch / group_seen)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                successful += 1
                if ema is not None:
                    ema.update(model)
            optimizer.zero_grad(set_to_none=True)
            group_seen = 0
        check_vram_budget(args)
        seen += count
        loss_sum += float(loss.detach()) * count
        for key, value in components.items():
            component_sums[key] += value * count
        if (index + 1) % 25 == 0 or index + 1 == len(batches):
            print(f"Epoch {epoch}/{args.epochs} batch {index+1}/{len(batches)} loss={loss_sum/seen:.5f} "
                  f"optimizer_steps={successful} EMA={'on' if ema else 'off'}", flush=True)
            if heartbeat:
                write_status(args.output, phase="training", epoch=epoch, batch=index+1, batches=len(batches),
                             loss=loss_sum/seen, seconds=time.perf_counter()-start,
                             allocated_GB=torch.cuda.max_memory_allocated()/1e9 if args.device == "cuda" else None,
                             reserved_GB=torch.cuda.max_memory_reserved()/1e9 if args.device == "cuda" else None)
    if not successful:
        raise FloatingPointError("No successful optimizer updates")
    return loss_sum/seen, {key: value/seen for key, value in component_sums.items()}, successful


def profile(args):
    torch.manual_seed(args.seed)
    model = make_model(args)
    # Include the future EMA allocation in calibration before the delayed start.
    ema = WeightEMA(model, args.ema_decay_max)
    optimizer = make_optimizer(model, args)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024, enabled=args.amp and args.device == "cuda")
    images = torch.randn(args.batch_size, 1, args.size, args.size)
    labels = (torch.rand(args.batch_size, args.size, args.size) > .9).long()
    labels[:, :8] = -1
    images[:, :, :8] = 0
    args.effective_batch = args.batch_size
    batches = [(images, labels, None)] * args.profile_steps
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    loss, components, steps = train_one_epoch(model, ema, batches, optimizer, scaler, args, max(150, args.ema_start), heartbeat=False)
    if args.device == "cuda":
        torch.cuda.synchronize()
    report = {"protocol": PROTOCOL, **model.parameter_counts, "batch_size": args.batch_size, "size": args.size,
              "successful_optimizer_steps": steps, "loss": loss, "components": components,
              "elapsed_seconds_including_cold_start": time.perf_counter()-start,
              "peak_allocated_GB": torch.cuda.max_memory_allocated()/1e9 if args.device == "cuda" else None,
              "peak_reserved_GB": torch.cuda.max_memory_reserved()/1e9 if args.device == "cuda" else None,
              "scope": "Cold forward/backward/AdamW plus future EMA allocation, trainable Swin and maximum scheduled boundary weight",
              "tokenizer_audit": model.tokenizer.audit}
    args.profile_output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.profile_output, report)
    print(json.dumps(report, indent=2), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--tokenizer-checkpoint", type=Path, default=TOKENIZER_CHECKPOINT)
    parser.add_argument("--output", type=Path, help="Run metadata folder; defaults to outputs/runs/<timestamp>_swin_hybrid")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--effective-batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--swin-lr-scale", type=float, default=.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=12.)
    parser.add_argument("--max-vram-gb", type=float, default=9.)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-fraction", type=float, default=.2)
    parser.add_argument("--augmentation", choices=("robust", "glioma", "legacy"), default="robust")
    parser.add_argument("--glioma-sampling-weight", type=float, default=2.)
    parser.add_argument("--pituitary-sampling-weight", type=float, default=1.3)
    parser.add_argument("--ema-start", type=int, default=20)
    parser.add_argument("--ema-decay-max", type=float, default=.9995)
    parser.add_argument("--early-start", type=int, default=100)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--nsd-tolerance", type=float, default=2.)
    parser.add_argument("--benchmark-warmup", type=int, default=3)
    parser.add_argument("--benchmark-steps", type=int, default=20)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--retry-empty-run", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--profile-output", type=Path, default=ROOT/"outputs/logs/calibration/swin_hybrid_512_v2.json")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke_test:
        # Keep the requested resolution and accumulation; only use nine images.
        args.epochs, args.workers = 2, 0
        args.benchmark_warmup, args.benchmark_steps = 1, 3
    if args.size < 64 or args.size % 32 or args.batch_size < 1 or args.effective_batch < 1 or args.effective_batch % args.batch_size:
        parser.error("Size must be >=64 divisible by 32; batch size must divide effective batch")
    if min(args.epochs, args.ema_start, args.patience, args.profile_steps) < 1 or args.early_start < 0 or args.workers < 0:
        parser.error("Invalid epoch, worker or stopping settings")
    if not 0 < args.max_vram_gb <= 10 or not 0 <= args.min_delta < 1:
        parser.error("VRAM budget must lie in (0,10] GB and min-delta in [0,1)")
    if not all(math.isfinite(x) for x in (args.lr, args.weight_decay, args.glioma_sampling_weight,
                                        args.pituitary_sampling_weight, args.swin_lr_scale, args.max_vram_gb)):
        parser.error("Hyperparameters must be finite")
    if min(args.lr, args.glioma_sampling_weight, args.pituitary_sampling_weight, args.swin_lr_scale) <= 0:
        parser.error("Learning rates and sampling weights must be positive")
    configure_run_paths(args, "swin_hybrid")
    args.data_root = args.data_root.resolve()
    args.tokenizer_checkpoint = args.tokenizer_checkpoint.resolve()
    return args


IMMUTABLE = ("epochs", "size", "effective_batch", "lr", "weight_decay", "seed", "val_fraction", "augmentation",
             "glioma_sampling_weight", "pituitary_sampling_weight", "swin_lr_scale", "ema_start",
             "ema_decay_max", "early_start", "patience", "min_delta", "smoke_test")


def validate_resume(args, checkpoint):
    if checkpoint["config"].get("protocol") != PROTOCOL:
        raise ValueError("Checkpoint belongs to a different training protocol; start a new 512/unfrozen run")
    for key in IMMUTABLE:
        if checkpoint["config"][key] != getattr(args, key):
            raise ValueError(f"Resume option {key} differs from checkpoint")


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2 if args.device == "cuda" else 4)
    torch.backends.cudnn.benchmark = args.device == "cuda"
    if args.profile_only:
        return profile(args)
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if checkpoint:
        validate_resume(args, checkpoint)
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        if not args.retry_empty_run or (history_path(args)).exists() or (checkpoint_path(args, "last.pt")).exists():
            raise FileExistsError("Use a new run directory or --resume")
    create_run_dirs(args)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(architecture="nnUNetSwinHybrid", protocol=PROTOCOL, kernel=3, depthwise_kernel=3,
                  freeze_tokenizer=False, tokenizer_checkpointing=True, parameter_limit=50_000_000,
                  scheduler="cosine shared LR multiplier; Swin/backbone ratio preserved",
                  weight_ema="disabled before ema_start", rl_enabled=False, policy_reward_holdout=False,
                  loss="alpha=1; beta=.5 to 1; gamma=.01 to .5 by epoch 150, then capped")
    write_json(args.output/"active_config.json", config)
    if not (args.output/"config.json").exists():
        write_json(args.output/"config.json", config)
    generator = torch.Generator().manual_seed(args.seed)
    source_train, source_test = discover(args.data_root, "train"), discover(args.data_root, "test")
    if checkpoint:
        train, val, test = restore_split(source_train, source_test, checkpoint["splits"])
    else:
        train, val = stratified_split(source_train, args.val_fraction, args.seed)
        test = source_test
    splits = split_manifest(train, val, test)
    write_json(args.output/"splits.json", splits)
    if args.smoke_test:
        train, val, test = (representative_subset(s, 9) for s in (train, val, test))
        print("SMOKE TEST ONLY: no accuracy claims.", flush=True)
    model = make_model(args)
    if checkpoint and model.tokenizer.audit["sha256"] != checkpoint["tokenizer_sha256"]:
        raise ValueError("Pretrained tokenizer checkpoint changed since training")
    write_json(args.output/"tokenizer_transfer.json", model.tokenizer.audit)
    write_json(args.output/"model_summary.json", {**model.parameter_counts, "architecture": "Swin + full nnU-Net ResEnc",
                                                 "depthwise_kernel": 3, "encoder_depths": [1, 3, 4, 6, 6, 6],
                                                 "trainable_swin_parameters": model.tokenizer.audit["trainable_parameters"]})
    print(f"MODEL {model.parameter_counts}; device={args.device}; batch={args.batch_size}; effective={args.effective_batch}; workers={args.workers}", flush=True)
    print(f"Slices fit={len(train)} val={len(val)} test={len(test)}; train classes={dict(Counter(s.tumor for s in train))}", flush=True)
    initial_swin_patch = model.tokenizer.encoder.patch_embed.proj.weight.detach().cpu().clone() if args.smoke_test else None
    optimizer = make_optimizer(model, args)
    scheduler = make_scheduler(optimizer, args.epochs)
    print(f"Optimizer groups: {[(g['name'], g['lr'], sum(p.numel() for p in g['params'])) for g in optimizer.param_groups]}", flush=True)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024, enabled=args.device == "cuda" and args.amp)
    early = EarlyStopping(args.early_start, args.patience, args.min_delta)
    ema, start_epoch, best_score, best_epoch, steps = None, 1, -math.inf, 0, 0
    training_peak = {"allocated": 0., "reserved": 0.}
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        if checkpoint["ema"] is not None:
            ema = WeightEMA(model, args.ema_decay_max)
            ema.load_state_dict(checkpoint["ema"])
        early.__dict__.update(checkpoint["early_stopping"])
        start_epoch, best_score, best_epoch = checkpoint["epoch"]+1, checkpoint["best_score"], checkpoint["best_epoch"]
        training_peak, steps = checkpoint["training_peak"], checkpoint["optimizer_steps"]
        restore_rng(checkpoint["rng"], generator)
        recover_best(args, checkpoint)
    train_batches = loader(train, args, generator, training=True)
    val_batches = loader(val, args, torch.Generator().manual_seed(args.seed+1))
    completed_epoch = start_epoch-1
    already_stopped = early.exhausted(completed_epoch)
    stop_reason = "early_stopping" if already_stopped else "epoch_budget"
    epochs = () if already_stopped else range(start_epoch, args.epochs+1)
    for epoch in epochs:
        previous_ema = ema
        ema = ema_for_epoch(model, ema, epoch, args.ema_start, args.ema_decay_max)
        if previous_ema is None and ema is not None:
            print(f"Weight EMA initialized from current student at epoch {epoch}.", flush=True)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        group_rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
        loss, components, updates = train_one_epoch(model, ema, train_batches, optimizer, scaler, args, epoch)
        steps += updates
        evaluation_model = ema.model if ema is not None else model
        write_status(args.output, phase="validation", epoch=epoch, best_glioma_mIoU=best_score)
        validation, _ = evaluate(evaluation_model, val_batches, args, f"Validation {'EMA' if ema else 'student'}")
        score = validation["per_class"]["Glioma"]["mIoU"]
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
        should_stop = early.update(score, epoch)
        scheduler.step()
        if args.device == "cuda":
            training_peak["allocated"] = max(training_peak["allocated"], torch.cuda.max_memory_allocated()/2**20)
            training_peak["reserved"] = max(training_peak["reserved"], torch.cuda.max_memory_reserved()/2**20)
        row = {"epoch": epoch, "train_loss": loss, "components": components, "lr": learning_rate,
               "learning_rates": group_rates,
               "optimizer_steps": steps, "ema_active": ema is not None, "ema_updates": ema.num_updates if ema else 0,
               "validation_weights": "EMA" if ema else "student", "validation": validation,
               "best_epoch": best_epoch, "best_glioma_mIoU": best_score, "early_bad_epochs": early.bad_epochs,
               "epoch_seconds": time.perf_counter()-start, "batch_size": args.batch_size, "workers": args.workers}
        state = {"epoch": epoch, "model": model.state_dict(), "ema": ema.state_dict() if ema else None,
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "evaluation_model": evaluation_model.state_dict(), "evaluation_weights": row["validation_weights"],
                 "best_score": best_score, "best_epoch": best_epoch, "optimizer_steps": steps,
                 "early_stopping": early.__dict__, "config": config, "splits": splits,
                 "rng": rng_state(generator), "training_peak": training_peak,
                 "history_row": row,
                 "validation": validation, "tokenizer_sha256": model.tokenizer.audit["sha256"]}
        atomic_save(checkpoint_path(args, "last.pt"), state)
        if improved:
            atomic_save(checkpoint_path(args, "best.pt"), state)
            write_json(args.output/"best_validation.json", validation)
        with (history_path(args)).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(json_safe(row), allow_nan=False)+"\n")
        display(validation, f"Epoch {epoch}: best glioma={best_score:.5f} @ {best_epoch}; bad={early.bad_epochs}/{args.patience}; components={components}")
        del state
        completed_epoch = epoch
        write_status(args.output, phase="epoch_complete", epoch=epoch, best_epoch=best_epoch, best_glioma_mIoU=best_score,
                     early_bad_epochs=early.bad_epochs, ema_active=ema is not None)
        if should_stop:
            stop_reason = "early_stopping"
            break
    smoke_checks = None
    if args.smoke_test:
        changed = not torch.equal(initial_swin_patch, model.tokenizer.encoder.patch_embed.proj.weight.detach().cpu())
        smoke_checks = {"completed_epochs": completed_epoch, "size": args.size,
                        "swin_patch_weight_changed": changed,
                        "trainable_swin_parameters": model.tokenizer.audit["trainable_parameters"],
                        "optimizer_group_lrs": {g["name"]: g["lr"] for g in optimizer.param_groups},
                        "swin_lr_ratio": optimizer.param_groups[1]["lr"]/optimizer.param_groups[0]["lr"],
                        "ema_start": args.ema_start, "ema_active": ema is not None}
        if not changed:
            raise AssertionError("Smoke test did not update the pretrained Swin patch embedding")
        write_json(args.output/"smoke_verification.json", smoke_checks)
    selected = torch.load(checkpoint_path(args, "best.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(selected["evaluation_model"], strict=True)
    selected_epoch, selected_weights = selected["epoch"], selected["evaluation_weights"]
    del selected, optimizer, scheduler, scaler, checkpoint, ema, train_batches, val_batches
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    write_status(args.output, phase="test", epoch=completed_epoch, selected_epoch=selected_epoch, selected_weights=selected_weights)
    test_batches = loader(test, args, torch.Generator().manual_seed(args.seed+2))
    metrics, rows = evaluate(model, test_batches, args, f"Final test {selected_weights} epoch {selected_epoch}")
    efficiency = profile_efficiency(model, args, training_peak)
    report = {"architecture": "nnUNetSwinHybrid", "protocol": PROTOCOL,
              "selected_epoch": selected_epoch, "selected_weights": selected_weights,
              "completed_epoch": completed_epoch, "stop_reason": stop_reason, "smoke_test": args.smoke_test,
              "selection": "validation glioma mIoU; test unused for tuning", "parameter_counts": model.parameter_counts,
              "tokenizer_transfer": model.tokenizer.audit, "training_config": config,
              "efficiency": efficiency, "smoke_checks": smoke_checks, **metrics}
    write_json(args.output/"test_metrics.json", report)
    write_rows(args.output/"test_per_image.csv", rows)
    text = (f"Swin/nnU-Net hybrid: {stop_reason} after {completed_epoch} epochs.\n"
            f"Selected {selected_weights} epoch {selected_epoch}; validation glioma mIoU {best_score:.5f}.\n"
            + "\n".join(f"Test {name}: mIoU {values['mIoU']:.5f}, Dice {values['Dice']:.5f}" for name, values in metrics["per_class"].items())
            + f"\nTest Weighted mIoU: {metrics['Weighted_mIoU']:.5f}.\nParameters: {model.parameter_counts}.\n"
            + f"Glioma target >=0.8: {metrics['per_class']['Glioma']['mIoU'] >= .8}.\n"
            + "3D checkpoint was transferred to a fully trainable 2D Swin encoder at 0.3x backbone LR by default.\n")
    (args.output/"summary.txt").write_text(text, encoding="utf-8")
    write_status(args.output, phase="completed", epoch=completed_epoch, stop_reason=stop_reason,
                 selected_epoch=selected_epoch, selected_weights=selected_weights,
                 test_glioma_mIoU=metrics["per_class"]["Glioma"]["mIoU"], weighted_mIoU=metrics["Weighted_mIoU"])
    print(text, flush=True)


def execute(args):
    try:
        run(args)
    except Exception as error:
        if not args.profile_only and not isinstance(error, FileExistsError):
            args.output.mkdir(parents=True, exist_ok=True)
            write_status(args.output, phase="failed", error=str(error), traceback=traceback.format_exc())
        raise


def main(argv=None):
    execute(parse_args(argv))


if __name__ == "__main__":
    main()
