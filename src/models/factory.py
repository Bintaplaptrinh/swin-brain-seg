"""Synthetic forward/backward/SGD benchmark; run from the project root.

python -m scripts.benchmark --variant depthwise --size 256 --amp --output outputs/runs/benchmark_depthwise.json
"""

import argparse
import gc
import json
import platform
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import torch
from torch.nn import functional as F

from .nnunet_resenc_kan import nnUNetResEncKAN
from .nnunet_resenc_depthwise import nnUNetResEncDepthwise
from .nnunet_biref import nnUNetBiRef, nnUNetBiRefEMASkip


VARIANTS = ("biref-ema", "biref", "depthwise", "kan-two-way", "kan-two-way-no-skips", "kan", "kan-no-skips", "kan-decoder", "conv", "resenc")
ATTENTION_DEFAULTS = {
    "attention_stages": [4, 5, 6], "attention_dim": 128, "attention_heads": 8,
    "attention_tokens": 16, "attention_depth": 2, "attention_dropout": 0.0,
}
DEPTHWISE_DEFAULTS = {"dw_kernel_size": 7, "dw_dilation": 3}
BIREF_DEFAULTS = {"biref_context": True, "biref_detail": True, "biref_boundary": True}
EMA_SKIP_DEFAULTS = {"ema_skip_stages": [3, 4, 5], "ema_skip_groups": 32}


def add_model_arguments(parser):
    parser.add_argument("--variant", choices=VARIANTS, default="biref-ema")
    parser.add_argument("--ema-skip-stages", type=int, nargs="*", default=[3, 4, 5])
    parser.add_argument("--ema-skip-groups", type=int, default=32)
    for name in BIREF_DEFAULTS:
        parser.add_argument("--" + name.replace("_", "-"), action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--groups", type=int, choices=(8, 16), default=8, help="Legacy KAN variants only")
    parser.add_argument("--num-basis", type=int, default=4, help="Legacy KAN variants only")
    parser.add_argument("--dw-kernel-size", type=int, default=7, help="Depthwise spatial kernel (positive odd integer)")
    parser.add_argument("--dw-dilation", type=int, default=None, help="Dilation on both axes; biref defaults to 1, legacy depthwise to 3")
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=False,
                        help="Use channels-last memory layout for convolution training")
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=None,
                        help="Tune convolution algorithms; defaults on for depthwise CUDA runs")
    parser.add_argument("--attention-stages", type=int, nargs="+", default=ATTENTION_DEFAULTS["attention_stages"],
                        help="One-based encoder stages for two-way context (subset of 4 5 6)")
    for name in ("attention_dim", "attention_heads", "attention_tokens", "attention_depth"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=ATTENTION_DEFAULTS[name])
    parser.add_argument("--attention-dropout", type=float, default=ATTENTION_DEFAULTS["attention_dropout"])


def attention_options(args):
    return {name: getattr(args, name) for name in ATTENTION_DEFAULTS}


def depthwise_options(args):
    options = {name: getattr(args, name) for name in DEPTHWISE_DEFAULTS}
    if options["dw_dilation"] is None:
        options["dw_dilation"] = 1 if args.variant.startswith("biref") else 3
    return options


def biref_options(args):
    return {name: getattr(args, name, default) for name, default in BIREF_DEFAULTS.items()}


def ema_skip_options(args):
    return {name: getattr(args, name, default) for name, default in EMA_SKIP_DEFAULTS.items()}


def segmentation_loss(outputs, target):
    """CE + foreground soft Dice, with normalized multiscale weights."""
    if isinstance(outputs, dict):
        from src.trainers.baseline import segmentation_loss as training_loss
        return training_loss(outputs, target)
    outputs = outputs if isinstance(outputs, list) else [outputs]
    weights = [2.0 ** -i for i in range(len(outputs))]
    loss = outputs[0].new_zeros((), dtype=torch.float32)
    for weight, logits in zip(weights, outputs):
        labels = F.interpolate(target[:, None].float(), size=logits.shape[-2:], mode="nearest")[:, 0].long()
        foreground = logits.float().softmax(1)[:, 1]
        labels_float = labels.float()
        dims = (1, 2)
        dice = (2 * (foreground * labels_float).sum(dims) + 1e-5) / (
            foreground.sum(dims) + labels_float.sum(dims) + 1e-5
        )
        loss = loss + weight * (F.cross_entropy(logits.float(), labels) + 1 - dice.mean())
    return loss / sum(weights)


def build_model(variant, groups=8, num_basis=4, **context_options):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown architecture variant: {variant}")
    spatial_options = {k: context_options.pop(k) for k in DEPTHWISE_DEFAULTS if k in context_options}
    reference_options = {k: context_options.pop(k) for k in BIREF_DEFAULTS if k in context_options}
    skip_options = {k: context_options.pop(k) for k in EMA_SKIP_DEFAULTS if k in context_options}
    if set(context_options) - ATTENTION_DEFAULTS.keys():
        raise ValueError("Unknown attention option")
    if variant == "depthwise":
        return nnUNetResEncDepthwise(deep_supervision=True, **spatial_options)
    if variant == "biref":
        return nnUNetBiRef(deep_supervision=True, **spatial_options, **reference_options)
    if variant == "biref-ema":
        return nnUNetBiRefEMASkip(deep_supervision=True, **spatial_options, **reference_options, **skip_options)
    kwargs = dict(kan_groups=groups, num_basis=num_basis, deep_supervision=True)
    if variant.startswith("kan-two-way"):
        kwargs.update(ATTENTION_DEFAULTS | context_options)
    if variant in {"resenc", "conv"}:
        kwargs.update(use_kan_encoder=False, skip_kan_stages=())
    if variant == "resenc":
        kwargs["n_blocks_per_stage"] = (1, 3, 4, 6, 6, 6)
    if variant in {"kan-no-skips", "kan-two-way-no-skips"}:
        kwargs["skip_kan_stages"] = ()
    if variant == "kan-decoder":
        kwargs["decoder_kan_stages"] = (4, 5)
    return nnUNetResEncKAN(**kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Use CUDA float16 autocast + GradScaler.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1 or args.batch_size < 1:
        parser.error("steps, warmup and batch-size must be positive.")
    if args.size < 64 or args.size % 32:
        parser.error("size must be >=64 and divisible by 32.")
    if args.device == "cpu" and args.amp:
        parser.error("--amp requires --device cuda.")
    if args.cudnn_benchmark is None:
        args.cudnn_benchmark = args.variant in {"biref-ema", "biref", "depthwise"} and args.device == "cuda"
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    device = torch.device(args.device)
    torch.manual_seed(2026)
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    # Measure eager execution; compile time and data loading are excluded.
    memory_format = torch.channels_last if args.channels_last else torch.contiguous_format
    model = build_model(args.variant, args.groups, args.num_basis, **attention_options(args),
                        **depthwise_options(args), **biref_options(args), **ema_skip_options(args)).to(device=device, memory_format=memory_format).train()
    parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.99, weight_decay=3e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    images = torch.randn(args.batch_size, 1, args.size, args.size, device=device).contiguous(memory_format=memory_format)
    # Sparse foreground resembles segmentation more closely than uniform labels.
    target = (torch.rand(args.batch_size, args.size, args.size, device=device) > 0.9).long()

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
            loss = segmentation_loss(model(images), target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss

    for _ in range(args.warmup):
        step()
    synchronize()
    # Clear warmup gradients before resetting the allocator peak; momentum
    # buffers stay allocated, reflecting steady-state training memory.
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(args.steps):
        synchronize()
        start = time.perf_counter()
        loss = step()
        synchronize()
        times.append(1000 * (time.perf_counter() - start))
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    finite = bool(torch.isfinite(loss).item()) and all(bool(torch.isfinite(g).all().item()) for g in grads)
    if not finite:
        raise RuntimeError("Nonfinite loss or gradients in benchmark.")
    result = {
        "variant": args.variant,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "dynamic_network_architectures": version("dynamic-network-architectures"),
        "input_shape": list(images.shape), "amp": args.amp,
        "groups": args.groups, "num_basis": args.num_basis,
        "attention_enabled": bool(getattr(model, "context_refiners", None)),
        "depthwise_enabled": args.variant in {"biref-ema", "biref", "depthwise"},
        **ema_skip_options(args),
        **biref_options(args),
        **depthwise_options(args), "channels_last": args.channels_last,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        **attention_options(args),
        "deep_supervision": True, "optimizer": "SGD(momentum=0.99)",
        "warmup_steps": args.warmup, "measured_steps": args.steps,
        "parameters": parameters, "median_step_ms": statistics.median(times),
        "min_step_ms": min(times), "max_step_ms": max(times),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
        "final_loss": float(loss.detach()), "finite_gradients": finite,
        "scope": "Synthetic eager CE+Dice forward/backward/optimizer; excludes data loading and validation.",
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
