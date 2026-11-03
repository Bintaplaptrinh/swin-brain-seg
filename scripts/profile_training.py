"""Bounded synthetic training-memory check including AdamW, EMA and auxiliary loss.

This does not load the dataset or launch an accuracy experiment.
"""

import argparse
import json
from pathlib import Path
import time

import torch

from src.models.factory import (add_model_arguments, attention_options, depthwise_options,
                                    biref_options, ema_skip_options, build_model)
from src.models.budget import enforce_parameter_budget
from src.trainers.baseline import WeightEMA, check_vram_budget, train_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-vram-gb", type=float, default=10)
    parser.add_argument("--boundary-weight", type=float, default=.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.size < 64 or args.size % 32 or min(args.batch_size, args.steps) < 1:
        parser.error("Use positive batch/step counts and size >=64 divisible by 32")
    if not 0 < args.max_vram_gb <= 10:
        parser.error("VRAM budget must be positive and at most 10 GB")
    args.epochs, args.grad_clip = 1, 12
    torch.manual_seed(2026)
    if args.device == "cpu":
        torch.set_num_threads(4)
    if args.cudnn_benchmark is None:
        args.cudnn_benchmark = args.device == "cuda" and args.variant in {"biref-ema", "biref", "depthwise"}
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    memory_format = torch.channels_last if args.channels_last else torch.contiguous_format
    model = build_model(args.variant, args.groups, args.num_basis, **attention_options(args),
                        **depthwise_options(args), **biref_options(args), **ema_skip_options(args))
    parameters = enforce_parameter_budget(model)
    model = model.to(device=args.device, memory_format=memory_format)
    ema = WeightEMA(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scaler = torch.amp.GradScaler("cuda", init_scale=128, enabled=args.device == "cuda" and args.amp)
    image = torch.randn(args.batch_size, 1, args.size, args.size)
    target = (torch.rand(args.batch_size, args.size, args.size) > .9).long()
    image[:, :, :8] = 0
    target[:, :8] = -1
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    loss = train_epoch(model, ema, [(image, target, None)] * args.steps, optimizer, scaler, args, epoch=1)
    if args.device == "cuda":
        torch.cuda.synchronize()
    check_vram_budget(args)
    if ema.num_updates != args.steps:
        raise RuntimeError("Some synthetic updates overflowed; resource check requires all updates to succeed")
    result = {"variant": args.variant, "parameters": parameters, "parameter_limit": 10_000_000,
              "input_shape": list(image.shape), "amp": args.amp, "optimizer": "AdamW(lr=.001, weight_decay=.0001)",
              "EMA_updates": ema.num_updates, "boundary_weight": args.boundary_weight,
              "EMA_unique_tensor_updates": ema.unique_updates,
              "dw_options": depthwise_options(args), "biref_options": biref_options(args),
              "ema_skip_options": ema_skip_options(args),
              "channels_last": args.channels_last, "cudnn_benchmark": args.cudnn_benchmark,
              "elapsed_seconds_including_warmup": time.perf_counter()-start, "final_loss": loss,
              "peak_allocated_GB": torch.cuda.max_memory_allocated()/1e9 if args.device == "cuda" else None,
              "peak_reserved_GB": torch.cuda.max_memory_reserved()/1e9 if args.device == "cuda" else None,
              "vram_budget_GB": args.max_vram_gb,
              "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
              "scope": "Synthetic train_epoch including student, AdamW, gradients, EMA, deep supervision and boundary loss; no dataset training. PyTorch memory excludes CUDA context and other processes."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
