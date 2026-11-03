"""Evaluate retained checkpoints on their saved test membership without training."""
import argparse
import gc
from pathlib import Path
import torch

from src.core.paths import ROOT, relocate_config
from src.datasets.brisc import discover, restore_split
from src.models.factory import (attention_options, biref_options, build_model,
                                depthwise_options, ema_skip_options)
from src.trainers.baseline import (MODEL_FIELDS, checkpoint_setting, evaluate, loader,
                                   parse_args as baseline_args, profile_efficiency, write_json, write_rows)
from src.trainers.swin_hybrid import PROTOCOL, make_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--benchmark", action="store_true")
    cli = parser.parse_args(argv)
    if (cli.output / "test_metrics.json").exists():
        raise FileExistsError("Evaluation report already exists; choose a new --output")
    checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=True)
    config = relocate_config(checkpoint["config"])
    if config.get("smoke_test"):
        raise ValueError("Use a full-data checkpoint for benchmark evaluation")
    torch.set_num_threads(2 if cli.device == "cuda" else 4)
    torch.backends.cudnn.benchmark = cli.device == "cuda"
    if config.get("protocol") == PROTOCOL:
        args = argparse.Namespace(**config)
        args.tokenizer_checkpoint = Path(config["tokenizer_checkpoint"])
        args.device = cli.device
        model = make_model(args)
        state = checkpoint["evaluation_model"]
        weights = checkpoint["evaluation_weights"]
    else:
        args = baseline_args(["--variant", config["variant"], "--device", cli.device])
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
        for key in MODEL_FIELDS:
            setattr(args, key, checkpoint_setting(config, key))
        model = build_model(args.variant, args.groups, args.num_basis, **attention_options(args),
                            **depthwise_options(args), **biref_options(args), **ema_skip_options(args))
        model.decoder.deep_supervision = args.deep_supervision
        state, weights = checkpoint["ema"]["model"], "EMA"
    args.device, args.batch_size, args.workers = cli.device, cli.batch_size, 0
    args.benchmark_warmup, args.benchmark_steps = 3, 20
    args.channels_last = False
    model.load_state_dict(state, strict=True)
    model.to(cli.device).eval()
    epoch = checkpoint["epoch"]
    train_peak = checkpoint.get("training_peak", {"allocated": None, "reserved": None})
    _, _, test = restore_split(discover(config["data_root"], "train"), discover(config["data_root"], "test"), checkpoint["splits"])
    del state, checkpoint
    gc.collect()
    if cli.device == "cuda":
        torch.cuda.empty_cache()
    cli.output.mkdir(parents=True, exist_ok=True)
    metrics, rows = evaluate(model, loader(test, args, torch.Generator().manual_seed(args.seed + 2)), args,
                             f"Test {cli.checkpoint.parent.name} epoch {epoch} {weights}")
    report = {"checkpoint": str(cli.checkpoint.resolve()), "selected_epoch": epoch,
              "selected_weights": weights, "selection": "explicit checkpoint; no test-set tuning",
              "split": "test", "smoke_test": False, "training_config": config,
              "evaluation_config": {"size": args.size, "device": cli.device, "batch_size": args.batch_size,
                                    "amp": args.amp, "workers": 0}, **metrics}
    if cli.benchmark:
        report["efficiency"] = profile_efficiency(model, args, train_peak)
    write_json(cli.output / "test_metrics.json", report)
    write_rows(cli.output / "test_per_image.csv", rows)
    print(f"epoch={epoch}; weights={weights}; weighted_mIoU={metrics['Weighted_mIoU']:.8f}; output={cli.output}", flush=True)


if __name__ == "__main__":
    main()
