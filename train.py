"""Configuration-driven entry point for Swin Hybrid and retained baselines."""
import argparse
import json
from pathlib import Path
import sys

from src.core.config import load_config, project_path
from src.core.paths import ROOT
from src.datasets.download import ensure_dataset


def configured_arguments(config):
    model = config["model"]["name"]
    values = dict(config["trainer"])
    optimizer = values.pop("optimizer", {"name": "AdamW"})
    scheduler = values.pop("lr_scheduler", {"name": "cosine", "minimum_lr_ratio": .01})
    if optimizer["name"].lower() != "adamw":
        raise ValueError("This checkpoint protocol supports the AdamW optimizer")
    if scheduler != {"name": "cosine", "minimum_lr_ratio": .01}:
        raise ValueError("This checkpoint protocol uses cosine decay to 1% of initial LR")
    values.update({k: v for k, v in optimizer.items() if k != "name"})
    values.update(config["model"].get("options", {}))
    values["data_root"] = str(project_path(config["dataset"]["root"]))
    values["size"] = config["dataset"]["input"]["size"]
    split = config["dataset"].get("split", {})
    values["seed"] = split.get("seed", values.get("seed", 2026))
    values["val_fraction"] = split.get("validation_fraction", values.get("val_fraction", .2))
    if model == "swin_hybrid":
        values["tokenizer_checkpoint"] = str(project_path(config["model"]["pretrained_checkpoint"]))
    else:
        values["variant"] = "depthwise" if model == "nnunet_v2_depthwise" else model
    if config.get("output"):
        values["output"] = str(project_path(config["output"]))
    arguments = []
    for key, value in values.items():
        if value is None:
            continue
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            # store_true actions have no --no-* counterpart.
            if key in {"smoke_test", "evaluate_only", "profile_only", "retry_empty_run"}:
                if value:
                    arguments.append(option)
            else:
                arguments.append(option if value else "--no-" + key.replace("_", "-"))
        elif isinstance(value, list):
            arguments += [option, *map(str, value)]
        else:
            arguments += [option, str(value)]
    return arguments


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", choices=("swin_hybrid", "kan", "depthwise", "nnunet_v2_depthwise"))
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("-h", "--help", action="store_true", help="Show configuration and trainer options")
    cli, remaining = parser.parse_known_args(argv)
    preset = "depthwise" if cli.model == "nnunet_v2_depthwise" else cli.model
    path = cli.config or ROOT / "configs" / (f"{preset}.yaml" if preset and preset != "swin_hybrid" else "default.yaml")
    config = load_config(path)
    requested_model = "depthwise" if cli.model == "nnunet_v2_depthwise" else cli.model
    if requested_model and requested_model != config["model"]["name"]:
        parser.error("--model differs from the selected config; select a matching config or omit --config")
    if cli.print_config:
        print(json.dumps(config, indent=2))
        return
    model = config["model"]["name"]
    if model == "swin_hybrid":
        from src.trainers.swin_hybrid import parse_args, execute
    else:
        from src.trainers.baseline import parse_args, main as baseline_main
    arguments = configured_arguments(config) + remaining
    if cli.help:
        parser.print_help()
        print("\nTrainer options:")
        arguments.append("--help")
    # Parse first so --help never triggers a dataset download.
    args = parse_args(arguments)
    ensure_dataset(args.data_root, auto_download=config["dataset"].get("auto_download", False),
                   kaggle_handle=config["dataset"].get("kaggle_handle", "briscdataset/brisc2025"))
    if model == "swin_hybrid":
        execute(args)
    else:
        baseline_main(arguments)


if __name__ == "__main__":
    main(sys.argv[1:])
