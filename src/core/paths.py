"""Portable project paths and separate run/checkpoint/log destinations."""
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data/data/brisc2025"
TOKENIZER_CHECKPOINT = ROOT / "outputs/checkpoints/pretrained/Brain_Swin_UNETR.pth"


def configure_run_paths(args, model):
    if args.output is None:
        if args.resume and not getattr(args, "evaluate_only", False):
            folder = args.resume.resolve().parent
            args.output = (folder.parent.parent / "runs" / folder.name
                           if folder.parent.name == "checkpoints" else
                           folder.parent if folder.name == "checkpoints" else folder)
        else:
            suffix = "_smoke" if args.smoke_test else ""
            args.output = ROOT / "outputs/runs" / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{model}{suffix}"
    args.output = args.output.resolve()
    if args.output.parent.name == "runs":
        base = args.output.parent.parent
        args.checkpoint_dir = base / "checkpoints" / args.output.name
        args.log_dir = base / "logs" / args.output.name
    else:
        args.checkpoint_dir = args.output / "checkpoints"
        args.log_dir = args.output / "logs"
    return args


def checkpoint_path(args, name):
    return Path(getattr(args, "checkpoint_dir", args.output)) / name


def history_path(args):
    return Path(getattr(args, "log_dir", args.output)) / "history.jsonl"


def previous_history(args, checkpoint):
    candidates = [history_path(args)]
    if args.resume:
        folder = args.resume.resolve().parent
        candidates += [folder.parent.parent / "logs" / folder.name / "history.jsonl",
                       folder.parent / "logs/history.jsonl",
                       folder / "history.jsonl"]
    configured = checkpoint.get("config", {}).get("log_dir")
    if configured:
        candidates.append(Path(configured) / "history.jsonl")
    return next((p for p in candidates if p.exists()), candidates[0])


def create_run_dirs(args):
    if not args.resume and (checkpoint_path(args, "best.pt").exists()
                            or checkpoint_path(args, "last.pt").exists() or history_path(args).exists()):
        raise FileExistsError("Checkpoint/log destinations already contain a run; choose a new --output or --resume")
    for path in (args.output, args.checkpoint_dir, args.log_dir):
        path.mkdir(parents=True, exist_ok=True)


def relocate_config(config):
    """Read historical checkpoints without modifying their original bytes."""
    config = dict(config)
    if not Path(config.get("data_root", "__missing__")).is_dir():
        config["data_root"] = str(DATA_ROOT)
    if not Path(config.get("tokenizer_checkpoint", "__missing__")).is_file():
        config["tokenizer_checkpoint"] = str(TOKENIZER_CHECKPOINT)
    return config
