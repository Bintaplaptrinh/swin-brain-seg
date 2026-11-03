import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.config import load_config
from src.core.paths import ROOT, configure_run_paths
from train import configured_arguments


def test_json_and_yaml_composition_have_cli_override_semantics(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"epochs": 300, "lr": .0003}))
    config = tmp_path / "run.yaml"
    config.write_text("defaults:\n  trainer: base.json\ntrainer:\n  epochs: 2\n")
    assert load_config(config)["trainer"] == {"epochs": 2, "lr": .0003}
    config.write_text("defaults:\n  trainer: run.yaml\n")
    with pytest.raises(ValueError, match="Cyclic"):
        load_config(config)


@pytest.mark.parametrize("preset,expected_size", [("default", 512), ("kan", 256), ("depthwise", 256)])
def test_presets_parse_with_their_actual_trainers(preset, expected_size):
    config = load_config(ROOT / "configs" / f"{preset}.yaml")
    if preset == "default":
        from src.trainers.swin_hybrid import parse_args
    else:
        from src.trainers.baseline import parse_args
    args = parse_args(configured_arguments(config) + ["--device", "cpu", "--epochs", "7"])
    assert args.epochs == 7 and args.size == expected_size
    assert args.output.parent == ROOT / "outputs/runs"
    assert args.checkpoint_dir == ROOT / "outputs/checkpoints" / args.output.name
    assert args.log_dir == ROOT / "outputs/logs" / args.output.name


def test_resume_finds_same_run_after_checkpoint_separation(tmp_path):
    for output in (tmp_path / "outputs/runs/run_123", tmp_path / "custom_run"):
        fresh = SimpleNamespace(output=output, resume=None, smoke_test=False)
        configure_run_paths(fresh, "swin_hybrid")
        resume = SimpleNamespace(output=None, resume=fresh.checkpoint_dir / "last.pt", smoke_test=False)
        configure_run_paths(resume, "swin_hybrid")
        assert resume.output == fresh.output
        assert resume.checkpoint_dir == fresh.checkpoint_dir
