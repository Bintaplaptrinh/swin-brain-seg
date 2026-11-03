import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from src.datasets.brisc import (BRISCDataset, Sample, discover, geometry, load_mask,
                      restore_split, split_manifest, stratified_split)
from src.utils.metrics import binary_metrics, json_safe, summarize
from src.trainers.baseline import (MODEL_FIELDS, TRAIN_FIELDS, WeightEMA, evaluate, prepare_resume,
                       atomic_save, checkpoint_setting, parse_args, restore_rng, rng_state, segmentation_loss, train_epoch,
                       validation_score, GLIOMA_MIOU_TARGET)


def test_ema_warmup_uses_optimizer_steps_and_restores_state():
    model = nn.BatchNorm1d(2)
    model.weight.data.zero_()
    ema = WeightEMA(model)
    model.weight.data.fill_(2)
    model.num_batches_tracked.fill_(7)
    ema.update(model)
    assert ema.num_updates == 1
    assert ema.decay == pytest.approx(2 / 11)
    torch.testing.assert_close(ema.model.weight, torch.full((2,), 2 * 9 / 11))
    assert ema.model.num_batches_tracked.item() == 7
    assert not any(p.requires_grad for p in ema.model.parameters())
    restored = WeightEMA(model)
    restored.load_state_dict(ema.state_dict())
    model.weight.data.fill_(4)
    ema.update(model)
    restored.update(model)
    assert restored.num_updates == 2 and restored.decay == 0.25
    torch.testing.assert_close(ema.model.weight, restored.model.weight)
    restored.num_updates = 20000
    restored.update(model)
    assert restored.decay == 0.9995


@pytest.mark.parametrize("ceiling", [0.998, 0.9999, float("nan")])
def test_ema_rejects_out_of_range_decay(ceiling):
    with pytest.raises(ValueError):
        WeightEMA(nn.Linear(1, 1), ceiling)


def test_surface_metrics_perfect_shifted_and_empty_masks():
    target = np.zeros((40, 40), dtype=bool)
    target[10:20, 10:20] = True
    perfect = binary_metrics(target, target)
    assert perfect["iou"] == perfect["dice"] == perfect["nsd"] == 1
    assert perfect["hd95"] == perfect["assd"] == 0
    shifted = np.roll(target, 2, axis=1)
    strict = binary_metrics(shifted, target, tolerance=0)
    tolerant = binary_metrics(shifted, target, tolerance=2)
    assert strict["hd95"] == pytest.approx(2)
    assert 0 < strict["assd"] <= 2
    assert strict["nsd"] < tolerant["nsd"] == 1
    empty = np.zeros_like(target)
    miss = binary_metrics(empty, target)
    assert miss["iou"] == miss["dice"] == miss["nsd"] == 0
    assert math.isinf(miss["hd95"]) and math.isinf(miss["assd"])
    both = binary_metrics(empty, empty)
    assert both["iou"] == both["dice"] == 1
    assert both["hd95"] is both["assd"] is both["nsd"] is None


def test_weighting_rates_and_infinite_failures_are_not_hidden():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:4, 2:4] = True
    rows = []
    for tumor, prediction in (("Glioma", mask), ("Glioma", ~mask), ("Meningioma", np.zeros_like(mask)), ("Pituitary", mask)):
        rows.append({"tumor": tumor, "plane": "Axial", **binary_metrics(prediction, mask)})
    result = summarize(rows)
    assert result["per_class"]["Glioma"]["mIoU"] == 0.5
    assert result["Weighted_mIoU"] == 0.5
    assert result["overall"]["Recall"] == 0.5
    assert result["overall"]["Precision"] == pytest.approx(8 / 104)
    assert result["per_plane"]["Coronal"]["images"] == 0
    assert math.isinf(result["overall"]["HD95_px"])
    assert result["overall"]["infinite_distance_images"] == 1
    encoded = json.dumps(json_safe(result), allow_nan=False)
    assert json.loads(encoded)["overall"]["HD95_px"] == "Infinity"


def fake_samples():
    return [Sample(Path(f"{tumor}_{plane}_{i}.jpg"), Path("mask.png"), tumor, plane)
            for tumor in ("Glioma", "Meningioma", "Pituitary")
            for plane in ("Axial", "Coronal", "Sagittal") for i in range(5)]


def test_split_disjoint_reproducible_and_roundtrips():
    pool = fake_samples()
    train, val = stratified_split(pool, 0.2, 2026)
    assert len(train) == 36 and len(val) == 9
    assert not {s.image.name for s in train} & {s.image.name for s in val}
    assert {s.stratum for s in val} == {s.stratum for s in pool}
    assert (train, val) == stratified_split(pool, 0.2, 2026)
    test = [Sample(Path("test.jpg"), Path("test.png"), "Glioma", "Axial")]
    manifest = split_manifest(train, val, test)
    assert restore_split(pool, test, manifest) == (train, val, test)
    manifest["val"][0] = manifest["train"][0]
    with pytest.raises(ValueError):
        restore_split(pool, test, manifest)


def test_threshold_letterbox_and_native_evaluation(tmp_path):
    image = np.full((32, 64), 100, dtype=np.uint8)
    mask = np.zeros_like(image)
    mask[8:24, 16:48] = 255
    mask[0, :3] = [1, 127, 128]
    image_path, mask_path = tmp_path / "image.jpg", tmp_path / "mask.png"
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)
    sample = Sample(image_path, mask_path, "Glioma", "Coronal")
    target = load_mask(mask_path)
    assert target[0, :3].tolist() == [False, False, True]
    dataset = BRISCDataset([sample], size=64)
    inputs, labels, index = dataset[0]
    assert inputs.shape == (1, 64, 64) and labels.shape == (64, 64) and index == 0
    assert geometry(32, 64, 64) == (32, 64, 16, 0)
    assert torch.all(labels[:16] == -1) and torch.all(labels[48:] == -1)

    class FixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = SimpleNamespace(deep_supervision=True)

        def forward(self, tensor):
            foreground = (labels == 1).to(tensor.device).float() * 20 - 10
            return torch.stack((-foreground, foreground), dim=0)[None].expand(tensor.shape[0], -1, -1, -1)

    model = FixedModel().train()
    args = SimpleNamespace(device="cpu", amp=False, size=64, nsd_tolerance=2)
    metrics, rows = evaluate(model, torch.utils.data.DataLoader(dataset), args, "Unit native restoration")
    assert metrics["Weighted_mIoU"] == 1
    assert rows[0]["tp"] == int(target.sum())
    assert sum(rows[0][key] for key in ("tp", "fp", "fn", "tn")) == target.size
    assert model.training and model.decoder.deep_supervision


def test_loss_ignores_padding_in_loss_and_gradients():
    logits = torch.randn(2, 2, 8, 8, requires_grad=True)
    targets = torch.zeros((2, 8, 8), dtype=torch.long)
    targets[:, 2:5, 2:5] = 1
    targets[:, :2] = -1
    changed = logits.detach().clone()
    changed[:, :, :2] = 100 * torch.randn_like(changed[:, :, :2])
    loss = segmentation_loss(logits, targets)
    torch.testing.assert_close(loss, segmentation_loss(changed, targets))
    loss.backward()
    assert torch.all(logits.grad[:, :, :2] == 0)
    assert torch.isfinite(logits.grad).all()
    multiscale = segmentation_loss([logits.detach(), torch.randn(2, 2, 4, 4)], targets)
    assert torch.isfinite(multiscale)


def test_rng_roundtrip_restores_loader_and_augmentation_streams():
    import random
    generator = torch.Generator().manual_seed(19)
    state = rng_state(generator)
    expected = (random.random(), np.random.random(), torch.rand(1), torch.rand(1, generator=generator))
    restore_rng(state, generator)
    actual = (random.random(), np.random.random(), torch.rand(1), torch.rand(1, generator=generator))
    assert expected[:2] == actual[:2]
    torch.testing.assert_close(expected[2], actual[2])
    torch.testing.assert_close(expected[3], actual[3])


def test_resume_keeps_prior_best_across_multiple_resumes(tmp_path):
    source, destination = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    destination.mkdir()
    config = {key: 1 for key in MODEL_FIELDS + TRAIN_FIELDS}
    best = {"epoch": 2, "best_epoch": 2, "best_score": 0.8, "splits": {"val": ["v.jpg"]},
            "config": {**config, "resume": None}, "validation": {"Weighted_mIoU": 0.8}}
    torch.save(best, source / "best.pt")
    checkpoint = {**best, "epoch": 3, "config": {**config, "resume": "last.pt"}}
    args = SimpleNamespace(output=destination, resume=source / "last.pt")
    prepare_resume(args, checkpoint)
    saved = torch.load(destination / "best.pt", weights_only=True)
    assert saved["epoch"] == 2
    assert json.loads((destination / "best_validation.json").read_text())["Weighted_mIoU"] == 0.8
    (destination / "history.jsonl").write_text('{"epoch": 4}\n')
    with pytest.raises(ValueError, match="newer"):
        prepare_resume(args, checkpoint)


def test_checkpoint_retries_transient_lock_without_rewriting_or_losing_old_state(tmp_path, monkeypatch):
    target = tmp_path / "best.pt"
    torch.save({"epoch": 21}, target)
    stale = tmp_path / "best.pt.tmp"
    stale.write_bytes(b"left from previous save")
    original_replace = Path.replace
    attempts, sleeps = [], []

    def locked_replace(source, destination):
        attempts.append(source)
        assert torch.load(source, weights_only=True)["epoch"] == 22
        assert torch.load(target, weights_only=True)["epoch"] == 21
        if len(attempts) < 3:
            raise PermissionError("Windows sharing violation")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", locked_replace)
    monkeypatch.setattr("src.trainers.baseline.time.sleep", sleeps.append)
    atomic_save(target, {"epoch": 22})
    assert len(attempts) == 3 and len(set(attempts)) == 1
    assert sleeps == [0.1, 0.2]
    assert torch.load(target, weights_only=True)["epoch"] == 22
    assert stale.read_bytes() == b"left from previous save"
    assert not attempts[0].exists()


def test_checkpoint_keeps_old_and_completed_new_state_when_lock_persists(tmp_path, monkeypatch):
    target = tmp_path / "best.pt"
    torch.save({"epoch": 21}, target)
    attempts = []

    def locked_replace(source, destination):
        attempts.append(source)
        raise PermissionError("Destination remains locked")

    monkeypatch.setattr(Path, "replace", locked_replace)
    monkeypatch.setattr("src.trainers.baseline.time.sleep", lambda _: None)
    with pytest.raises(PermissionError, match="completed new checkpoint") as failure:
        atomic_save(target, {"epoch": 22})
    assert len(attempts) == 10
    assert torch.load(target, weights_only=True)["epoch"] == 21
    assert torch.load(attempts[0], weights_only=True)["epoch"] == 22
    assert str(attempts[0]) in str(failure.value)


def test_failed_checkpoint_serialization_preserves_old_state(tmp_path, monkeypatch):
    target = tmp_path / "last.pt"
    torch.save({"epoch": 21}, target)

    def failed_save(state, stream):
        stream.write(b"incomplete checkpoint")
        raise OSError("disk full")

    monkeypatch.setattr("src.trainers.baseline.torch.save", failed_save)
    with pytest.raises(OSError, match="disk full"):
        atomic_save(target, {"epoch": 22})
    assert torch.load(target, weights_only=True)["epoch"] == 21


def test_resume_recovers_best_after_save_failed_on_improving_epoch(tmp_path):
    checkpoint = {"epoch": 22, "best_epoch": 22, "best_score": 0.7973,
                  "validation": {"Weighted_mIoU": 0.7973}}
    torch.save({"epoch": 21}, tmp_path / "best.pt")
    (tmp_path / "history.jsonl").write_text('{"epoch": 21}\n')
    prepare_resume(SimpleNamespace(output=tmp_path, resume=tmp_path / "last.pt"), checkpoint)
    assert torch.load(tmp_path / "best.pt", weights_only=True)["epoch"] == 22
    assert json.loads((tmp_path / "best_validation.json").read_text())["Weighted_mIoU"] == 0.7973


def test_skipped_optimizer_step_does_not_advance_ema():
    model = nn.Conv2d(1, 2, 1)
    ema = WeightEMA(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    class OverflowScaler:
        scale_value = 8

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def get_scale(self):
            return self.scale_value

        def step(self, optimizer):
            pass

        def update(self):
            self.scale_value /= 2

    args = SimpleNamespace(device="cpu", amp=False, epochs=1, grad_clip=12)
    batches = [(torch.randn(1, 1, 8, 8), torch.ones(1, 8, 8, dtype=torch.long), None)]
    with pytest.raises(FloatingPointError, match="No optimizer update"):
        train_epoch(model, ema, batches, optimizer, OverflowScaler(), args, epoch=1)
    assert ema.num_updates == 0


def test_local_dataset_contract():
    root = Path(__file__).resolve().parents[1] / "data/data/brisc2025"
    if not root.exists():
        pytest.skip("Local dataset not present")
    assert len(discover(root, "train")) == 3933
    assert len(discover(root, "test")) == 860


def test_depthwise_training_defaults_and_legacy_checkpoint_settings():
    args = parse_args(["--device", "cpu", "--variant", "depthwise"])
    assert args.variant == "depthwise"
    assert args.selection_metric == "glioma-miou"
    assert GLIOMA_MIOU_TARGET == 0.8
    assert (args.dw_kernel_size, args.dw_dilation) == (7, 3)
    assert checkpoint_setting({"variant": "depthwise"}, "dw_kernel_size") == 3
    assert checkpoint_setting({"variant": "depthwise"}, "dw_dilation") == 1
    assert checkpoint_setting({"variant": "kan"}, "dw_kernel_size") == 7
    assert all(name in MODEL_FIELDS for name in ("dw_kernel_size", "dw_dilation"))
    assert (args.epochs, args.batch_size, args.size, args.workers, args.seed) == (50, 32, 256, 8, 2026)
    assert not args.channels_last and not args.cudnn_benchmark
    assert args.attention_stages == [4, 5, 6]
    assert args.attention_dim == 128 and args.attention_tokens == 16
    assert all(name in MODEL_FIELDS for name in ("attention_stages", "attention_dim", "attention_depth"))
    legacy = {"variant": "kan"}
    assert checkpoint_setting(legacy, "attention_stages") == [4, 5, 6]
    assert checkpoint_setting(legacy, "selection_metric") == "weighted-miou"
    assert parse_args(["--device", "cpu", "--variant", "kan-two-way"]).selection_metric == "weighted-miou"
    with pytest.raises(KeyError):
        checkpoint_setting({"variant": "kan-two-way"}, "attention_dim")


@pytest.mark.parametrize("options", [
    ["--attention-stages", "3"], ["--attention-stages", "4", "4"],
    ["--attention-heads", "3"], ["--attention-tokens", "0"], ["--attention-depth", "0"],
    ["--attention-dropout", "nan"],
])
def test_training_rejects_invalid_attention_options(options):
    with pytest.raises(SystemExit):
        parse_args(["--device", "cpu", "--variant", "kan-two-way", *options])


def test_glioma_selection_follows_objective_when_weighted_score_disagrees():
    earlier = {"per_class": {"Glioma": {"mIoU": 0.81}}, "Weighted_mIoU": 0.84}
    later = {"per_class": {"Glioma": {"mIoU": 0.77}}, "Weighted_mIoU": 0.87}
    assert validation_score(earlier, "glioma-miou") > validation_score(later, "glioma-miou")
    assert validation_score(earlier, "weighted-miou") < validation_score(later, "weighted-miou")
    with pytest.raises(ValueError, match="finite score"):
        validation_score({"per_class": {"Glioma": {"mIoU": None}}}, "glioma-miou")


@pytest.mark.parametrize("options", [["--dw-kernel-size", "6"], ["--dw-kernel-size", "0"], ["--dw-dilation", "0"]])
def test_training_rejects_invalid_depthwise_settings(options):
    with pytest.raises(SystemExit):
        parse_args(["--device", "cpu", *options])


def test_biref_defaults_and_legacy_training_protocol():
    args = parse_args(["--device", "cpu", "--variant", "biref"])
    assert args.variant == "biref" and args.dw_dilation == 1
    assert args.augmentation == "robust" and args.boundary_weight == .1
    assert args.max_vram_gb == 10 and args.selection_metric == "glioma-miou"
    legacy = {"variant": "depthwise"}
    assert checkpoint_setting(legacy, "augmentation") == "legacy"
    assert checkpoint_setting(legacy, "biref_detail") is True


def test_vram_budget_fails_before_excessive_training_continues(monkeypatch):
    from src.trainers.baseline import check_vram_budget
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 10_100_000_000)
    with pytest.raises(RuntimeError, match="Reduce --batch-size"):
        check_vram_budget(SimpleNamespace(device="cuda", max_vram_gb=10))


def test_robust_scaling_retains_border_lesion_on_zoom_in(monkeypatch):
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, :10] = 255
    image = Image.fromarray(mask)
    monkeypatch.setattr("src.datasets.brisc.random.uniform", lambda *_: 1.15)
    _, transformed_mask = BRISCDataset.robust_spatial(image, image)
    np.testing.assert_array_equal(np.asarray(transformed_mask), mask)


def test_biref_ema_updates_aliased_tensors_once():
    model = nn.Module()
    model.first = nn.Linear(2, 2, bias=False)
    model.second = model.first
    model.ema_unique_updates = True
    model.first.weight.data.zero_()
    ema = WeightEMA(model)
    model.first.weight.data.fill_(2)
    ema.update(model)
    torch.testing.assert_close(ema.model.first.weight, torch.full((2, 2), 2 * 9 / 11))
    assert ema.state_dict()["unique_updates"] is True
