from pathlib import Path
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.models.swin_hybrid import CoplanarPatchMerging, nnUNetSwinHybrid
from src.trainers.hybrid_loss import dynamic_loss, dynamic_weights
from src.trainers.swin_hybrid import (EarlyStopping, ema_for_epoch, make_optimizer, make_scheduler,
                                   parse_args, recover_history, validate_resume)
from src.datasets.brisc import Sample
from src.trainers.baseline import loader


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_coplanar_merge_exactly_preserves_3d_projection_on_identical_planes():
    from monai.networks.nets.swin_unetr import PatchMerging
    original = PatchMerging(4, spatial_dims=3)
    converted = CoplanarPatchMerging(4)
    converted.load_state_dict(original.state_dict(), strict=True)
    x = torch.randn(2, 8, 10, 4)
    repeated = x[:, None].repeat(1, 2, 1, 1, 1)
    torch.testing.assert_close(converted(x), original(repeated)[:, 0])


def test_actual_checkpoint_is_trainable_and_fused_into_large_model():
    path = Path(__file__).parents[2]/"outputs/checkpoints/pretrained/Brain_Swin_UNETR.pth"
    if not path.exists():
        pytest.skip("Local pretrained checkpoint absent")
    model = nnUNetSwinHybrid(path)
    assert 20_000_000 <= model.parameter_counts["trainable"] <= 50_000_000
    assert model.parameter_counts["total"] <= 50_000_000
    assert model.parameter_counts == {"total": 38_801_022, "trainable": 38_801_022, "frozen": 0}
    assert model.tokenizer.audit["trainable_parameters"] == 1_988_850
    assert model.tokenizer.audit["encoder_tensor_count"] == 126
    before = model.tokenizer.encoder.patch_embed.proj.weight.detach().clone()
    model.train()
    assert model.tokenizer.training and model.tokenizer.encoder.training
    outputs = model(torch.randn(1, 1, 64, 96))
    assert outputs["segmentation"][0].shape == (1, 2, 64, 96)
    loss, _ = dynamic_loss(outputs, (torch.rand(1, 64, 96) > .8).long(), 150)
    loss.backward()
    assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.tokenizer.parameters())
    assert model.tokenizer.encoder.patch_embed.proj.weight.grad.abs().sum() > 0
    assert all(module[0].weight.grad is not None and module[0].weight.grad.abs().sum() > 0 for module in model.adapters)
    torch.testing.assert_close(before, model.tokenizer.encoder.patch_embed.proj.weight)
    assert all(m.path[0].kernel_size == (3, 3) for m in model.large_kernels.values())
    model.eval()
    assert not model.tokenizer.training and not model.tokenizer.encoder.training


def test_ema_is_not_created_before_epoch_20():
    model = nn.Conv2d(1, 2, 1)
    assert ema_for_epoch(model, None, 19, 20, .9995) is None
    ema = ema_for_epoch(model, None, 20, 20, .9995)
    assert ema is not None and ema.num_updates == 0
    assert ema_for_epoch(model, ema, 21, 20, .9995) is ema


def test_early_stopping_starts_counting_after_100_epochs():
    stopping = EarlyStopping()
    for epoch in range(1, 150):
        assert not stopping.update(.7, epoch)
    assert stopping.bad_epochs == 49
    assert stopping.update(.7, 150)
    restored = EarlyStopping()
    restored.__dict__.update(stopping.__dict__)
    assert restored.exhausted(150)
    assert not stopping.update(.71, 151)
    assert stopping.bad_epochs == 0


def test_optimizer_groups_are_disjoint_and_cosine_preserves_swin_lr_ratio():
    model = nn.Module()
    model.tokenizer = nn.Conv2d(1, 2, 1)
    model.head = nn.Conv2d(2, 2, 1)
    args = parse_args(["--device", "cpu"])
    optimizer = make_optimizer(model, args)
    groups = [{id(p) for p in g["params"]} for g in optimizer.param_groups]
    assert not groups[0] & groups[1]
    assert groups[0] | groups[1] == {id(p) for p in model.parameters()}
    assert groups[1] == {id(p) for p in model.tokenizer.parameters()}
    scheduler = make_scheduler(optimizer, args.epochs)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(9e-5)
    for _ in range(300):
        optimizer.step()  # No gradients or parameter updates; exercise the LR schedule only.
        scheduler.step()
        assert optimizer.param_groups[1]["lr"] == pytest.approx(.3*optimizer.param_groups[0]["lr"])
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-6)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(9e-7)


def test_sampling_weights_and_defaults_match_new_protocol():
    args = parse_args(["--device", "cpu"])
    assert (args.size, args.epochs, args.batch_size, args.effective_batch) == (512, 300, 2, 16)
    assert (args.ema_start, args.early_start, args.patience) == (20, 100, 50)
    samples = [Sample(Path(t+".jpg"), Path("mask.png"), t, "Axial")
               for t in ("Glioma", "Meningioma", "Pituitary")]
    batches = loader(samples, args, torch.Generator().manual_seed(1), training=True)
    assert batches.sampler.weights.tolist() == [2., 1., 1.3]
    assert len(batches.sampler) == len(samples)
    evaluation = loader(samples, args, torch.Generator().manual_seed(1))
    assert isinstance(evaluation.sampler, torch.utils.data.SequentialSampler)
    smoke = parse_args(["--device", "cpu", "--smoke-test"])
    assert smoke.size == 512 and smoke.epochs == 2 and smoke.ema_start == 20


def test_new_protocol_rejects_legacy_resume_and_boundary_weight_is_capped():
    args = parse_args(["--device", "cpu"])
    with pytest.raises(ValueError, match="different training protocol"):
        validate_resume(args, {"config": {"size": 256}})
    assert dynamic_weights(1) == {"alpha": 1., "beta": .5, "gamma": .01}
    assert dynamic_weights(150)["gamma"] == pytest.approx(.5)
    assert dynamic_weights(300)["gamma"] == pytest.approx(.5)
    assert all(.01 <= dynamic_weights(epoch)["gamma"] <= .5 for epoch in range(1, 301))


def test_interrupted_history_tail_is_reconstructed_from_checkpoint(tmp_path):
    history = tmp_path/"history.jsonl"
    history.write_text('{"epoch": 1}\n{"epoch": 2', encoding="utf-8")
    args = SimpleNamespace(output=tmp_path, resume=tmp_path/"last.pt")
    recover_history(args, {"epoch": 2, "history_row": {"epoch": 2, "train_loss": .5}})
    rows = [json.loads(line) for line in history.read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [1, 2]
    assert rows[-1]["train_loss"] == .5
    assert len(list(tmp_path.glob("history.interrupted.*.jsonl"))) == 1


def test_resume_refuses_newer_history_or_corruption_before_tail(tmp_path):
    history = tmp_path/"history.jsonl"
    args = SimpleNamespace(output=tmp_path, resume=tmp_path/"last.pt")
    history.write_text('{"epoch": 3}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="newer"):
        recover_history(args, {"epoch": 2})
    history.write_text('damaged\n{"epoch": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="corruption"):
        recover_history(args, {"epoch": 2})


def test_boundary_error_receives_heavier_penalty_and_padding_is_ignored():
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    target[:, 8:24, 8:24] = 1
    target[:, :4] = -1
    fg = (target == 1).float()*10-5
    perfect = torch.stack((-fg, fg), 1)
    edge_error, interior_error = perfect.clone(), perfect.clone()
    edge_error[:, :, 8, 12] *= -1
    interior_error[:, :, 12, 12] *= -1
    edge_loss, edge_parts = dynamic_loss(edge_error, target, 150)
    interior_loss, interior_parts = dynamic_loss(interior_error, target, 150)
    assert edge_parts["boundary"] > interior_parts["boundary"]
    assert edge_loss > interior_loss
    logits = torch.randn(1, 2, 32, 32, requires_grad=True)
    changed = logits.detach().clone()
    changed[:, :, :4] = 100
    loss, _ = dynamic_loss(logits, target, 1)
    torch.testing.assert_close(loss, dynamic_loss(changed, target, 1)[0])
    loss.backward()
    assert torch.all(logits.grad[:, :, :4] == 0)
    assert dynamic_weights(150)["gamma"] > dynamic_weights(1)["gamma"]
