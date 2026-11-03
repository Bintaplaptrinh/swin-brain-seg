import pytest
import torch
from torch import nn

from src.models import nnUNetBiRef
from src.models.factory import build_model
from src.models.budget import enforce_parameter_budget
from src.models.nnunet_biref import ImageDetailReference
from src.models.kan import GroupedFastKAN2d
from src.models.two_way import TwoWayFeatureRefiner
from src.trainers.baseline import segmentation_loss, boundary_loss, boundary_targets


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(11)
    yield
    torch.set_num_threads(previous)


def small_model(**kwargs):
    return nnUNetBiRef(features_per_stage=(8, 16, 32, 64, 64, 64),
                       n_blocks_per_stage=(1,) * 6, **kwargs)


@pytest.mark.parametrize("deep_supervision", [True, False])
def test_reference_paths_and_auxiliary_heads_receive_segmentation_gradients(deep_supervision):
    model = small_model(deep_supervision=deep_supervision).train()
    outputs = model(torch.randn(2, 1, 64, 96))
    assert set(outputs) == {"segmentation", "boundary"}
    assert [tuple(x.shape) for x in outputs["boundary"]] == [(2, 1, 16, 24), (2, 1, 32, 48)]
    labels = (torch.rand(2, 64, 96) > .8).long()
    labels[:, :5] = -1
    loss = segmentation_loss(outputs, labels)
    loss.backward()
    assert torch.isfinite(loss)
    for group in (model.decoder.context, model.decoder.details, model.decoder.boundary_heads):
        for name, parameter in group.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name


def test_inference_interface_and_checkpoint_roundtrip():
    training = small_model(deep_supervision=True).eval()
    inference = small_model(deep_supervision=False).eval()
    inference.load_state_dict(training.state_dict(), strict=True)
    inputs = torch.randn(1, 1, 64, 96)
    with torch.no_grad():
        torch.testing.assert_close(training(inputs)[0], inference(inputs))
        training.decoder.deep_supervision = False
        torch.testing.assert_close(training(inputs), inference(inputs))
    assert not any(isinstance(m, (GroupedFastKAN2d, TwoWayFeatureRefiner)) for m in training.modules())


def test_default_candidate_obeys_parameter_budget():
    model = build_model("biref")
    assert enforce_parameter_budget(model) <= 10_000_000
    assert model.dw_kernel_size == (7, 7) and model.dw_dilation == (1, 1)
    with pytest.raises(ValueError, match="limit is 10,000,000"):
        enforce_parameter_budget(nn.Linear(1000, 10001, device="meta"))


def test_reference_ablations_keep_region_output():
    model = small_model(deep_supervision=True, biref_context=False, biref_detail=False, biref_boundary=False)
    assert model.decoder.context is None
    assert not model.decoder.details and not model.decoder.boundary_heads
    assert len(model(torch.randn(1, 1, 64, 64))) == 5


def test_image_detail_reference_has_direct_image_gradient():
    layer = ImageDetailReference(1, 8, factor=4)
    image = torch.randn(1, 1, 64, 96, requires_grad=True)
    output = layer(torch.randn(1, 8, 16, 24), image)
    output.square().mean().backward()
    assert image.grad is not None and image.grad.abs().sum() > 0


def test_boundary_loss_excludes_padding_and_has_no_padding_contour():
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    target[:, :8] = -1
    target[:, 12:20, 12:20] = 1
    truth, valid = boundary_targets(target, (32, 32))
    assert not truth[:, :, :10].any()
    assert not valid[:, :, :9].any()
    logits = torch.randn(1, 1, 32, 32, requires_grad=True)
    changed = logits.detach().clone()
    changed[valid == 0] = 100
    torch.testing.assert_close(boundary_loss([logits], target), boundary_loss([changed], target))
    boundary_loss([logits], target).backward()
    assert torch.all(logits.grad[valid == 0] == 0)
    assert logits.grad[valid == 1].abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_candidate_amp_optimizer_update():
    model = small_model(deep_supervision=True).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scaler = torch.amp.GradScaler("cuda", init_scale=128)
    before = model.decoder.context.mix.weight.detach().clone()
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(torch.randn(2, 1, 64, 96, device="cuda"))
        loss = segmentation_loss(output, (torch.rand(2, 64, 96, device="cuda") > .9).long())
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    scaler.step(optimizer)
    scaler.update()
    assert not torch.equal(before, model.decoder.context.mix.weight)
