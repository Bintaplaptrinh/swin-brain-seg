from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest
import torch

from src.models.ema_attention import EfficientMultiScaleAttention, ResidualEMASkip
from src.models.nnunet_biref import nnUNetBiRefEMASkip
from src.models.factory import build_model
from src.trainers.baseline import segmentation_loss


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(47)
    yield
    torch.set_num_threads(before)


def small_model(**kwargs):
    return nnUNetBiRefEMASkip(features_per_stage=(8, 16, 32, 64, 64, 64),
                             n_blocks_per_stage=(1,)*6, ema_skip_groups=8, **kwargs)


def test_core_matches_local_user_supplied_ema_reference():
    path = Path(__file__).parents[2] / "third_party/EMA-attention-module/EMA_attention_module"
    if not path.exists():
        pytest.skip("Reference repository unavailable")
    reference_class = SourceFileLoader("ema_attention_reference", str(path)).load_module().EMA
    reference = reference_class(32, factor=8).double()
    implementation = EfficientMultiScaleAttention(32, groups=8).double()
    implementation.load_state_dict(reference.state_dict())
    inputs = torch.randn(2, 32, 7, 11, dtype=torch.float64)
    # FP32 score accumulation is intentional for AMP; allow its rounding.
    torch.testing.assert_close(implementation(inputs), reference(inputs), rtol=1e-6, atol=1e-7)


def test_residual_skip_preserves_bypass_and_gradients():
    module = ResidualEMASkip(32, 8)
    inputs = torch.randn(2, 32, 8, 12, requires_grad=True)
    output = module(inputs)
    assert ((output-inputs).abs() <= .011 * inputs.abs() + 1e-7).all()
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in module.parameters())
    with torch.no_grad():
        module.strength.zero_()
        torch.testing.assert_close(module(inputs), inputs, atol=0, rtol=0)


def test_skip_attention_is_decoder_only_and_learns_on_all_selected_stages():
    model = small_model(deep_supervision=True)
    assert set(model.decoder.skip_attention) == {"3", "4", "5"}
    assert not any(isinstance(m, ResidualEMASkip) for m in model.encoder.modules())
    outputs = model(torch.randn(2, 1, 64, 96))
    loss = segmentation_loss(outputs, (torch.rand(2, 64, 96) > .9).long())
    loss.backward()
    for attention in model.decoder.skip_attention.values():
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in attention.parameters())
    inference = small_model(deep_supervision=False).eval()
    inference.load_state_dict(model.state_dict())
    assert inference(torch.randn(1, 1, 64, 96)).shape == (1, 2, 64, 96)


def test_default_ema_variant_stays_under_parameter_limit():
    model = build_model("biref-ema")
    assert sum(p.numel() for p in model.parameters()) < 10_000_000


@pytest.mark.parametrize("channels,groups", [(32, 7), (8, 32), (32, 0)])
def test_invalid_groups(channels, groups):
    with pytest.raises(ValueError):
        EfficientMultiScaleAttention(channels, groups)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_ema_skip_amp_update():
    model = small_model(deep_supervision=True).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scaler = torch.amp.GradScaler("cuda", init_scale=128)
    before = model.decoder.skip_attention["3"].attention.conv3x3.weight.detach().clone()
    with torch.autocast("cuda", dtype=torch.float16):
        outputs = model(torch.randn(2, 1, 64, 96, device="cuda"))
        loss = segmentation_loss(outputs, (torch.rand(2, 64, 96, device="cuda") > .9).long())
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    scaler.step(optimizer)
    scaler.update()
    assert not torch.equal(before, model.decoder.skip_attention["3"].attention.conv3x3.weight)
