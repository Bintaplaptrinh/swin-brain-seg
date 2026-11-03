import pytest
import torch

from src.models import nnUNetResEncKAN
from src.models.factory import build_model
from src.models.two_way import ProjectedAttention, TwoWayFeatureRefiner, spatial_position_encoding
from src.trainers.baseline import segmentation_loss


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(19)
    yield
    torch.set_num_threads(previous)


def small_model(**kwargs):
    return nnUNetResEncKAN(
        features_per_stage=(8, 16, 32, 64, 64, 64), n_blocks_per_stage=(1,) * 6,
        attention_stages=(4, 5, 6), attention_dim=32, attention_heads=4,
        attention_tokens=4, **kwargs,
    )


def test_projected_attention_matches_explicit_softmax():
    layer = ProjectedAttention(16, 2, downsample=2).double()
    query, key, value = torch.randn(2, 7, 16).double(), torch.randn(2, 3, 16).double(), torch.randn(2, 3, 16).double()
    q = layer.query(query).reshape(2, 7, 2, 4).transpose(1, 2)
    k = layer.key(key).reshape(2, 3, 2, 4).transpose(1, 2)
    v = layer.value(value).reshape(2, 3, 2, 4).transpose(1, 2)
    weights = ((q @ k.transpose(-1, -2)) / 2).softmax(-1)
    expected = layer.output((weights @ v).transpose(1, 2).reshape(2, 7, 8))
    torch.testing.assert_close(layer(query, key, value), expected)


def test_refiner_preserves_bypass_and_both_attention_directions_learn():
    layer = TwoWayFeatureRefiner(8, dim=32, heads=4, num_tokens=4)
    inputs = torch.randn(2, 8, 4, 6, requires_grad=True)
    output = layer(inputs)
    assert output.shape == inputs.shape
    assert (output - inputs).square().mean().sqrt() < 0.1
    output.square().mean().backward()
    for name, parameter in layer.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for block in layer.layers:
        for attention in (block.token_to_image, block.image_to_token):
            assert attention.query.weight.grad.abs().sum() > 0
            assert attention.key.weight.grad.abs().sum() > 0
            assert attention.value.weight.grad.abs().sum() > 0
    assert layer.context_tokens.grad.abs().sum() > 0
    with torch.no_grad():
        layer.scale.zero_()
        torch.testing.assert_close(layer(inputs), inputs, rtol=0, atol=0)


def test_remote_image_content_changes_local_refinement():
    layer = TwoWayFeatureRefiner(8, dim=32, heads=4, num_tokens=4).eval()
    image = torch.randn(1, 8, 4, 6)
    changed = image.clone()
    changed[:, :, -1, -1] += 10
    with torch.no_grad():
        original, remote_changed = layer(image), layer(changed)
    assert not torch.equal(original[:, :, 0, 0], remote_changed[:, :, 0, 0])


def test_rectangular_positions_distinguish_both_axes():
    positions = spatial_position_encoding(3, 5, 32, torch.device("cpu"), torch.float32)
    assert positions.shape == (1, 15, 32)
    torch.testing.assert_close(positions[0, 0, :16], positions[0, 1, :16])
    assert not torch.equal(positions[0, 0, 16:], positions[0, 1, 16:])
    assert not torch.equal(positions[0, 0, :16], positions[0, 5, :16])
    torch.testing.assert_close(positions[0, 0, 16:], positions[0, 5, 16:])


def test_two_way_segmentation_backward_and_checkpoint_toggle():
    model = small_model(deep_supervision=True)
    image = torch.randn(2, 1, 64, 96)
    outputs = model(image)
    assert [tuple(out.shape) for out in outputs] == [(2, 2, 64 // 2**i, 96 // 2**i) for i in range(5)]
    labels = (torch.rand(2, 64, 96) > 0.9).long()
    segmentation_loss(outputs, labels).backward()
    for refiner in model.context_refiners.values():
        assert refiner.scale.grad.abs().sum() > 0
        assert refiner.layers[0].image_to_token.value.weight.grad.abs().sum() > 0
    restored = small_model(deep_supervision=False).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(image)[0], restored(image))


def test_two_way_no_skip_ablation_and_baseline_are_distinct():
    model = build_model("kan-two-way-no-skips", attention_dim=32, attention_heads=4, attention_depth=1)
    assert not model.skip_calibrators
    assert set(model.context_refiners) == {"4", "5", "6"}
    baseline = build_model("kan")
    assert not baseline.context_refiners
    assert set(baseline.skip_calibrators) == {"3", "4"}
    # Original backbone state names are unchanged by the context modules.
    assert set(baseline.state_dict()) == {k for k in build_model("kan-two-way").state_dict()
                                          if not k.startswith("context_refiners.")}


@pytest.mark.parametrize("kwargs", [
    {"dim": 30}, {"heads": 3}, {"num_tokens": 0}, {"depth": 0}, {"dropout": float("nan")},
])
def test_invalid_refiner_configuration(kwargs):
    with pytest.raises(ValueError):
        TwoWayFeatureRefiner(8, **({"dim": 32, "heads": 4} | kwargs))


@pytest.mark.parametrize("stages", [(3,), (4, 4), (7,)])
def test_attention_rejects_high_resolution_or_duplicate_stages(stages):
    with pytest.raises(ValueError, match="attention_stages"):
        nnUNetResEncKAN(attention_stages=stages)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_amp_has_finite_two_way_gradients():
    model = small_model(deep_supervision=True).cuda()
    image = torch.randn(2, 1, 64, 96, device="cuda")
    labels = (torch.rand(2, 64, 96, device="cuda") > 0.9).long()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = segmentation_loss(model(image), labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    for refiner in model.context_refiners.values():
        assert refiner.layers[0].token_to_image.value.weight.grad.abs().sum() > 0
        assert refiner.layers[0].image_to_token.value.weight.grad.abs().sum() > 0


def test_dropout_is_disabled_during_evaluation():
    layer = TwoWayFeatureRefiner(8, dim=32, heads=4, dropout=0.5).eval()
    image = torch.randn(1, 8, 4, 6)
    with torch.no_grad():
        torch.testing.assert_close(layer(image), layer(image), rtol=0, atol=0)
