import json
import pydoc
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder

from src.models import nnUNetResEncKAN
from src.models.kan import FastKANSkipCalibrator, GroupedFastKAN2d
from src.models.factory import segmentation_loss


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(7)
    yield
    torch.set_num_threads(previous)


def small_model(**kwargs):
    return nnUNetResEncKAN(
        features_per_stage=(8, 16, 32, 64, 64, 64),
        n_blocks_per_stage=(1, 1, 1, 1, 1, 1), **kwargs,
    )


@pytest.mark.parametrize("groups", [8, 16])
@pytest.mark.parametrize("decoder_stages", [(), (4, 5)])
def test_shapes_loss_and_kan_gradients(groups, decoder_stages):
    model = small_model(kan_groups=groups, decoder_kan_stages=decoder_stages, deep_supervision=True)
    outputs = model(torch.randn(1, 1, 64, 96))
    assert [tuple(t.shape) for t in outputs] == [
        (1, 2, 64 // (2**i), 96 // (2**i)) for i in range(5)
    ]
    target = (torch.rand(1, 64, 96) > 0.9).long()
    loss = segmentation_loss(outputs, target)
    loss.backward()
    kan_gradients = [p.grad for n, p in model.named_parameters() if "rbf_projection.weight" in n]
    assert kan_gradients
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in kan_gradients)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(isinstance(m, BasicBlockD) for m in model.encoder.stages[2].modules())
    assert isinstance(model.decoder, UNetDecoder)


def test_deep_supervision_toggle_keeps_full_resolution_prediction():
    model = small_model(deep_supervision=True).eval()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        high_resolution = model(x)[0]
        model.decoder.deep_supervision = False
        torch.testing.assert_close(model(x), high_resolution)


def test_grouped_projection_matches_explicit_edge_function_sum():
    layer = GroupedFastKAN2d(4, 6, groups=2, num_basis=4).double()
    x = torch.randn(1, 4, 3, 5, dtype=torch.float64, requires_grad=True)
    normalized = layer.norm(x)
    phi = layer.basis(normalized)
    weights = layer.rbf_projection.weight[:, :, 0, 0].reshape(2, 3, 2, 4)
    expected_groups = []
    for group in range(2):
        # Independent reference sums all input-channel and RBF edge terms.
        terms = phi[:, group * 2 : group * 2 + 2]
        expected_groups.append(torch.einsum("bikhw,oik->bohw", terms, weights[group]))
    expected = torch.cat(expected_groups, 1) + layer.base_projection(F.silu(normalized))
    torch.testing.assert_close(layer(x), expected)
    torch.autograd.gradcheck(layer, (x,), fast_mode=True)


def test_bounded_basis_with_extreme_or_constant_features():
    layer = GroupedFastKAN2d(8, 8)
    inputs = torch.tensor([-1e6, -10, 0, 10, 1e6]).reshape(1, 1, 1, 5).expand(1, 8, 2, 5)
    basis = layer.basis(inputs)
    assert torch.isfinite(basis).all()
    assert basis.min() >= 0 and basis.max() <= 1
    # Even extreme inputs retain an active edge RBF due to explicit bounding.
    assert (basis.max(dim=2).values > 0.5).all()
    output = layer(torch.ones(1, 8, 4, 4))
    assert torch.isfinite(output).all()


def test_skip_starts_near_identity_and_has_gradient():
    layer = FastKANSkipCalibrator(8)
    skip = torch.randn(1, 8, 4, 4, requires_grad=True)
    result = layer(skip)
    assert (result - skip).abs().max() <= 0.011 * skip.abs().max()
    result.square().mean().backward()
    assert layer.strength.grad.abs().sum() > 0
    assert layer.kan.rbf_projection.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("shape", [(1, 1, 65, 64), (1, 1, 32, 64), (1, 3, 64, 64)])
def test_invalid_inputs_fail_before_encoder(shape):
    with pytest.raises(ValueError):
        small_model()(torch.randn(*shape))


@pytest.mark.parametrize("kwargs", [
    {"n_stages": 5}, {"kan_groups": 7}, {"skip_kan_stages": (1,)},
    {"decoder_kan_stages": (3,)}, {"conv_op": nn.Conv3d}, {"num_basis": 1},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        small_model(**kwargs)


@pytest.mark.parametrize("filename", ["nnunet_architecture.json", "nnunet_architecture_two_way.json", "nnunet_architecture_depthwise.json"])
def test_plans_fragment_resolves_and_checkpoint_loads_with_ds_disabled(filename):
    plans = json.loads((Path(__file__).parents[2] / "configs/model/plans" / filename).read_text())
    kwargs = plans["arch_kwargs"]
    for key in plans["_kw_requires_import"]:
        kwargs[key] = pydoc.locate(kwargs[key]) if kwargs[key] is not None else None
    kwargs.update(features_per_stage=(8, 16, 32, 64, 64, 64), n_blocks_per_stage=(1,) * 6)
    cls = pydoc.locate(plans["network_class_name"])
    training = cls(input_channels=1, num_classes=2, deep_supervision=True, **kwargs)
    inference = cls(input_channels=1, num_classes=2, deep_supervision=False, **kwargs)
    inference.load_state_dict(training.state_dict(), strict=True)
    training.eval()
    inference.eval()
    with torch.no_grad():
        x = torch.randn(1, 1, 64, 64)
        torch.testing.assert_close(training(x)[0], inference(x))


def test_convolution_ablation_removes_all_kan_modules():
    model = small_model(use_kan_encoder=False, skip_kan_stages=())
    assert not any(isinstance(m, GroupedFastKAN2d) for m in model.modules())
    assert model(torch.randn(1, 1, 64, 64)).shape == (1, 2, 64, 64)
