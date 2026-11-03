import pytest
import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder

from src.models import nnUNetResEncDepthwise
from src.models.factory import build_model
from src.models.kan import GroupedFastKAN2d, FastKANSkipCalibrator, TokenizedKANBlock
from src.models.nnunet_resenc_depthwise import ResidualDepthwiseBlock
from src.models.two_way import TwoWayAttentionBlock, TwoWayFeatureRefiner
from src.trainers.baseline import segmentation_loss


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(29)
    yield
    torch.set_num_threads(previous)


def small_model(**kwargs):
    return nnUNetResEncDepthwise(
        features_per_stage=(8, 16, 32, 64, 64, 64),
        n_blocks_per_stage=(1, 1, 1, 2, 2, 2), **kwargs,
    )


def test_depthwise_backbone_has_no_kan_or_attention_and_all_deep_stages_learn():
    model = small_model(deep_supervision=True)
    assert isinstance(model, ResidualEncoderUNet)
    assert isinstance(model.decoder, UNetDecoder)
    for stage in model.encoder.stages[:3]:
        assert any(isinstance(m, BasicBlockD) for m in stage.modules())
    forbidden = (GroupedFastKAN2d, FastKANSkipCalibrator, TokenizedKANBlock,
                 TwoWayAttentionBlock, TwoWayFeatureRefiner, nn.MultiheadAttention)
    assert not any(isinstance(m, forbidden) for m in model.modules())
    blocks = [m for m in model.modules() if isinstance(m, ResidualDepthwiseBlock)]
    assert len(blocks) == 6
    for block in blocks:
        spatial = block.spatial[0]
        assert spatial.groups == spatial.in_channels == spatial.out_channels
        assert spatial.kernel_size == (7, 7)
        assert spatial.dilation == (3, 3)
        assert spatial.padding == (9, 9)
        assert block.pointwise_in[0].groups == block.pointwise_out.groups == 1
    outputs = model(torch.randn(2, 1, 64, 96))
    assert [tuple(x.shape) for x in outputs] == [(2, 2, 64 // 2**i, 96 // 2**i) for i in range(5)]
    target = (torch.rand(2, 64, 96) > 0.9).long()
    target[:, :4, :] = -1
    loss = segmentation_loss(outputs, target)
    loss.backward()
    assert torch.isfinite(loss)
    for block in blocks:
        for layer in (block.pointwise_in[0], block.spatial[0], block.pointwise_out):
            assert layer.weight.grad is not None
            assert torch.isfinite(layer.weight.grad).all()
            assert layer.weight.grad.abs().sum() > 0


def test_skips_reach_decoder_without_any_calibration():
    model = small_model().eval()
    encoded, decoded = [], []
    encoder_hook = model.encoder.register_forward_hook(lambda _m, _args, output: encoded.append(output))
    decoder_hook = model.decoder.register_forward_pre_hook(lambda _m, args: decoded.append(args[0]))
    with torch.no_grad():
        model(torch.randn(1, 1, 64, 64))
    encoder_hook.remove()
    decoder_hook.remove()
    assert encoded[0] is decoded[0]


def test_depthwise_factory_and_legacy_controls():
    model = build_model("depthwise", groups=16, num_basis=8, attention_stages=[4, 5, 6])
    assert isinstance(model, nnUNetResEncDepthwise)
    assert model.decoder.deep_supervision
    assert not hasattr(model, "context_refiners")
    assert not hasattr(model, "skip_calibrators")
    assert sum(isinstance(m, ResidualDepthwiseBlock) for m in model.modules()) == 6


def test_explicit_kernel_and_dilation_support_legacy_checkpoints():
    model = small_model(dw_kernel_size=(3, 3), dw_dilation=(1, 1))
    assert model.encoder.stages[3][1].spatial[0].weight.shape[-2:] == (3, 3)
    model = build_model("depthwise", dw_kernel_size=7, dw_dilation=1)
    assert model.encoder.stages[3][1].spatial[0].dilation == (1, 1)


@pytest.mark.parametrize("shape", [(1, 1, 65, 64), (1, 1, 32, 64), (1, 3, 64, 64)])
def test_invalid_inputs(shape):
    with pytest.raises(ValueError):
        small_model()(torch.randn(*shape))


@pytest.mark.parametrize("kwargs", [
    {"n_stages": 5}, {"conv_op": nn.Conv3d}, {"kernel_sizes": 5},
    {"strides": (1,) * 6}, {"num_classes": 0}, {"dropout_op": nn.Dropout2d},
    {"dw_kernel_size": 6}, {"dw_dilation": 0}, {"dw_dilation": (3,)},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        small_model(**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_depthwise_cuda_amp_update_is_finite():
    model = small_model(deep_supervision=True).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", init_scale=128)
    with torch.autocast("cuda", dtype=torch.float16):
        outputs = model(torch.randn(2, 1, 64, 96, device="cuda"))
        loss = segmentation_loss(outputs, torch.ones(2, 64, 96, dtype=torch.long, device="cuda"))
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    before = model.encoder.stages[3][1].spatial[0].weight.detach().clone()
    scaler.step(optimizer)
    scaler.update()
    assert not torch.equal(before, model.encoder.stages[3][1].spatial[0].weight)
