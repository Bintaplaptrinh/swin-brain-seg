"""nnU-Net v2 ResEnc backbone with convolution-only deep encoder stages."""

from collections.abc import Sequence

import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks


def _per_stage(value, count, name):
    values = (value,) * count if isinstance(value, int) else tuple(value)
    if len(values) != count:
        raise ValueError(f"{name} must have {count} entries.")
    return values


def _pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


class ResidualDepthwiseBlock(nn.Module):
    """Dense channel mixing around a dilated depthwise spatial convolution.

    The channel width stays fixed. The residual scale starts at 0.01, matching
    the previous deep-block initialization, without any KAN or attention path.
    """

    def __init__(self, channels, conv_bias=True, norm_kwargs=None, nonlin_kwargs=None,
                 kernel_size=7, dilation=3):
        super().__init__()
        kernel_size, dilation = _pair(kernel_size), _pair(dilation)
        if (len(kernel_size) != 2 or len(dilation) != 2
                or any(not isinstance(k, int) or k < 1 or k % 2 == 0 for k in kernel_size)
                or any(not isinstance(d, int) or d < 1 for d in dilation)):
            raise ValueError("Depthwise kernels must be positive odd integers and dilations positive integers.")
        padding = tuple(d * (k - 1) // 2 for k, d in zip(kernel_size, dilation))
        norm_kwargs = {"eps": 1e-5, "affine": True} if norm_kwargs is None else dict(norm_kwargs)
        nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True} if nonlin_kwargs is None else dict(nonlin_kwargs)
        self.pointwise_in = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=conv_bias),
            nn.InstanceNorm2d(channels, **norm_kwargs),
            nn.LeakyReLU(**nonlin_kwargs),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, dilation=dilation,
                      groups=channels, bias=False),
            nn.InstanceNorm2d(channels, **norm_kwargs),
            nn.LeakyReLU(**nonlin_kwargs),
        )
        self.pointwise_out = nn.Conv2d(channels, channels, 1, bias=conv_bias)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 1e-2))

    def forward(self, x):
        return x + self.scale * self.pointwise_out(self.spatial(self.pointwise_in(x)))


class nnUNetResEncDepthwise(ResidualEncoderUNet):
    """Six-stage 2D nnU-Net ResEnc + depthwise separable convolution.

    Retains the library's stem, E1-E3 residual stages, transpose upsampling,
    concatenated unmodified skips, plain convolution decoder and supervision
    heads. E4-E6 each use a strided convolution followed by depthwise blocks.
    No KAN, skip calibrator, context tokens or two-way attention is created.

    Accepts the normal nnU-Net architecture kwargs for a compatible 2D plans
    copy; this fixed topology is not an automatic nnU-Net memory planner.
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        n_stages: int = 6,
        features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
        conv_op=nn.Conv2d,
        kernel_sizes=3,
        strides=(1, 2, 2, 2, 2, 2),
        n_blocks_per_stage=(1, 3, 4, 2, 2, 2),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        conv_bias: bool = True,
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs=None,
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=None,
        deep_supervision: bool = False,
        dw_kernel_size=7,
        dw_dilation=3,
    ):
        if n_stages != 6:
            raise ValueError("This design requires exactly six encoder stages.")
        if conv_op is not nn.Conv2d or norm_op is not nn.InstanceNorm2d or nonlin is not nn.LeakyReLU:
            raise ValueError("Use Conv2d, InstanceNorm2d and LeakyReLU for this 2D src.models.")
        if input_channels < 1 or num_classes < 1:
            raise ValueError("input_channels and num_classes must be positive.")
        if dropout_op is not None:
            raise ValueError("This depthwise experiment uses no dropout.")
        features = _per_stage(features_per_stage, 6, "features_per_stage")
        blocks = _per_stage(n_blocks_per_stage, 6, "n_blocks_per_stage")
        decoder_depth = _per_stage(n_conv_per_stage_decoder, 5, "n_conv_per_stage_decoder")
        kernels = tuple(_pair(k) for k in _per_stage(kernel_sizes, 6, "kernel_sizes"))
        strides = tuple(_pair(s) for s in _per_stage(strides, 6, "strides"))
        if kernels != ((3, 3),) * 6 or strides != ((1, 1),) + ((2, 2),) * 5:
            raise ValueError("Expected 3x3 kernels and strides [1,2,2,2,2,2].")
        if any(not isinstance(c, int) or c < 1 for c in (*features, *blocks, *decoder_depth)):
            raise ValueError("Feature counts and block counts must be positive integers.")
        norm_kwargs = {"eps": 1e-5, "affine": True} if norm_op_kwargs is None else dict(norm_op_kwargs)
        nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True} if nonlin_kwargs is None else dict(nonlin_kwargs)
        super().__init__(
            input_channels=input_channels, n_stages=6, features_per_stage=features,
            conv_op=conv_op, kernel_sizes=kernels, strides=strides,
            n_blocks_per_stage=blocks[:3] + (1, 1, 1),
            num_classes=num_classes, n_conv_per_stage_decoder=decoder_depth,
            conv_bias=conv_bias, norm_op=norm_op, norm_op_kwargs=norm_kwargs,
            dropout_op=None, dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, deep_supervision=deep_supervision,
        )
        self.input_channels = input_channels
        self.dw_kernel_size = _pair(dw_kernel_size)
        self.dw_dilation = _pair(dw_dilation)
        for idx in range(3, 6):
            transition = StackedConvBlocks(
                1, conv_op, features[idx - 1], features[idx], 3, 2,
                conv_bias, norm_op, norm_kwargs, None, None, nonlin, nonlin_kwargs,
            )
            self.encoder.stages[idx] = nn.Sequential(
                transition,
                *(ResidualDepthwiseBlock(features[idx], conv_bias, norm_kwargs, nonlin_kwargs,
                                        self.dw_kernel_size, self.dw_dilation)
                  for _ in range(blocks[idx])),
            )
        self.apply(self.initialize)

    def forward(self, x: torch.Tensor):
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError(f"Expected [B,{self.input_channels},H,W].")
        h, w = x.shape[-2:]
        if h < 64 or w < 64 or h % 32 or w % 32:
            raise ValueError("H and W must each be >=64 and divisible by 32; pad/crop outside the network.")
        return self.decoder(self.encoder(x))

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError(
            "The original estimator does not describe the replaced depthwise stages. "
            "Plan with the residual baseline, then profile with src.models.factory."
        )
