"""Six-stage 2D nnU-Net ResEnc with grouped FastKAN and optional two-way context."""

from collections.abc import Sequence

import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks

from .kan import FastKANSkipCalibrator, GroupedFastKAN2d, TokenizedKANBlock
from .two_way import TwoWayFeatureRefiner


def _per_stage(value, count, name):
    values = (value,) * count if isinstance(value, int) else tuple(value)
    if len(values) != count:
        raise ValueError(f"{name} must have {count} entries.")
    return values


def _pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


class nnUNetResEncKAN(ResidualEncoderUNet):
    """Real nnU-Net residual stem/shallow blocks and plain convolution decoder.

    Stage numbers in configuration are ONE-BASED: KAN encoder stages 4/5/6,
    calibrated skips 3/4, optional decoder refinements 5/4, optional two-way
    global context after encoder stages 4/5/6. Fixed topology:
    six stages, 3x3 kernels, strides [1,2,2,2,2,2], two spatial dimensions.

    The constructor accepts nnU-Net's standard architecture kwargs. Use its
    fully qualified class name in an existing six-stage 2D plans configuration.
    See README.md for limits of the planner's convolution-only memory model.
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
        kan_groups: int = 8,
        num_basis: int = 4,
        kan_expansion: int = 1,
        skip_kan_stages: Sequence[int] = (3, 4),
        decoder_kan_stages: Sequence[int] = (),
        use_kan_encoder: bool = True,
        attention_stages: Sequence[int] = (),
        attention_dim: int = 128,
        attention_heads: int = 8,
        attention_tokens: int = 16,
        attention_depth: int = 2,
        attention_dropout: float = 0.0,
    ):
        if n_stages != 6:
            raise ValueError("This design requires exactly six encoder stages.")
        if conv_op is not nn.Conv2d or norm_op is not nn.InstanceNorm2d or nonlin is not nn.LeakyReLU:
            raise ValueError("Use Conv2d, InstanceNorm2d and LeakyReLU for this 2D src.models.")
        if input_channels < 1 or num_classes < 1:
            raise ValueError("input_channels and num_classes must be positive.")
        features = _per_stage(features_per_stage, 6, "features_per_stage")
        blocks = _per_stage(n_blocks_per_stage, 6, "n_blocks_per_stage")
        decoder_depth = _per_stage(n_conv_per_stage_decoder, 5, "n_conv_per_stage_decoder")
        kernels = tuple(_pair(k) for k in _per_stage(kernel_sizes, 6, "kernel_sizes"))
        strides = tuple(_pair(s) for s in _per_stage(strides, 6, "strides"))
        if kernels != ((3, 3),) * 6 or strides != ((1, 1),) + ((2, 2),) * 5:
            raise ValueError("Expected 3x3 kernels and strides [1,2,2,2,2,2].")
        if any(not isinstance(c, int) or c < 1 for c in (*features, *blocks, *decoder_depth)):
            raise ValueError("Feature counts and block counts must be positive integers.")
        skip_stages = tuple(skip_kan_stages)
        decoder_stages = tuple(decoder_kan_stages)
        context_stages = tuple(attention_stages)
        if len(set(skip_stages)) != len(skip_stages) or not set(skip_stages) <= {3, 4}:
            raise ValueError("skip_kan_stages must be a subset of (3, 4), without duplicates.")
        if len(set(decoder_stages)) != len(decoder_stages) or not set(decoder_stages) <= {4, 5}:
            raise ValueError("decoder_kan_stages must be a subset of (4, 5), without duplicates.")
        if len(set(context_stages)) != len(context_stages) or not set(context_stages) <= {4, 5, 6}:
            raise ValueError("attention_stages must be a subset of (4, 5, 6), without duplicates.")
        kan_stages = set(skip_stages) | set(decoder_stages) | ({4, 5, 6} if use_kan_encoder else set())
        if kan_groups < 1 or any(features[s - 1] % kan_groups for s in kan_stages):
            raise ValueError("Channels of all KAN stages must be divisible by kan_groups.")
        norm_kwargs = {"eps": 1e-5, "affine": True} if norm_op_kwargs is None else dict(norm_op_kwargs)
        nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True} if nonlin_kwargs is None else dict(nonlin_kwargs)
        # Retain the library's stem, BasicBlockD shallow stages, and UNetDecoder.
        # Deep residual stages are placeholders replaced immediately below.
        super().__init__(
            input_channels=input_channels, n_stages=6, features_per_stage=features,
            conv_op=conv_op, kernel_sizes=kernels, strides=strides,
            n_blocks_per_stage=blocks[:3] + ((1, 1, 1) if use_kan_encoder else blocks[3:]),
            num_classes=num_classes, n_conv_per_stage_decoder=decoder_depth,
            conv_bias=conv_bias, norm_op=norm_op, norm_op_kwargs=norm_kwargs,
            dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, deep_supervision=deep_supervision,
        )
        self.input_channels = input_channels
        eps = norm_kwargs.get("eps", 1e-5)
        if use_kan_encoder:
            for idx in range(3, 6):
                transition = StackedConvBlocks(
                    1, nn.Conv2d, features[idx - 1], features[idx], 3, 2,
                    conv_bias, norm_op, norm_kwargs, dropout_op, dropout_op_kwargs,
                    nonlin, nonlin_kwargs,
                )
                self.encoder.stages[idx] = nn.Sequential(
                    transition,
                    *(TokenizedKANBlock(features[idx], kan_groups, num_basis, kan_expansion, eps)
                      for _ in range(blocks[idx])),
                )
        self.skip_calibrators = nn.ModuleDict({
            str(s): FastKANSkipCalibrator(features[s - 1], kan_groups, num_basis, eps)
            for s in skip_stages
        })
        self.context_refiners = nn.ModuleDict({
            str(s): TwoWayFeatureRefiner(features[s - 1], attention_dim, attention_heads,
                                         attention_tokens, attention_depth, attention_dropout)
            for s in context_stages
        })
        for stage in decoder_stages:
            # Decoder order is 5,4,3,2,1, the reverse of encoder skip order.
            idx = 5 - stage
            self.decoder.stages[idx] = nn.Sequential(
                self.decoder.stages[idx],
                TokenizedKANBlock(features[stage - 1], kan_groups, num_basis, kan_expansion, eps),
            )
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        # nnU-Net may invoke this again through get_network_from_plans.
        if getattr(module, "_kan_rbf_projection", False):
            GroupedFastKAN2d.reset_rbf_parameters(module)
        else:
            ResidualEncoderUNet.initialize(module)

    def forward(self, x: torch.Tensor):
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError(f"Expected [B,{self.input_channels},H,W].")
        h, w = x.shape[-2:]
        if h < 64 or w < 64 or h % 32 or w % 32:
            raise ValueError("H and W must each be >=64 and divisible by 32; pad/crop outside the network.")
        if self.context_refiners:
            # Interleave context with encoding so both the next stage and the
            # corresponding decoder skip receive the refined feature map.
            if self.encoder.stem is not None:
                x = self.encoder.stem(x)
            skips = []
            for stage, block in enumerate(self.encoder.stages, start=1):
                x = block(x)
                if str(stage) in self.context_refiners:
                    x = self.context_refiners[str(stage)](x)
                skips.append(x)
        else:
            skips = self.encoder(x)
        for stage, calibrator in self.skip_calibrators.items():
            idx = int(stage) - 1
            skips[idx] = calibrator(skips[idx])
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError(
            "Convolution-only estimates omit RBF expansion/autograd buffers. "
            "Plan with the residual baseline, then profile this model with src.models.factory."
        )
