"""Compact BiRefNet-inspired reconstruction for the nnU-Net depthwise backbone.

Independent implementation for grayscale binary segmentation: no RMBG weights,
Swin backbone, deformable operators, KAN or two-way attention are imported.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .budget import enforce_parameter_budget
from .nnunet_resenc_depthwise import nnUNetResEncDepthwise
from .ema_attention import ResidualEMASkip


def pointwise(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                         nn.InstanceNorm2d(out_channels, affine=True), nn.LeakyReLU(.01, inplace=True))


class CompactContext(nn.Module):
    """Fuse pooled E3-E6 features and a global descriptor at bottleneck size."""

    def __init__(self, features, width=16):
        super().__init__()
        self.projections = nn.ModuleList([pointwise(c, width) for c in features[2:]])
        self.global_projection = nn.Conv2d(features[-1], width, 1)
        self.mix = nn.Conv2d(width * 5, features[-1], 1, bias=False)
        self.scale = nn.Parameter(torch.full((1, features[-1], 1, 1), .01))

    def forward(self, skips):
        size = skips[-1].shape[-2:]
        maps = [proj(F.adaptive_avg_pool2d(x, size)) for proj, x in zip(self.projections, skips[2:])]
        global_map = self.global_projection(F.adaptive_avg_pool2d(skips[-1], 1)).expand(-1, -1, *size)
        return skips[-1] + self.scale * self.mix(torch.cat([*maps, global_map], 1))


class ImageDetailReference(nn.Module):
    """Preserve local grayscale samples with pixel-unshuffle, then project.

    Unlike BiRefNet's nonlocal patch stacking, neighboring pixels are packed
    into channels here. This explicit adaptation preserves rectangular shapes.
    """

    def __init__(self, input_channels, feature_channels, factor, width=8):
        super().__init__()
        self.factor = factor
        self.project = pointwise(input_channels * factor * factor, width)
        self.spatial = nn.Sequential(nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
                                     nn.InstanceNorm2d(width, affine=True), nn.LeakyReLU(.01, inplace=True))
        self.fuse = nn.Conv2d(width, feature_channels, 1, bias=False)
        self.scale = nn.Parameter(torch.full((1, feature_channels, 1, 1), .01))

    def forward(self, features, image):
        packed = F.pixel_unshuffle(image, self.factor) if self.factor > 1 else image
        return features + self.scale * self.fuse(self.spatial(self.project(packed)))


class BiReferenceDecoder(nn.Module):
    def __init__(self, base, features, input_channels, context=True, detail=True, boundary=True):
        super().__init__()
        self.base = base
        self.skip_attention = nn.ModuleDict()
        self.context = CompactContext(features) if context else None
        self.details = nn.ModuleDict({str(s): ImageDetailReference(input_channels, features[s-1], 2**(s-1))
                                      for s in (1, 2, 3)}) if detail else nn.ModuleDict()
        # Two low-cost training heads. Region features are never multiplied by
        # an edge gate, preserving a direct path for weak tumor interiors.
        self.boundary_heads = nn.ModuleDict({str(s): nn.Sequential(pointwise(features[s-1], 8), nn.Conv2d(8, 1, 1))
                                             for s in (2, 3)}) if boundary else nn.ModuleDict()

    @property
    def deep_supervision(self):
        return self.base.deep_supervision

    @deep_supervision.setter
    def deep_supervision(self, value):
        self.base.deep_supervision = value

    def forward(self, skips, image):
        x = self.context(skips) if self.context is not None else skips[-1]
        masks, boundaries = [], []
        for i, block in enumerate(self.base.stages):
            stage = str(5 - i)
            skip = skips[-(i+2)]
            if stage in self.skip_attention:
                skip = self.skip_attention[stage](skip)
            x = block(torch.cat((self.base.transpconvs[i](x), skip), 1))
            if stage in self.details:
                x = self.details[stage](x, image)
            if self.training and stage in self.boundary_heads:
                boundaries.append(self.boundary_heads[stage](x))
            if self.deep_supervision or i == len(self.base.stages) - 1:
                masks.append(self.base.seg_layers[i](x))
        masks.reverse()
        segmentation = masks if self.deep_supervision else masks[0]
        if self.training and boundaries:
            return {"segmentation": segmentation, "boundary": boundaries}
        return segmentation


class nnUNetBiRef(nnUNetResEncDepthwise):
    """nnU-Net + dense 7x7 depthwise + compact context/detail reconstruction."""

    ema_unique_updates = True

    def __init__(self, input_channels=1, features_per_stage=(32, 64, 128, 256, 320, 320),
                 dw_kernel_size=7, dw_dilation=1, biref_context=True, biref_detail=True,
                 biref_boundary=True, **kwargs):
        super().__init__(input_channels=input_channels, features_per_stage=features_per_stage,
                         dw_kernel_size=dw_kernel_size, dw_dilation=dw_dilation, **kwargs)
        features = self.encoder.output_channels
        self.decoder = BiReferenceDecoder(self.decoder, features, input_channels,
                                           biref_context, biref_detail, biref_boundary)
        # Initialize only additions; keep the existing backbone initialization.
        if self.decoder.context is not None:
            self.decoder.context.apply(self.initialize)
        self.decoder.details.apply(self.initialize)
        self.decoder.boundary_heads.apply(self.initialize)
        enforce_parameter_budget(self)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError(f"Expected [B,{self.input_channels},H,W].")
        h, w = x.shape[-2:]
        if h < 64 or w < 64 or h % 32 or w % 32:
            raise ValueError("H and W must each be >=64 and divisible by 32.")
        return self.decoder(self.encoder(x), x)


class nnUNetBiRefEMASkip(nnUNetBiRef):
    """BiRef adaptation with residual EMA attention on decoder-only skips."""

    def __init__(self, ema_skip_stages=(3, 4, 5), ema_skip_groups=32, **kwargs):
        stages = tuple(ema_skip_stages)
        if len(set(stages)) != len(stages) or not set(stages) <= {1, 2, 3, 4, 5}:
            raise ValueError("EMA skip stages must be a subset of 1..5 without duplicates.")
        super().__init__(**kwargs)
        self.decoder.skip_attention = nn.ModuleDict({
            str(stage): ResidualEMASkip(self.encoder.output_channels[stage-1], ema_skip_groups)
            for stage in stages
        })
        self.decoder.skip_attention.apply(self.initialize)
        enforce_parameter_budget(self)
