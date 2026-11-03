"""Efficient multi-scale attention, distinct from exponential weight averaging.

Pattern: third_party/EMA-attention-module/EMA_attention_module, upstream
YOLOonMe/EMA-attention-module at afc37ecf52454c03421a86105fc5c16eb4ea1ad3.
Ouyang et al., ICASSP 2023, https://arxiv.org/abs/2305.13563.
"""

import torch
from torch import nn


class EfficientMultiScaleAttention(nn.Module):
    def __init__(self, channels, groups=32):
        super().__init__()
        if groups < 1 or channels < groups or channels % groups:
            raise ValueError("EMA attention groups must divide the channel count.")
        self.groups = groups
        width = channels // groups
        self.conv1x1 = nn.Conv2d(width, width, 1)
        self.conv3x3 = nn.Conv2d(width, width, 3, padding=1)
        self.gn = nn.GroupNorm(width, width)

    def forward(self, x):
        b, c, h, w = x.shape
        grouped = x.reshape(b * self.groups, c // self.groups, h, w)
        axes = torch.cat((grouped.mean(3, keepdim=True), grouped.mean(2, keepdim=True).transpose(2, 3)), 2)
        horizontal, vertical = self.conv1x1(axes).split((h, w), dim=2)
        coordinate = self.gn(grouped * horizontal.sigmoid() * vertical.transpose(2, 3).sigmoid())
        local = self.conv3x3(grouped)
        # Accumulate cross-spatial scores in FP32 under AMP.
        with torch.autocast(x.device.type, enabled=False):
            first, second = coordinate.float(), local.float()
            query1 = first.mean((2, 3)).softmax(1).unsqueeze(1)
            query2 = second.mean((2, 3)).softmax(1).unsqueeze(1)
            scores = query1 @ second.flatten(2) + query2 @ first.flatten(2)
            gate = scores.reshape(b * self.groups, 1, h, w).sigmoid()
        return (grouped * gate.to(grouped.dtype)).reshape(b, c, h, w)


class ResidualEMASkip(nn.Module):
    """Near-identity residual use of EMA to preserve weak skip features."""

    def __init__(self, channels, groups=32):
        super().__init__()
        self.attention = EfficientMultiScaleAttention(channels, groups)
        self.strength = nn.Parameter(torch.full((1, channels, 1, 1), .01))

    def forward(self, x):
        return x + self.strength.tanh() * (2 * self.attention(x) - x)
