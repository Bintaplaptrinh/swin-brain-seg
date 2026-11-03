"""Grouped, pointwise Gaussian RBF KANs on NCHW tensors.

Conceptually related to FastKAN (https://github.com/ZiyaoLi/fast-kan).
This implementation uses grouped 1x1 convolutions, spatial InstanceNorm,
explicit range bounding, and residual scaling; it does not copy its code.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class GroupedFastKAN2d(nn.Module):
    """An independent KAN channel transform at every spatial token.

    Channels are partitioned into groups. Each within-group edge learns its
    own RBF coefficients; the fixed basis centers are shared, not the learned
    edge functions. Grouping reduces weights/MACs, not basis activation size.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 8,
        num_basis: int = 4,
        grid_limit: float = 2.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        if groups < 1 or min(in_channels, out_channels) < 1:
            raise ValueError("Channels and groups must be positive.")
        if in_channels % groups or out_channels % groups:
            raise ValueError("Both channel counts must be divisible by groups.")
        if num_basis < 2 or not math.isfinite(grid_limit) or grid_limit <= 0:
            raise ValueError("Use at least two RBF centers and a positive finite grid_limit.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_basis = num_basis
        self.grid_limit = float(grid_limit)
        self.width = 2 * grid_limit / (num_basis - 1)
        # Non-affine: a learned scale/shift here could undo standardization.
        self.norm = nn.InstanceNorm2d(in_channels, eps=norm_eps, affine=False)
        self.register_buffer("centers", torch.linspace(-grid_limit, grid_limit, num_basis))
        self.rbf_projection = nn.Conv2d(
            in_channels * num_basis, out_channels, 1, groups=groups, bias=False
        )
        self.rbf_projection._kan_rbf_projection = True
        self.base_projection = nn.Conv2d(in_channels, out_channels, 1, groups=groups, bias=True)
        self.reset_rbf_parameters(self.rbf_projection)

    @staticmethod
    def reset_rbf_parameters(module):
        # Keep the nonlinear correction modest relative to the base branch.
        nn.init.normal_(module.weight, std=0.1 / math.sqrt(module.weight.shape[1]))

    def basis(self, normalized: torch.Tensor) -> torch.Tensor:
        """Return [B,C,K,H,W]; evaluate exp in FP32 under mixed precision."""
        with torch.autocast(device_type=normalized.device.type, enabled=False):
            z = normalized if normalized.dtype == torch.float64 else normalized.float()
            z = self.grid_limit * torch.tanh(z / self.grid_limit)
            centers = self.centers.to(dtype=z.dtype).view(1, 1, -1, 1, 1)
            return torch.exp(-((z.unsqueeze(2) - centers) / self.width).square())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        basis = self.basis(normalized).to(dtype=normalized.dtype)
        # Flatten only C and K, preserving each contiguous channel group.
        expanded = basis.flatten(1, 2)
        return self.rbf_projection(expanded) + self.base_projection(F.silu(normalized))


class TokenizedKANBlock(nn.Module):
    """Grouped token transforms + local spatial mixing + cross-group mixing.

    A token is one spatial location. Pointwise convolutions implement exactly
    the per-token channel maps without a costly NCHW -> BNC -> NCHW permutation.
    DWConv provides spatial context; the KAN alone is not global attention.
    """

    def __init__(self, channels: int, groups: int = 8, num_basis: int = 4,
                 expansion: int = 1, norm_eps: float = 1e-5):
        super().__init__()
        if not isinstance(expansion, int) or expansion < 1:
            raise ValueError("expansion must be a positive integer.")
        hidden = channels * expansion
        self.kan1 = GroupedFastKAN2d(channels, hidden, groups, num_basis, norm_eps=norm_eps)
        self.spatial = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.InstanceNorm2d(hidden, eps=norm_eps, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )
        self.kan2 = GroupedFastKAN2d(hidden, channels, groups, num_basis, norm_eps=norm_eps)
        # A dense pointwise map lets different KAN groups exchange information.
        self.mix = nn.Conv2d(channels, channels, 1, bias=False)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.mix(self.kan2(self.spatial(self.kan1(x))))


class FastKANSkipCalibrator(nn.Module):
    """Encoder-only residual calibration; no decoder-conditioned attention.

    Starts near identity, allowing both amplification and suppression without
    abruptly discarding weak tumor features through a sigmoid gate.
    """

    def __init__(self, channels: int, groups: int = 8, num_basis: int = 4,
                 norm_eps: float = 1e-5):
        super().__init__()
        self.kan = GroupedFastKAN2d(channels, channels, groups, num_basis, norm_eps=norm_eps)
        self.strength = nn.Parameter(torch.full((1, channels, 1, 1), 1e-2))

    def forward(self, skip: torch.Tensor) -> torch.Tensor:
        # tanh(strength) keeps the multiplicative factor strictly in (0, 2).
        return skip * (1 + torch.tanh(self.strength) * torch.tanh(self.kan(skip)))
