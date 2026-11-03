"""Prompt-free two-way feature refinement, inspired by SAM's attention decoder.

Reference: third_party/sam3/sam3/sam/transformer.py (TwoWayAttentionBlock).
This is a standalone implementation: no SAM3 imports, prompts or pretrained
weights. Sparse learned context tokens exchange information with dense image
features; the dense result returns through a small residual branch.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ProjectedAttention(nn.Module):
    """Multihead attention with a smaller internal width for cross attention."""

    def __init__(self, dim, heads, downsample=1, dropout=0.0):
        super().__init__()
        if min(dim, heads, downsample) < 1 or dim % downsample or (dim // downsample) % heads:
            raise ValueError("Attention width must be divisible by downsample; projected width must be divisible by heads.")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("Attention dropout must be finite and in [0, 1).")
        self.heads = heads
        self.head_dim = dim // downsample // heads
        self.dropout = dropout
        width = self.heads * self.head_dim
        self.query = nn.Linear(dim, width)
        self.key = nn.Linear(dim, width)
        self.value = nn.Linear(dim, width)
        self.output = nn.Linear(width, dim)

    def _heads(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value):
        attended = F.scaled_dot_product_attention(
            self._heads(self.query(query)), self._heads(self.key(key)), self._heads(self.value(value)),
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.output(attended.transpose(1, 2).flatten(2))


class TwoWayAttentionBlock(nn.Module):
    """Sparse self attention -> token-to-image -> MLP -> image-to-token.

    Names describe the querying side, as in SAM3: image-to-token uses dense
    image queries and sparse token keys/values, so it updates the IMAGE map.
    Unlike SAM's first block, sparse self attention always keeps its residual.
    No final token-only attention is needed because this model consumes the
    dense image output rather than prompt/mask tokens.
    """

    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.self_attention = ProjectedAttention(dim, heads, dropout=dropout)
        self.token_to_image = ProjectedAttention(dim, heads, downsample=2, dropout=dropout)
        self.image_to_token = ProjectedAttention(dim, heads, downsample=2, dropout=dropout)
        self.token_mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, dim))
        self.norm_self = nn.LayerNorm(dim)
        self.norm_cross = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.norm_image = nn.LayerNorm(dim)

    def forward(self, tokens, image, token_pe, image_pe):
        query = tokens + token_pe
        tokens = self.norm_self(tokens + self.self_attention(query, query, tokens))
        tokens = self.norm_cross(tokens + self.token_to_image(tokens + token_pe, image + image_pe, image))
        tokens = self.norm_mlp(tokens + self.token_mlp(tokens))
        image = self.norm_image(image + self.image_to_token(image + image_pe, tokens + token_pe, tokens))
        return tokens, image


def spatial_position_encoding(height, width, dim, device, dtype):
    """Dynamic 2D sine/cosine positions, including rectangular feature maps."""
    frequencies = torch.arange(dim // 4, device=device, dtype=torch.float32)
    frequencies = 10000.0 ** (-frequencies / (dim // 4))
    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height * (2 * math.pi)
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width * (2 * math.pi)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    py, px = yy.flatten()[:, None] * frequencies, xx.flatten()[:, None] * frequencies
    return torch.cat((py.sin(), py.cos(), px.sin(), px.cos()), dim=-1)[None].to(dtype=dtype)


class TwoWayFeatureRefiner(nn.Module):
    """O(HW * token_count) global context with a local feature bypass.

    Dense image self attention is intentionally absent. Each stage owns its
    context tokens; they are trained with segmentation loss and require no
    image-level labels or inference prompts. Spatial positions are added to
    queries/keys only. The original feature map bypasses all normalizations.
    """

    def __init__(self, channels, dim=128, heads=8, num_tokens=16, depth=2, dropout=0.0):
        super().__init__()
        if any(not isinstance(v, int) or v < 1 for v in (channels, dim, heads, num_tokens, depth)):
            raise ValueError("Attention channels, width, heads, token count and depth must be positive integers.")
        if dim % 4:
            raise ValueError("Attention width must be divisible by 4 for 2D position encoding.")
        self.input_projection = nn.Conv2d(channels, dim, 1)
        self.context_tokens = nn.Parameter(torch.empty(1, num_tokens, dim))
        nn.init.normal_(self.context_tokens, std=0.02)
        self.layers = nn.ModuleList(TwoWayAttentionBlock(dim, heads, dropout) for _ in range(depth))
        self.output_projection = nn.Conv2d(dim, channels, 1)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 1e-2))

    def forward(self, x):
        features = self.input_projection(x)
        batch, dim, height, width = features.shape
        image = features.flatten(2).transpose(1, 2)
        image_pe = spatial_position_encoding(height, width, dim, image.device, image.dtype)
        token_pe = self.context_tokens.to(dtype=image.dtype).expand(batch, -1, -1)
        tokens = token_pe
        for layer in self.layers:
            tokens, image = layer(tokens, image, token_pe, image_pe)
        refined = image.transpose(1, 2).reshape(batch, dim, height, width)
        return x + self.scale * self.output_projection(refined)
