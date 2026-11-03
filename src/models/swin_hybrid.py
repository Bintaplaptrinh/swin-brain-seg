"""Large 2D nnU-Net with an audited, trainable 3D-to-2D Brain Swin encoder."""

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

from .nnunet_biref import BiReferenceDecoder, pointwise
from .ema_attention import ResidualEMASkip
from .budget import enforce_parameter_budget


class CoplanarPatchMerging(nn.Module):
    """Reuse 3D merge weights on two identical depth planes, without 3D tensors.

    Ordering exactly follows MONAI's legacy 3D PatchMerging: A,A,B,C,B,C,D,D.
    The original 8C LayerNorm and 8C->2C projection are loaded unchanged.
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, spatial_dims=2):
        super().__init__()
        self.norm = norm_layer(8 * dim)
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)

    def forward(self, x):
        _, h, w, _ = x.shape
        x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
        a, b = x[:, 0::2, 0::2], x[:, 1::2, 0::2]
        c, d = x[:, 0::2, 1::2], x[:, 1::2, 1::2]
        return self.reduction(self.norm(torch.cat((a, a, b, c, b, c, d, d), -1)))


class BrainTokenizer(nn.Module):
    def __init__(self, checkpoint, frozen=False, use_checkpoint=True):
        super().__init__()
        from monai.networks.nets.swin_unetr import SwinTransformer
        path = Path(checkpoint).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        original = torch.load(path, map_location="cpu", weights_only=True)
        state = {k.removeprefix("module.swinViT."): v for k, v in original.items()
                 if k.startswith("module.swinViT.")}
        if not state:
            raise ValueError("Brain checkpoint has no module.swinViT encoder weights")
        self.encoder = SwinTransformer(in_chans=1, embed_dim=24, window_size=(7, 7), patch_size=(2, 2),
                                      depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24), spatial_dims=2,
                                      downsample=CoplanarPatchMerging, use_checkpoint=use_checkpoint)
        converted, changes = {}, {}
        for key, target in self.encoder.state_dict().items():
            if key.endswith("relative_position_index"):
                converted[key] = target
                changes[key] = "regenerated 2D relative-position index (nonlearned buffer)"
                continue
            source = state[key]
            if key == "patch_embed.proj.weight":
                if tuple(source.shape) != (24, 3, 2, 2, 2):
                    raise ValueError(f"Unexpected pretrained patch embedding: {source.shape}")
                source = source.sum(2).sum(1, keepdim=True)
                changes[key] = "sum depth kernel and input channels: constant-depth, repeated-grayscale projection"
            elif key.endswith("relative_position_bias_table"):
                source = source.reshape(13, 13, 13, -1)[6].reshape(169, -1)
                changes[key] = "central depth-offset plane of 3D relative-position bias"
            if source.shape != target.shape:
                raise ValueError(f"Tokenizer transfer mismatch: {key}: {source.shape} versus {target.shape}")
            converted[key] = source
        self.encoder.load_state_dict(converted, strict=True)
        self.frozen = frozen
        self.requires_grad_(not frozen)
        if frozen:
            self.eval()
        parameter_count = sum(p.numel() for p in self.parameters())
        self.audit = {"source": str(path), "sha256": digest, "source_format": "3D, 3 input channels, feature_size=24",
                      "encoder_tensor_count": len(converted), "converted_or_regenerated": changes,
                      "unchanged_tensor_count": len(converted)-len(changes),
                      "frozen_parameters": parameter_count if frozen else 0,
                      "trainable_parameters": 0 if frozen else parameter_count,
                      "gradient_checkpointing": use_checkpoint,
                      "notes": "Swin encoder only; source convolution encoders/decoder/output are not used. This is an explicit 3D-to-2D transfer, not native 2D pretraining. Source training modalities/provenance cannot be established from the state dictionary."}

    def train(self, mode=True):
        return super().train(False if self.frozen else mode)

    def forward(self, x):
        if self.frozen:
            with torch.no_grad():
                return self.encoder(x, normalize=True)
        return self.encoder(x, normalize=True)


class LargeKernelRefiner(nn.Module):
    def __init__(self, channels, kernel=3):
        super().__init__()
        self.path = nn.Sequential(nn.Conv2d(channels, channels, kernel, padding=kernel//2, groups=channels, bias=False),
                                  nn.InstanceNorm2d(channels, affine=True), nn.LeakyReLU(.01, inplace=True),
                                  nn.Conv2d(channels, channels, 1, bias=False))
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), .01))

    def forward(self, x):
        return x + self.scale * self.path(x)


class nnUNetSwinHybrid(ResidualEncoderUNet):
    parameter_limit = 50_000_000
    ema_unique_updates = True

    def __init__(self, tokenizer_checkpoint, deep_supervision=True, *, depthwise_kernel=3,
                 freeze_tokenizer=False, tokenizer_checkpointing=True):
        if depthwise_kernel < 1 or depthwise_kernel % 2 != 1:
            raise ValueError("Depthwise kernel must be a positive odd integer")
        features = (32, 64, 128, 256, 320, 320)
        super().__init__(input_channels=1, n_stages=6, features_per_stage=features, conv_op=nn.Conv2d,
                         kernel_sizes=3, strides=(1, 2, 2, 2, 2, 2), n_blocks_per_stage=(1, 3, 4, 6, 6, 6),
                         num_classes=2, n_conv_per_stage_decoder=(2, 2, 2, 2, 2), conv_bias=True,
                         norm_op=nn.InstanceNorm2d, norm_op_kwargs={"eps": 1e-5, "affine": True},
                         nonlin=nn.LeakyReLU, nonlin_kwargs={"negative_slope": .01, "inplace": True},
                         deep_supervision=deep_supervision)
        self.apply(self.initialize)
        self.decoder = BiReferenceDecoder(self.decoder, features, 1)
        self.decoder.skip_attention = nn.ModuleDict({str(s): ResidualEMASkip(features[s-1], 32) for s in (3, 4, 5)})
        self.adapters = nn.ModuleList([pointwise(c, f) for c, f in zip((24, 48, 96, 192, 384), features[1:])])
        self.token_scales = nn.ParameterList([nn.Parameter(torch.full((1, f, 1, 1), .1)) for f in features[1:]])
        self.large_kernels = nn.ModuleDict({str(s): LargeKernelRefiner(features[s-1], depthwise_kernel) for s in (3, 4, 5, 6)})
        for module in (self.decoder.context, self.decoder.details, self.decoder.boundary_heads,
                       self.decoder.skip_attention, self.adapters, self.large_kernels):
            module.apply(self.initialize)
        # Add the pretrained module last so no He initializer touches its weights.
        self.tokenizer = BrainTokenizer(tokenizer_checkpoint, frozen=freeze_tokenizer, use_checkpoint=tokenizer_checkpointing)
        total = enforce_parameter_budget(self)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if not 20_000_000 <= trainable <= 50_000_000:
            raise ValueError(f"Trainable branch must contain 20-50M parameters; got {trainable:,}")
        self.parameter_counts = {"total": total, "trainable": trainable, "frozen": total-trainable}

    def forward(self, x):
        h, w = x.shape[-2:]
        if x.ndim != 4 or x.shape[1] != 1 or min(h, w) < 64 or h % 32 or w % 32:
            raise ValueError("Expected [B,1,H,W], H/W >=64 divisible by 32")
        tokens = self.tokenizer(x)
        value = self.encoder.stem(x)
        skips = []
        for index, block in enumerate(self.encoder.stages):
            value = block(value)
            if index > 0:
                projected = self.adapters[index-1](tokens[index-1])
                value = value + self.token_scales[index-1].tanh() * projected
            stage = str(index + 1)
            if stage in self.large_kernels:
                value = self.large_kernels[stage](value)
            skips.append(value)
        return self.decoder(skips, x)

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("Profile the hybrid; the original estimator omits Swin and fusion features")
