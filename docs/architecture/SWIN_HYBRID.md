# Swin Hybrid architecture and checkpoint protocol

Current protocol: `swin_hybrid_512_unfrozen_v2`. Source: [model](../../src/models/swin_hybrid.py), [decoder](../../src/models/nnunet_biref.py), [attention](../../src/models/ema_attention.py), [trainer](../../src/trainers/swin_hybrid.py), [loss](../../src/trainers/hybrid_loss.py).

The input is `[B, 1, H, W]`, with dimensions at least 64 and divisible by 32. The retained experiment uses 512×512. Encoder channels are `[32,64,128,256,320,320]`, residual block counts `[1,3,4,6,6,6]`, and strides `[1,2,2,2,2,2]`. Five decoder levels each use two convolution blocks and produce optional deep-supervision logits.

The MONAI Swin encoder uses embedding width 24, four stages of depth 2, heads `[3,6,12,24]`, patch size 2 and window size 7. Its five feature scales have channels `[24,48,96,192,384]`. Pointwise projections with instance normalization and LeakyReLU map these features to CNN stages E2–E6. Each fusion is `cnn + tanh(scale) * projected_swin`, with scale initialized to 0.1. Residual depthwise refiners at E3–E6 use 3×3 kernels, dilation 1, a pointwise projection and scale initialized to 0.01.

The decoder pools E3–E6 to the bottleneck resolution, projects each to 16 channels, adds a global descriptor, and combines them through a residual context path. Image-detail references pack raw neighboring grayscale pixels by pixel-unshuffle at E1–E3. Efficient Multi-Scale Attention affects E3–E5 decoder skips through near-identity residual gates. Boundary heads at E2–E3 are used only while training. Efficient Multi-Scale Attention is unrelated to the exponential moving average of model weights.

Parameter counts, including auxiliary heads:

| Part | Trainable parameters |
|---|---:|
| Swin encoder | 1,988,850 |
| CNN, fusion, refiners and decoder | 36,812,172 |
| Total | 38,801,022 |

The original `Brain_Swin_UNETR.pth` is retained at `outputs/checkpoints/pretrained/`. Its SHA-256 is `75756f5cd01970d7a6d2a004bd7e03bd5a24c0b6d03eb84df2a21abd46c8d13f`. This is a 3D, three-channel state dictionary. The explicit 2D grayscale transfer sums input/depth patch-embedding kernels, takes the central depth-offset relative-position bias plane, regenerates 2D relative-position indices, and retains 8C patch-merging weights by repeating coplanar 2D samples in the original merge ordering. There are 126 encoder state tensors; 109 transfer unchanged. Source convolutional encoder/decoder weights are unused, and original pretraining provenance is not inferred from tensor names.

All Swin parameters remain trainable, with gradient checkpointing during training. AdamW uses CNN LR 0.0003, Swin LR 0.3 times the CNN LR, and weight decay 0.0001. A shared cosine multiplier preserves the ratio through the 1% LR floor. Physical batch 2 accumulates to effective batch 16. Robust augmentation and sampling weights 2.0/1.0/1.3 apply to Glioma/Meningioma/Pituitary respectively.

The loss is deep-supervised Dice + Focal + boundary supervision. With `p=clamp((epoch-1)/149,0,1)`, alpha=1, beta=0.5+0.5p, and gamma=0.01+0.49p. Letterbox padding is ignored. Weight EMA starts at epoch 20 and follows `min(0.9995, (1+t)/(10+t))`, where t counts successful EMA optimizer updates. Each unique tensor is updated once; AMP-overflow steps do not advance EMA.

Validation Glioma mIoU selects `best.pt`. Early stopping counts non-improving epochs after epoch 100, with patience 50 and delta 0.0001. The retained experiment stopped at epoch 159, selecting EMA epoch 109. Final EMA epoch 159 is also retained and is the default Rust export. Exact results and split definitions are in the [benchmark report](../benchmarks/README.md).

New training writes checkpoints to `outputs/checkpoints/<run>`, history to `outputs/logs/<run>`, and metadata/reports to `outputs/runs/<run>`. Checkpoint tensors and saved training protocol remain compatible after the code move. Immutable resume settings, initialization checksum, earlier best companion, saved memberships and history chronology are validated. The current CLI is `python train.py`; see the [README](../../README.md) for commands. Older design documents are preserved under [archive](../archive/architecture/).

Rust deployment uses a fixed-size ONNX graph. The export copy replaces divisible adaptive context pooling with mathematically equivalent fixed pooling, disables gradient checkpointing and returns full-resolution logits only. Original training modules and checkpoint bytes are unchanged. ONNX metadata records the source epoch, weight type, checkpoint/model hashes and parity check. Native Rust handles grayscale conversion, Pillow-compatible bilinear resize, nonzero normalization, letterboxing, continuous-logit restoration, argmax and mask writing. CPU runtime and numerical parity have been verified; CUDA compilation is checked separately.
