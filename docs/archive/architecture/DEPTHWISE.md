> Historical document from before the 2026-11-03 reorganization. Commands, results and relative paths describe the original workspace. Use the current [README](../../../README.md) for supported commands and retained artifacts.

# Previous nnU-Net v2 + depthwise convolution experiment

This previous experiment is `architecture.nnUNetResEncDepthwise`, selected explicitly by `--variant depthwise`. The current default is the compact BiRef adaptation in [README.md](README.md). The depthwise model uses `ResidualEncoderUNet`, residual stem/shallow blocks and `UNetDecoder` from `dynamic-network-architectures`, the network library used by the local nnU-Net v2 repository. Deep stages use depthwise separable convolution. All skips feed the decoder directly.

The model contains **no KAN transforms, KAN skip calibrators, two-way attention, learned context tokens or SAM3 components**. Earlier implementations remain available under explicit legacy variants to load their checkpoints; see [LEGACY_KAN.md](LEGACY_KAN.md).

This is an nnU-Net v2 **network adaptation**, trained by this project's BRISC trainer. It does not claim to run nnU-Net v2's complete automatic planning, preprocessing, augmentation or training recipe.

## Architecture

| Stage | Channels | Resolution at 256 input | Blocks | Decoder connection |
|---|---:|---:|---|---|
| Stem | 32 | 256 x 256 | nnU-Net Conv3x3 + IN + LeakyReLU | — |
| E1 | 32 | 256 x 256 | 1 x nnU-Net BasicBlockD | Direct to D1 |
| E2 | 64 | 128 x 128 | 3 x nnU-Net BasicBlockD, first downsamples | Direct to D2 |
| E3 | 128 | 64 x 64 | 4 x nnU-Net BasicBlockD, first downsamples | Direct to D3 |
| E4 | 256 | 32 x 32 | Strided Conv3x3-IN-LeakyReLU + 2 depthwise blocks | Direct to D4 |
| E5 | 320 | 16 x 16 | Strided Conv3x3-IN-LeakyReLU + 2 depthwise blocks | Direct to D5 |
| E6 | 320 | 8 x 8 | Strided Conv3x3-IN-LeakyReLU + 2 depthwise blocks | Decoder input |

Each deep block is:

```text
x ---------------------------------------------------------- + --> output
 \-> dense Conv1x1 -> IN -> LeakyReLU                          |
     -> depthwise Conv7x7 (dilation=3, padding=9, groups=channels)
     -> IN -> LeakyReLU
     -> dense Conv1x1 -> per-channel residual scale (init 0.01) -/
```

Both pointwise convolutions mix channels densely; the depthwise convolution mixes spatial neighbors independently within each channel. Hidden width stays equal to the stage width. Native PyTorch `Conv2d(groups=channels)` works on CPU and CUDA without a custom extension. Stage widths, block counts, transitions and residual-scale initialization match the previous hybrid's budget; the KAN and attention paths are removed.

All six depthwise layers at E4-E6 default to kernel `(7,7)`, dilation `(3,3)`, stride `(1,1)` and padding `(9,9)`. Their effective spatial span is 19 x 19 with 49 learned weights per channel; no dense 19 x 19 kernel or explicit unfold is allocated. At small bottleneck resolutions some samples fall into padding. The nnU-Net stem, shallow residual stages, downsampling and decoder retain their 3 x 3 convolutions; pointwise mixing remains 1 x 1. Override with `--dw-kernel-size` and `--dw-dilation`; both settings are saved and checked before loading a checkpoint.

The decoder retains five ConvTranspose2d upsamplers, concatenated direct skips, one Conv3x3-IN-LeakyReLU per decoder stage, and two-channel logit heads. Deep supervision returns five tensors at input, 1/2, 1/4, 1/8 and 1/16 resolution. Inference returns the full-resolution tensor. All heads are always instantiated, so supervision can be toggled after checkpoint loading.

Input is grayscale `[B,1,H,W]`; H and W must be at least 64 and divisible by 32. Rectangular images are supported. Targets are binary background/tumor masks (`mask >= 128`), with padding ignored by the trainer. Glioma/meningioma/pituitary scores group images by tumor type; these are not three pixel labels. Preprocessing, native-size logit restoration and surface metric definitions are documented in the [project README](../README.md).

## Next experiment

The goal is **glioma mIoU >= 0.80**. `depthwise` selects `best.pt` by validation glioma mIoU; the supplied test set is evaluated only after selection. Legacy variants retain weighted-mIoU selection. Use `--selection-metric weighted-miou` to reproduce the earlier criterion, and state which criterion was used when comparing runs.

```powershell
.\run.bat --variant depthwise --dw-kernel-size 7 --dw-dilation 3 --epochs 50 --batch-size 32 --size 256 --workers 8 --seed 2026 --output runs\depthwise_k7_d3_seed2026
.\run.bat --variant depthwise --resume runs\depthwise_k7_d3_seed2026\last.pt
.\run.bat --variant depthwise --evaluate-only --resume runs\depthwise_k7_d3_seed2026\best.pt
```

The epoch budget, seed, stratified split procedure, CE + Dice, deep-supervision weights, AdamW, cosine decay, AMP and EMA match the saved two-way run; the current batch default is 32 (the earlier run used 16). The checkpoint-selection criterion reflects the glioma objective. Start a fresh run for the new kernel. To load earlier 3 x 3 depthwise checkpoints, pass `--dw-kernel-size 3 --dw-dilation 1` and their original training settings. Missing kernel/dilation fields in older depthwise checkpoints are interpreted as 3/1, preventing silent dilation changes.

CUDA depthwise runs enable cuDNN algorithm tuning by default, retaining AMP and contiguous NCHW layout. `--no-cudnn-benchmark` disables tuning; `--channels-last` is available for hardware comparisons but is slower in the local check. Performance settings are recorded in the run configuration. No full training job is started automatically by these changes.

## Verification and resource measurement

```powershell
python -m pytest src/tests architecture/tests -q
.\run.bat --smoke-test
python -m architecture.benchmark --variant depthwise --size 256 --batch-size 16 --amp --warmup 2 --steps 5 --output architecture/measurements/depthwise_k7_d3_b16_autotune.json
```

On the local RTX 3060, PyTorch 2.10/CUDA 13, the 7 x 7/dilation-3 model has **8,804,746 parameters**. Short synthetic batch-16 AMP forward/backward/SGD checks at 256 x 256 measured median steps of **221 ms** for NCHW, **279 ms** for channels-last, and **206 ms** for NCHW with cuDNN tuning. The tuned run used approximately **2,103 MiB peak allocated memory**. Each used two warmups and five measured steps, excludes data loading, EMA, AdamW state and validation, and is not an accuracy result or stable throughput study. Raw comparisons are in `measurements/depthwise_k7_d3_b16_*.json`. The older `depthwise_256_b16_amp.json` measures the previous 3 x 3 model.

Tests cover the absence of KAN/attention, unchanged skip tensors, depthwise grouping, rectangular multiscale output, nonzero gradients in each deep block, CUDA AMP updates and plans/checkpoint loading with supervision toggled. The training smoke check exercises AdamW, EMA, native-size evaluation and report generation on tiny subsets. The 0.80 objective requires a full accuracy experiment.

A separate three-step synthetic check at the current **batch 32, 256 x 256** setting completed AdamW + AMP + EMA updates with finite loss and **5,469 MiB (5.34 GiB) peak allocated memory**. It includes the student, EMA copy, optimizer and activations, and excludes data loading/validation. See [measurements/depthwise_k7_d3_b32_adamw_ema.json](measurements/depthwise_k7_d3_b32_adamw_ema.json). The test process exited; no full dataset training run was launched.

## Using the network with nnU-Net v2

`nnunet_architecture_depthwise.json` is an architecture fragment for a separate, compatible six-stage 2D nnU-Net plans copy. It is not a complete plans file or dataset conversion. Create baseline plans/preprocessing first, use a new plans name, keep the matching stride schedule `[1,2,2,2,2,2]` and preprocessing identifier, and expose the project root on `PYTHONPATH` for training and inference. The trainer supplies input channels, output classes and deep supervision.

The constructor supports the fragment's standard architecture kwargs. The old feature-map estimator is disabled because the replaced stages need independent profiling; this network is not registered as an automatic memory planner. Full nnU-Net preprocessing/training/inference integration has not been run.

## Files

- `nnunet_resenc_depthwise.py`: current convolution-only model and depthwise blocks.
- `nnunet_architecture_depthwise.json`: nnU-Net architecture fragment.
- `benchmark.py`: shared variant factory and synthetic resource benchmark; default `depthwise`.
- `tests/test_depthwise.py`: topology, skip flow, learning and CUDA AMP checks.
- `nnunet_resenc_kan.py`, `kan.py`, `two_way.py`: explicit legacy variants.
- [LEGACY_KAN.md](LEGACY_KAN.md): previous model details, references and measurements.
