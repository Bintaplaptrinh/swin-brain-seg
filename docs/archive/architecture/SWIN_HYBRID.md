> Historical document from before the 2026-11-03 reorganization. Commands, results and relative paths describe the original workspace. Use the current [README](../../../README.md) for supported commands and retained artifacts.

# Brain Swin + nnU-Net: 512x512 fine-tuning protocol

The current protocol is `swin_hybrid_512_unfrozen_v2`. `run_hybrid.bat` starts its independent trainer, `src.train_swin_hybrid`. The revised full experiment is for the user to launch. Preparation runs only a two-epoch subset smoke test.

## Current architecture

- **38,801,022 parameters, all trainable**: **1,988,850** Swin parameters and **36,812,172** nnU-Net/fusion/decoder parameters. The student remains below 50M.
- nnU-Net residual encoder widths `[32,64,128,256,320,320]`, block counts `[1,3,4,6,6,6]`, standard 3x3 residual convolutions and a full decoder.
- Local depthwise refiners at E3–E6 now use **3x3, dilation 1**, replacing historical 9x9 kernels. Pretrained Swin attention windows remain 7x7.
- Swin features fuse through adapters at E2–E6 before the next encoder stage. BiRef detail/context reconstruction, auxiliary boundary heads and Efficient Multi-Scale Attention on E3–E5 decoder skips remain.
- The entire Swin encoder is unfrozen and follows train/eval mode normally. Swin gradient checkpointing reduces activation memory at 512x512. Weight EMA is separate from skip attention.

The local `architecture/swinbrain/Brain_Swin_UNETR.pth` contains a 3D, three-input-channel Swin encoder. Its original data/modalities cannot be established from the state dictionary. Transfer to 2D grayscale sums patch-embedding depth/input-channel weights, takes the central depth-offset plane of relative-position bias, and preserves pretrained 8C patch-merging normalization/projection through coplanar merging. All 126 Swin tensors load with strict shape checks; 109 are unchanged during transfer. The transferred parameters are subsequently fine-tuned. Source convolution encoders and decoder are unused. SHA256 and transfer settings are recorded per run.

## Training defaults

| Setting | Value |
|---|---|
| Input | **512x512** |
| Epoch budget | 300 |
| Physical / effective batch | **2 / 16** (8 microbatches per full optimizer update) |
| Data workers | 4; smoke test uses 0 |
| Optimizer | AdamW, weight decay 0.0001, gradient clipping 12 |
| nnU-Net LR | 0.0003 |
| Swin LR | **0.3 x nnU-Net LR**, initially 0.00009 |
| LR schedule | Shared cosine multiplier down to 1% of each group's initial LR |
| Weight EMA | Starts at **epoch 20**, decay ceiling 0.9995 |
| Early stopping | Count after **epoch 100**, patience **50**, min delta 0.0001 |
| Checkpoint selection | Validation Glioma mean foreground IoU |
| Sampling weights | **Glioma 2.0, Meningioma 1.0, Pituitary 1.3** |
| Augmentation | Robust, all classes; RL disabled |
| VRAM budget | 9 decimal GB of PyTorch reserved memory |
| Output | `runs/swin_hybrid_512_unfrozen_seed2026` |

`LambdaLR` implements the shared cosine multiplier so the Swin/backbone LR ratio stays 0.3 throughout, including the LR floor. Two disjoint optimizer groups include all student parameters exactly once. Both group rates are logged and restored on resume.

For `alpha*Dice + beta*Focal + gamma*Boundary`, epoch progress is `(epoch-1)/149` clipped to `[0,1]`. **Alpha stays 1**, beta rises **0.5 to 1**, and **gamma rises from 0.01 to a maximum of 0.5** at epoch 150, then remains capped. Gamma controls the boundary contribution while preserving its gradients. Contour pixels retain 5x regional error weight plus auxiliary boundary supervision; padding remains excluded.

The original stratified split is unchanged: fit 3,146 images (917 Glioma), validation 787, test 860. Sampling weights alter expected frequency with replacement while preserving epoch length. Validation/test are not oversampled. Earliest early stop for a constant score is epoch 150, after weight EMA starts at 20. Final test uses the weights recorded in the selected checkpoint.

## Run and smoke test

```powershell
# Full revised experiment: the user launches this command.
.\run_hybrid.bat

# Functional check only: two epochs, nine representative slices per split,
# requested resolution and the same optimizer groups/accumulation.
.\run_hybrid.bat --smoke-test --output runs\verification_swin_hybrid_512_unfrozen_cuda

# Resume only a checkpoint created by the revised protocol.
.\run_hybrid.bat --resume runs\swin_hybrid_512_unfrozen_seed2026\last.pt
```

The full-run folder is fresh and separate from the historical frozen-Swin experiment. Old 256/frozen/9x9 checkpoints are explicitly rejected for resume because architecture, loss and optimizer protocol changed. Fresh runs initialize Swin from the original pretrained Brain checkpoint. Repeated smoke tests need a new output folder.

The smoke records that pretrained Swin patch weights actually update, both optimizer group rates, their ratio, Swin trainable count, resolution and completed epochs in `smoke_verification.json`. EMA's start at 20 and earliest early stop at 150 are covered by unit tests; the two-epoch smoke remains before EMA starts. Subset accuracy values do not establish model quality or long-run memory behavior.

Preparation verification passed: **64 related tests** and exactly **two CUDA smoke epochs at 512x512**. Swin patch weights changed, all 1,988,850 Swin parameters were trainable, the LR ratio stayed 0.3, and peak PyTorch reserved memory was **3.173 GB** before EMA was enabled. See [the smoke audit](../runs/verification_swin_hybrid_512_unfrozen_cuda/smoke_verification.json). The full-run output folder has not been created.

Checkpoint writes flush to disk before atomic replacement. `last.pt` stores student/optional EMA, optimizer groups, scheduler, scaler, early stopping and RNG state. The saved epoch row can reconstruct an interrupted final history line. Physical batch/workers may change on resume while effective batch stays fixed.

The optional supervisor `analysis/supervise_swin_hybrid.py` now requires a new **512/unfrozen v2** CUDA calibration and rejects the obsolete 256/frozen/batch-16 calibration. No supervisor or monitoring schedule is started during this change. Direct `run_hybrid.bat` is the manual entry point.

## Historical experiments

The previous 256/frozen/9x9 run stopped at epoch 103, selected student epoch 73, and achieved validation Glioma mIoU 0.733687 / test Glioma 0.673706. Its [report](../runs/swin_hybrid_300_seed2026/summary_vi.md) and checkpoints remain unchanged.

The independent component-filter experiment is entirely under [storage/postprocessing_10pct](../storage/postprocessing_10pct/README.md), including code, tests, results and saved masks. The main training/inference pipeline does not import or apply it.
