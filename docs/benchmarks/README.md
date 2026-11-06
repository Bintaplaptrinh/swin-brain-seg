# Benchmark evidence

The comparison in the main README is generated from the full-data reports below. [comparison.json](comparison.json) and [comparison.csv](comparison.csv) retain unrounded values and SHA-256 hashes of their source reports. [split_audit.json](split_audit.json) confirms identical ordered training, validation and test memberships across the three retained model families.

| Experiment | Checkpoint | Full test report | Per-image evidence |
|---|---|---|---|
| Swin Hybrid, validation-selected EMA epoch 109 | [best.pt](../../outputs/checkpoints/swin_hybrid_512_unfrozen_seed2026/best.pt) | [original test report](../../outputs/runs/swin_hybrid_512_unfrozen_seed2026/test_metrics.json) | [CSV](../../outputs/runs/swin_hybrid_512_unfrozen_seed2026/test_per_image.csv) |
| Swin Hybrid, final EMA epoch 159 | [last.pt](../../outputs/checkpoints/swin_hybrid_512_unfrozen_seed2026/last.pt) | [epoch-159 test report](../../outputs/runs/swin_hybrid_512_unfrozen_seed2026/epoch159_test/test_metrics.json) | [CSV](../../outputs/runs/swin_hybrid_512_unfrozen_seed2026/epoch159_test/test_per_image.csv) |
| KAN, validation-selected EMA epoch 54 | [best.pt](../../outputs/checkpoints/kan_20261101_090911_204358/best.pt) | [test report](../../outputs/runs/kan_20261101_090911_204358/test_best/test_metrics.json) | [CSV](../../outputs/runs/kan_20261101_090911_204358/test_best/test_per_image.csv) |
| Custom 2D nnU-Net v2 + depthwise 7×7, EMA epoch 23 | [best.pt](../../outputs/checkpoints/depthwise7_seed2026/best.pt) | [original test report](../../outputs/runs/depthwise7_seed2026/test_metrics.json) | [CSV](../../outputs/runs/depthwise7_seed2026/test_per_image.csv) |

Each report evaluates 860 original segmentation test pairs. Tumor type and imaging plane come from filenames. Checkpoint selection uses validation Glioma mIoU for Swin/depthwise and validation weighted mIoU for KAN. The epoch-159 row is an explicitly evaluated final checkpoint; it does not replace validation-based model selection.

Weighted mIoU is the per-image foreground IoU mean. Masks are evaluated at native size after bilinear restoration of continuous logits. No postprocessing, threshold search, TTA or classification-only no-tumor images enter this table. These are slice-level comparisons; patient-independent generalization cannot be established from the available filenames.

The final Swin checkpoint stores **validation weighted mIoU 0.8264233872**. Its newly measured **test weighted mIoU is 0.8114617485**. The original end-of-training test report evaluates the best EMA checkpoint at epoch 109, obtaining **0.8146507907**. These values refer to different checkpoint/split combinations and must remain distinct.

The retained depthwise 7×7 model has test weighted mIoU **0.7937759536** and selected validation Glioma mIoU **0.6906380635**. The removed 3×3 predecessor obtained **0.7900836023** test weighted mIoU and **0.6852363186** selected validation Glioma mIoU. Its [evaluation report](historical/depthwise_seed2026/test_best/test_metrics.json) remains available; its binaries were removed after comparison.

GPU efficiency numbers are forward-only PyTorch measurements using CUDA AMP and an RTX 3060, batch 1. They exclude preprocessing, restoration, metrics and I/O. Different timing iteration counts are recorded in the reports. FLOPs cover supported operators only, with multiply-add = 2 FLOPs. They are not total arithmetic counts. Rust uses the exported float32 graph and has separate [functional parity evidence](../../outputs/runs/rust_parity_verified/parity.json).

Historical BiRef, two-way KAN, frozen-Swin and epoch-58 reports are under [historical](historical/). Their checkpoint binaries are intentionally absent. The earlier 1,000-image epoch-58 experiment adds classification-labeled no-tumor cases with generated empty masks and is not comparable to the 860-image segmentation benchmark. Synthetic profiling records under [measurements](measurements/) measure resource use, not accuracy.
