# Workspace reorganization — 2026-11-03

The workspace now uses `configs`, `data`, `src`, `scripts`, `outputs`, `tests`, and a Rust `inference` crate. The output directory is spelled **outputs** consistently.

| Previous location | Current location |
|---|---|
| `dataset/brisc2025` | `data/data/brisc2025` |
| `architecture/*.py` | `src/models/` |
| `src/data.py` | `src/datasets/brisc.py` |
| `src/train.py`, `src/train_swin_hybrid.py` | `src/trainers/baseline.py`, `src/trainers/swin_hybrid.py` |
| `src/metrics.py` | `src/utils/metrics.py` |
| `analysis/*.py` | `scripts/` |
| `architecture/tests`, `src/tests` | `tests/models`, `tests` |
| `architecture/*.json` | `configs/model/plans/` |
| `architecture/measurements` | `docs/benchmarks/measurements/` |
| `repo` | `third_party/` |
| `storage` | `outputs/logs/legacy_storage/` |
| Run checkpoints | `outputs/checkpoints/<run>/` |
| Run histories | `outputs/logs/<run>/history.jsonl` |
| Run configs, splits, metrics | `outputs/runs/<run>/` |

Retained checkpoint families:

- `swin_hybrid_512_unfrozen_seed2026`: best EMA epoch 109 and final epoch 159, both fully resumable.
- `kan_20261101_090911_204358`: best EMA epoch 54 and last epoch 57, original budget 100 epochs.
- `depthwise7_seed2026`: best EMA epoch 23 and last epoch 50, 7×7 dilation-3 refinement.
- The original Brain Swin initialization is separately retained under `pretrained/`; the epoch-159 ONNX deployment artifact is under `exports/`.

Verification, two-way KAN, BiRef variants, frozen-Swin 300, epoch-58, and the lower-scoring depthwise checkpoint binaries were removed. Historical full-data reports remain available for provenance. Generated smoke/resume checkpoint binaries were removed after successful verification. No dataset images or masks were deleted.

The initial checkpoint inventory, file sizes, deletion decisions and retained-file SHA-256 values are in `outputs/logs/reorganization/checkpoints.json`. Current checkpoint paths, epochs and checksums are in `outputs/checkpoints/manifest.json`. Original checkpoint tensors/config payloads remain byte-for-byte unchanged; the moved dataset and initialization paths are resolved at load time. Original metadata JSON files remain historical records, including the paths and commands used at training time. Runtime paths are defined in `src/core/paths.py`.

Source before the reorganization is backed up as `outputs/logs/reorganization/source_before.zip`, excluding redundant model binaries. The one-time migration script is retained with the reorganization records and refuses to rerun after migration.

Validation: 129 Python tests; 3 Rust tests; Rust release build and CUDA-feature compilation; two-epoch CUDA Swin smoke with delayed EMA; Swin resume into a separate destination; one-epoch depthwise smoke; evaluation of final Swin, retained KAN and the lower-scoring depthwise predecessor; ONNX graph validation and real-input PyTorch parity; six native Rust input/mask parity cases; editable Python package installation. Compact numerical results are in `outputs/logs/reorganization/validation.json`.
