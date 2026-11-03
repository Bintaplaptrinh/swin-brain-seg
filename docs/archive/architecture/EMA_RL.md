> Historical document from before the 2026-11-03 reorganization. Commands, results and relative paths describe the original workspace. Use the current [README](../../../README.md) for supported commands and retained artifacts.

# EMA skips and glioma-only learned augmentation/sampling

The new default is **`biref-ema`**. The previous `biref_seed2026` experiment improved test glioma mIoU from 0.662226 to **0.682290**; its selected validation glioma mIoU was **0.697262**. This motivates the next experiment but does not establish that the new additions will reach the 0.80 objective.

No full training run was launched during implementation. All checks were bounded unit tests, synthetic updates or smoke/evaluation runs.

## Architecture and budgets

The BiRef backbone, context fusion, raw-image detail references, mask-boundary supervision and unique-tensor weight EMA remain in place. The new model adds **Efficient Multi-Scale Attention** to decoder skips **E3, E4 and E5**, before concatenation with upsampled decoder features. E1/E2 are left unmodified by default to preserve local detail and control activation memory. Encoder features passed to subsequent encoder stages are unchanged.

This attention is distinct from **exponential moving averaging of model weights**, which remains the validation/test protocol.

The attention pattern was checked against the user-supplied [EMA_attention_module](../repo/EMA-attention-module/EMA_attention_module), upstream `YOLOonMe/EMA-attention-module` at commit `afc37ecf52454c03421a86105fc5c16eb4ea1ad3`. It follows grouped coordinate/spatial branches and cross-spatial interaction from the [EMA paper](https://arxiv.org/abs/2305.13563). Cross-spatial score accumulation is FP32 under AMP. A residual wrapper uses:

```text
refined_skip = skip + tanh(strength) * (2 * EMA_attention(skip) - skip)
strength starts at 0.01 per channel
```

This near-identity adaptation keeps a direct feature path while the attention learns. It is not a claim that the original attention module used this wrapper. Default groups are 32; groups must divide the channels of every selected skip. `--ema-skip-stages` with no stage numbers disables all EMA skip modules.

The segmentation network has **8,858,988 parameters**, including boundary heads, an increase of 2,592 over the BiRef baseline. The policy controller has nine additional scalar preferences on CPU. Both are well below the 10M budget. A three-step batch-32, 256x256 check using actual `train_epoch`, AdamW, AMP, weight EMA and boundary loss measured **6.82 GB allocated / 7.38 GB reserved** on the RTX 3060. These are decimal GB tracked by PyTorch; CUDA context and other applications are additional. The synthetic check is not a full-run memory guarantee.

The existing 10M parameter guard, warning near 9 GB reserved memory, and stop above the configured maximum of 10 GB remain active. Measure memory again when increasing input size or adding high-resolution attention stages. Raw report: [biref_ema_b32_adamw_ema.json](measurements/biref_ema_b32_adamw_ema.json).

## Glioma-only augmentation

`--augmentation glioma` applies augmentation **only to images whose training tumor label is Glioma**. Meningioma and pituitary images receive deterministic preprocessing, with no flips, rotations or intensity augmentation. All validation, policy-reward and test images use deterministic preprocessing. The tumor label controls training data operations only; the inference model does not require that label.

Three bounded profiles are available: `mild`, `standard`, and `strong`. They vary rotation, centered scaling, gamma, blur/noise and simulated low resolution. Geometric transforms retain the existing lesion-area fallback, and masks use nearest-neighbor interpolation. No random crops or changes to annotation content are introduced. Augmentation transforms the whole glioma image/mask pair, not just tumor pixels.

Without the controller, select a fixed profile with `--glioma-policy` and a fixed sampling multiplier with `--glioma-sampling-weight`. The old `robust` and `legacy` profiles remain available unchanged for earlier models and ablations.

## Asynchronous policy learning

The user clarified that RL should learn glioma augmentation/sampling choices. The implementation is a **single-step policy-gradient bandit**, not a pixel-action segmentation agent or a full A3C implementation. Segmentation weights are still optimized by supervised CE + Dice + boundary loss.

- **Actions:** nine combinations of `{mild, standard, strong}` augmentation and `{1, 1.5, 2}` glioma sampling weight. Each action is held for one epoch.
- **Sampling:** non-glioma images have weight 1. A multiplier above 1 uses weighted sampling with replacement; epoch length remains the number of model-fitting slices. Multiplier 1 uses the existing shuffled traversal without replacement. A 2x sampling weight does not mean glioma is exactly half the batch.
- **Reward:** change in deterministic EMA glioma mIoU on the reserved policy holdout, multiplied by 100 and clipped to [-5,5]. Previous reward score is checkpointed. The controller never reads model-selection validation or test metrics as rewards.
- **Learning:** REINFORCE with an exponential reward baseline and an epsilon-mixture exploration policy. Default learning rate is 0.1 and exploration is 0.2. The first three epochs use standard augmentation/1x sampling and update the reward baseline without changing policy preferences.
- **Asynchrony:** a single CPU thread computes policy updates while the GPU/main process evaluates model-selection validation. The update is completed before checkpoint serialization and before choosing the next epoch's action. There are no GPU actors or extra model copies for RL.
- **Resume:** preferences, baseline, action visits, observation count, controller RNG, learning options, previous holdout score and exact reward membership are saved with the usual optimizer/model/RNG state. All pending work is flushed at checkpoint boundaries.

Reward-driven augmentation has precedent in [medical segmentation augmentation research](https://arxiv.org/abs/2002.09703). This small controller is an independent, resource-limited experiment, not a reproduction of that method or of [asynchronous actor–critic](https://arxiv.org/abs/1602.01783). Rewards are noisy and nonstationary because the segmenter is learning; 50 epochs may provide too little evidence to reliably distinguish all nine actions. Action/preference history is retained so this can be evaluated rather than assumed.

## Four data roles

Original source assignments remain **3,146 training / 787 validation / 860 supplied test** slices. With the default policy fraction 0.1, **92 glioma images** are reserved from the original training pool, stratified by plane. Effective model fitting uses **3,054 images**, including **825 glioma**, with the policy adjusting their sampling frequency.

| Data role | Used for |
|---|---|
| Effective fit set | Segmentation optimizer updates and augmentation |
| Reserved policy holdout | Deterministic glioma reward for the controller |
| Original validation set | Best checkpoint selection by glioma mIoU |
| Supplied test set | Final evaluation after checkpoint selection |

`splits.json` preserves the original source partition. **`policy_split.json` is authoritative for effective fit/reward membership** when a reward fraction is enabled. The two lists are disjoint, cover the original training membership, and cannot include validation/test images. No reward images reach the supervised training loader. Patient independence remains unestablished because patient IDs are unavailable.

The holdout reduces the number of distinct glioma images used for gradient fitting; oversampling changes exposure, not that count. Even with `--no-glioma-rl`, the new variant retains the same reserved split by default for a matched static-policy comparison. Set `--policy-reward-fraction 0` only with the controller disabled to use the complete source training pool.

## Run manually

```powershell
# New experiment, current defaults: 50 epochs, batch 32, 256x256, seed 2026
.\run.bat --variant biref-ema --output runs\biref_ema_rl_seed2026

# Resume the same experiment
.\run.bat --resume runs\biref_ema_rl_seed2026\last.pt

# Evaluate its selected checkpoint
.\run.bat --evaluate-only --resume runs\biref_ema_rl_seed2026\best.pt

# Previous BiRef baseline and its original augmentation protocol
.\run.bat --variant biref --output runs\biref_repeat_seed2026
```

Start fresh for the new architecture. Existing BiRef checkpoints remain compatible with explicit `--variant biref`; they are not silently converted into attention/policy checkpoints.

## Ablations

Use fresh output folders and compare validation glioma mIoU, recall/precision, missed-region counts, other tumor groups and memory, before testing the selected model once. The following controls retain the new variant's reward reservation unless explicitly stated:

| Experiment | Extra arguments |
|---|---|
| Static glioma augmentation and sampling, no attention | `--no-glioma-rl --ema-skip-stages` |
| Attention + static glioma policy | `--no-glioma-rl` |
| Learned glioma policy, no attention | `--ema-skip-stages` |
| Full candidate | Defaults |
| Attention alone against the previous full-data BiRef protocol | `--no-glioma-rl --policy-reward-fraction 0 --augmentation robust` |

Static-profile settings are ignored when RL controls them; use `--glioma-policy`/`--glioma-sampling-weight` with `--no-glioma-rl`. Resume checks the immutable architecture/training options, including policy settings.

## Verification and logs

Tests cover numerical agreement with the local EMA reference, attention gradients/bypass, decoder-only placement, AMP, parameter budget, glioma-only augmentation, sampler frequency/epoch length, disjoint policy splits, positive policy-gradient behavior, asynchronous thread execution and controller checkpoint/RNG restoration. An end-to-end CUDA smoke run performed an actual policy update with `--rl-warmup 0`; checkpoint resume and old BiRef evaluation were also verified.

```powershell
python -m pytest src/tests architecture/tests -q
.\run.bat --smoke-test --rl-warmup 0
python -m analysis.profile_training --variant biref-ema --size 256 --batch-size 32 --steps 3 --output architecture/measurements/biref_ema_b32_adamw_ema.json
```

`history.jsonl` includes the chosen profile, sampling weight, behavior probability, reward mIoU, clipped reward and learned preferences for each epoch. `last.pt` and `best.pt` contain full controller state and policy split assignments. Smoke scores are not accuracy results. See [the BiRef backbone description](README.md) for region/boundary loss and decoder details.
