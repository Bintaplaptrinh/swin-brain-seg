"""Read-only checkpoint diagnostics; never fits weights or changes run files.

Run from the project root: python -m scripts.diagnose_glioma --run outputs/checkpoints/depthwise7_seed2026
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from src.models.factory import attention_options, build_model, depthwise_options, biref_options, ema_skip_options
from src.datasets.brisc import BRISCDataset, discover, geometry, load_mask, restore_split
from src.utils.metrics import json_safe
from src.trainers.baseline import evaluate, image_to_device, loader, write_rows
from src.trainers.glioma_policy import policy_split


def error_summary(rows):
    result = {}
    for tumor in sorted({r["tumor"] for r in rows}):
        group = [r for r in rows if r["tumor"] == tumor]
        tp, fp, fn = (sum(r[k] for r in group) for k in ("tp", "fp", "fn"))
        result[tumor] = {
            "images": len(group), "mIoU": float(np.mean([r["iou"] for r in group])),
            "pooled_recall": tp / (tp + fn), "pooled_precision": tp / (tp + fp),
            "iou_below_0.5": sum(r["iou"] < .5 for r in group),
            "per_plane": {p: {"images": sum(r["plane"] == p for r in group),
                              "mIoU": float(np.mean([r["iou"] for r in group if r["plane"] == p]))}
                          for p in sorted({r["plane"] for r in group})},
        }
    return result


def enrich(rows, samples, size):
    lookup = {s.image.name: s for s in samples}
    for row in rows:
        sample = lookup[row["filename"]]
        mask = load_mask(sample.mask)
        h, w = mask.shape
        rh, rw, *_ = geometry(h, w, size)
        small = Image.fromarray(mask.astype(np.uint8) * 255).resize((rw, rh), Image.Resampling.NEAREST)
        restored = np.asarray(small.resize((w, h), Image.Resampling.BILINEAR)) >= 128
        row["native_tumor_pixels"] = int(mask.sum())
        row["tumor_fraction"] = float(mask.mean())
        row["label_resize_roundtrip_iou"] = float((restored & mask).sum() / (restored | mask).sum())
    return rows


@torch.inference_mode()
def plot_examples(model, rows, samples, args, output):
    group = sorted((r for r in rows if r["tumor"] == "Glioma"), key=lambda r: r["iou"])
    selected = [group[i] for i in (0, 1, len(group)//4, len(group)//2, 3*len(group)//4, len(group)-1)]
    lookup = {s.image.name: s for s in samples}
    fig, axes = plt.subplots(len(selected), 3, figsize=(12, 3.5 * len(selected)), constrained_layout=True)
    for row_axes, row in zip(axes, selected):
        sample = lookup[row["filename"]]
        image = np.asarray(Image.open(sample.image).convert("L"))
        target = load_mask(sample.mask)
        inputs, _, _ = BRISCDataset([sample], args.size)[0]
        with torch.autocast(args.device, dtype=torch.float16, enabled=args.amp and args.device == "cuda"):
            logits = model(image_to_device(inputs[None], args))
        rh, rw, top, left = geometry(*target.shape, args.size)
        logits = F.interpolate(logits[:, :, top:top+rh, left:left+rw].float(), size=target.shape,
                               mode="bilinear", align_corners=False)
        pred = logits.argmax(1)[0].cpu().numpy().astype(bool)
        colors = np.zeros((*target.shape, 4), dtype=float)
        colors[target & pred] = (0.1, .95, .2, .6)
        colors[~target & pred] = (1, .15, .1, .75)
        colors[target & ~pred] = (.1, .6, 1, .8)
        for ax in row_axes:
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
        row_axes[0].contour(target, levels=[.5], colors=["cyan"], linewidths=.8)
        if pred.any():
            row_axes[1].contour(pred, levels=[.5], colors=["orange"], linewidths=.8)
        row_axes[2].imshow(colors)
        row_axes[0].set_title(f"GT (cyan): {sample.image.stem}\n{row['plane']}", fontsize=8)
        row_axes[1].set_title(f"EMA prediction (orange); IoU={row['iou']:.3f}")
        row_axes[2].set_title("Green: TP; red: FP; blue: FN")
    fig.suptitle("Validation glioma examples: worst, quartiles and best (native coordinates)", fontsize=14)
    fig.savefig(output / "validation_glioma_examples.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/runs/analysis/depthwise7"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli = parser.parse_args()
    cli.output.mkdir(parents=True, exist_ok=True)
    from src.core.paths import relocate_config
    checkpoint = torch.load(cli.run / "best.pt", map_location="cpu", weights_only=True)
    args = SimpleNamespace(**relocate_config(checkpoint["config"]))
    args.device, args.workers, args.batch_size = cli.device, 0, 16
    torch.manual_seed(2026)
    if args.device == "cpu":
        torch.set_num_threads(4)
    model = build_model(args.variant, args.groups, args.num_basis, **attention_options(args),
                        **depthwise_options(args), **biref_options(args), **ema_skip_options(args)).to(args.device)
    model.load_state_dict(checkpoint["ema"]["model"], strict=True)
    model.decoder.deep_supervision = False
    model.eval().requires_grad_(False)
    train_pool, test_pool = discover(args.data_root, "train"), discover(args.data_root, "test")
    train, val, _ = restore_split(train_pool, test_pool, checkpoint["splits"])
    if checkpoint.get("policy_split") is not None:
        train, _, _ = policy_split(train, args.policy_reward_fraction, args.seed, checkpoint["policy_split"])
    metrics, val_rows = evaluate(model, loader(val, args, torch.Generator().manual_seed(2026)), args, "Audit validation EMA")
    enrich(val_rows, val, args.size)
    write_rows(cli.output / "validation_per_image.csv", val_rows)
    glioma_train = [s for s in train if s.tumor == "Glioma"]
    chosen = np.random.default_rng(2026).choice(len(glioma_train), min(120, len(glioma_train)), replace=False)
    train_subset = [glioma_train[int(i)] for i in chosen]
    _, train_rows = evaluate(model, loader(train_subset, args, torch.Generator().manual_seed(2026)), args, "Audit training subset (no augmentation)")
    enrich(train_rows, train_subset, args.size)
    write_rows(cli.output / "training_glioma_subset.csv", train_rows)
    summary = {"checkpoint": str((cli.run / "best.pt").resolve()), "epoch": checkpoint["epoch"],
               "weights": "EMA", "training_updated": False,
               "train_counts": dict(Counter(s.tumor for s in train)),
               "validation": error_summary(val_rows), "training_subset": error_summary(train_rows),
               "roundtrip_mean_iou": {t: float(np.mean([r["label_resize_roundtrip_iou"] for r in val_rows if r["tumor"] == t]))
                                      for t in ("Glioma", "Meningioma", "Pituitary")},
               "notes": ["Training subset is a seeded random sample, not full training-set performance.",
                         "Mask resizing roundtrip is a reference, not a theoretical model accuracy ceiling.",
                         "Checkpoint was selected on validation; this audit does not establish causal mechanisms."]}
    for tumor in summary["validation"]:
        group = [r for r in val_rows if r["tumor"] == tumor]
        summary["validation"][tumor]["area_bins"] = {
            label: {"images": len(rr), "mIoU": float(np.mean([r["iou"] for r in rr])) if rr else None}
            for label, rr in [("<1%", [r for r in group if r["tumor_fraction"] < .01]),
                              ("1-5%", [r for r in group if .01 <= r["tumor_fraction"] < .05]),
                              (">=5%", [r for r in group if r["tumor_fraction"] >= .05])]}
    (cli.output / "diagnosis.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n")
    plot_examples(model, val_rows, val, args, cli.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
