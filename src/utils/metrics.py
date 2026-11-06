"""Native-pixel binary overlap and contour-length-weighted surface metrics."""

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from surface_distance import metrics as surface

from src.datasets.brisc import PLANES, TUMORS


def binary_metrics(prediction, target, tolerance=2.0):
    prediction, target = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if prediction.shape != target.shape or target.ndim != 2:
        raise ValueError("Metrics require equal-sized 2D masks")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("NSD tolerance must be finite and nonnegative")
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    tn = int(target.size - tp - fp - fn)
    pred_empty, target_empty = not prediction.any(), not target.any()
    result = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "empty_prediction": pred_empty, "empty_target": target_empty,
    }
    if pred_empty and target_empty:
        # There is no contour to measure. Do not invent a zero-distance surface.
        result.update(hd95=None, assd=None, nsd=None)
    elif pred_empty or target_empty:
        # A missed lesion remains a failure in the aggregate distance score.
        result.update(hd95=math.inf, assd=math.inf, nsd=0.0)
    else:
        distances = surface.compute_surface_distances(target, prediction, spacing_mm=(1.0, 1.0))
        dg, dp = distances["distances_gt_to_pred"], distances["distances_pred_to_gt"]
        ag, ap = distances["surfel_areas_gt"], distances["surfel_areas_pred"]
        result.update(
            hd95=float(surface.compute_robust_hausdorff(distances, 95)),
            assd=float((np.sum(dg * ag) + np.sum(dp * ap)) / (ag.sum() + ap.sum())),
            nsd=float(surface.compute_surface_dice_at_tolerance(distances, tolerance)),
        )
    return result


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def summarize(rows):
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")

    def group(items):
        if not items:
            return {"images": 0, "mIoU": None, "Dice": None}
        return {"images": len(items), "mIoU": mean([r["iou"] for r in items]), "Dice": mean([r["dice"] for r in items])}

    per_class = {name: group([r for r in rows if r["tumor"] == name]) for name in TUMORS.values()}
    per_plane = {name: group([r for r in rows if r["plane"] == name]) for name in PLANES.values()}
    tp, fp, fn, tn = (sum(r[key] for r in rows) for key in ("tp", "fp", "fn", "tn"))
    overall = {
        **group(rows),
        "Recall": tp / (tp + fn) if tp + fn else None,
        "Precision": tp / (tp + fp) if tp + fp else None,
        "Specificity": tn / (tn + fp) if tn + fp else None,
        "micro_IoU": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "micro_Dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "HD95_px": mean([r["hd95"] for r in rows]),
        "ASSD_px": mean([r["assd"] for r in rows]),
        "NSD": mean([r["nsd"] for r in rows]),
        "empty_predictions": sum(r["empty_prediction"] for r in rows),
        "empty_targets": sum(r["empty_target"] for r in rows),
        "both_empty": sum(r["empty_prediction"] and r["empty_target"] for r in rows),
        "undefined_distance_images": sum(r["hd95"] is None for r in rows),
        "infinite_distance_images": sum(r["hd95"] == math.inf for r in rows),
        "confusion_pixels": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
    # Supplementary diagnostics; the headline distances above still include infinity.
    for metric in ("hd95", "assd"):
        overall[f"{metric.upper()}_finite_only_px"] = mean([r[metric] for r in rows if r[metric] is not None and math.isfinite(r[metric])])
    weighted = sum(v["images"] * v["mIoU"] for v in per_class.values() if v["images"]) / len(rows)
    return {"per_class": per_class, "Weighted_mIoU": weighted, "overall": overall, "per_plane": per_plane}


def json_safe(value):
    """Strict JSON: preserve infinite-distance failures explicitly as strings."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else None
    return value


# ---------------------------------------------------------------------------
# Image-level confusion matrix for Stage 1 (Classifier: Tumor vs No-Tumor)
# ---------------------------------------------------------------------------

def classifier_confusion_matrix(
    rows: List[Dict],
    *,
    tumor_key: str = "tumor",
    predicted_key: str = "segmenter_ran",
    no_tumor_label: str = "No Tumor",
) -> Dict:
    """Compute image-level confusion matrix for the two-stage CADx pipeline.

    Each ``row`` in *rows* must have:
    - ``tumor``         : ground-truth class string (``"No Tumor"`` for negatives)
    - ``segmenter_ran`` : bool — True when Stage 1 classified the image as positive
                          and Stage 2 was invoked.

    The classifier decision is treated as a binary positive/negative signal:
    - **Predicted Positive** → ``segmenter_ran is True``
    - **Predicted Negative** → ``segmenter_ran is False``

    Ground-truth label:
    - **Actual Positive** → ``tumor != no_tumor_label``
    - **Actual Negative** → ``tumor == no_tumor_label``

    Returns a dict containing:
    - ``tp``, ``fp``, ``fn``, ``tn`` : raw counts
    - ``sensitivity`` (Recall)       : TP / (TP + FN)
    - ``specificity``                : TN / (TN + FP)
    - ``precision`` (PPV)            : TP / (TP + FP)
    - ``npv``                        : TN / (TN + FN)
    - ``f1``                         : harmonic mean of precision and sensitivity
    - ``accuracy``                   : (TP + TN) / total
    - ``matrix``                     : 2×2 list [[TN, FP], [FN, TP]]  (actual × predicted)
    - ``labels``                     : axis labels for the matrix
    """
    if not rows:
        raise ValueError("classifier_confusion_matrix requires at least one row")
    missing_predicted = [i for i, r in enumerate(rows) if predicted_key not in r]
    if missing_predicted:
        raise ValueError(
            f"Rows at indices {missing_predicted[:5]} lack '{predicted_key}'. "
            "Pass --classifier-checkpoint to enable two-stage CADx evaluation."
        )

    tp = fp = fn = tn = 0
    for row in rows:
        actual_positive = row[tumor_key] != no_tumor_label
        predicted_positive = bool(row[predicted_key])
        if actual_positive and predicted_positive:
            tp += 1
        elif not actual_positive and predicted_positive:
            fp += 1
        elif actual_positive and not predicted_positive:
            fn += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    sensitivity = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    precision   = tp / (tp + fp) if (tp + fp) else None
    npv         = tn / (tn + fn) if (tn + fn) else None
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision is not None and sensitivity is not None and (precision + sensitivity) > 0
        else None
    )
    accuracy = (tp + tn) / total if total else None

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total": total,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "f1": f1,
        "accuracy": accuracy,
        # 2×2 matrix: rows = Actual (Negative first), cols = Predicted (Negative first)
        "matrix": [[tn, fp], [fn, tp]],
        "labels": {
            "rows": ["Actual Negative (No Tumor)", "Actual Positive (Tumor)"],
            "cols": ["Predicted Negative", "Predicted Positive"],
        },
    }


def print_classifier_confusion_matrix(cm: Dict, *, threshold: Optional[float] = None) -> None:
    """Print an image-level confusion matrix in ASCII table format."""
    sep = "-" * 58
    threshold_str = f" (threshold={threshold:.2f})" if threshold is not None else ""
    print(f"\n{'=' * 58}")
    print(f"  Stage 1 Classifier — Image-level Confusion Matrix{threshold_str}")
    print(sep)
    print(f"  {'':30s}  {'Pred NEG':>8}  {'Pred POS':>8}")
    print(sep)
    tn, fp = cm["matrix"][0]
    fn, tp = cm["matrix"][1]
    print(f"  {'Actual NEG (No Tumor)':30s}  {tn:>8}  {fp:>8}")
    print(f"  {'Actual POS (Tumor)':30s}  {fn:>8}  {tp:>8}")
    print(sep)
    print(f"  Total images : {cm['total']}")
    print(sep)

    def fmt(v):
        return f"{v:.5f}" if v is not None else "N/A"

    print(f"  Sensitivity (Recall)   : {fmt(cm['sensitivity'])}")
    print(f"  Specificity            : {fmt(cm['specificity'])}")
    print(f"  Precision (PPV)        : {fmt(cm['precision'])}")
    print(f"  NPV                    : {fmt(cm['npv'])}")
    print(f"  F1 Score               : {fmt(cm['f1'])}")
    print(f"  Accuracy               : {fmt(cm['accuracy'])}")
    print(f"{'=' * 58}\n", flush=True)


def plot_classifier_confusion_matrix(
    cm: Dict,
    output_path: Path,
    *,
    threshold: Optional[float] = None,
    title: str = "Stage 1 Classifier — Image-level Confusion Matrix",
) -> Path:
    """Render the image-level confusion matrix as a PNG file.

    Requires ``matplotlib`` (listed in requirements-dev.txt).
    Returns the resolved path of the saved PNG.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless — no display required
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for PNG output. "
            "Install it with: pip install matplotlib>=3.8"
        ) from exc

    output_path = Path(output_path)
    tn, fp = cm["matrix"][0]
    fn, tp_val = cm["matrix"][1]
    matrix = [[tn, fp], [fn, tp_val]]

    row_labels = ["No Tumor\n(Actual NEG)", "Tumor\n(Actual POS)"]
    col_labels = ["Predicted NEG\n(No Tumor)", "Predicted POS\n(Tumor)"]

    # Colour map: white→green (correct), white→red (errors)
    correct_cmap = LinearSegmentedColormap.from_list("correct", ["#ffffff", "#2ecc71"])
    error_cmap   = LinearSegmentedColormap.from_list("error",   ["#ffffff", "#e74c3c"])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_aspect("equal")

    total = cm["total"] or 1  # guard /0
    cell_colors = [
        [correct_cmap(tn / total),  error_cmap(fp / total)],
        [error_cmap(fn / total),    correct_cmap(tp_val / total)],
    ]

    for i in range(2):
        for j in range(2):
            count = matrix[i][j]
            pct   = 100.0 * count / total
            color = cell_colors[i][j]
            ax.add_patch(mpatches.FancyBboxPatch(
                (j + 0.05, 1 - i + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.02",
                linewidth=1.5,
                edgecolor="#555555",
                facecolor=color,
            ))
            # Count (large)
            ax.text(j + 0.5, 1 - i + 0.58, str(count),
                    ha="center", va="center", fontsize=28, fontweight="bold", color="#1a1a1a")
            # Percentage (small)
            ax.text(j + 0.5, 1 - i + 0.30, f"{pct:.1f}%",
                    ha="center", va="center", fontsize=13, color="#444444")
            # Cell label (TP / FP / FN / TN)
            cell_name = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}[(i, j)]
            ax.text(j + 0.88, 1 - i + 0.88, cell_name,
                    ha="right", va="top", fontsize=9, color="#888888",
                    style="italic")

    # Axis labels
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(row_labels[::-1], fontsize=11)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    # Title
    thr_str = f"  (threshold = {threshold:.2f})" if threshold is not None else ""
    ax.set_title(f"{title}{thr_str}", fontsize=12, fontweight="bold", pad=16)

    # Metrics footer
    def fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"

    footer = (
        f"Sensitivity={fmt(cm['sensitivity'])}  "
        f"Specificity={fmt(cm['specificity'])}  "
        f"Precision={fmt(cm['precision'])}  "
        f"F1={fmt(cm['f1'])}  "
        f"Accuracy={fmt(cm['accuracy'])}"
    )
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=9, color="#333333")

    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()
