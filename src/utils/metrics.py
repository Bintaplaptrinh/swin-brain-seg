"""Native-pixel binary overlap and contour-length-weighted surface metrics."""

import math

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
