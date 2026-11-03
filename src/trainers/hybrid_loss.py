"""Dynamic Dice + Focal + boundary-region objective, with letterbox masking."""

import torch
from torch.nn import functional as F

from .baseline import boundary_loss, boundary_targets


def dynamic_weights(epoch):
    progress = min(1., max(0., (epoch - 1) / 149))
    return {"alpha": 1., "beta": .5 + .5 * progress, "gamma": .01 + .49 * progress}


def dynamic_loss(outputs, target, epoch):
    config = dynamic_weights(epoch)
    boundaries = outputs.get("boundary", []) if isinstance(outputs, dict) else []
    predictions = outputs["segmentation"] if isinstance(outputs, dict) else outputs
    predictions = predictions if isinstance(predictions, (list, tuple)) else [predictions]
    dice_total, focal_total = predictions[0].new_zeros((), dtype=torch.float32), predictions[0].new_zeros((), dtype=torch.float32)
    weights = [2. ** -i for i in range(len(predictions))]
    for weight, logits in zip(weights, predictions):
        logits = logits.float()
        labels = F.interpolate(target[:, None].float(), size=logits.shape[-2:], mode="nearest")[:, 0].long()
        valid = labels != -1
        if not valid.any():
            raise ValueError("No valid pixels at supervision scale")
        truth = (labels == 1).float()
        foreground = logits.softmax(1)[:, 1] * valid
        dice = (2 * (foreground * truth).sum((1, 2)) + 1e-5) / (foreground.sum((1, 2)) + truth.sum((1, 2)) + 1e-5)
        ce = F.cross_entropy(logits, labels, ignore_index=-1, reduction="none")
        class_weight = torch.where(labels == 1, .75, .25)
        focal = (class_weight * (1 - torch.exp(-ce)).square() * ce * valid).sum() / valid.sum()
        dice_total = dice_total + weight * (1 - dice.mean())
        focal_total = focal_total + weight * focal
    dice_total, focal_total = dice_total / sum(weights), focal_total / sum(weights)
    # Penalize region logits around true contours directly, not only auxiliary heads.
    main = predictions[0].float()
    contour, valid = boundary_targets(target, main.shape[-2:])
    error = (main.softmax(1)[:, 1:2] - (target == 1)[:, None].float()).abs()
    border_weight = (1 + 4 * contour) * valid
    region_boundary = (error * border_weight).sum() / border_weight.sum().clamp_min(1)
    edge = boundary_loss(boundaries, target) if boundaries else region_boundary.new_zeros(())
    boundary = region_boundary + .5 * edge
    result = config["alpha"] * dice_total + config["beta"] * focal_total + config["gamma"] * boundary
    return result, {"dice": float(dice_total.detach()), "focal": float(focal_total.detach()),
                    "boundary": float(boundary.detach()), **config}
