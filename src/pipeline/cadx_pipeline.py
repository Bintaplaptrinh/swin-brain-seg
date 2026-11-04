"""Two-Stage CADx Pipeline: Swin Transformer V2-Tiny Classifier -> Swin Hybrid Segmenter.

Workflow:
    Stage 1 (Screening / Gating):
        Input MRI slice -> Swin Transformer V2-Tiny binary classifier.
        Calculates P(Tumor).
        - If P(Tumor) < threshold (default 0.40):
            Mark as Negative / No Tumor.
            Pipeline halts immediately, saving compute and eliminating false positive masks.
        - If P(Tumor) >= threshold:
            Mark as Positive / Tumor Detected.
            Proceeds to Stage 2.

    Stage 2 (Localization & Delineation):
        Input MRI slice -> Swin Hybrid (nnU-Net + Swin Transformer 2D).
        Extracts pixel-accurate binary mask, unpads and restores to native slice dimensions,
        and computes clinical metrics (area, percentage, bounding box).

Usage:
    from src.pipeline import CADxPipeline

    pipeline = CADxPipeline(
        classifier_checkpoint="outputs/classifier/best_swinv2_tiny_binary.pt",
        segmenter_checkpoint="outputs/checkpoints/swin_hybrid_512_unfrozen_seed2026/best.pt",
        tokenizer_checkpoint="outputs/checkpoints/pretrained/model_swinvit.pt",
        threshold=0.40,
    )
    result = pipeline.predict("path/to/brain_mri.jpg")
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch

from src.models.swin_classifier import (
    DEFAULT_IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SwinTumorClassifier,
)


def geometry(height: int, width: int, size: int) -> Tuple[int, int, int, int]:
    """Computes aspect-preserving letterbox dimensions and offsets."""
    scale = min(size / height, size / width)
    rh = max(1, round(height * scale))
    rw = max(1, round(width * scale))
    return rh, rw, (size - rh) // 2, (size - rw) // 2


class CADxPipeline:
    """Full two-stage CADx inference pipeline."""

    def __init__(
        self,
        classifier_checkpoint: Optional[Union[str, Path]] = None,
        segmenter_checkpoint: Optional[Union[str, Path]] = None,
        tokenizer_checkpoint: Optional[Union[str, Path]] = None,
        threshold: float = 0.40,
        device: Optional[str] = None,
        classifier_img_size: int = DEFAULT_IMG_SIZE,
        segmenter_img_size: int = 512,
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.threshold = threshold
        self.cls_size = classifier_img_size
        self.seg_size = segmenter_img_size

        # 1. Load Stage 1 Classifier (Swin Transformer V2-Tiny)
        self.classifier = SwinTumorClassifier(
            pretrained=False,
            dropout=0.0,
            checkpoint_path=classifier_checkpoint,
        ).to(self.device).eval()

        # 2. Load Stage 2 Segmenter (optional if only screening is needed)
        self.segmenter = None
        if segmenter_checkpoint is not None and Path(segmenter_checkpoint).is_file():
            self._load_segmenter(segmenter_checkpoint, tokenizer_checkpoint)

    def _load_segmenter(
        self,
        segmenter_checkpoint: Union[str, Path],
        tokenizer_checkpoint: Optional[Union[str, Path]],
    ) -> None:
        try:
            from src.models.swin_hybrid import nnUNetSwinHybrid
        except ImportError as err:
            print(f"Warning: Could not import nnUNetSwinHybrid ({err}). Segmenter disabled.")
            return

        seg_path = Path(segmenter_checkpoint)
        ckpt = torch.load(seg_path, map_location="cpu", weights_only=False)

        cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        dw_kernel = cfg.get("dw_kernel_size", 3)

        tok_ckpt = tokenizer_checkpoint or cfg.get("tokenizer_checkpoint")
        if tok_ckpt is None or not Path(tok_ckpt).is_file():
            print(f"Warning: Swin tokenizer checkpoint not found ({tok_ckpt}). Segmenter disabled.")
            return

        model = nnUNetSwinHybrid(
            tokenizer_checkpoint=tok_ckpt,
            deep_supervision=False,
            depthwise_kernel=dw_kernel,
            freeze_tokenizer=True,
            tokenizer_checkpointing=False,
        )

        state = ckpt.get("model", ckpt)
        filtered_state = {k: v for k, v in state.items() if not k.startswith("aux_heads.")}
        model.load_state_dict(filtered_state, strict=False)
        self.segmenter = model.to(self.device).eval()

    def _preprocess_for_classifier(self, img_pil: Image.Image) -> torch.Tensor:
        """Direct resize 224x224 and z-score normalization for classifier."""
        resized = img_pil.resize((self.cls_size, self.cls_size), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0

        foreground = arr > 0
        if foreground.any():
            vals = arr[foreground]
            arr[foreground] = (vals - vals.mean()) / max(float(vals.std()), 1e-6)

        tensor_3c = np.stack([arr, arr, arr], axis=0)
        for c in range(3):
            tensor_3c[c] = (tensor_3c[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

        return torch.from_numpy(tensor_3c).unsqueeze(0).float().to(self.device)

    def _preprocess_for_segmenter(self, img_pil: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """Letterbox padding to 512x512 and z-score normalization for segmenter."""
        w, h = img_pil.size
        rh, rw, top, left = geometry(h, w, self.seg_size)

        resized = img_pil.resize((rw, rh), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0

        foreground = arr > 0
        if foreground.any():
            vals = arr[foreground]
            arr[foreground] = (vals - vals.mean()) / max(float(vals.std()), 1e-6)

        canvas = np.zeros((self.seg_size, self.seg_size), dtype=np.float32)
        canvas[top:top + rh, left:left + rw] = arr

        tensor_1c = torch.from_numpy(canvas).unsqueeze(0).unsqueeze(0).float().to(self.device)
        return tensor_1c, (rh, rw, top, left)

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        force_segmentation: bool = False,
    ) -> Dict[str, Any]:
        """Runs the two-stage CADx diagnosis on a single 2D MRI slice."""
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            with Image.open(img_path) as pil_img:
                img_pil = pil_img.convert("L")
            source_name = img_path.name
        elif isinstance(image_input, np.ndarray):
            img_pil = Image.fromarray(image_input).convert("L")
            source_name = "numpy_array"
        elif isinstance(image_input, Image.Image):
            img_pil = image_input.convert("L")
            source_name = "pil_image"
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        orig_w, orig_h = img_pil.size

        # -------------------------------------------------------------
        # STAGE 1: Screening Gating (Swin Transformer V2-Tiny)
        # -------------------------------------------------------------
        cls_input = self._preprocess_for_classifier(img_pil)
        with torch.no_grad():
            cls_out = self.classifier(cls_input)
            tumor_prob = float(torch.sigmoid(cls_out.squeeze()).item())

        is_positive = bool(tumor_prob >= self.threshold)

        result: Dict[str, Any] = {
            "source": source_name,
            "dimensions": (orig_w, orig_h),
            "stage1_screening": {
                "tumor_probability": round(tumor_prob, 4),
                "threshold": self.threshold,
                "is_positive": is_positive,
                "status": "POSITIVE (Có u)" if is_positive else "NEGATIVE (Không u / Não lành)",
                "classifier_architecture": "Swin-Transformer-V2-Tiny",
            },
            "stage2_segmentation": None,
            "diagnosis": "Não lành tính (No Tumor)" if not is_positive else "Phát hiện khối u - Đã phân đoạn",
        }

        # Early exit if negative
        if not is_positive and not force_segmentation:
            return result

        # -------------------------------------------------------------
        # STAGE 2: Tumor Delineation (Swin Hybrid Segmentation)
        # -------------------------------------------------------------
        if self.segmenter is None:
            result["stage2_segmentation"] = {
                "error": "Segmenter checkpoint not provided. Only screening result available."
            }
            return result

        seg_input, (rh, rw, top, left) = self._preprocess_for_segmenter(img_pil)
        with torch.no_grad():
            seg_logits = self.segmenter(seg_input)
            seg_prob = torch.sigmoid(seg_logits).squeeze().cpu().numpy()

        # Unpad letterbox back to native dimensions
        cropped_prob = seg_prob[top:top + rh, left:left + rw]
        prob_pil = Image.fromarray(cropped_prob).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
        native_prob = np.asarray(prob_pil, dtype=np.float32)
        native_mask = (native_prob >= 0.50).astype(np.uint8)

        # Compute clinical metrics
        tumor_pixel_count = int(native_mask.sum())
        total_pixel_count = orig_w * orig_h
        tumor_percentage = round((tumor_pixel_count / total_pixel_count) * 100, 3)

        # Bounding box
        if tumor_pixel_count > 0:
            rows = np.any(native_mask, axis=1)
            cols = np.any(native_mask, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            bbox = [int(xmin), int(ymin), int(xmax - xmin + 1), int(ymax - ymin + 1)]
        else:
            bbox = None

        result["stage2_segmentation"] = {
            "mask": native_mask,
            "probability_map": native_prob,
            "tumor_pixel_count": tumor_pixel_count,
            "tumor_area_percentage": tumor_percentage,
            "bounding_box_xywh": bbox,
            "segmenter_architecture": "Swin-Hybrid (nnU-Net + Swin Transformer 2D)",
        }
        return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="Path to input brain MRI slice (.jpg, .png)")
    parser.add_argument("--classifier-checkpoint", default="outputs/classifier/best_swinv2_tiny_binary.pt")
    parser.add_argument("--segmenter-checkpoint", default="outputs/checkpoints/swin_hybrid_512_unfrozen_seed2026/best.pt")
    parser.add_argument("--tokenizer-checkpoint", default="outputs/checkpoints/pretrained/model_swinvit.pt")
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_arguments()
    pipe = CADxPipeline(
        classifier_checkpoint=cli_args.classifier_checkpoint,
        segmenter_checkpoint=cli_args.segmenter_checkpoint,
        tokenizer_checkpoint=cli_args.tokenizer_checkpoint,
        threshold=cli_args.threshold,
    )
    res = pipe.predict(cli_args.image)
    print(json.dumps({k: v for k, v in res.items() if k != "stage2_segmentation"}, indent=2))
