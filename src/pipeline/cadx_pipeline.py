"""Two-Stage CADx Pipeline: Swin Transformer V2-Tiny Classifier -> Swin Hybrid Segmenter.

Workflow:
    Stage 1 (Screening / Gating):
        Input MRI slice -> Swin Transformer V2-Tiny binary classifier.
        Calculates P(Tumor).
        - If P(Tumor) < threshold (default 0.40):
            Return the negative screening result and skip segmentation.
        - If P(Tumor) >= threshold:
            Mark as Positive / Tumor Detected.
            Proceeds to Stage 2.

    Stage 2 (Localization & Delineation):
        Input MRI slice -> Swin Hybrid (nnU-Net + Swin Transformer 2D).
        Restores the segmentation logits to native slice dimensions and returns the binary
        mask, foreground area, and bounding box.

Usage:
    from src.pipeline import CADxPipeline

    pipeline = CADxPipeline()
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
from torch.nn import functional as F

from src.core.paths import ROOT, relocate_config
from src.datasets.brisc import geometry
from src.models.swin_classifier import (
    DEFAULT_BACKBONE,
    DEFAULT_CLASSIFIER_CHECKPOINT,
    DEFAULT_CLASSIFIER_THRESHOLD,
    DEFAULT_IMG_SIZE,
    SwinTumorClassifier,
)


DEFAULT_SEGMENTER_CHECKPOINT = ROOT / "outputs/checkpoints/swin_hybrid_512_unfrozen_seed2026/last.pt"


class CADxPipeline:
    """Full two-stage CADx inference pipeline."""

    def __init__(
        self,
        classifier_checkpoint: Union[str, Path] = DEFAULT_CLASSIFIER_CHECKPOINT,
        segmenter_checkpoint: Optional[Union[str, Path]] = DEFAULT_SEGMENTER_CHECKPOINT,
        tokenizer_checkpoint: Optional[Union[str, Path]] = None,
        threshold: float = DEFAULT_CLASSIFIER_THRESHOLD,
        device: Optional[str] = None,
        classifier_img_size: int = DEFAULT_IMG_SIZE,
        segmenter_img_size: Optional[int] = None,
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Classifier threshold must be between 0 and 1")
        if classifier_checkpoint is None:
            raise ValueError("A trained classifier checkpoint is required")
        self.threshold = threshold
        self.cls_size = classifier_img_size
        self.seg_size = segmenter_img_size

        # 1. Load Stage 1 Classifier (Swin Transformer V2-Tiny)
        self.classifier = SwinTumorClassifier(
            pretrained=False,
            backbone_name=DEFAULT_BACKBONE,
            dropout=0.0,
            checkpoint_path=classifier_checkpoint,
        ).to(self.device).eval()

        # 2. Load Stage 2 Segmenter (optional if only screening is needed)
        self.segmenter = None
        if segmenter_checkpoint is not None:
            self._load_segmenter(segmenter_checkpoint, tokenizer_checkpoint)

    def _load_segmenter(
        self,
        segmenter_checkpoint: Union[str, Path],
        tokenizer_checkpoint: Optional[Union[str, Path]],
    ) -> None:
        from src.models.swin_hybrid import nnUNetSwinHybrid

        checkpoint = torch.load(segmenter_checkpoint, map_location="cpu", weights_only=True)
        config = relocate_config(checkpoint.get("config", {}))
        tok_ckpt = tokenizer_checkpoint or config["tokenizer_checkpoint"]
        self.seg_size = self.seg_size or config.get("size", 512)
        model = nnUNetSwinHybrid(
            tokenizer_checkpoint=tok_ckpt,
            depthwise_kernel=config.get("dw_kernel_size", 3),
            tokenizer_checkpointing=False,
        )
        if "tokenizer_sha256" in checkpoint and model.tokenizer.audit["sha256"] != checkpoint["tokenizer_sha256"]:
            raise ValueError("Tokenizer source differs from training")
        state = checkpoint.get("evaluation_model", checkpoint.get("model", checkpoint))
        model.load_state_dict(state, strict=True)
        model.decoder.deep_supervision = False
        self.segmenter = model.to(self.device).eval()

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
        screening = self.classifier.predict_slice(img_pil, threshold=self.threshold, img_size=self.cls_size)
        is_positive = screening["has_tumor"]

        result: Dict[str, Any] = {
            "source": source_name,
            "dimensions": (orig_w, orig_h),
            "stage1_screening": {
                "tumor_probability": screening["tumor_probability"],
                "threshold": self.threshold,
                "is_positive": is_positive,
                "status": "POSITIVE (Có u)" if is_positive else "NEGATIVE (Không u / Não lành)",
                "classifier_architecture": "Swin-Transformer-V2-Tiny",
            },
            "stage2_segmentation": None,
            "diagnosis": "Không phát hiện khối u" if not is_positive else "Phát hiện khối u",
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
        with torch.inference_mode():
            logits = self.segmenter(seg_input)
            cropped = logits[:, :, top:top + rh, left:left + rw]
            restored = F.interpolate(cropped.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            native_mask = restored.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            native_prob = restored.softmax(dim=1)[0, 1].cpu().numpy()

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
    parser.add_argument("--classifier-checkpoint", type=Path, default=DEFAULT_CLASSIFIER_CHECKPOINT)
    parser.add_argument("--segmenter-checkpoint", type=Path, default=DEFAULT_SEGMENTER_CHECKPOINT)
    parser.add_argument("--tokenizer-checkpoint", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=DEFAULT_CLASSIFIER_THRESHOLD)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    cli_args = parse_arguments()
    pipe = CADxPipeline(
        classifier_checkpoint=cli_args.classifier_checkpoint,
        segmenter_checkpoint=cli_args.segmenter_checkpoint,
        tokenizer_checkpoint=cli_args.tokenizer_checkpoint,
        threshold=cli_args.threshold,
    )
    res = pipe.predict(cli_args.image)
    segmentation = res["stage2_segmentation"]
    if segmentation is not None:
        res["stage2_segmentation"] = {
            key: value for key, value in segmentation.items() if key not in {"mask", "probability_map"}
        }
    if cli_args.output_json:
        report = json.dumps(res, ensure_ascii=False, indent=2)
        cli_args.output_json.write_text(report + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
