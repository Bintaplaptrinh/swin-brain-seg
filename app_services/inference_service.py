"""Model Inference Service for Swin V2-Tiny Classification and Swin Hybrid Segmentation."""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

from app_utils.imaging import compute_mask_metrics

# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]

CLASSIFIER_CKPT = ROOT_DIR / "outputs/checkpoints/classifer_swinv2_tiny/classifer_swinv2_tiny_binary.pt"
CLASSIFIER_ONNX = ROOT_DIR / "outputs/checkpoints/classifer_swinv2_tiny/classifer_swinv2_tiny_binary.onnx"
SEGMENTER_CKPT = ROOT_DIR / "outputs/checkpoints/swin_hybrid_512_unfrozen_seed2026/last.pt"
SEGMENTER_ONNX = ROOT_DIR / "outputs/checkpoints/exports/swin_hybrid_epoch159.onnx"

# Global operating mode: "auto" (detect real weights or fallback to simulation), "real", or "mock"
MODEL_MODE = os.getenv("MODEL_MODE", "auto").lower()
SCREENING_THRESHOLD = 0.40


def _is_git_lfs_pointer(path: Path) -> bool:
    """Check if a checkpoint file is just a text Git LFS pointer instead of binary weights."""
    if not path.exists():
        return True
    try:
        with open(path, "rb") as f:
            header = f.read(30)
            return b"version https://git-lfs" in header
    except Exception:
        return True


class MedicalInferenceEngine:
    """CADx Inference Engine supporting PyTorch/ONNX Runtime and Clinical Benchmark Emulation."""

    def __init__(self, mode: str = MODEL_MODE):
        if mode not in {"auto", "real", "mock"}:
            raise ValueError("MODEL_MODE must be auto, real, or mock")
        self.mode = mode
        self.error = None
        self.is_real_loaded = False
        self._initialize_backends()

    def _initialize_backends(self):
        """Check availability of actual PyTorch/ONNX checkpoints."""
        if self.mode == "mock":
            self.is_real_loaded = False
            return

        classifier_ok = CLASSIFIER_CKPT.exists() and not _is_git_lfs_pointer(CLASSIFIER_CKPT)
        segmenter_ok = SEGMENTER_CKPT.exists() and not _is_git_lfs_pointer(SEGMENTER_CKPT)

        if classifier_ok and segmenter_ok:
            try:
                # Attempt importing CADx pipeline from src
                from src.pipeline import CADxPipeline
                self.pipeline = CADxPipeline(
                    classifier_checkpoint=CLASSIFIER_CKPT,
                    segmenter_checkpoint=SEGMENTER_CKPT,
                    threshold=SCREENING_THRESHOLD,
                )
                self.is_real_loaded = True
            except Exception as exc:
                self.error = f"Model initialization failed: {exc}"
                self.is_real_loaded = False
        else:
            self.error = "Checkpoint files are missing or Git LFS pointers. Download binary weights to enable real inference."
            self.is_real_loaded = False

    def predict_classification(
        self,
        image: Image.Image,
        threshold: float = SCREENING_THRESHOLD,
    ) -> Dict[str, Any]:
        """Giai đoạn 1: Screen 2D MRI slice using Swin Transformer V2-Tiny.

        Returns:
            dict containing:
            - probability: float [0.0, 1.0]
            - is_positive: bool
            - threshold: float
            - model_name: str
            - status_text: str
        """
        gray_img = image.convert("L")

        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        if self.mode == "real" and not self.is_real_loaded:
            raise RuntimeError(self.error)
        if self.is_real_loaded:
            result = self.pipeline.classifier.predict_slice(
                gray_img, threshold=threshold, img_size=self.pipeline.cls_size
            )
            prob = float(result["tumor_probability"])
            return {
                "probability": prob, "is_positive": prob >= threshold,
                "threshold": threshold, "model_name": "Swin V2-Tiny",
                "execution_mode": "Suy luận bằng mô hình thật",
            }

        # Medical Image Analysis Simulation (Dynamic based on slice intensity distribution)
        arr = np.asarray(gray_img, dtype=np.float32)
        fg = arr > 25
        if not fg.any():
            prob = 0.005
        else:
            # High-intensity lesion search
            upper_percentile = np.percentile(arr[fg], 96) if len(arr[fg]) > 100 else 0
            if upper_percentile > 200:
                # Likely hyperintense tumor lesion
                contrast_factor = float(upper_percentile - arr[fg].mean()) / max(float(arr[fg].std()), 1.0)
                prob = min(0.995, max(0.65, 0.50 + contrast_factor * 0.15))
            else:
                prob = max(0.012, min(0.35, float(arr[fg].mean() / 400.0)))

        is_pos = prob >= threshold
        return {
            "probability": round(prob, 4),
            "is_positive": is_pos,
            "threshold": threshold,
            "model_name": "Thuật toán mô phỏng sàng lọc",
            "execution_mode": "MÔ PHỎNG — dựa trên cường độ ảnh, không chạy mô hình đã huấn luyện",
        }

    def predict_segmentation(
        self,
        image: Image.Image,
        threshold: float = 0.50,
        preset_mask: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        """Giai đoạn 2: Segment suspicious tumor region using Swin Hybrid (nnU-Net + 2D Swin).

        Returns:
            dict containing:
            - mask: PIL.Image (binary 0 / 255)
            - tumor_detected: bool
            - metrics: dict (tumor_pixels, tumor_ratio_percent, bbox)
            - model_name: str
        """
        gray_img = image.convert("L")

        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        if self.mode == "real" and not self.is_real_loaded:
            raise RuntimeError(self.error)
        if self.is_real_loaded:
            result = self.pipeline.predict(gray_img, force_segmentation=True)
            seg = result.get("stage2_segmentation")
            if seg is None or "probability_map" not in seg:
                raise RuntimeError("Segmenter did not return a probability map")
            mask_arr = (seg["probability_map"] >= threshold).astype(np.uint8) * 255
            metrics = compute_mask_metrics(mask_arr)
            return {
                "mask": Image.fromarray(mask_arr), "tumor_detected": metrics["tumor_pixels"] > 0,
                "metrics": metrics, "model_name": "Swin Hybrid",
                "execution_mode": "Suy luận bằng mô hình thật",
            }

        # Reference masks are intentionally never used as predictions.
        # Otsu/Intensity adaptive segmentation fallback for custom uploaded images
        arr = np.asarray(gray_img, dtype=np.float32)
        fg = arr > 25
        mask_arr = np.zeros_like(arr, dtype=np.uint8)

        if fg.any():
            threshold_val = np.percentile(arr[fg], 93)
            mask_arr = ((arr > max(threshold_val, 180)) & fg).astype(np.uint8) * 255
            # Keep largest connected component approximately
            from scipy import ndimage
            labeled, num_features = ndimage.label(mask_arr)
            if num_features > 0:
                sizes = ndimage.sum(mask_arr, labeled, range(1, num_features + 1))
                largest_label = np.argmax(sizes) + 1
                mask_arr = (labeled == largest_label).astype(np.uint8) * 255

        mask_img = Image.fromarray(mask_arr)
        metrics = compute_mask_metrics(mask_arr)

        return {
            "mask": mask_img,
            "tumor_detected": metrics["tumor_pixels"] > 0,
            "metrics": metrics,
            "model_name": "Thuật toán mô phỏng phân vùng",
            "execution_mode": "MÔ PHỎNG — dựa trên cường độ ảnh, không chạy mô hình đã huấn luyện",
        }


# Singleton instance for the Streamlit session
engine = MedicalInferenceEngine()
