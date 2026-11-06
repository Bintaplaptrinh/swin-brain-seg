"""Medical Imaging Utilities for MRI Preprocessing, Segmentation Overlay, and Analysis."""

from __future__ import annotations

import io
from typing import Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


# Medical Red / Coral for high-contrast tumor overlay on grayscale MRI
DEFAULT_TUMOR_COLOR = (225, 29, 72)  # #E11D48 (Clinical Rose/Crimson)
CONTOUR_COLOR = (244, 63, 94)        # #F43F5E (Lighter edge highlight)


def load_mri_image(image_input: Union[bytes, io.BytesIO, Image.Image, str]) -> Image.Image:
    """Load an image and convert it to single-channel Grayscale (L)."""
    if isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_input))
    else:
        img = Image.open(image_input)
    return img.convert("L")


def apply_contrast_enhancement(gray_img: Image.Image, clip_limit: float = 1.35) -> Image.Image:
    """Enhance soft tissue contrast while preserving natural MRI anatomical gradients."""
    arr = np.asarray(gray_img, dtype=np.float32)
    fg_mask = arr > 0
    if not fg_mask.any():
        return gray_img

    mean_val = float(arr[fg_mask].mean())
    std_val = max(float(arr[fg_mask].std()), 1e-5)
    
    # Adaptive normalization
    norm = (arr - mean_val) / std_val
    enhanced = norm * clip_limit * 45.0 + 115.0
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    enhanced[~fg_mask] = 0
    return Image.fromarray(enhanced)


def compute_mask_metrics(mask: np.ndarray) -> Dict[str, Union[int, float, Optional[Tuple[int, int, int, int]]]]:
    """Compute lesion area, image total area, tumor ratio, and bounding box [x1, y1, x2, y2]."""
    binary = mask >= 128
    tumor_pixels = int(np.count_nonzero(binary))
    total_pixels = int(mask.size)
    ratio = (tumor_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    bbox = None
    if tumor_pixels > 0:
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox = (int(cmin), int(rmin), int(cmax), int(rmax))

    from scipy import ndimage
    _, region_count = ndimage.label(binary)
    return {
        "region_count": int(region_count),
        "tumor_pixels": tumor_pixels,
        "total_pixels": total_pixels,
        "tumor_ratio_percent": round(ratio, 2),
        "bbox": bbox,
    }


def create_clinical_overlay(
    raw_img: Image.Image,
    mask_img: Optional[Image.Image],
    opacity: float = 0.50,
    tumor_color: Tuple[int, int, int] = DEFAULT_TUMOR_COLOR,
    draw_bbox: bool = True,
) -> Image.Image:
    """Render medical overlay: raw MRI preserved in grayscale + translucent tumor region + contour + bbox."""
    rgb_base = raw_img.convert("RGB")
    if mask_img is None:
        return rgb_base

    mask_resized = mask_img.resize(raw_img.size, Image.Resampling.NEAREST)
    mask_arr = np.asarray(mask_resized, dtype=np.uint8) >= 128
    base_arr = np.asarray(rgb_base, dtype=np.float32)

    if not mask_arr.any():
        return rgb_base

    # Alpha blending on tumor pixels only
    blended = base_arr.copy()
    for c in range(3):
        blended[mask_arr, c] = (1.0 - opacity) * base_arr[mask_arr, c] + opacity * tumor_color[c]

    output_img = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(output_img)

    # Compute bounding box
    rows = np.any(mask_arr, axis=1)
    cols = np.any(mask_arr, axis=0)
    if np.any(rows) and np.any(cols) and draw_bbox:
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        # Clean subtle dashed or solid medical box
        draw.rectangle([cmin - 2, rmin - 2, cmax + 2, rmax + 2], outline="#0EA5E9", width=2)
        # Bounding box tag
        # The Vietnamese region label is provided by the viewer, avoiding raster font limitations.

    return output_img


# ---------------------------------------------------------------------------
# Benchmark Presets for Instant One-Click Medical AI Validation
# ---------------------------------------------------------------------------
BENCHMARK_PRESETS = {
    "glioma": {
        "id": "glioma",
        "name": "Ảnh mô phỏng — vùng sáng không đều",
        "subtype": "Glioma",
        "plane": "Coronal",
        "expected_result": "Tumor Dương tính",
        "bbox": (180, 140, 310, 260),
    },
    "meningioma": {
        "id": "meningioma",
        "name": "Ảnh mô phỏng — vùng sáng tròn",
        "subtype": "Meningioma",
        "plane": "Sagittal",
        "expected_result": "Tumor Dương tính",
        "bbox": (220, 120, 335, 230),
    },
    "pituitary": {
        "id": "pituitary",
        "name": "Ảnh mô phỏng — vùng sáng nhỏ",
        "subtype": "Pituitary",
        "plane": "Coronal",
        "expected_result": "Tumor Dương tính",
        "bbox": (230, 250, 305, 315),
    },
    "normal": {
        "id": "normal",
        "name": "Ảnh mô phỏng — không có vùng sáng",
        "subtype": "No Tumor",
        "plane": "Axial",
        "expected_result": "Tumor Âm tính",
        "bbox": None,
    },
}


def generate_benchmark_slice(preset_key: str, size: int = 512) -> Tuple[Image.Image, Optional[Image.Image]]:
    """Synthesize high-resolution anatomical MRI slice and matching binary mask for immediate testing."""
    preset = BENCHMARK_PRESETS[preset_key]
    img = Image.new("L", (size, size), 6)
    draw = ImageDraw.Draw(img)

    # Skull outer contour
    margin = 42
    draw.ellipse([margin, margin + 12, size - margin, size - margin - 12], fill=125, outline=215, width=4)

    # Brain parenchyma inner contour
    draw.ellipse([margin + 12, margin + 22, size - margin - 12, size - margin - 22], fill=72, outline=35, width=2)

    # Symmetrical lateral ventricles
    cx, cy = size // 2, size // 2 - 12
    draw.ellipse([cx - 42, cy - 35, cx - 12, cy + 35], fill=15)
    draw.ellipse([cx + 12, cy - 35, cx + 42, cy + 35], fill=15)

    # Anatomical texture noise
    arr = np.asarray(img, dtype=np.float32)
    noise = np.random.default_rng(2026).normal(0, 6.0, (size, size))
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=0.7))

    mask = None
    if preset["bbox"] is not None:
        mask = Image.new("L", (size, size), 0)
        m_draw = ImageDraw.Draw(mask)
        t_draw = ImageDraw.Draw(img)
        x1, y1, x2, y2 = preset["bbox"]

        if preset["subtype"] == "Glioma":
            # Irregular infiltrative morphology
            points = [
                (x1 + 10, y1 + 15), (x2 - 15, y1 + 5),
                (x2, (y1 + y2) // 2), (x2 - 25, y2),
                (x1 + 20, y2 - 10), (x1, (y1 + y2) // 2 + 10)
            ]
            t_draw.polygon(points, fill=225)
            m_draw.polygon(points, fill=255)
        elif preset["subtype"] == "Meningioma":
            # Solid circumscribed mass
            t_draw.ellipse([x1, y1, x2, y2], fill=238, outline=255, width=2)
            m_draw.ellipse([x1, y1, x2, y2], fill=255)
        else:
            # Sellar adenoma
            t_draw.ellipse([x1, y1, x2, y2], fill=198)
            m_draw.ellipse([x1, y1, x2, y2], fill=255)

        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=0.3))
        mask = Image.fromarray((np.asarray(mask) >= 128).astype(np.uint8) * 255)

    return img, mask
