"""Swin Hybrid, classification, and segmentation architectures."""

try:
    from .nnunet_resenc_depthwise import nnUNetResEncDepthwise
    from .nnunet_biref import nnUNetBiRef, nnUNetBiRefEMASkip
    from .nnunet_resenc_kan import nnUNetResEncKAN
    from .swin_hybrid import nnUNetSwinHybrid
except ImportError:
    nnUNetResEncDepthwise = None  # type: ignore
    nnUNetBiRef = None  # type: ignore
    nnUNetBiRefEMASkip = None  # type: ignore
    nnUNetResEncKAN = None  # type: ignore
    nnUNetSwinHybrid = None  # type: ignore

from .swin_classifier import SwinTumorClassifier, SwinTumorDataset

__all__ = [
    "nnUNetSwinHybrid",
    "nnUNetBiRefEMASkip",
    "nnUNetBiRef",
    "nnUNetResEncDepthwise",
    "nnUNetResEncKAN",
    "SwinTumorClassifier",
    "SwinTumorDataset",
]
