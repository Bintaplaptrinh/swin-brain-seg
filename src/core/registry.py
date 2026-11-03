"""Public model registry used by the training entry point and tooling."""
from src.models.factory import VARIANTS, build_model as build_baseline

MODEL_NAMES = ("swin_hybrid", "kan", "depthwise", "nnunet_v2_depthwise")


def build_model(name, **options):
    if name == "swin_hybrid":
        from src.models.swin_hybrid import nnUNetSwinHybrid
        return nnUNetSwinHybrid(**options)
    if name == "nnunet_v2_depthwise":
        name = "depthwise"
    if name not in VARIANTS:
        raise ValueError(f"Unknown model {name!r}; available: {MODEL_NAMES}")
    return build_baseline(name, **options)
