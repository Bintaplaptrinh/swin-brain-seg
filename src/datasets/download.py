"""Fetch BRISC only when absent; retain the existing dataset without copying it."""
from pathlib import Path
import shutil

from .brisc import discover


def ensure_dataset(root, *, auto_download=False, kaggle_handle="briscdataset/brisc2025"):
    root = Path(root)
    if (root / "segmentation_task/train/images").is_dir():
        return root
    if not auto_download:
        raise FileNotFoundError(f"BRISC dataset not found at {root}; enable dataset.auto_download or run python -m scripts.download_data")
    import kagglehub
    cached = Path(kagglehub.dataset_download(kaggle_handle))
    candidates = [cached, *[p.parent for p in cached.rglob("segmentation_task") if p.is_dir()]]
    source = next((p for p in candidates if (p / "segmentation_task/train/images").is_dir()), None)
    if source is None:
        raise FileNotFoundError("Downloaded Kaggle dataset has no supported segmentation_task layout")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to merge a download into a nonempty dataset directory: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, root, dirs_exist_ok=True)
    discover(root, "train")
    discover(root, "test")
    return root
