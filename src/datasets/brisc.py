"""BRISC binary masks, saved stratified splits, and reversible letterboxing."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset


TUMORS = {"gl": "Glioma", "me": "Meningioma", "pi": "Pituitary"}
PLANES = {"ax": "Axial", "co": "Coronal", "sa": "Sagittal"}
NAME = re.compile(r"brisc2025_(train|test)_\d+_(gl|me|pi)_(ax|co|sa)_t1$", re.I)
GLIOMA_POLICIES = {
    "mild": dict(rotation=5, scaling=.08, probability=.1, gamma=(.9, 1.1), noise=(.01, .04), lowres=(.8, .95)),
    "standard": dict(rotation=10, scaling=.15, probability=.2, gamma=(.7, 1.5), noise=(.01, .1), lowres=(.5, .85)),
    "strong": dict(rotation=15, scaling=.2, probability=.3, gamma=(.6, 1.6), noise=(.02, .12), lowres=(.5, .8)),
}


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path
    tumor: str
    plane: str

    @property
    def stratum(self):
        return self.tumor, self.plane


def discover(root, split):
    folder = Path(root) / "segmentation_task" / split
    images = sorted(p for p in (folder / "images").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise FileNotFoundError(f"No images in {folder / 'images'}")
    samples = []
    for image in images:
        match = NAME.fullmatch(image.stem)
        if not match or match[1].lower() != split:
            raise ValueError(f"Unrecognized BRISC segmentation filename: {image.name}")
        mask = folder / "masks" / (image.stem + ".png")
        if not mask.is_file():
            raise FileNotFoundError(f"Missing paired mask: {mask}")
        samples.append(Sample(image, mask, TUMORS[match[2].lower()], PLANES[match[3].lower()]))
    mask_names = {p.stem for p in (folder / "masks").glob("*.png")}
    if mask_names != {s.image.stem for s in samples}:
        raise ValueError(f"Unpaired masks in {folder}")
    return samples


def stratified_split(samples, val_fraction, seed):
    """Split by tumor x plane. These filenames do not contain patient IDs."""
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.stratum].append(sample)
    rng = random.Random(seed)
    train, val = [], []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda s: s.image.name)
        if len(group) < 2:
            raise ValueError(f"Need at least two samples in stratum {key}")
        rng.shuffle(group)
        count = min(len(group) - 1, max(1, round(len(group) * val_fraction)))
        val.extend(group[:count])
        train.extend(group[count:])
    return sorted(train, key=lambda s: s.image.name), sorted(val, key=lambda s: s.image.name)


def split_manifest(train, val, test):
    return {key: [s.image.name for s in samples] for key, samples in (("train", train), ("val", val), ("test", test))}


def restore_split(train_pool, test_pool, manifest):
    """Reject changed memberships and overlapping training/validation samples."""
    if set(manifest) != {"train", "val", "test"}:
        raise ValueError("Checkpoint must contain train, val, and test splits")
    for names in manifest.values():
        if not names or len(names) != len(set(names)):
            raise ValueError("Saved splits must be nonempty and contain no duplicates")
    train_names, val_names = set(manifest["train"]), set(manifest["val"])
    if train_names & val_names or train_names | val_names != {s.image.name for s in train_pool}:
        raise ValueError("Training/validation membership differs from saved split")
    if set(manifest["test"]) != {s.image.name for s in test_pool}:
        raise ValueError("Test membership differs from saved split")
    lookup = {s.image.name: s for s in train_pool + test_pool}
    return tuple([lookup[name] for name in manifest[key]] for key in ("train", "val", "test"))


def representative_subset(samples, limit):
    """Round-robin tumor/plane strata for the explicitly marked smoke test."""
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.stratum].append(sample)
    result = []
    for offset in range(max(map(len, groups.values()))):
        for key in sorted(groups):
            if offset < len(groups[key]):
                result.append(groups[key][offset])
                if len(result) == limit:
                    return result
    return result


def geometry(height, width, size):
    scale = min(size / height, size / width)
    rh, rw = max(1, round(height * scale)), max(1, round(width * scale))
    return rh, rw, (size - rh) // 2, (size - rw) // 2


def load_mask(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) >= 128


class BRISCDataset(Dataset):
    def __init__(self, samples, size=256, augment=False, augmentation="legacy", glioma_policy="standard"):
        if augmentation not in {"legacy", "robust", "glioma"} or glioma_policy not in GLIOMA_POLICIES:
            raise ValueError("Unknown augmentation profile")
        self.samples, self.size, self.augment = samples, size, augment
        self.augmentation = augmentation
        self.glioma_policy = glioma_policy

    @staticmethod
    def robust_spatial(image, mask, scale_range=(.85, 1.15)):
        """Centered scaling with a lesion-retention fallback; no random crops."""
        scale = random.uniform(*scale_range)
        w, h = image.size
        transform = (1 / scale, 0, (w - w / scale) / 2, 0, 1 / scale, (h - h / scale) / 2)
        resized_mask = mask.transform((w, h), Image.Transform.AFFINE, transform,
                                      resample=Image.Resampling.NEAREST, fillcolor=0)
        expected = (np.asarray(mask) >= 128).sum() * scale**2
        retained = (np.asarray(resized_mask) >= 128).sum()
        if expected and retained < .97 * expected:
            return image, mask
        return image.transform((w, h), Image.Transform.AFFINE, transform,
                               resample=Image.Resampling.BILINEAR, fillcolor=0), resized_mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        augment = self.augment and (self.augmentation != "glioma" or sample.tumor == "Glioma")
        robust = augment and self.augmentation in {"robust", "glioma"}
        policy = GLIOMA_POLICIES[self.glioma_policy if self.augmentation == "glioma" else "standard"]
        with Image.open(sample.image) as source:
            image = source.convert("L")
        with Image.open(sample.mask) as source:
            mask = Image.fromarray((np.asarray(source.convert("L")) >= 128).astype(np.uint8) * 255)
        if image.size != mask.size:
            raise ValueError(f"Image/mask size mismatch: {sample.image}")
        rh, rw, top, left = geometry(image.height, image.width, self.size)
        image = image.resize((rw, rh), Image.Resampling.BILINEAR)
        mask = mask.resize((rw, rh), Image.Resampling.NEAREST)
        if augment:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            rotation = policy["rotation"] if robust else 10
            angle = random.uniform(-rotation, rotation)
            original_image, original_mask = image, mask
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
            mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
            if robust:
                original_area = (np.asarray(original_mask) >= 128).sum()
                if original_area and (np.asarray(mask) >= 128).sum() < .97 * original_area:
                    image, mask = original_image, original_mask
                image, mask = self.robust_spatial(image, mask, (1-policy["scaling"], 1+policy["scaling"]))
                if random.random() < policy["probability"]:
                    image = image.filter(ImageFilter.GaussianBlur(random.uniform(.3, .9)))
                if random.random() < policy["probability"]:
                    scale = random.uniform(*policy["lowres"])
                    original_size = image.size
                    image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                                         Image.Resampling.BILINEAR).resize(original_size, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        foreground = array > 0
        if robust and random.random() < .3:
            array = array ** random.uniform(*policy["gamma"])
        if foreground.any():
            values = array[foreground]
            array[foreground] = (values - values.mean()) / max(float(values.std()), 1e-6)
        if augment:
            array[foreground] = array[foreground] * random.uniform(0.9, 1.1) + random.uniform(-0.1, 0.1)
            if robust and random.random() < policy["probability"]:
                array[foreground] += np.random.normal(0, random.uniform(*policy["noise"]), int(foreground.sum())).astype(np.float32)
        inputs = np.zeros((1, self.size, self.size), dtype=np.float32)
        targets = np.full((self.size, self.size), -1, dtype=np.int64)
        inputs[0, top:top + rh, left:left + rw] = array
        targets[top:top + rh, left:left + rw] = np.asarray(mask) >= 128
        return torch.from_numpy(inputs), torch.from_numpy(targets), index
