"""Swin Transformer V2-Tiny Binary Tumor Classifier (Stage 1 Screening Gating).

Task:
    Binary Classification of 2D Brain MRI Slices:
    - Label 0: Không có khối u (Não lành / No Tumor)
    - Label 1: Có khối u (Tumor present: Glioma, Meningioma, Pituitary)

Role in CADx System:
    - If P(Tumor) < threshold (default 0.40):
        Kết luận não lành tính. Dừng pipeline ngay lập tức, tiết kiệm 100% chi phí tính toán.
    - If P(Tumor) >= threshold:
        Phát hiện có khối u. Chuyển lát cắt sang Giai đoạn 2 (Swin-Hybrid) để phân đoạn.

Architecture:
    Swin Transformer V2-Tiny (timm 'swinv2_cr_tiny_ns_224', ~27.57M parameters).
    - Đầu ra: 1 logit duy nhất -> torch.sigmoid(logit) = P(Tumor).
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import timm

from src.core.paths import ROOT


# ---------------------------------------------------------------------------
# Default Constants
# ---------------------------------------------------------------------------
SEED = 2026
DEFAULT_IMG_SIZE = 224
DEFAULT_BACKBONE = "swinv2_cr_tiny_ns_224"
DEFAULT_CLASSIFIER_THRESHOLD = 0.40
DEFAULT_CLASSIFIER_CHECKPOINT = (
    ROOT / "outputs/checkpoints/classifer_swinv2_tiny/classifer_swinv2_tiny_binary.pt"
)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SWIN_BACKBONE_CANDIDATES = [
    DEFAULT_BACKBONE,
    "swin_tiny_patch4_window7_224",
    "swin_s3_tiny_224",
]


def preprocess_classifier(img_pil: Image.Image, img_size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    resized = img_pil.resize((img_size, img_size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0

    foreground = arr > 0
    if foreground.any():
        vals = arr[foreground]
        arr[foreground] = (vals - vals.mean()) / max(float(vals.std()), 1e-6)

    tensor_3c = np.stack([arr, arr, arr], axis=0)
    for c in range(3):
        tensor_3c[c] = (tensor_3c[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    return tensor_3c


# ---------------------------------------------------------------------------
# Dataset Nhị Phân (0: Không u, 1: Có u)
# ---------------------------------------------------------------------------
class SwinTumorDataset(Dataset):
    """Nạp lát cắt MRI nhị phân: 0 (No Tumor) vs 1 (Tumor)."""

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        img_size: int = DEFAULT_IMG_SIZE,
        augment: bool = False,
    ):
        self.img_size = img_size
        self.augment = augment
        self.samples: List[Tuple[Path, int]] = []

        root_path = Path(root)
        split_dir = root_path / "classification_task" / split
        if not split_dir.is_dir():
            split_dir = root_path / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Thư mục không tồn tại: {split_dir}")

        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            folder_name = cls_dir.name.lower().strip()
            # Nhãn 0: Não lành (no_tumor) | Nhãn 1: Có u (glioma, meningioma, pituitary)
            label = 0 if folder_name in ("no_tumor", "notumor", "healthy", "normal") else 1

            for img_path in sorted(cls_dir.glob("*.jpg")):
                self.samples.append((img_path, label))

        if not self.samples:
            raise RuntimeError(f"Không tìm thấy ảnh nào trong: {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def class_counts(self) -> Dict[str, int]:
        neg = sum(1 for _, lbl in self.samples if lbl == 0)
        pos = sum(1 for _, lbl in self.samples if lbl == 1)
        return {"no_tumor (label 0)": neg, "tumor (label 1)": pos}

    @property
    def pos_weight(self) -> float:
        """Tỷ lệ neg / pos dùng cho BCEWithLogitsLoss."""
        neg = sum(1 for _, lbl in self.samples if lbl == 0)
        pos = sum(1 for _, lbl in self.samples if lbl == 1)
        return float(neg / max(pos, 1))

    def _augment(self, img_pil: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            img_pil = img_pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.3:
            img_pil = img_pil.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            angle = random.uniform(-15.0, 15.0)
            img_pil = img_pil.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
        if random.random() < 0.3:
            factor = random.uniform(0.85, 1.15)
            img_pil = ImageEnhance.Brightness(img_pil).enhance(factor)
        if random.random() < 0.3:
            factor = random.uniform(0.85, 1.15)
            img_pil = ImageEnhance.Contrast(img_pil).enhance(factor)
        return img_pil

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            img_l = img.convert("L")
        if self.augment:
            img_l = self._augment(img_l)
        tensor_3c = preprocess_classifier(img_l, self.img_size)
        return torch.from_numpy(tensor_3c).float(), torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Kiến Trúc Mô Hình Swin Transformer V2-Tiny Nhị Phân
# ---------------------------------------------------------------------------
class SwinTumorClassifier(nn.Module):
    """Mô hình phân loại nhị phân Swin Transformer V2-Tiny (1 logit output)."""

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.2,
        checkpoint_path: Optional[Union[str, Path]] = None,
        backbone_name: Optional[str] = None,
    ):
        super().__init__()
        selected_backbone = backbone_name
        if selected_backbone is None:
            for cand in SWIN_BACKBONE_CANDIDATES:
                try:
                    _ = timm.create_model(cand, pretrained=pretrained, num_classes=0)
                    selected_backbone = cand
                    break
                except Exception:
                    continue
        if selected_backbone is None:
            selected_backbone = "swin_tiny_patch4_window7_224"

        self.backbone_name = selected_backbone
        try:
            self.backbone = timm.create_model(
                self.backbone_name,
                pretrained=pretrained,
                num_classes=0,
                drop_rate=dropout,
            )
        except Exception:
            self.backbone = timm.create_model(
                self.backbone_name,
                pretrained=False,
                num_classes=0,
                drop_rate=dropout,
            )

        num_features = self.backbone.num_features
        # Head nhị phân: 1 logit duy nhất
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 1),
        )

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat).squeeze(-1)

    def predict_slice(
        self,
        img: Union[Image.Image, np.ndarray, Path, str],
        device: Optional[torch.device] = None,
        threshold: float = DEFAULT_CLASSIFIER_THRESHOLD,
        img_size: int = DEFAULT_IMG_SIZE,
    ) -> Dict[str, Any]:
        """Dự đoán nhị phân cho 1 lát cắt MRI: trả về P(Tumor) và cờ is_positive."""
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        if isinstance(img, (str, Path)):
            with Image.open(img) as pil_img:
                img_l = pil_img.convert("L")
        elif isinstance(img, np.ndarray):
            img_l = Image.fromarray(img).convert("L")
        else:
            img_l = img.convert("L")

        x_tensor = torch.from_numpy(preprocess_classifier(img_l, img_size)).unsqueeze(0).to(device)

        with torch.no_grad():
            logit = self.forward(x_tensor)
            prob = float(torch.sigmoid(logit).item())
            if not math.isfinite(prob):
                raise ValueError("Classifier produced a nonfinite tumor probability")
            has_tumor = prob >= threshold

        return {
            "has_tumor": has_tumor,
            "tumor_probability": round(prob, 4),
            "threshold": threshold,
            "status": "POSITIVE (Có u)" if has_tumor else "NEGATIVE (Não lành)",
            "model": "Swin Transformer V2-Tiny",
        }

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint không tồn tại: {p}")
        state = torch.load(p, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.load_state_dict(state, strict=True)


# ---------------------------------------------------------------------------
# CLI Entrypoint Huấn Luyện Nhị Phân
# ---------------------------------------------------------------------------
def train_binary_swin(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị huấn luyện: {device}")

    output_dir = Path(args.output)
    save_path = output_dir / DEFAULT_CLASSIFIER_CHECKPOINT.name
    if save_path.exists():
        raise FileExistsError(f"Checkpoint already exists: {save_path}. Choose another --output.")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SwinTumorDataset(root=args.data_root, split="train", img_size=args.img_size, augment=True)
    test_ds = SwinTumorDataset(root=args.data_root, split="test", img_size=args.img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    print(f"Dữ liệu: Train={len(train_ds)} ảnh, Test={len(test_ds)} ảnh")
    print(f"Phân bố nhãn Train: {train_ds.class_counts}")

    model = SwinTumorClassifier(pretrained=True, dropout=args.dropout, backbone_name=DEFAULT_BACKBONE).to(device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Khởi tạo Swin Transformer V2-Tiny ({model.backbone_name}, {params_m:.2f}M tham số)")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([train_ds.pos_weight], device=device))
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
            {"params": model.head.parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(labels)
        scheduler.step()

        if epoch % 3 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch {epoch:2d}/{args.epochs} | Train Loss: {total_loss / len(train_ds):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Huấn luyện hoàn tất. Đã lưu checkpoint tại: {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/data/brisc2025")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_CLASSIFIER_CHECKPOINT.parent)
    return parser.parse_args()


if __name__ == "__main__":
    train_binary_swin(parse_args())
