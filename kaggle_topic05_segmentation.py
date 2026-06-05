# %% [markdown]
# # Topic 05 — Investigating Modern Encoder–Decoder Architectures for Image Segmentation
#
# **Notebook final assignment** — Phiên bản hoàn chỉnh cho Kaggle (T4 GPU) & Google Colab.
#
# ## Mục tiêu
#
# Notebook này khảo sát và so sánh **03 kiến trúc encoder–decoder** hiện đại cho image segmentation,
# sử dụng **pretrained encoders** từ torchvision:
#
# | # | Model | Encoder | Đặc trưng chính |
# |---|-------|---------|-----------------|
# | 1 | **U-Net** | ResNet-34 (ImageNet) | Skip connections concatenation |
# | 2 | **SegNet** | VGG16-BN (ImageNet) | MaxPool indices → MaxUnpool |
# | 3 | **DeepLabV3+** | ResNet-34 (ImageNet) | ASPP + Low-level feature fusion |
#
# ## Datasets
#
# | Dataset | Domain | Classes | Source |
# |---------|--------|---------|--------|
# | Oxford-IIIT Pet | Natural images | 3 (bg/pet/boundary) | torchvision auto-download |
# | Kvasir-SEG | Medical endoscopy | 2 (bg/polyp) | Kaggle dataset |
#
# ## Controlled Experiment Design
#
# | Component | Design |
# |-----------|--------|
# | Data split | Fixed seed (70/15/15) |
# | Input size | 256×256 |
# | Optimizer | AdamW (weight_decay=1e-4) |
# | Loss | CrossEntropy + Dice (hybrid) |
# | Scheduler | CosineAnnealingLR |
# | AMP | Mixed precision training |
# | Early stopping | Patience=7 on val IoU |
# | Reliability | 3 seeds × 3 data fractions × 2 LRs |
# | Metrics | Pixel Accuracy, Mean IoU, Mean Dice, Per-class IoU/Dice |
#
# ## Output Structure
# ```
# outputs_topic05/
#     OxfordPet/
#         all_results.csv, summary_mean_std.csv
#         history/, checkpoints/, figures/
#     KvasirSEG/
#         all_results.csv, summary_mean_std.csv
#         history/, checkpoints/, figures/
#     combined_all_results.csv
#     combined_summary_mean_std.csv
#     profiling.csv
#     complexity_results.csv
# ```

# %% [markdown]
# # 1. Research Questions & Hypotheses
#
# ## Research Questions
#
# **RQ1.** Trong ba kiến trúc encoder–decoder (U-Net, SegNet, DeepLabV3+) sử dụng pretrained encoder,
# mô hình nào đạt hiệu năng segmentation tốt nhất theo Mean IoU và Dice trên từng domain?
#
# **RQ2.** Khi giảm lượng dữ liệu huấn luyện (100% → 50% → 25%), mô hình nào duy trì
# hiệu năng ổn định nhất? Pretrained encoder có giúp giảm suy giảm không?
#
# **RQ3.** Khi thay đổi learning rate (3e-4 vs 1e-3), mô hình nào hội tụ ổn định hơn?
# LR nào phù hợp hơn khi fine-tune pretrained encoder?
#
# **RQ4.** Mô hình nào có trade-off tốt nhất giữa accuracy, inference latency và
# memory footprint? Encoder size ảnh hưởng thế nào?
#
# **RQ5.** Hiệu năng thay đổi như thế nào giữa natural image (OxfordPet) và
# medical image segmentation (KvasirSEG)? ASPP có lợi thế ở domain nào?
#
# ## Hypotheses
#
# **H1.** U-Net + ResNet34 sẽ đạt IoU cao nhất nhờ dense skip connections giữa encoder-decoder
# giúp phục hồi spatial detail tốt hơn, đặc biệt ở boundary classes.
#
# **H2.** SegNet + VGG16-BN sẽ có inference latency thấp hơn U-Net nhờ memory-efficient
# unpooling (chỉ lưu indices thay vì toàn bộ feature maps), nhưng có thể kém hơn ở
# vùng biên phức tạp do thiếu full skip connection features.
#
# **H3.** DeepLabV3+ sẽ có lợi thế ở các object lớn nhờ ASPP capture multi-scale context,
# nhưng decoder nhẹ có thể gây mất detail ở boundary.
#
# **H4.** Khi giảm training data, tất cả mô hình sẽ suy giảm, nhưng mô hình có pretrained
# encoder mạnh (ResNet34) sẽ suy giảm ít hơn SegNet (VGG16-BN lớn hơn, dễ overfit).
#
# **H5.** Không có mô hình tối ưu tuyệt đối — U-Net tốt cho accuracy, SegNet cho efficiency,
# DeepLabV3+ cho large-scale objects. Lựa chọn phụ thuộc vào domain và constraints.

# %% [markdown]
# # 2. Environment Setup

# %%
# ============================================================
# 2. Environment Setup & Imports
# ============================================================

import subprocess
import sys

# Cài đặt albumentations nếu chưa có (Kaggle/Colab)
try:
    import albumentations
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "albumentations>=1.3.0"])

try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "thop"])
        from thop import profile as thop_profile
        HAS_THOP = True
    except Exception:
        HAS_THOP = False

import os
import json
import time
import math
import random
import shutil
import zipfile
import warnings
import gc
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision
from torchvision import transforms, models
from torchvision.datasets import OxfordIIITPet
from tqdm.auto import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Environment detection
IN_KAGGLE = os.path.exists("/kaggle/working")
IN_COLAB = False
try:
    from google.colab import drive  # noqa: F401
    IN_COLAB = True
except ImportError:
    pass

print("=" * 60)
print(f"PyTorch:      {torch.__version__}")
print(f"Torchvision:  {torchvision.__version__}")
print(f"CUDA:         {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:          {torch.cuda.get_device_name(0)}")
print(f"Kaggle:       {IN_KAGGLE}")
print(f"Colab:        {IN_COLAB}")
print(f"Albumentations: {albumentations.__version__}")
print(f"thop (FLOPs): {HAS_THOP}")
print("=" * 60)

# %%
# ============================================================
# 3. Configuration
# ============================================================

@dataclass
class Config:
    """Cấu hình trung tâm cho toàn bộ notebook."""
    PROJECT_NAME: str = "Topic05_EncoderDecoder_Segmentation"

    # --- Paths (auto-set by detect_environment) ---
    DATA_ROOT: str = ""
    OUTPUT_ROOT: str = ""
    KVASIR_IMAGE_DIR: str = ""
    KVASIR_MASK_DIR: str = ""

    # --- Data ---
    IMG_SIZE: int = 256
    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 2

    # --- Training ---
    QUICK_RUN: bool = False       # True → chạy nhanh để test pipeline
    QUICK_EPOCHS: int = 2
    FINAL_EPOCHS: int = 25

    # --- Experiment grid ---
    MODELS: Tuple[str, ...] = ("unet_resnet34", "segnet_vgg16", "deeplabv3plus_resnet34")
    SEEDS: Tuple[int, ...] = (42, 2025, 3407)
    DATA_FRACTIONS: Tuple[float, ...] = (1.0, 0.5, 0.25)
    LEARNING_RATES: Tuple[float, ...] = (3e-4,)

    # --- Training details ---
    USE_AMP: bool = True
    EARLY_STOP_PATIENCE: int = 7
    SCHEDULER: str = "cosine"
    WEIGHT_DECAY: float = 1e-4
    DICE_WEIGHT: float = 0.5

    # --- Kaggle dataset slug (for KvasirSEG) ---
    KAGGLE_KVASIR_SLUG: str = "debeshjha1/kvasirseg"


def detect_environment(cfg: Config) -> Config:
    """Tự động phát hiện môi trường và thiết lập paths."""
    if IN_KAGGLE:
        cfg.DATA_ROOT = "/kaggle/working/data"
        cfg.OUTPUT_ROOT = "/kaggle/working/outputs_topic05"

        # === Kvasir-SEG: Auto-scan toàn bộ /kaggle/input/ ===
        # Dataset Kvasir-SEG có sẵn trên Kaggle (slug: debeshjha1/kvasirseg)
        # Cấu trúc phổ biến:
        #   /kaggle/input/<slug>/Kvasir-SEG/images/
        #   /kaggle/input/<slug>/Kvasir-SEG/masks/
        #   hoặc: /kaggle/input/<slug>/images/ + /kaggle/input/<slug>/masks/
        kaggle_input = Path("/kaggle/input")
        kvasir_found = False

        if kaggle_input.exists():
            import os
            for root, dirs, files in os.walk(kaggle_input):
                root_path = Path(root)
                if root_path.name == "images":
                    masks_path = root_path.parent / "masks"
                    if masks_path.is_dir():
                        img_files = list(root_path.glob("*.jpg")) + list(root_path.glob("*.png"))
                        mask_files = list(masks_path.glob("*.jpg")) + list(masks_path.glob("*.png"))
                        if len(img_files) > 0 and len(mask_files) > 0:
                            cfg.KVASIR_IMAGE_DIR = str(root_path)
                            cfg.KVASIR_MASK_DIR = str(masks_path)
                            kvasir_found = True
                            print(f"[Config] Kvasir-SEG auto-detected: {root_path}")
                            print(f"         images: {len(img_files)}, masks: {len(mask_files)}")
                            break
                if kvasir_found:
                    break
                if kvasir_found:
                    break

        # Fallback: đường dẫn mặc định (slug phổ biến nhất)
        if not kvasir_found:
            cfg.KVASIR_IMAGE_DIR = "/kaggle/input/kvasirseg/Kvasir-SEG/images"
            cfg.KVASIR_MASK_DIR = "/kaggle/input/kvasirseg/Kvasir-SEG/masks"
            print("[Config] Kvasir-SEG: using default path (debeshjha1/kvasirseg)")
            print("         Nếu lỗi, hãy kiểm tra: Add Data → Search 'kvasirseg'")

    elif IN_COLAB:
        cfg.DATA_ROOT = "/content/datasets"
        cfg.OUTPUT_ROOT = "/content/outputs_topic05"
        cfg.KVASIR_IMAGE_DIR = "/content/datasets/kvasir_seg/images"
        cfg.KVASIR_MASK_DIR = "/content/datasets/kvasir_seg/masks"
    else:
        # Local
        base = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
        cfg.DATA_ROOT = os.path.join(base, "data")
        cfg.OUTPUT_ROOT = os.path.join(base, "outputs_topic05")
        cfg.KVASIR_IMAGE_DIR = os.path.join(base, "data", "kvasir_seg", "images")
        cfg.KVASIR_MASK_DIR = os.path.join(base, "data", "kvasir_seg", "masks")

    Path(cfg.DATA_ROOT).mkdir(parents=True, exist_ok=True)
    Path(cfg.OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    print(f"[Config] DATA_ROOT:    {cfg.DATA_ROOT}")
    print(f"[Config] OUTPUT_ROOT:  {cfg.OUTPUT_ROOT}")
    print(f"[Config] KVASIR_IMG:   {cfg.KVASIR_IMAGE_DIR}")
    print(f"[Config] KVASIR_MASK:  {cfg.KVASIR_MASK_DIR}")
    return cfg


def set_seed(seed: int):
    """Đặt seed cho reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- Initialize ---
cfg = Config()
cfg = detect_environment(cfg)
device = get_device()
print(f"[Device] {device}")

# Save config
with open(Path(cfg.OUTPUT_ROOT) / "config.json", "w", encoding="utf-8") as f:
    json.dump({k: str(v) for k, v in cfg.__dict__.items()}, f, indent=2, ensure_ascii=False)

# %% [markdown]
# # 3. Dataset Policy
#
# - **OxfordPet**: tự download qua torchvision, 3-class segmentation (bg/pet/boundary)
# - **KvasirSEG**: binary segmentation polyp, cần có sẵn trên Kaggle hoặc Drive
# - Cả hai dataset đều dùng **albumentations** cho data augmentation

# %%
# ============================================================
# 4. Dataset Classes (with Albumentations)
# ============================================================

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_augmentation(img_size: int) -> A.Compose:
    """Augmentation mạnh cho training."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.5,
                           border_mode=0),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_augmentation(img_size: int) -> A.Compose:
    """Chỉ resize + normalize cho val/test."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class OxfordPetSegDataset(Dataset):
    """Oxford-IIIT Pet dataset cho segmentation.

    Trimap mapping:
        Original: 1=pet, 2=background, 3=boundary
        Mapped:   0=background, 1=pet, 2=boundary
    """
    def __init__(self, root: str, split: str, img_size: int,
                 download: bool = True, transform=None):
        self.dataset = OxfordIIITPet(
            root=root, split=split,
            target_types="segmentation",
            download=download,
        )
        self.transform = transform
        self.img_size = img_size
        # Fallback transform nếu không truyền albumentations
        self._fallback_img_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self._fallback_mask_tf = transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.NEAREST,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        img = img.convert("RGB")
        img_np = np.array(img)
        mask_np = np.array(mask).astype(np.int64)

        # Map trimap: 2→0(bg), 1→1(pet), 3→2(boundary)
        mapped = np.zeros_like(mask_np, dtype=np.int64)
        mapped[mask_np == 2] = 0  # background
        mapped[mask_np == 1] = 1  # pet
        mapped[mask_np == 3] = 2  # boundary

        if self.transform is not None:
            augmented = self.transform(image=img_np, mask=mapped)
            img_t = augmented["image"]       # [C, H, W] float
            mask_t = augmented["mask"].long()  # [H, W] long
        else:
            img_t = self._fallback_img_tf(img)
            mask_resized = self._fallback_mask_tf(Image.fromarray(mapped.astype(np.uint8)))
            mask_t = torch.from_numpy(np.array(mask_resized).astype(np.int64)).long()

        return img_t, mask_t


class KvasirSEGDataset(Dataset):
    """Kvasir-SEG binary segmentation dataset.

    Mask: grayscale > 127 → 1 (polyp), else → 0 (background).
    """
    def __init__(self, image_dir: str, mask_dir: str, img_size: int,
                 transform=None):
        self.image_paths = sorted([
            p for p in Path(image_dir).glob("*")
            if p.suffix.lower() in IMAGE_EXTS
        ])
        self.mask_paths = sorted([
            p for p in Path(mask_dir).glob("*")
            if p.suffix.lower() in IMAGE_EXTS
        ])

        if len(self.image_paths) == 0 or len(self.mask_paths) == 0:
            raise RuntimeError(
                f"KvasirSEG dataset rỗng.\n"
                f"  image_dir: {image_dir} ({len(self.image_paths)} files)\n"
                f"  mask_dir:  {mask_dir} ({len(self.mask_paths)} files)"
            )

        n = min(len(self.image_paths), len(self.mask_paths))
        if len(self.image_paths) != len(self.mask_paths):
            warnings.warn(
                f"Image/mask count mismatch: {len(self.image_paths)} vs {len(self.mask_paths)}. "
                f"Using first {n} pairs."
            )
        self.image_paths = self.image_paths[:n]
        self.mask_paths = self.mask_paths[:n]
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        img_np = np.array(img)
        mask_np = (np.array(mask) > 127).astype(np.int64)

        if self.transform is not None:
            augmented = self.transform(image=img_np, mask=mask_np)
            img_t = augmented["image"]
            mask_t = augmented["mask"].long()
        else:
            tf = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
            img_t = tf(img)
            mask_resized = transforms.Resize(
                (self.img_size, self.img_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            )(mask)
            mask_t = torch.from_numpy((np.array(mask_resized) > 127).astype(np.int64)).long()

        return img_t, mask_t


# %%
# ============================================================
# 5. Dataset Split, Build & Visualization
# ============================================================

def split_dataset(dataset, seed: int, train_ratio=0.70, val_ratio=0.15):
    """Split dataset thành train/val/test theo seed cố định."""
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=generator)


def build_dataset(name: str, seed: int, cfg: Config, split_data: bool = True):
    """Factory function để tạo dataset pack.

    Returns:
        dict với keys: name, train, val, test, num_classes, class_names
    """
    train_tf = get_train_augmentation(cfg.IMG_SIZE)
    val_tf = get_val_augmentation(cfg.IMG_SIZE)

    if name == "OxfordPet":
        # OxfordPet dùng torchvision → download tự động
        # Tạo full dataset rồi split
        full_trainval = OxfordPetSegDataset(
            root=cfg.DATA_ROOT, split="trainval",
            img_size=cfg.IMG_SIZE, download=True, transform=train_tf,
        )
        full_test = OxfordPetSegDataset(
            root=cfg.DATA_ROOT, split="test",
            img_size=cfg.IMG_SIZE, download=True, transform=val_tf,
        )
        if split_data:
            train_ds, val_ds_raw, _ = split_dataset(full_trainval, seed=seed)
            # Val subset cần val transform → wrap lại
            # Vì Subset không cho thay transform, ta dùng class wrapper
            val_ds = _SubsetWithTransform(full_trainval, val_ds_raw.indices, val_tf)
            test_ds = full_test
        else:
            train_ds = full_trainval
            val_ds = full_test
            test_ds = full_test

        return {
            "name": "OxfordPet",
            "train": train_ds,
            "val": val_ds,
            "test": test_ds,
            "num_classes": 3,
            "class_names": ["background", "pet", "boundary"],
        }

    elif name == "KvasirSEG":
        if not (Path(cfg.KVASIR_IMAGE_DIR).exists() and Path(cfg.KVASIR_MASK_DIR).exists()):
            hint = ""
            if IN_KAGGLE:
                hint = (
                    "\n\n  [Hướng dẫn Kaggle]\n"
                    "  1. Bấm 'Add Data' ở bảng bên phải\n"
                    "  2. Search: kvasirseg\n"
                    "  3. Add dataset 'debeshjha1/kvasirseg'\n"
                    "  4. Restart notebook và chạy lại"
                )
            raise RuntimeError(
                f"KvasirSEG dataset không tìm thấy.\n"
                f"  Expected images: {cfg.KVASIR_IMAGE_DIR}\n"
                f"  Expected masks:  {cfg.KVASIR_MASK_DIR}"
                f"{hint}"
            )
        full = KvasirSEGDataset(
            cfg.KVASIR_IMAGE_DIR, cfg.KVASIR_MASK_DIR,
            img_size=cfg.IMG_SIZE, transform=train_tf,
        )
        if split_data:
            train_ds, val_ds_raw, test_ds_raw = split_dataset(full, seed=seed)
            val_ds = _SubsetWithTransform(full, val_ds_raw.indices, val_tf)
            test_ds = _SubsetWithTransform(full, test_ds_raw.indices, val_tf)
        else:
            train_ds = full
            val_ds = full
            test_ds = full

        return {
            "name": "KvasirSEG",
            "train": train_ds,
            "val": val_ds,
            "test": test_ds,
            "num_classes": 2,
            "class_names": ["background", "polyp"],
        }

    raise ValueError(f"Unknown dataset: {name}")


class _SubsetWithTransform(Dataset):
    """Subset wrapper cho phép thay đổi transform (dùng cho val/test split)."""
    def __init__(self, base_dataset, indices, new_transform):
        self.base = base_dataset
        self.indices = indices
        self.new_transform = new_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        # Lấy raw data từ base dataset underlying
        if hasattr(self.base, 'dataset'):
            # OxfordPetSegDataset wraps torchvision
            orig_ds = self.base
            img_pil, mask_pil = orig_ds.dataset[real_idx]
            img_pil = img_pil.convert("RGB")
            img_np = np.array(img_pil)
            mask_np = np.array(mask_pil).astype(np.int64)
            # Map trimap
            mapped = np.zeros_like(mask_np, dtype=np.int64)
            mapped[mask_np == 2] = 0
            mapped[mask_np == 1] = 1
            mapped[mask_np == 3] = 2
            mask_np = mapped
        elif hasattr(self.base, 'image_paths'):
            # KvasirSEGDataset
            img = Image.open(self.base.image_paths[real_idx]).convert("RGB")
            mask = Image.open(self.base.mask_paths[real_idx]).convert("L")
            img_np = np.array(img)
            mask_np = (np.array(mask) > 127).astype(np.int64)
        else:
            # Fallback: use base __getitem__ (sẽ dùng transform cũ)
            return self.base[real_idx]

        augmented = self.new_transform(image=img_np, mask=mask_np)
        return augmented["image"], augmented["mask"].long()


def get_available_datasets(cfg: Config) -> List[str]:
    """Kiểm tra dataset nào khả dụng."""
    available = []
    for name in ["OxfordPet", "KvasirSEG"]:
        try:
            _ = build_dataset(name, seed=42, cfg=cfg)
            available.append(name)
            print(f"[Dataset] ✓ {name} available")
        except Exception as e:
            print(f"[Dataset] ✗ {name}: {e}")
    if not available:
        raise RuntimeError("Không có dataset nào khả dụng!")
    return available


def make_dataset_output_dirs(cfg: Config, dataset_name: str) -> Dict[str, Path]:
    """Tạo cấu trúc thư mục output cho từng dataset."""
    base = Path(cfg.OUTPUT_ROOT) / dataset_name
    dirs = {
        "base": base,
        "history": base / "history",
        "checkpoints": base / "checkpoints",
        "figures": base / "figures",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# --- Visualization helpers ---

def denormalize(img_tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return torch.clamp(img_tensor.cpu().float() * std + mean, 0, 1)


def show_dataset_samples(dataset_pack: dict, n: int = 4, save_path: str = None):
    """Hiển thị mẫu dữ liệu từ dataset."""
    ds = dataset_pack["train"]
    class_names = dataset_pack["class_names"]
    nc = dataset_pack["num_classes"]
    n = min(n, len(ds))

    fig, axes = plt.subplots(n, 2, figsize=(8, 3.5 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(n):
        img, mask = ds[i]
        axes[i, 0].imshow(denormalize(img).permute(1, 2, 0).numpy())
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        im = axes[i, 1].imshow(mask.numpy(), vmin=0, vmax=nc - 1, cmap="tab10")
        axes[i, 1].set_title("Mask")
        axes[i, 1].axis("off")

    plt.suptitle(f"Dataset: {dataset_pack['name']}  |  Classes: {class_names}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_class_distribution(dataset_pack: dict, max_samples: int = 200,
                            save_path: str = None):
    """Phân tích và vẽ phân bố pixel theo class."""
    ds = dataset_pack["train"]
    nc = dataset_pack["num_classes"]
    class_names = dataset_pack["class_names"]
    counts = np.zeros(nc, dtype=np.int64)

    n = min(len(ds), max_samples)
    for i in tqdm(range(n), desc=f"Class dist - {dataset_pack['name']}", leave=False):
        _, mask = ds[i]
        mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
        for c in range(nc):
            counts[c] += int((mask_np == c).sum())

    total = counts.sum()
    ratios = counts / total if total > 0 else counts

    df = pd.DataFrame({"class": class_names, "pixel_count": counts, "ratio": ratios})
    print(f"\n[Class Distribution] {dataset_pack['name']}:")
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(df["class"], df["ratio"], color=plt.cm.Set2.colors[:nc])
    for bar, ratio in zip(bars, df["ratio"]):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f"{ratio:.1%}", ha="center", fontsize=10)
    ax.set_ylabel("Pixel Ratio")
    ax.set_title(f"Class Distribution — {dataset_pack['name']}")
    ax.set_ylim(0, max(ratios) * 1.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return df


# --- Run dataset checks ---
available_datasets = get_available_datasets(cfg)
print(f"\n✓ Available datasets: {available_datasets}")

# Visualize each available dataset
for ds_name in available_datasets:
    pack = build_dataset(ds_name, seed=42, cfg=cfg)
    print(f"\n{'=' * 60}")
    print(f"Dataset: {pack['name']}")
    print(f"  Num classes: {pack['num_classes']}")
    print(f"  Class names: {pack['class_names']}")
    print(f"  Train: {len(pack['train'])}, Val: {len(pack['val'])}, Test: {len(pack['test'])}")
    print(f"{'=' * 60}")

    out_dirs = make_dataset_output_dirs(cfg, ds_name)
    show_dataset_samples(pack, n=3,
                         save_path=str(out_dirs["figures"] / "dataset_samples.png"))
    plot_class_distribution(pack, max_samples=150,
                            save_path=str(out_dirs["figures"] / "class_distribution.png"))

# %% [markdown]
# # 5. Model Architectures
#
# Triển khai 3 mô hình FROM SCRATCH sử dụng pretrained encoders từ torchvision:
#
# 1. **U-Net + ResNet34**: Skip connections qua concatenation
# 2. **SegNet + VGG16-BN**: MaxPool indices → MaxUnpool (true SegNet)
# 3. **DeepLabV3+ + ResNet34**: ASPP + low-level feature fusion

# %%
# ============================================================
# 5a. Helper Blocks
# ============================================================

class DoubleConv(nn.Module):
    """Double convolution block: Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    """Decoder block cho U-Net: ConvTranspose2d upsample + skip concat + DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch * 2, out_ch)  # after concat with skip

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle potential size mismatch (e.g., odd input dimensions)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# %%
# ============================================================
# 5b. U-Net with ResNet34 Pretrained Encoder
# ============================================================

class UNetResNet34(nn.Module):
    """U-Net sử dụng pretrained ResNet34 làm encoder.

    Encoder stages (pretrained):
        enc0: conv1 + bn1 + relu     → 64ch,  stride /2
        pool0: maxpool                → 64ch,  stride /4
        enc1: layer1                  → 64ch,  stride /4
        enc2: layer2                  → 128ch, stride /8
        enc3: layer3                  → 256ch, stride /16
        enc4: layer4 (bottleneck)     → 512ch, stride /32

    Decoder stages (learned):
        up4: 512 → 256, skip=enc3(256)    → /16
        up3: 256 → 128, skip=enc2(128)    → /8
        up2: 128 → 64,  skip=enc1(64)     → /4
        up1: 64  → 64,  skip=enc0(64)     → /2
        up0: ConvTranspose 64→32          → /1
        final: DoubleConv(32) → Conv1x1(num_classes)
    """
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet34(
            weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # === Encoder (pretrained ResNet34) ===
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64ch, /2
        self.pool0 = resnet.maxpool  # /4
        self.enc1 = resnet.layer1    # 64ch,  /4
        self.enc2 = resnet.layer2    # 128ch, /8
        self.enc3 = resnet.layer3    # 256ch, /16
        self.enc4 = resnet.layer4    # 512ch, /32  (bottleneck)

        # === Decoder (learned from scratch) ===
        self.up4 = UpBlock(512, 256)   # 512→256, concat enc3(256), DoubleConv(512→256)
        self.up3 = UpBlock(256, 128)   # 256→128, concat enc2(128), DoubleConv(256→128)
        self.up2 = UpBlock(128, 64)    # 128→64,  concat enc1(64),  DoubleConv(128→64)
        self.up1 = UpBlock(64, 64)     # 64→64,   concat enc0(64),  DoubleConv(128→64)

        # Final upsample from /2 → /1
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            DoubleConv(32, 32),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        e0 = self.enc0(x)          # [B, 64,  H/2, W/2]
        e0p = self.pool0(e0)       # [B, 64,  H/4, W/4]
        e1 = self.enc1(e0p)        # [B, 64,  H/4, W/4]
        e2 = self.enc2(e1)         # [B, 128, H/8, W/8]
        e3 = self.enc3(e2)         # [B, 256, H/16,W/16]
        e4 = self.enc4(e3)         # [B, 512, H/32,W/32]

        # Decoder path with skip connections
        d4 = self.up4(e4, e3)      # [B, 256, H/16,W/16]
        d3 = self.up3(d4, e2)      # [B, 128, H/8, W/8]
        d2 = self.up2(d3, e1)      # [B, 64,  H/4, W/4]
        d1 = self.up1(d2, e0)      # [B, 64,  H/2, W/2]

        d0 = self.up0(d1)          # [B, 32,  H,   W]
        out = self.final_conv(d0)  # [B, C,   H,   W]

        # Ensure output matches input spatial size
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode="bilinear",
                                align_corners=False)
        return out


# %%
# ============================================================
# 5c. SegNet with VGG16-BN Pretrained Encoder
# ============================================================

class SegNetVGG16(nn.Module):
    """SegNet sử dụng pretrained VGG16-BN làm encoder.

    Đặc trưng chính: MaxPool indices được lưu trong encoding và dùng cho
    MaxUnpool trong decoding — đây là kiến trúc TRUE SegNet.

    VGG16-BN feature structure (with BatchNorm):
        Block 1: features[0:6]   → Conv64-BN-ReLU × 2   (indices 0-5)
        MaxPool at features[6]
        Block 2: features[7:13]  → Conv128-BN-ReLU × 2  (indices 7-12)
        MaxPool at features[13]
        Block 3: features[14:23] → Conv256-BN-ReLU × 3  (indices 14-22)
        MaxPool at features[23]
        Block 4: features[24:33] → Conv512-BN-ReLU × 3  (indices 24-32)
        MaxPool at features[33]
        Block 5: features[34:43] → Conv512-BN-ReLU × 3  (indices 34-42)
        MaxPool at features[43]
    """
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        vgg = models.vgg16_bn(
            weights=models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # === Encoder: 5 blocks from VGG16-BN (pretrained) ===
        # Tách features thành các block, KHÔNG bao gồm MaxPool layers
        # VGG16-BN features layout:
        # 0:Conv, 1:BN, 2:ReLU, 3:Conv, 4:BN, 5:ReLU, 6:MaxPool,
        # 7:Conv, 8:BN, 9:ReLU, 10:Conv, 11:BN, 12:ReLU, 13:MaxPool,
        # 14:Conv, 15:BN, 16:ReLU, 17:Conv, 18:BN, 19:ReLU, 20:Conv, 21:BN, 22:ReLU, 23:MaxPool,
        # 24:Conv, 25:BN, 26:ReLU, 27:Conv, 28:BN, 29:ReLU, 30:Conv, 31:BN, 32:ReLU, 33:MaxPool,
        # 34:Conv, 35:BN, 36:ReLU, 37:Conv, 38:BN, 39:ReLU, 40:Conv, 41:BN, 42:ReLU, 43:MaxPool

        features = list(vgg.features.children())
        self.enc1 = nn.Sequential(*features[0:6])    # 3→64,   2 conv blocks
        self.enc2 = nn.Sequential(*features[7:13])    # 64→128, 2 conv blocks
        self.enc3 = nn.Sequential(*features[14:23])   # 128→256, 3 conv blocks
        self.enc4 = nn.Sequential(*features[24:33])   # 256→512, 3 conv blocks
        self.enc5 = nn.Sequential(*features[34:43])   # 512→512, 3 conv blocks

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # === Decoder: Mirror of encoder (learned from scratch) ===
        # Decoder 5: 512 → 512 (3 conv blocks)
        self.dec5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        # Decoder 4: 512 → 256 (3 conv blocks)
        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        # Decoder 3: 256 → 128 (3 conv blocks)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        # Decoder 2: 128 → 64 (2 conv blocks)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        # Decoder 1: 64 → 64 (2 conv blocks)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )

        # Final classifier
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # === Encoding: save pool indices and sizes ===
        e1 = self.enc1(x)
        s1 = e1.size()
        e1p, idx1 = self.pool(e1)

        e2 = self.enc2(e1p)
        s2 = e2.size()
        e2p, idx2 = self.pool(e2)

        e3 = self.enc3(e2p)
        s3 = e3.size()
        e3p, idx3 = self.pool(e3)

        e4 = self.enc4(e3p)
        s4 = e4.size()
        e4p, idx4 = self.pool(e4)

        e5 = self.enc5(e4p)
        s5 = e5.size()
        e5p, idx5 = self.pool(e5)

        # === Decoding: use saved indices for unpooling ===
        d5 = self.unpool(e5p, idx5, output_size=s5)
        d5 = self.dec5(d5)

        d4 = self.unpool(d5, idx4, output_size=s4)
        d4 = self.dec4(d4)

        d3 = self.unpool(d4, idx3, output_size=s3)
        d3 = self.dec3(d3)

        d2 = self.unpool(d3, idx2, output_size=s2)
        d2 = self.dec2(d2)

        d1 = self.unpool(d2, idx1, output_size=s1)
        d1 = self.dec1(d1)

        return self.classifier(d1)


# %%
# ============================================================
# 5d. DeepLabV3+ with ResNet34 Encoder + ASPP
# ============================================================

class ASPPConv(nn.Module):
    """Single ASPP convolution branch."""
    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__()
        if dilation == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)


class ASPPPooling(nn.Module):
    """Global Average Pooling branch trong ASPP."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[2:]
        x = self.pool(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module.

    5 branches:
        1. 1×1 conv (rate=1)
        2. 3×3 conv, dilation=6
        3. 3×3 conv, dilation=12
        4. 3×3 conv, dilation=18
        5. Global Average Pooling
    → Concatenate → 1×1 projection → Dropout
    """
    def __init__(self, in_ch: int, out_ch: int, rates: Tuple[int, ...] = (1, 6, 12, 18)):
        super().__init__()
        modules = []
        for rate in rates:
            modules.append(ASPPConv(in_ch, out_ch, rate))
        modules.append(ASPPPooling(in_ch, out_ch))
        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [conv(x) for conv in self.convs]
        x = torch.cat(features, dim=1)
        return self.project(x)


class DeepLabV3PlusResNet34(nn.Module):
    """DeepLabV3+ sử dụng pretrained ResNet34.

    Architecture:
        Encoder: ResNet34 backbone (output stride ≈ 16)
            layer0: conv1 + bn1 + relu + maxpool  → stride /4
            layer1: 64ch  (low-level features)     → stride /4
            layer2: 128ch                           → stride /8
            layer3: 256ch                           → stride /16
            layer4: 512ch (high-level features)     → stride /32

        ASPP: Applied on layer4 output
            rates = (1, 6, 12, 18) + global avg pool
            → 256ch output

        Decoder:
            1. Upsample ASPP output 4× → match layer1 size
            2. Project layer1 (64→48ch)
            3. Concatenate [ASPP_upsampled(256) + low_proj(48)] = 304ch
            4. Two 3×3 convs → num_classes
            5. Upsample 4× → original size
    """
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet34(
            weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # === Backbone (pretrained) ===
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )  # /4
        self.layer1 = resnet.layer1   # 64ch,  /4  (low-level features)
        self.layer2 = resnet.layer2   # 128ch, /8
        self.layer3 = resnet.layer3   # 256ch, /16
        self.layer4 = resnet.layer4   # 512ch, /32

        # === ASPP module ===
        self.aspp = ASPP(512, 256, rates=(1, 6, 12, 18))

        # === Low-level feature projection ===
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(64, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # === Decoder ===
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        # Encoder
        x0 = self.layer0(x)       # [B, 64,  H/4,  W/4]
        low = self.layer1(x0)     # [B, 64,  H/4,  W/4]  ← low-level features
        x2 = self.layer2(low)     # [B, 128, H/8,  W/8]
        x3 = self.layer3(x2)      # [B, 256, H/16, W/16]
        x4 = self.layer4(x3)      # [B, 512, H/32, W/32]

        # ASPP
        aspp_out = self.aspp(x4)  # [B, 256, H/32, W/32]

        # Upsample ASPP output to match low-level feature size
        aspp_up = F.interpolate(aspp_out, size=low.shape[2:],
                                mode="bilinear", align_corners=False)  # [B, 256, H/4, W/4]

        # Low-level feature projection
        low_proj = self.low_level_proj(low)  # [B, 48, H/4, W/4]

        # Concatenate and decode
        concat = torch.cat([aspp_up, low_proj], dim=1)  # [B, 304, H/4, W/4]
        dec = self.decoder(concat)  # [B, C, H/4, W/4]

        # Final upsample to input size
        out = F.interpolate(dec, size=input_size,
                            mode="bilinear", align_corners=False)  # [B, C, H, W]
        return out


# %%
# ============================================================
# 5e. Model Factory & Summary
# ============================================================

def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Factory function để tạo model theo tên."""
    name_lower = name.lower()
    if name_lower == "unet_resnet34":
        return UNetResNet34(num_classes, pretrained=pretrained)
    elif name_lower == "segnet_vgg16":
        return SegNetVGG16(num_classes, pretrained=pretrained)
    elif name_lower == "deeplabv3plus_resnet34":
        return DeepLabV3PlusResNet34(num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    """Đếm số trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model: nn.Module) -> int:
    """Đếm tổng parameters (trainable + frozen)."""
    return sum(p.numel() for p in model.parameters())


def verify_model(name: str, num_classes: int, img_size: int, dev: torch.device):
    """Verify model output shape và in parameter count."""
    print(f"\n{'=' * 60}")
    print(f"Model: {name}")

    model = build_model(name, num_classes).to(dev)
    model.eval()

    trainable = count_parameters(model)
    total = count_all_parameters(model)
    print(f"  Trainable params: {trainable:,}")
    print(f"  Total params:     {total:,}")
    print(f"  Param memory:     {total * 4 / 1024**2:.1f} MB")

    x = torch.randn(2, 3, img_size, img_size).to(dev)
    with torch.no_grad():
        y = model(x)
    expected = (2, num_classes, img_size, img_size)
    actual = tuple(y.shape)
    assert actual == expected, f"Output shape mismatch: {actual} != {expected}"
    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {actual} ✓")

    # FLOPs estimation
    if HAS_THOP:
        try:
            x1 = torch.randn(1, 3, img_size, img_size).to(dev)
            flops, params = thop_profile(model, inputs=(x1,), verbose=False)
            print(f"  FLOPs:  {flops / 1e9:.2f} G")
        except Exception as e:
            print(f"  FLOPs:  [Error] {e}")

    print(f"{'=' * 60}")
    del model
    torch.cuda.empty_cache() if dev.type == "cuda" else None


# Verify all models
for model_name in cfg.MODELS:
    verify_model(model_name, num_classes=3, img_size=cfg.IMG_SIZE, dev=device)

# %% [markdown]
# # 6. Metrics & Loss Functions

# %%
# ============================================================
# 6. Metrics & Loss
# ============================================================

@torch.no_grad()
def calculate_metrics(preds: torch.Tensor, targets: torch.Tensor,
                      num_classes: int, eps: float = 1e-7) -> dict:
    """Tính segmentation metrics.

    Args:
        preds: [B, C, H, W] logits hoặc [B, H, W] class indices
        targets: [B, H, W] ground truth class indices
        num_classes: số lượng class

    Returns:
        dict: pixel_acc, mean_iou, mean_dice, per_class_iou, per_class_dice
    """
    if preds.dim() == 4:
        preds = torch.argmax(preds, dim=1)
    preds = preds.long()
    targets = targets.long()

    pixel_acc = (preds == targets).float().mean().item()

    ious, dices = [], []
    for c in range(num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)

        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        pred_sum = pred_c.sum().float()
        target_sum = target_c.sum().float()

        # Handle absent classes: if neither pred nor target has class c
        if union.item() < eps:
            ious.append(float('nan'))
            dices.append(float('nan'))
        else:
            iou = (intersection + eps) / (union + eps)
            dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
            ious.append(iou.item())
            dices.append(dice.item())

    # Mean ignoring NaN (absent classes)
    valid_ious = [v for v in ious if not math.isnan(v)]
    valid_dices = [v for v in dices if not math.isnan(v)]

    return {
        "pixel_acc": pixel_acc,
        "mean_iou": float(np.mean(valid_ious)) if valid_ious else 0.0,
        "mean_dice": float(np.mean(valid_dices)) if valid_dices else 0.0,
        "per_class_iou": ious,
        "per_class_dice": dices,
    }


@torch.no_grad()
def compute_confusion_matrix(preds: torch.Tensor, targets: torch.Tensor,
                             num_classes: int) -> np.ndarray:
    """Tính confusion matrix cho segmentation."""
    if preds.dim() == 4:
        preds = torch.argmax(preds, dim=1)
    preds = preds.cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def dice_loss(logits: torch.Tensor, targets: torch.Tensor,
              num_classes: int, eps: float = 1e-7) -> torch.Tensor:
    """Soft Dice loss cho multi-class segmentation."""
    probs = torch.softmax(logits, dim=1)
    targets_onehot = F.one_hot(targets.long(), num_classes=num_classes)
    targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)  # batch, height, width
    intersection = torch.sum(probs * targets_onehot, dims)
    cardinality = torch.sum(probs + targets_onehot, dims)
    dice_per_class = (2 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice_per_class.mean()


class HybridLoss(nn.Module):
    """Hybrid Loss = CrossEntropy + λ × Dice Loss."""
    def __init__(self, num_classes: int, dice_weight: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, targets.long())
        dl = dice_loss(logits, targets, self.num_classes)
        return ce + self.dice_weight * dl


def estimate_boundary_f1(pred: torch.Tensor, target: torch.Tensor,
                         dilation: int = 2) -> float:
    """Estimate boundary F1 score (simplified).

    Extracts boundary pixels via gradient and computes F1.
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    pred_np = pred.cpu().numpy().astype(np.uint8)
    target_np = target.cpu().numpy().astype(np.uint8)

    def get_boundary(mask):
        h_diff = np.abs(np.diff(mask, axis=-1))
        v_diff = np.abs(np.diff(mask, axis=-2))
        boundary = np.zeros_like(mask)
        boundary[:, :, 1:] |= (h_diff > 0).astype(np.uint8)
        boundary[:, 1:, :] |= (v_diff > 0).astype(np.uint8)
        return boundary

    pred_b = get_boundary(pred_np)
    target_b = get_boundary(target_np)

    tp = (pred_b & target_b).sum()
    fp = (pred_b & ~target_b).sum()
    fn = (~pred_b & target_b).sum()

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    return float(f1)


# Test metrics
print("\n[Test] Metrics verification:")
test_logits = torch.randn(2, 3, cfg.IMG_SIZE, cfg.IMG_SIZE)
test_masks = torch.randint(0, 3, (2, cfg.IMG_SIZE, cfg.IMG_SIZE))
m = calculate_metrics(test_logits, test_masks, 3)
print(f"  Random baseline → IoU={m['mean_iou']:.4f}, Dice={m['mean_dice']:.4f}")

loss_fn = HybridLoss(num_classes=3)
test_loss = loss_fn(test_logits, test_masks)
print(f"  Hybrid loss: {test_loss.item():.4f}")

# %% [markdown]
# # 7. Training Engine

# %%
# ============================================================
# 7. Training Engine
# ============================================================

def make_fraction_subset(dataset, fraction: float, seed: int):
    """Tạo subset theo tỷ lệ dữ liệu."""
    if fraction >= 1.0:
        return dataset
    n = len(dataset)
    k = max(1, int(n * fraction))
    rng = np.random.default_rng(seed)
    indices = rng.choice(np.arange(n), size=k, replace=False).tolist()
    return Subset(dataset, indices)


def make_loaders(dataset_pack: dict, fraction: float, seed: int, cfg: Config):
    """Tạo DataLoader cho train/val/test."""
    train_subset = make_fraction_subset(dataset_pack["train"], fraction, seed)
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        dataset_pack["val"], batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=pin,
    )
    test_loader = DataLoader(
        dataset_pack["test"], batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=pin,
    )
    return train_loader, val_loader, test_loader


class EarlyStopping:
    """Early stopping monitor dựa trên val metric."""
    def __init__(self, patience: int = 7, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer,
                    criterion, num_classes: int, device: torch.device,
                    scaler=None, use_amp: bool = False) -> dict:
    """Train one epoch với optional AMP."""
    model.train()
    total_loss = 0.0
    sums = {"pixel_acc": 0.0, "mean_iou": 0.0, "mean_dice": 0.0}
    n_samples = 0

    for imgs, masks in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = imgs.size(0)
        with torch.no_grad():
            metrics = calculate_metrics(logits.detach(), masks, num_classes)
        total_loss += loss.item() * bs
        for k in sums:
            sums[k] += metrics[k] * bs
        n_samples += bs

    return {
        "loss": total_loss / max(n_samples, 1),
        **{k: v / max(n_samples, 1) for k, v in sums.items()},
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion,
             num_classes: int, device: torch.device,
             use_amp: bool = False, desc: str = "Eval") -> dict:
    """Evaluate model trên validation/test set."""
    model.eval()
    total_loss = 0.0
    sums = {"pixel_acc": 0.0, "mean_iou": 0.0, "mean_dice": 0.0}
    per_iou = np.zeros(num_classes)
    per_dice = np.zeros(num_classes)
    n_samples = 0

    for imgs, masks in tqdm(loader, desc=desc, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                loss = criterion(logits, masks)
        else:
            logits = model(imgs)
            loss = criterion(logits, masks)

        bs = imgs.size(0)
        metrics = calculate_metrics(logits, masks, num_classes)
        total_loss += loss.item() * bs
        for k in sums:
            sums[k] += metrics[k] * bs

        # Per-class metrics (handle NaN from absent classes)
        pci = np.array(metrics["per_class_iou"])
        pcd = np.array(metrics["per_class_dice"])
        pci = np.nan_to_num(pci, nan=0.0)
        pcd = np.nan_to_num(pcd, nan=0.0)
        per_iou += pci * bs
        per_dice += pcd * bs
        n_samples += bs

    n = max(n_samples, 1)
    return {
        "loss": total_loss / n,
        **{k: v / n for k, v in sums.items()},
        "per_class_iou": (per_iou / n).tolist(),
        "per_class_dice": (per_dice / n).tolist(),
    }


def train_model_one_run(dataset_pack: dict, model_name: str, seed: int,
                        fraction: float, lr: float, cfg: Config,
                        device: torch.device) -> dict:
    """Train một model configuration hoàn chỉnh.

    Bao gồm: training loop, early stopping, checkpoint, evaluation.
    """
    set_seed(seed)

    ds_name = dataset_pack["name"]
    num_classes = dataset_pack["num_classes"]
    out_dirs = make_dataset_output_dirs(cfg, ds_name)

    # Run ID for logging
    run_id = (f"{ds_name}_{model_name}_seed{seed}"
              f"_frac{str(fraction).replace('.', 'p')}"
              f"_lr{str(lr).replace('.', 'p')}")

    ckpt_path = out_dirs["checkpoints"] / f"best_{run_id}.pt"
    history_path = out_dirs["history"] / f"history_{run_id}.csv"

    # Check for resume
    if ckpt_path.exists() and history_path.exists():
        print(f"[SKIP] {run_id} — checkpoint exists, loading results...")
        # Try to reload test results from existing checkpoint
        try:
            model = build_model(model_name, num_classes).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                             weights_only=True))
            _, _, test_loader = make_loaders(dataset_pack, fraction, seed, cfg)
            criterion = HybridLoss(num_classes, cfg.DICE_WEIGHT)
            use_amp = cfg.USE_AMP and device.type == "cuda"
            test_stats = evaluate(model, test_loader, criterion, num_classes,
                                  device, use_amp=use_amp, desc="Test")
            latency_ms = measure_inference_latency(model, cfg.IMG_SIZE, device)
            mem = measure_memory_footprint(model, cfg.IMG_SIZE, device)

            hist_df = pd.read_csv(history_path)
            best_val_iou = hist_df["val_iou"].max() if "val_iou" in hist_df else 0

            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None

            return _build_result_dict(
                ds_name, model_name, seed, fraction, lr,
                len(hist_df), best_val_iou, test_stats, latency_ms, mem,
                str(ckpt_path), str(history_path), dataset_pack["class_names"],
            )
        except Exception as e:
            print(f"  [WARN] Resume failed: {e}, retraining...")

    # Build components
    train_loader, val_loader, test_loader = make_loaders(
        dataset_pack, fraction, seed, cfg
    )

    model = build_model(model_name, num_classes).to(device)
    criterion = HybridLoss(num_classes, cfg.DICE_WEIGHT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=cfg.WEIGHT_DECAY)

    epochs = cfg.QUICK_EPOCHS if cfg.QUICK_RUN else cfg.FINAL_EPOCHS
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    early_stop = EarlyStopping(patience=cfg.EARLY_STOP_PATIENCE, mode="max")

    # AMP setup
    use_amp = cfg.USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Training loop
    history = []
    best_val_iou = -1.0
    start_time = time.time()

    print(f"\n[Train] {run_id}")
    print(f"  Epochs: {epochs}, LR: {lr}, Fraction: {fraction}")
    print(f"  Train samples: {len(train_loader.dataset)}, AMP: {use_amp}")

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion,
            num_classes, device, scaler=scaler, use_amp=use_amp,
        )
        val_stats = evaluate(
            model, val_loader, criterion, num_classes,
            device, use_amp=use_amp, desc="Val",
        )
        scheduler.step()

        row = {
            "dataset": ds_name, "model": model_name, "seed": seed,
            "fraction": fraction, "lr": lr, "epoch": epoch,
            "train_loss": train_stats["loss"], "val_loss": val_stats["loss"],
            "train_iou": train_stats["mean_iou"], "val_iou": val_stats["mean_iou"],
            "train_dice": train_stats["mean_dice"], "val_dice": val_stats["mean_dice"],
            "train_pixel_acc": train_stats["pixel_acc"],
            "val_pixel_acc": val_stats["pixel_acc"],
            "lr_current": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(f"  Epoch {epoch:02d}/{epochs} | "
              f"Train IoU={train_stats['mean_iou']:.4f} Dice={train_stats['mean_dice']:.4f} | "
              f"Val IoU={val_stats['mean_iou']:.4f} Dice={val_stats['mean_dice']:.4f} | "
              f"LR={optimizer.param_groups[0]['lr']:.2e}")

        # Save best checkpoint
        if val_stats["mean_iou"] > best_val_iou:
            best_val_iou = val_stats["mean_iou"]
            torch.save(model.state_dict(), ckpt_path)

        # Early stopping
        if early_stop(val_stats["mean_iou"]):
            print(f"  [Early Stop] at epoch {epoch}")
            break

    train_time = time.time() - start_time
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"  Training time: {train_time:.1f}s | Best val IoU: {best_val_iou:.4f}")

    # Load best model & evaluate on test
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    test_stats = evaluate(model, test_loader, criterion, num_classes,
                          device, use_amp=use_amp, desc="Test")

    # Profiling
    latency_ms = measure_inference_latency(
        model, cfg.IMG_SIZE, device,
        repeats=10 if cfg.QUICK_RUN else 50,
    )
    mem = measure_memory_footprint(model, cfg.IMG_SIZE, device)

    print(f"  Test IoU={test_stats['mean_iou']:.4f} Dice={test_stats['mean_dice']:.4f} "
          f"PixAcc={test_stats['pixel_acc']:.4f}")
    print(f"  Latency: {latency_ms:.1f}ms | Params: {mem['num_params']:,}")

    # Cleanup
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None
    gc.collect()

    return _build_result_dict(
        ds_name, model_name, seed, fraction, lr,
        len(history), best_val_iou, test_stats, latency_ms, mem,
        str(ckpt_path), str(history_path), dataset_pack["class_names"],
        train_time=train_time,
    )


def _build_result_dict(ds_name, model_name, seed, fraction, lr,
                       epochs, best_val_iou, test_stats, latency_ms, mem,
                       ckpt_path, history_path, class_names,
                       train_time=None):
    """Helper để tạo result dictionary."""
    result = {
        "dataset": ds_name,
        "model": model_name,
        "seed": seed,
        "fraction": fraction,
        "lr": lr,
        "epochs": epochs,
        "best_val_iou": best_val_iou,
        "test_loss": test_stats["loss"],
        "test_iou": test_stats["mean_iou"],
        "test_dice": test_stats["mean_dice"],
        "test_pixel_acc": test_stats["pixel_acc"],
        "per_class_iou": json.dumps(test_stats.get("per_class_iou", [])),
        "per_class_dice": json.dumps(test_stats.get("per_class_dice", [])),
        "latency_ms": latency_ms,
        "num_params": mem["num_params"],
        "param_memory_mb": mem["param_memory_mb"],
        "peak_cuda_memory_mb": mem.get("peak_cuda_memory_mb", float("nan")),
        "checkpoint_path": ckpt_path,
        "history_path": history_path,
        "class_names": json.dumps(class_names),
    }
    if train_time is not None:
        result["train_time_s"] = train_time
    return result

# %% [markdown]
# # 8. Profiling (Latency, Memory, FLOPs)

# %%
# ============================================================
# 8. Profiling
# ============================================================

@torch.no_grad()
def measure_inference_latency(model: nn.Module, img_size: int,
                              device: torch.device,
                              repeats: int = 50, warmup: int = 10) -> float:
    """Đo inference latency (ms/image) với warmup."""
    model.eval()
    x = torch.randn(1, 3, img_size, img_size).to(device)

    # Warmup
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure
    start = time.perf_counter()
    for _ in range(repeats):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / repeats * 1000  # ms


@torch.no_grad()
def measure_memory_footprint(model: nn.Module, img_size: int,
                             device: torch.device) -> dict:
    """Đo memory footprint."""
    n_params = count_parameters(model)
    param_memory_mb = n_params * 4 / (1024 ** 2)  # float32
    peak_cuda_mb = float("nan")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(1, 3, img_size, img_size).to(device)
        _ = model(x)
        torch.cuda.synchronize()
        peak_cuda_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "num_params": n_params,
        "param_memory_mb": round(param_memory_mb, 2),
        "peak_cuda_memory_mb": round(peak_cuda_mb, 2) if not math.isnan(peak_cuda_mb) else peak_cuda_mb,
    }


def profile_all_models(cfg: Config, device: torch.device) -> pd.DataFrame:
    """Profile tất cả models và trả về DataFrame."""
    rows = []
    for name in cfg.MODELS:
        for nc in [2, 3]:
            model = build_model(name, nc).to(device)
            model.eval()
            latency = measure_inference_latency(model, cfg.IMG_SIZE, device, repeats=30)
            mem = measure_memory_footprint(model, cfg.IMG_SIZE, device)

            row = {"model": name, "num_classes": nc, "latency_ms": latency, **mem}

            if HAS_THOP:
                try:
                    x = torch.randn(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE).to(device)
                    flops, _ = thop_profile(model, inputs=(x,), verbose=False)
                    row["gflops"] = round(flops / 1e9, 2)
                except Exception:
                    row["gflops"] = float("nan")

            rows.append(row)
            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None

    df = pd.DataFrame(rows)
    df.to_csv(Path(cfg.OUTPUT_ROOT) / "profiling.csv", index=False)
    print("\n[Profiling] Model comparison:")
    print(df.to_string(index=False))
    return df


profiling_df = profile_all_models(cfg, device)

# %% [markdown]
# # 9. Experiment Runner

# %%
# ============================================================
# 9. Experiment Runner
# ============================================================

def save_dataset_results(cfg: Config, dataset_name: str,
                         results: List[dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Save results cho một dataset và tạo summary."""
    out_dirs = make_dataset_output_dirs(cfg, dataset_name)
    df = pd.DataFrame(results)

    # Save all results
    all_path = out_dirs["base"] / "all_results.csv"
    df.to_csv(all_path, index=False)

    # Create summary (mean ± std across seeds)
    agg_dict = {
        "test_iou": ["mean", "std"],
        "test_dice": ["mean", "std"],
        "test_pixel_acc": ["mean", "std"],
        "latency_ms": ["mean", "std"],
        "param_memory_mb": ["mean"],
        "num_params": ["mean"],
    }
    if "train_time_s" in df.columns:
        agg_dict["train_time_s"] = ["mean"]

    summary = (
        df.groupby(["dataset", "model", "fraction", "lr"])
        .agg(**{
            "mean_iou": ("test_iou", "mean"),
            "std_iou": ("test_iou", "std"),
            "mean_dice": ("test_dice", "mean"),
            "std_dice": ("test_dice", "std"),
            "mean_pixel_acc": ("test_pixel_acc", "mean"),
            "std_pixel_acc": ("test_pixel_acc", "std"),
            "mean_latency_ms": ("latency_ms", "mean"),
            "std_latency_ms": ("latency_ms", "std"),
            "mean_param_memory_mb": ("param_memory_mb", "mean"),
            "num_params": ("num_params", "mean"),
        })
        .reset_index()
    )

    summary_path = out_dirs["base"] / "summary_mean_std.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[Saved] {dataset_name}:\n  {all_path}\n  {summary_path}")

    return df, summary


def run_experiments(cfg: Config, device: torch.device):
    """Run full experiment grid."""
    available = get_available_datasets(cfg)

    if cfg.QUICK_RUN:
        run_datasets = available
        run_models = cfg.MODELS
        run_seeds = cfg.SEEDS[:1]
        run_fractions = cfg.DATA_FRACTIONS[:1]
        run_lrs = cfg.LEARNING_RATES[:1]
    else:
        run_datasets = available
        run_models = cfg.MODELS
        run_seeds = cfg.SEEDS
        run_fractions = cfg.DATA_FRACTIONS
        run_lrs = cfg.LEARNING_RATES

    # Count total experiments
    total = (len(run_datasets) * len(run_models) * len(run_seeds) *
             len(run_fractions) * len(run_lrs))

    print("\n" + "=" * 80)
    print("EXPERIMENT PLAN")
    print("=" * 80)
    print(f"  Datasets:   {run_datasets}")
    print(f"  Models:     {list(run_models)}")
    print(f"  Seeds:      {list(run_seeds)}")
    print(f"  Fractions:  {list(run_fractions)}")
    print(f"  LRs:        {list(run_lrs)}")
    print(f"  Total runs: {total}")
    print(f"  Epochs/run: {cfg.QUICK_EPOCHS if cfg.QUICK_RUN else cfg.FINAL_EPOCHS}")
    print("=" * 80)

    all_combined = []
    run_count = 0

    for ds_name in run_datasets:
        dataset_pack = build_dataset(ds_name, seed=42, cfg=cfg)
        ds_results = []

        for model_name in run_models:
            for seed in run_seeds:
                for fraction in run_fractions:
                    for lr in run_lrs:
                        run_count += 1
                        print(f"\n{'─' * 80}")
                        print(f"Run {run_count}/{total}: "
                              f"{ds_name} | {model_name} | seed={seed} | "
                              f"frac={fraction} | lr={lr}")
                        print(f"{'─' * 80}")

                        try:
                            result = train_model_one_run(
                                dataset_pack, model_name, seed,
                                fraction, lr, cfg, device,
                            )
                            ds_results.append(result)
                            all_combined.append(result)
                        except Exception as e:
                            print(f"  [ERROR] Run failed: {e}")
                            import traceback
                            traceback.print_exc()

        # Save per-dataset results
        if ds_results:
            save_dataset_results(cfg, ds_name, ds_results)

    # Save combined results
    combined_df = pd.DataFrame(all_combined)
    combined_path = Path(cfg.OUTPUT_ROOT) / "combined_all_results.csv"
    combined_df.to_csv(combined_path, index=False)

    # Combined summary
    if len(combined_df) > 0:
        combined_summary = (
            combined_df.groupby(["dataset", "model", "fraction", "lr"])
            .agg(**{
                "mean_iou": ("test_iou", "mean"),
                "std_iou": ("test_iou", "std"),
                "mean_dice": ("test_dice", "mean"),
                "std_dice": ("test_dice", "std"),
                "mean_pixel_acc": ("test_pixel_acc", "mean"),
                "std_pixel_acc": ("test_pixel_acc", "std"),
                "mean_latency_ms": ("latency_ms", "mean"),
                "mean_param_memory_mb": ("param_memory_mb", "mean"),
                "num_params": ("num_params", "mean"),
            })
            .reset_index()
        )
        combined_summary_path = Path(cfg.OUTPUT_ROOT) / "combined_summary_mean_std.csv"
        combined_summary.to_csv(combined_summary_path, index=False)
        print(f"\n[Saved] Combined: {combined_path}")
        print(f"[Saved] Summary:  {combined_summary_path}")
    else:
        combined_summary = pd.DataFrame()

    return combined_df, combined_summary


# === RUN EXPERIMENTS ===
results_df, summary_df = run_experiments(cfg, device)
print("\n" + "=" * 80)
print("EXPERIMENT RESULTS")
print("=" * 80)
if len(results_df) > 0:
    print(results_df[["dataset", "model", "seed", "fraction", "lr",
                       "test_iou", "test_dice", "test_pixel_acc",
                       "latency_ms"]].to_string(index=False))
if len(summary_df) > 0:
    print("\nSUMMARY (mean ± std):")
    print(summary_df.to_string(index=False))

# %% [markdown]
# # 10. Visualization

# %%
# ============================================================
# 10. Visualization Functions
# ============================================================

def load_results_if_needed(cfg: Config):
    """Load results từ CSV nếu biến chưa có."""
    output_root = Path(cfg.OUTPUT_ROOT)
    results_df = None
    summary_df = None

    # Try combined files first
    combined_path = output_root / "combined_all_results.csv"
    combined_summary_path = output_root / "combined_summary_mean_std.csv"

    if combined_path.exists():
        results_df = pd.read_csv(combined_path)
    if combined_summary_path.exists():
        summary_df = pd.read_csv(combined_summary_path)

    # Fallback: load per-dataset files
    if results_df is None:
        dfs = []
        for f in sorted(output_root.glob("*/all_results.csv")):
            try:
                dfs.append(pd.read_csv(f))
            except Exception:
                pass
        if dfs:
            results_df = pd.concat(dfs, ignore_index=True)

    if summary_df is None:
        dfs = []
        for f in sorted(output_root.glob("*/summary_mean_std.csv")):
            try:
                dfs.append(pd.read_csv(f))
            except Exception:
                pass
        if dfs:
            summary_df = pd.concat(dfs, ignore_index=True)

    return results_df, summary_df


def plot_training_curves(results_df: pd.DataFrame, cfg: Config,
                         max_runs: int = 9):
    """Vẽ training curves (loss, IoU, dice) cho các runs."""
    if results_df is None or len(results_df) == 0:
        print("[SKIP] No results for training curves.")
        return

    if "history_path" not in results_df.columns:
        print("[SKIP] No history_path column.")
        return

    # Group by dataset, pick representative runs
    for ds_name in results_df["dataset"].unique():
        ds_rows = results_df[results_df["dataset"] == ds_name]
        # Pick best run per model
        best_per_model = (
            ds_rows.sort_values("test_iou", ascending=False)
            .groupby("model").head(1)
        ).head(max_runs)

        n_models = len(best_per_model)
        if n_models == 0:
            continue

        fig, axes = plt.subplots(n_models, 3, figsize=(15, 4 * n_models))
        if n_models == 1:
            axes = axes.reshape(1, -1)

        for i, (_, row) in enumerate(best_per_model.iterrows()):
            hist_path = Path(row["history_path"])
            if not hist_path.exists():
                continue

            hist = pd.read_csv(hist_path)
            titles = ["Loss", "Mean IoU", "Mean Dice"]
            train_cols = ["train_loss", "train_iou", "train_dice"]
            val_cols = ["val_loss", "val_iou", "val_dice"]

            for j, (title, tc, vc) in enumerate(zip(titles, train_cols, val_cols)):
                if tc in hist.columns and vc in hist.columns:
                    axes[i, j].plot(hist["epoch"], hist[tc], "b-", label="Train", linewidth=1.5)
                    axes[i, j].plot(hist["epoch"], hist[vc], "r-", label="Val", linewidth=1.5)
                    axes[i, j].set_title(f"{row['model']} — {title}")
                    axes[i, j].legend(fontsize=8)
                    axes[i, j].grid(True, alpha=0.3)
                    axes[i, j].set_xlabel("Epoch")

        plt.suptitle(f"Training Curves — {ds_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_path = Path(cfg.OUTPUT_ROOT) / ds_name / "figures" / "training_curves.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


def plot_model_comparison(summary_df: pd.DataFrame, cfg: Config):
    """Bar charts so sánh IoU và Dice giữa các models."""
    if summary_df is None or len(summary_df) == 0:
        print("[SKIP] No summary data.")
        return

    for ds_name in summary_df["dataset"].unique():
        sub = summary_df[summary_df["dataset"] == ds_name]
        base = sub.groupby("model")[["mean_iou", "mean_dice"]].mean().reset_index()

        x = np.arange(len(base))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        b1 = ax.bar(x - width/2, base["mean_iou"], width, label="Mean IoU",
                     color="steelblue", edgecolor="white")
        b2 = ax.bar(x + width/2, base["mean_dice"], width, label="Mean Dice",
                     color="coral", edgecolor="white")

        # Add value labels
        for bar in b1:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', fontsize=9)
        for bar in b2:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(base["model"], rotation=15)
        ax.set_ylabel("Score")
        ax.set_title(f"Model Comparison — {ds_name}")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        save_path = Path(cfg.OUTPUT_ROOT) / ds_name / "figures" / "model_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


def plot_accuracy_latency_tradeoff(summary_df: pd.DataFrame, cfg: Config):
    """Scatter plot: IoU vs Latency."""
    if summary_df is None or "mean_latency_ms" not in summary_df.columns:
        print("[SKIP] No latency data.")
        return

    for ds_name in summary_df["dataset"].unique():
        sub = summary_df[summary_df["dataset"] == ds_name]
        base = sub.groupby("model").agg(
            mean_iou=("mean_iou", "mean"),
            mean_latency_ms=("mean_latency_ms", "mean"),
        ).reset_index()

        fig, ax = plt.subplots(figsize=(7, 5))
        colors = plt.cm.Set1.colors
        for i, (_, r) in enumerate(base.iterrows()):
            ax.scatter(r["mean_latency_ms"], r["mean_iou"], s=200,
                       c=[colors[i]], zorder=5, edgecolors="black")
            ax.annotate(r["model"], (r["mean_latency_ms"], r["mean_iou"]),
                        textcoords="offset points", xytext=(10, 5), fontsize=10)

        ax.set_xlabel("Inference Latency (ms/image)")
        ax.set_ylabel("Mean IoU")
        ax.set_title(f"Accuracy–Latency Trade-off — {ds_name}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = Path(cfg.OUTPUT_ROOT) / ds_name / "figures" / "accuracy_latency.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


def plot_accuracy_memory_tradeoff(summary_df: pd.DataFrame, cfg: Config):
    """Scatter plot: IoU vs Memory."""
    if summary_df is None or "mean_param_memory_mb" not in summary_df.columns:
        print("[SKIP] No memory data.")
        return

    for ds_name in summary_df["dataset"].unique():
        sub = summary_df[summary_df["dataset"] == ds_name]
        base = sub.groupby("model").agg(
            mean_iou=("mean_iou", "mean"),
            mean_param_memory_mb=("mean_param_memory_mb", "mean"),
            num_params=("num_params", "mean"),
        ).reset_index()

        fig, ax = plt.subplots(figsize=(7, 5))
        colors = plt.cm.Set1.colors
        for i, (_, r) in enumerate(base.iterrows()):
            ax.scatter(r["mean_param_memory_mb"], r["mean_iou"], s=200,
                       c=[colors[i]], zorder=5, edgecolors="black")
            ax.annotate(f"{r['model']}\n({r['num_params']/1e6:.1f}M params)",
                        (r["mean_param_memory_mb"], r["mean_iou"]),
                        textcoords="offset points", xytext=(10, -10), fontsize=9)

        ax.set_xlabel("Parameter Memory (MB)")
        ax.set_ylabel("Mean IoU")
        ax.set_title(f"Accuracy–Memory Trade-off — {ds_name}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = Path(cfg.OUTPUT_ROOT) / ds_name / "figures" / "accuracy_memory.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


def plot_data_fraction_degradation(results_df: pd.DataFrame, cfg: Config):
    """Vẽ IoU suy giảm theo data fraction."""
    if results_df is None or len(results_df) == 0:
        print("[SKIP] No results for fraction analysis.")
        return

    fractions = sorted(results_df["fraction"].unique())
    if len(fractions) <= 1:
        print("[SKIP] Only 1 fraction — skip degradation plot.")
        return

    for ds_name in results_df["dataset"].unique():
        sub = results_df[results_df["dataset"] == ds_name]

        fig, ax = plt.subplots(figsize=(8, 5))
        for model_name in sub["model"].unique():
            msub = sub[sub["model"] == model_name]
            agg = msub.groupby("fraction")["test_iou"].agg(["mean", "std"]).reset_index()
            ax.errorbar(agg["fraction"], agg["mean"], yerr=agg["std"].fillna(0),
                        marker="o", capsize=4, linewidth=2, label=model_name)

        ax.set_xlabel("Training Data Fraction")
        ax.set_ylabel("Test IoU")
        ax.set_title(f"Data Fraction Degradation — {ds_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(fractions)
        plt.tight_layout()

        save_path = Path(cfg.OUTPUT_ROOT) / ds_name / "figures" / "data_fraction.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


def plot_lr_comparison(results_df: pd.DataFrame, cfg: Config):
    """So sánh IoU theo learning rate."""
    if results_df is None or len(results_df) == 0:
        print("[SKIP] No results for LR comparison.")
        return

    lrs = sorted(results_df["lr"].unique())
    if len(lrs) <= 1:
        print("[SKIP] Only 1 LR — skip comparison plot.")
        return

    for ds_name in results_df["dataset"].unique():
        sub = results_df[(results_df["dataset"] == ds_name) &
                         (results_df["fraction"] == 1.0)]

        if len(sub) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        models = sub["model"].unique()
        x = np.arange(len(models))
        width = 0.8 / len(lrs)

        for i, lr_val in enumerate(lrs):
            lr_sub = sub[sub["lr"] == lr_val]
            means = [lr_sub[lr_sub["model"] == m]["test_iou"].mean() for m in models]
            offset = (i - len(lrs)/2 + 0.5) * width
            bars = ax.bar(x + offset, means, width, label=f"LR={lr_val}")
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
                        f'{bar.get_height():.3f}', ha='center', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15)
        ax.set_ylabel("Test IoU")
        ax.set_title(f"Learning Rate Comparison (frac=1.0) — {ds_name}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        save_path = Path(cfg.OUTPUT_ROOT) / ds_name / "figures" / "lr_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


def plot_confusion_matrix_fig(results_df: pd.DataFrame, cfg: Config, device: torch.device):
    """Vẽ confusion matrix cho best model mỗi dataset."""
    if results_df is None or len(results_df) == 0:
        print("[SKIP] No results for confusion matrix.")
        return

    for ds_name in results_df["dataset"].unique():
        ds_sub = results_df[results_df["dataset"] == ds_name]
        dataset_pack = build_dataset(ds_name, seed=42, cfg=cfg)
        num_classes = dataset_pack["num_classes"]
        class_names = dataset_pack["class_names"]

        best_rows = (
            ds_sub.sort_values("test_iou", ascending=False)
            .groupby("model").head(1)
        )

        n_models = len(best_rows)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        if n_models == 1:
            axes = [axes]

        test_loader = DataLoader(
            dataset_pack["test"], batch_size=cfg.BATCH_SIZE,
            shuffle=False, num_workers=cfg.NUM_WORKERS,
        )

        for ax_idx, (_, row) in enumerate(best_rows.iterrows()):
            try:
                model = build_model(row["model"], num_classes).to(device)
                model.load_state_dict(
                    torch.load(row["checkpoint_path"], map_location=device,
                               weights_only=True)
                )
                model.eval()

                cm = np.zeros((num_classes, num_classes), dtype=np.int64)
                max_batches = 30 if cfg.QUICK_RUN else 100
                for batch_idx, (imgs, masks) in enumerate(test_loader):
                    if batch_idx >= max_batches:
                        break
                    imgs = imgs.to(device)
                    with torch.no_grad():
                        logits = model(imgs)
                    cm += compute_confusion_matrix(logits, masks, num_classes)

                # Normalize
                cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-7)

                im = axes[ax_idx].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
                axes[ax_idx].set_xticks(range(num_classes))
                axes[ax_idx].set_yticks(range(num_classes))
                axes[ax_idx].set_xticklabels(class_names, rotation=45)
                axes[ax_idx].set_yticklabels(class_names)
                axes[ax_idx].set_xlabel("Predicted")
                axes[ax_idx].set_ylabel("True")
                axes[ax_idx].set_title(f"{row['model']}\nIoU={row['test_iou']:.3f}")

                # Add text annotations
                for i in range(num_classes):
                    for j in range(num_classes):
                        color = "white" if cm_norm[i, j] > 0.5 else "black"
                        axes[ax_idx].text(j, i, f"{cm_norm[i, j]:.2f}",
                                          ha="center", va="center", color=color, fontsize=10)

                del model
                torch.cuda.empty_cache() if device.type == "cuda" else None

            except Exception as e:
                axes[ax_idx].set_title(f"{row['model']}\n[Error: {e}]")

        plt.suptitle(f"Confusion Matrix — {ds_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path = Path(cfg.OUTPUT_ROOT) / ds_name / "figures" / "confusion_matrix.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


@torch.no_grad()
def visualize_predictions(results_df: pd.DataFrame, cfg: Config,
                          device: torch.device, n: int = 4):
    """Visualize predictions: Image | GT | Pred | Error Map."""
    if results_df is None or len(results_df) == 0:
        print("[SKIP] No results for predictions.")
        return

    for ds_name in results_df["dataset"].unique():
        dataset_pack = build_dataset(ds_name, seed=42, cfg=cfg)
        test_ds = dataset_pack["test"]
        num_classes = dataset_pack["num_classes"]

        best_rows = (
            results_df[results_df["dataset"] == ds_name]
            .sort_values("test_iou", ascending=False)
            .groupby("model").head(1)
        )

        for _, row in best_rows.iterrows():
            try:
                model = build_model(row["model"], num_classes).to(device)
                model.load_state_dict(
                    torch.load(row["checkpoint_path"], map_location=device,
                               weights_only=True)
                )
                model.eval()

                n_show = min(n, len(test_ds))
                fig, axes = plt.subplots(n_show, 4, figsize=(16, 4 * n_show))
                if n_show == 1:
                    axes = axes.reshape(1, -1)

                for i in range(n_show):
                    img, mask = test_ds[i]
                    logits = model(img.unsqueeze(0).to(device))
                    pred = torch.argmax(logits, dim=1).squeeze(0).cpu()
                    err = (pred != mask).numpy().astype(np.float32)

                    axes[i, 0].imshow(denormalize(img).permute(1, 2, 0).numpy())
                    axes[i, 0].set_title("Input Image")
                    axes[i, 0].axis("off")

                    axes[i, 1].imshow(mask.numpy(), vmin=0, vmax=num_classes-1,
                                      cmap="tab10")
                    axes[i, 1].set_title("Ground Truth")
                    axes[i, 1].axis("off")

                    axes[i, 2].imshow(pred.numpy(), vmin=0, vmax=num_classes-1,
                                      cmap="tab10")
                    axes[i, 2].set_title("Prediction")
                    axes[i, 2].axis("off")

                    axes[i, 3].imshow(err, cmap="Reds", vmin=0, vmax=1)
                    axes[i, 3].set_title(f"Error Map (err={err.mean():.3f})")
                    axes[i, 3].axis("off")

                plt.suptitle(
                    f"{ds_name} — {row['model']} | IoU={row['test_iou']:.4f}",
                    fontsize=14, fontweight="bold",
                )
                plt.tight_layout()
                save_path = (Path(cfg.OUTPUT_ROOT) / ds_name / "figures" /
                             f"predictions_{row['model']}.png")
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.show()

                del model
                torch.cuda.empty_cache() if device.type == "cuda" else None

            except Exception as e:
                print(f"  [Error] {row['model']}: {e}")


def plot_cross_dataset_comparison(summary_df: pd.DataFrame, cfg: Config):
    """So sánh cross-dataset performance."""
    if summary_df is None or len(summary_df) == 0:
        print("[SKIP] No summary for cross-dataset comparison.")
        return

    datasets = summary_df["dataset"].unique()
    if len(datasets) < 2:
        print("[SKIP] Need ≥ 2 datasets for cross-dataset comparison.")
        return

    # Aggregate: best config per model per dataset
    agg = (
        summary_df[summary_df["fraction"] == 1.0]
        .groupby(["dataset", "model"])
        .agg(mean_iou=("mean_iou", "max"), mean_dice=("mean_dice", "max"))
        .reset_index()
    )

    models = sorted(agg["model"].unique())
    n_datasets = len(datasets)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in [(axes[0], "mean_iou", "Mean IoU"),
                               (axes[1], "mean_dice", "Mean Dice")]:
        x = np.arange(len(models))
        width = 0.8 / n_datasets

        for i, ds in enumerate(datasets):
            ds_data = agg[agg["dataset"] == ds]
            vals = [ds_data[ds_data["model"] == m][metric].values[0]
                    if len(ds_data[ds_data["model"] == m]) > 0 else 0
                    for m in models]
            offset = (i - n_datasets/2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=ds)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15)
        ax.set_ylabel(title)
        ax.set_title(f"Cross-Dataset: {title}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = Path(cfg.OUTPUT_ROOT) / "cross_dataset_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# %%
# ============================================================
# 10b. Complexity-Aware Analysis
# ============================================================

def estimate_boundary_ratio(mask: torch.Tensor) -> float:
    """Ước lượng tỷ lệ boundary pixels."""
    arr = mask.numpy() if isinstance(mask, torch.Tensor) else mask
    h_diff = np.abs(np.diff(arr.astype(float), axis=-1))
    v_diff = np.abs(np.diff(arr.astype(float), axis=-2))
    boundary_pixels = (h_diff > 0).sum() + (v_diff > 0).sum()
    return float(boundary_pixels / max(arr.size, 1))


def compute_sample_complexity(mask: torch.Tensor) -> float:
    """Tính complexity score cho 1 sample (boundary + foreground ratio)."""
    arr = mask.numpy() if isinstance(mask, torch.Tensor) else mask
    fg_ratio = (arr != 0).sum() / max(arr.size, 1)
    boundary = estimate_boundary_ratio(mask)
    return boundary + 0.3 * (1.0 - fg_ratio)


@torch.no_grad()
def complexity_analysis(results_df: pd.DataFrame, cfg: Config,
                        device: torch.device, max_samples: int = 120) -> pd.DataFrame:
    """Phân tích IoU theo độ phức tạp mẫu (Dễ/Trung bình/Khó)."""
    if results_df is None or len(results_df) == 0:
        print("[SKIP] No results for complexity analysis.")
        return pd.DataFrame()

    rows = []
    for ds_name in results_df["dataset"].unique():
        dataset_pack = build_dataset(ds_name, seed=42, cfg=cfg)
        test_ds = dataset_pack["test"]
        num_classes = dataset_pack["num_classes"]

        # Compute complexity for test samples
        n = min(len(test_ds), max_samples)
        complexities = []
        for idx in range(n):
            _, mask = test_ds[idx]
            complexities.append({
                "idx": idx,
                "complexity": compute_sample_complexity(mask),
            })
        cdf = pd.DataFrame(complexities)
        cdf["group"] = pd.qcut(cdf["complexity"], q=3,
                                labels=["Easy", "Medium", "Hard"],
                                duplicates="drop")

        # Evaluate best model per architecture
        best_rows = (
            results_df[results_df["dataset"] == ds_name]
            .sort_values("test_iou", ascending=False)
            .groupby("model").head(1)
        )

        for _, row in best_rows.iterrows():
            try:
                model = build_model(row["model"], num_classes).to(device)
                model.load_state_dict(
                    torch.load(row["checkpoint_path"], map_location=device,
                               weights_only=True)
                )
                model.eval()

                for _, cr in tqdm(cdf.iterrows(), total=len(cdf),
                                  desc=f"Complexity {ds_name}-{row['model']}",
                                  leave=False):
                    img, mask = test_ds[int(cr["idx"])]
                    logits = model(img.unsqueeze(0).to(device))
                    metrics = calculate_metrics(
                        logits, mask.unsqueeze(0).to(device), num_classes
                    )
                    rows.append({
                        "dataset": ds_name,
                        "model": row["model"],
                        "idx": int(cr["idx"]),
                        "complexity": cr["complexity"],
                        "group": cr["group"],
                        "iou": metrics["mean_iou"],
                        "dice": metrics["mean_dice"],
                    })

                del model
                torch.cuda.empty_cache() if device.type == "cuda" else None
            except Exception as e:
                print(f"  [Error] Complexity analysis {row['model']}: {e}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.to_csv(Path(cfg.OUTPUT_ROOT) / "complexity_results.csv", index=False)

    # Summary
    summary = df.groupby(["dataset", "model", "group"])[["iou", "dice"]].mean().reset_index()
    print("\n[Complexity Analysis] Summary:")
    print(summary.to_string(index=False))

    # Plot
    for ds_name in summary["dataset"].unique():
        sub = summary[summary["dataset"] == ds_name]
        fig, ax = plt.subplots(figsize=(8, 5))

        group_order = ["Easy", "Medium", "Hard"]
        for model_name in sub["model"].unique():
            msub = sub[sub["model"] == model_name].copy()
            msub["group"] = pd.Categorical(msub["group"], categories=group_order, ordered=True)
            msub = msub.sort_values("group")
            ax.plot(msub["group"], msub["iou"], marker="o", linewidth=2,
                    markersize=8, label=model_name)

        ax.set_xlabel("Sample Complexity")
        ax.set_ylabel("Mean IoU")
        ax.set_title(f"IoU vs. Sample Complexity — {ds_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = Path(cfg.OUTPUT_ROOT) / ds_name / "figures" / "complexity_analysis.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    return df


# %%
# ============================================================
# 10c. Run All Visualizations
# ============================================================

# Ensure results are loaded
if results_df is None or len(results_df) == 0:
    results_df, summary_df = load_results_if_needed(cfg)

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

plot_training_curves(results_df, cfg)
plot_model_comparison(summary_df, cfg)
plot_accuracy_latency_tradeoff(summary_df, cfg)
plot_accuracy_memory_tradeoff(summary_df, cfg)
plot_data_fraction_degradation(results_df, cfg)
plot_lr_comparison(results_df, cfg)
plot_confusion_matrix_fig(results_df, cfg, device)
visualize_predictions(results_df, cfg, device, n=4)
plot_cross_dataset_comparison(summary_df, cfg)

# Complexity analysis
complexity_df = complexity_analysis(results_df, cfg, device)

# %% [markdown]
# # 11. Critical Discussion
#
# ## 11.1. Phân tích kết quả chính (Main Results)
#
# Sau khi chạy notebook với `QUICK_RUN=False`, sử dụng bảng `combined_summary_mean_std.csv`
# để phân tích. Các điểm cần chú ý:
#
# | Metric | Ý nghĩa |
# |--------|---------|
# | **Mean IoU** | Đo overlap giữa prediction và GT, robust với class imbalance |
# | **Mean Dice** | Tương tự IoU nhưng nhấn mạnh agreement (thường ≥ IoU) |
# | **Pixel Accuracy** | Dễ bị bias bởi background class chiếm đa số |
# | **Per-class IoU** | Quan trọng cho phân tích class khó (boundary, polyp) |
#
# ### Kết quả dự kiến (điền sau khi chạy full):
# - **U-Net + ResNet34**: Expected IoU cao nhất nhờ dense skip connections,
#   đặc biệt ở boundary class (OxfordPet) và small polyps (Kvasir).
# - **SegNet + VGG16-BN**: Competitive nhưng có thể kém ở boundary
#   do thiếu full feature skip (chỉ dùng pooling indices).
# - **DeepLabV3+**: Tốt ở large-scale regions nhờ ASPP multi-scale,
#   nhưng decoder nhẹ có thể miss fine detail.
#
# ## 11.2. Hypothesis Validation Framework
#
# | Hypothesis | Evidence | Method | Expected Outcome |
# |-----------|----------|--------|-----------------|
# | **H1**: U-Net tốt nhờ skip connections | Per-class IoU, error maps | So sánh boundary IoU | U-Net IoU boundary > SegNet |
# | **H2**: SegNet efficient but weak at boundary | Latency, memory, boundary F1 | Profiling + error analysis | Latency ↓, boundary IoU ↓ |
# | **H3**: DeepLabV3+ good at large objects | IoU vs complexity | Complexity-aware analysis | IoU cao ở "Easy" samples |
# | **H4**: Pretrained helps with less data | Data fraction curves | Degradation rate comparison | Slope nhỏ hơn scratch models |
# | **H5**: No universal best model | All metrics combined | Radar chart / trade-off plots | Different winners per metric |
#
# ## 11.3. Cross-Domain Analysis
#
# - **OxfordPet** (natural): 3 classes, boundary class khó → U-Net expected best
# - **KvasirSEG** (medical): Binary, polyp size varies → DeepLabV3+ may benefit from ASPP
# - **Key question**: Model ranking có thay đổi giữa domains không?
#
# ## 11.4. Trade-offs
#
# | Model | Accuracy | Latency | Memory | Best for |
# |-------|----------|---------|--------|----------|
# | U-Net+ResNet34 | ★★★ | ★★ | ★★ | Accuracy-first tasks |
# | SegNet+VGG16-BN | ★★ | ★★ | ★★★ | Balanced efficiency |
# | DeepLabV3++ResNet34 | ★★★ | ★★ | ★★ | Multi-scale objects |
#
# ## 11.5. Limitations
#
# 1. **Encoder pretrained, decoder random**: Decoder không pretrained → cần đủ epochs hội tụ
# 2. **IMG_SIZE=256**: Nhỏ hơn real-world (512-1024), có thể underestimate performance
# 3. **No test-time augmentation (TTA)**: Could improve results by 1-2%
# 4. **SegNet decoder**: Không dùng BN initialization từ encoder (chỉ indices)
# 5. **DeepLabV3+ output stride**: Thực tế là /32 (không dùng dilated resnet),
#    paper gốc dùng /16 hoặc /8 → performance có thể thấp hơn
# 6. **Hardware-dependent latency**: Kết quả latency chỉ valid trên T4 GPU
#
# ## 11.6. Future Work
#
# 1. **Larger input resolution** (512×512) với output stride 16
# 2. **More encoders**: EfficientNet-B0, MobileNetV3 cho mobile deployment
# 3. **Advanced losses**: Boundary Loss, Tversky Loss, Focal Loss
# 4. **More datasets**: CamVid, Cityscapes cho urban scene segmentation
# 5. **Post-processing**: CRF, test-time augmentation
# 6. **Model optimization**: ONNX export, INT8 quantization, knowledge distillation

# %%
# ============================================================
# 12. Export & Summary
# ============================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

# Print final results table
if results_df is not None and len(results_df) > 0:
    print("\n📊 All Results:")
    display_cols = ["dataset", "model", "seed", "fraction", "lr",
                    "test_iou", "test_dice", "test_pixel_acc",
                    "latency_ms", "num_params"]
    available_cols = [c for c in display_cols if c in results_df.columns]
    print(results_df[available_cols].to_string(index=False))

if summary_df is not None and len(summary_df) > 0:
    print("\n📈 Summary (mean ± std across seeds):")
    print(summary_df.to_string(index=False))

# List all saved files
print("\n📁 Output files:")
output_root = Path(cfg.OUTPUT_ROOT)
for f in sorted(output_root.rglob("*")):
    if f.is_file():
        size_kb = f.stat().st_size / 1024
        print(f"  {f.relative_to(output_root)} ({size_kb:.0f} KB)")

# Zip outputs
zip_path = output_root.parent / "topic05_outputs.zip"
try:
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", str(output_root))
    print(f"\n📦 Exported: {zip_path}")
except Exception as e:
    print(f"\n⚠️ Zip failed: {e}")

print("\n✅ Notebook completed!")
print("=" * 80)
