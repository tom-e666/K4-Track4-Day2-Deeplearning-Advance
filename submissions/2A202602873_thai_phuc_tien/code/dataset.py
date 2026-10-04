"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.
Giao diện giữ nguyên để kết nối với notebook, train.py và eval.py:
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1)."""
    p = Path(labels_dir)
    train_df = pd.read_csv(p / f"train_subset{fold}.csv")
    val_df = pd.read_csv(p / f"val_subset{fold}.csv")
    test_df = pd.read_csv(p / f"test_subset{fold}.csv")
    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1)."""
    img_dir = Path(images_dir)
    n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    total = n_train + n_val + n_test

    assert total == 17509, f"Tổng số ảnh phải bằng 17509, thực tế: {total}"

    train_set = set(train_df["Filename"])
    val_set = set(val_df["Filename"])
    test_set = set(test_df["Filename"])

    ov_train_val = len(train_set & val_set)
    ov_train_test = len(train_set & test_set)
    ov_val_test = len(val_set & test_set)

    assert ov_train_val == 0, f"Giao train và val không rỗng: {ov_train_val} ảnh trùng"
    assert ov_train_test == 0, f"Giao train và test không rỗng: {ov_train_test} ảnh trùng"
    assert ov_val_test == 0, f"Giao val và test không rỗng: {ov_val_test} ảnh trùng"

    all_files = train_set | val_set | test_set
    assert len(all_files) == 17509, f"Hợp ba tập phải bằng 17509, thực tế: {len(all_files)}"

    missing = [f for f in all_files if not (img_dir / f).is_file()]
    assert len(missing) == 0, f"Thiếu {len(missing)} ảnh trên đĩa tại {img_dir}. Ví dụ: {missing[:3]}"

    per_class = {
        "train": train_df["Label"].value_counts().sort_index().to_dict(),
        "val": val_df["Label"].value_counts().sort_index().to_dict(),
        "test": test_df["Label"].value_counts().sort_index().to_dict(),
    }

    return {
        "n": {"train": n_train, "val": n_val, "test": n_test, "total": total},
        "per_class": per_class,
        "overlap": {"train_val": ov_train_val, "train_test": ov_train_test, "val_test": ov_val_test},
    }


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo torchvision transforms cho huấn luyện hoặc đánh giá."""
    if train:
        aug_ops = []
        if aug == "basic":
            aug_ops = [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
            ]
        elif aug == "color":
            aug_ops = [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        elif aug == "trivial":
            aug_ops = [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.TrivialAugmentWide(),
            ]
        elif aug == "randaug":
            aug_ops = [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(),
            ]
        else:
            aug_ops = [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
            ]
        return transforms.Compose(aug_ops + [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ images_dir theo DataFrame (Filename, Label)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, str]:
        row = self.df.iloc[i]
        filename = str(row["Filename"])
        label = int(row["Label"])
        img_path = self.images_dir / filename
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
        return img, label, filename


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2) -> DataLoader:
    """Tạo DataLoader cho tập dữ liệu."""
    ds = DeepWeedsDataset(df, images_dir, transform=transform)

    if train:
        if sampler == "balanced":
            class_counts = df["Label"].value_counts().to_dict()
            weights = [1.0 / class_counts[y] for y in df["Label"]]
            torch_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            shuffle = False
        else:
            torch_sampler = None
            shuffle = True
        drop_last = len(df) > batch_size
    else:
        torch_sampler = None
        shuffle = False
        drop_last = False

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=torch_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        worker_init_fn=_seed_worker if train else None,
    )
