"""CIFAR-10 loaders with fixed train/val/test splits and seed control."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import StratifiedShuffleSplit

CIFAR_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)

# Official CIFAR-10 on the Hub (same 50k/10k splits as Toronto).
HF_CIFAR10_ID = "uoft-cs/cifar10"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch (CPU + CUDA) for reproducible splits/runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class SplitSizes:
    train: int
    val: int
    test: int


class _HFCifarDataset(Dataset):
    """Thin Dataset wrapper around a Hugging Face CIFAR-10 split."""

    def __init__(self, hf_split, transform=None):
        self._ds = hf_split
        self.transform = transform
        # Cache labels for stratified indexing without decoding images.
        self.targets = [int(x) for x in hf_split["label"]]

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, index: int):
        row = self._ds[int(index)]
        img = row["img"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"])


def _build_transform(image_size: int = 224):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD),
        ]
    )


def _load_cifar10_splits(data_root: str, transform, download: bool):
    """
    Prefer local torchvision CIFAR-10 if already extracted; otherwise load from
    the Hugging Face Hub (faster CDN than the Toronto mirror in many environments).
    """
    from pathlib import Path

    extracted = Path(data_root) / "cifar-10-batches-py"
    if extracted.is_dir():
        from torchvision import datasets

        train_full = datasets.CIFAR10(
            root=data_root, train=True, download=False, transform=transform
        )
        test_full = datasets.CIFAR10(
            root=data_root, train=False, download=False, transform=transform
        )
        return train_full, test_full

    if not download:
        raise FileNotFoundError(
            f"CIFAR-10 not found under {data_root} and download=False."
        )

    from datasets import load_dataset

    raw = load_dataset(HF_CIFAR10_ID)
    train_full = _HFCifarDataset(raw["train"], transform=transform)
    test_full = _HFCifarDataset(raw["test"], transform=transform)
    return train_full, test_full


def _stratified_indices(
    labels: np.ndarray,
    n_per_class: Optional[int],
    seed: int,
) -> np.ndarray:
    """Return indices covering all classes, optionally capped per class."""
    labels = np.asarray(labels)
    if n_per_class is None:
        return np.arange(len(labels))

    chosen = []
    rng = np.random.RandomState(seed)
    for cls in range(10):
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) < n_per_class:
            raise ValueError(
                f"Class {cls} has only {len(cls_idx)} samples; need {n_per_class}."
            )
        chosen.append(rng.choice(cls_idx, size=n_per_class, replace=False))
    return np.concatenate(chosen)


def _train_val_split(
    indices: np.ndarray,
    labels: np.ndarray,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified train/val split over the provided index pool."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1).")
    y = labels[indices]
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed
    )
    train_rel, val_rel = next(splitter.split(indices, y))
    return indices[train_rel], indices[val_rel]


def build_cifar10_loaders(
    data_root: str = "data/cifar10",
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
    image_size: int = 224,
    val_fraction: float = 0.1,
    train_per_class: Optional[int] = None,
    test_per_class: Optional[int] = None,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, SplitSizes]:
    """
    Build fixed train / val / test DataLoaders for CIFAR-10.

    Contract:
    - Official CIFAR-10 test set is never used for training or model selection.
    - Val is carved from the official train set (stratified).
    - Optional ``train_per_class`` / ``test_per_class`` draw a stratified subset
      (still preserving the train∪val vs test boundary).
    """
    set_seed(seed)
    transform = _build_transform(image_size)
    train_full, test_full = _load_cifar10_splits(data_root, transform, download)

    train_labels = np.asarray(train_full.targets)
    test_labels = np.asarray(test_full.targets)

    train_pool = _stratified_indices(train_labels, train_per_class, seed=seed)
    effective_val = val_fraction
    if train_per_class is not None:
        if train_per_class < 2:
            raise ValueError("train_per_class must be >= 2 to form a val split.")
        # Tiny smoke pools: keep >=1 sample/class in val when n is small.
        if train_per_class < max(2, int(np.ceil(1.0 / val_fraction))):
            effective_val = 1.0 / train_per_class

    train_idx, val_idx = _train_val_split(
        train_pool, train_labels, effective_val, seed=seed
    )
    test_idx = _stratified_indices(test_labels, test_per_class, seed=seed + 1)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        Subset(train_full, train_idx.tolist()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(train_full, val_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        Subset(test_full, test_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    sizes = SplitSizes(train=len(train_idx), val=len(val_idx), test=len(test_idx))
    return train_loader, val_loader, test_loader, sizes
