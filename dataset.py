"""
dataset.py
ImageNet-100 dataset loader with augmentation.

ImageNet-100 is a 100-class subset of ImageNet-1k.
Expected directory structure:
    data/imagenet100/
        train/
            class_folder_1/  *.JPEG
            class_folder_2/  *.JPEG
            ...
        val/
            class_folder_1/  *.JPEG
            ...

You can download the class list from:
https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt
Then symlink the relevant class folders from your full ImageNet download.
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import config as C


# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────

def get_train_transform():
    """
    Standard ImageNet augmentation pipeline.
    Random crop + flip matches the paper's augmentation for CIFAR-10,
    scaled to 224×224 for ViT.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(C.IMAGE_SIZE, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet mean
            std =[0.229, 0.224, 0.225]    # ImageNet std
        ),
    ])


def get_val_transform():
    """Clean center-crop for validation — no augmentation."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(C.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ),
    ])


def get_reconstruction_transform():
    """
    For PSNR evaluation we need pixel values in [0,1] without normalization.
    We apply this on top of val images by denormalizing.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return mean, std


# ─────────────────────────────────────────────
# DATASET & DATALOADER BUILDERS
# ─────────────────────────────────────────────

def get_dataloaders(batch_size: int, num_workers: int = C.NUM_WORKERS):
    """
    Returns train and val DataLoaders for ImageNet-100.

    Args:
        batch_size:  Mini-batch size.
        num_workers: DataLoader worker count.

    Returns:
        train_loader, val_loader, num_classes
    """
    train_dir = os.path.join(C.DATA_ROOT, "train")
    val_dir   = os.path.join(C.DATA_ROOT, "val")

    assert os.path.isdir(train_dir), \
        f"Training data not found at {train_dir}. See dataset.py docstring."
    assert os.path.isdir(val_dir), \
        f"Validation data not found at {val_dir}. See dataset.py docstring."

    train_dataset = datasets.ImageFolder(
        root      = train_dir,
        transform = get_train_transform()
    )
    val_dataset = datasets.ImageFolder(
        root      = val_dir,
        transform = get_val_transform()
    )

    assert len(train_dataset.classes) == C.NUM_CLASSES, (
        f"Expected {C.NUM_CLASSES} classes, "
        f"found {len(train_dataset.classes)}. "
        "Check your ImageNet-100 class folders."
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,    # Needed for coding rate reduction batch stats
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = C.EVAL_BATCH,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    return train_loader, val_loader


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """
    Reverse ImageNet normalization. Used for PSNR computation.
    x: (N, 3, H, W) normalized tensor
    Returns: (N, 3, H, W) tensor in [0, 1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406],
                        device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225],
                        device=x.device).view(1, 3, 1, 1)
    return torch.clamp(x * std + mean, 0.0, 1.0)
