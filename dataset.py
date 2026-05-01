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

import glob
import json
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

def _find_labels_json(root: str):
    for name in ["Labels.json", "labels.json", "label.json"]:
        path = os.path.join(root, name)
        if os.path.isfile(path):
            return path
    return None


def _find_sharded_dirs(root: str, split: str):
    """Return directories under root whose name starts with split, e.g. train.X1."""
    if not os.path.isdir(root):
        return []
    dirs = []
    for entry in os.listdir(root):
        if entry.lower().startswith(split.lower()):
            candidate = os.path.join(root, entry)
            if os.path.isdir(candidate):
                dirs.append(candidate)
    return sorted(dirs)


def _collect_image_paths(root: str, split: str):
    """Collect image files from either standard or sharded train/val directories."""
    paths = []
    base_dir = os.path.join(root, split)
    patterns = []
    if os.path.isdir(base_dir):
        patterns = [os.path.join(base_dir, "**", "*.JPEG"),
                    os.path.join(base_dir, "**", "*.jpg"),
                    os.path.join(base_dir, "**", "*.png")]
    else:
        shard_dirs = _find_sharded_dirs(root, split)
        for shard_dir in shard_dirs:
            patterns.extend([
                os.path.join(shard_dir, "**", "*.JPEG"),
                os.path.join(shard_dir, "**", "*.jpg"),
                os.path.join(shard_dir, "**", "*.png"),
            ])
    for pattern in patterns:
        paths.extend(glob.glob(pattern, recursive=True))
    return sorted([p for p in paths if os.path.isfile(p)])


def _load_label_map(labels_path: str) -> dict:
    with open(labels_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Labels.json must contain a dictionary mapping image names to labels.")
    return data


def _normalize_key(path: str, root: str) -> str:
    rel = os.path.relpath(path, root).replace("\\", "/")
    rel = rel.lstrip("./")
    return rel


def _find_label(label_map: dict, img_path: str, root: str):
    candidates = []
    rel = _normalize_key(img_path, root)
    candidates.append(rel)
    candidates.append(os.path.basename(rel))
    if "/" in rel:
        candidates.append(rel.split("/", 1)[-1])
        candidates.append(rel.split("/", 2)[-1])
    if rel.lower().endswith(".jpeg") or rel.lower().endswith(".jpg"):
        candidates.append(os.path.splitext(os.path.basename(rel))[0])
    for key in candidates:
        if key in label_map:
            return label_map[key]
    raise KeyError(
        f"Could not find label for image path '{img_path}'. "
        f"Tried keys: {candidates}."
    )


class JsonImageDataset(Dataset):
    def __init__(self, root: str, split: str, label_map: dict, transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.label_map = label_map

        self.image_paths = _collect_image_paths(root, split)
        if len(self.image_paths) == 0:
            raise ValueError(
                f"No images found for split '{split}' under root '{root}'."
            )

        self.classes = sorted(set(label_map.values()))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        for path in self.image_paths:
            try:
                label = _find_label(label_map, path, root)
            except KeyError as e:
                raise KeyError(
                    f"Could not assign a label for '{path}'. "
                    f"Make sure Labels.json keys match image filenames or relative paths.\n{e}"
                )
            if label not in self.class_to_idx:
                raise ValueError(
                    f"Label '{label}' from Labels.json is not one of the known classes."
                )
            self.samples.append((path, self.class_to_idx[label]))

        if len(self.classes) != C.NUM_CLASSES:
            print(
                f"[Warning] Labels.json defines {len(self.classes)} classes, "
                f"expected {C.NUM_CLASSES}."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = datasets.folder.default_loader(path)
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(batch_size: int, num_workers: int = C.NUM_WORKERS):
    """
    Returns train and val DataLoaders for ImageNet-100.

    Args:
        batch_size:  Mini-batch size.
        num_workers: DataLoader worker count.

    Returns:
        train_loader, val_loader
    """
    train_dir = os.path.join(C.DATA_ROOT, "train")
    val_dir = os.path.join(C.DATA_ROOT, "val")
    labels_path = _find_labels_json(C.DATA_ROOT)

    sharded_train_dirs = _find_sharded_dirs(C.DATA_ROOT, "train")
    sharded_val_dirs = _find_sharded_dirs(C.DATA_ROOT, "val")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        train_dataset = datasets.ImageFolder(
            root=train_dir,
            transform=get_train_transform()
        )
        val_dataset = datasets.ImageFolder(
            root=val_dir,
            transform=get_val_transform()
        )
        assert len(train_dataset.classes) == C.NUM_CLASSES, (
            f"Expected {C.NUM_CLASSES} classes, "
            f"found {len(train_dataset.classes)}. "
            "Check your ImageNet-100 class folders."
        )
    elif sharded_train_dirs and sharded_val_dirs:
        from torchvision.datasets.folder import default_loader

        class ShardedImageFolderDataset(Dataset):
            def __init__(self, roots, transform=None):
                self.roots = roots
                self.transform = transform
                self.samples = []
                classes = set()
                for root_dir in roots:
                    for class_name in sorted(os.listdir(root_dir)):
                        class_path = os.path.join(root_dir, class_name)
                        if os.path.isdir(class_path):
                            classes.add(class_name)
                self.classes = sorted(classes)
                self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
                for root_dir in roots:
                    for class_name in self.classes:
                        class_path = os.path.join(root_dir, class_name)
                        if not os.path.isdir(class_path):
                            continue
                        for ext in ["*.JPEG", "*.jpg", "*.png"]:
                            for img_path in glob.glob(os.path.join(class_path, "**", ext), recursive=True):
                                if os.path.isfile(img_path):
                                    self.samples.append((img_path, self.class_to_idx[class_name]))
                if len(self.classes) != C.NUM_CLASSES:
                    print(
                        f"[Warning] Found {len(self.classes)} classes in sharded folders, "
                        f"expected {C.NUM_CLASSES}."
                    )

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                path, label = self.samples[idx]
                image = default_loader(path)
                if self.transform:
                    image = self.transform(image)
                return image, label

        train_dataset = ShardedImageFolderDataset(sharded_train_dirs, transform=get_train_transform())
        val_dataset = ShardedImageFolderDataset(sharded_val_dirs, transform=get_val_transform())
    elif labels_path is not None:
        label_map = _load_label_map(labels_path)
        train_dataset = JsonImageDataset(
            root=C.DATA_ROOT,
            split="train",
            label_map=label_map,
            transform=get_train_transform(),
        )
        val_dataset = JsonImageDataset(
            root=C.DATA_ROOT,
            split="val",
            label_map=label_map,
            transform=get_val_transform(),
        )
    else:
        raise AssertionError(
            f"Training data not found at {train_dir} and no Labels.json found in {C.DATA_ROOT}. "
            "Please provide a dataset with standard ImageFolder layout or a label JSON file."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=C.EVAL_BATCH,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
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
