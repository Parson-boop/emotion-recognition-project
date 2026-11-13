"""Chargement de FER2013 (format dossiers train/ et test/ avec un sous-dossier par classe)."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN, STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


def build_transforms(img_size: int, augment: bool):
    base = [transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size))]
    norm = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    aug = []
    if augment:
        aug = [transforms.RandomHorizontalFlip(),
               transforms.RandomRotation(10),
               transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0))]
    return transforms.Compose(base + aug + norm), transforms.Compose(base + norm)


def get_loaders(data_dir: str, img_size: int, batch_size: int, augment: bool,
                val_fraction: float = 0.1, seed: int = 42, num_workers: int = 2):
    train_t, eval_t = build_transforms(img_size, augment)
    # Deux ImageFolder sur le même dossier : un avec augmentation (train), un sans (val)
    full_train = datasets.ImageFolder(f"{data_dir}/train", transform=train_t)
    full_val = datasets.ImageFolder(f"{data_dir}/train", transform=eval_t)
    test_ds = datasets.ImageFolder(f"{data_dir}/test", transform=eval_t)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(full_train), generator=g).tolist()
    n_val = int(len(perm) * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_val, val_idx)

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(val_ds, shuffle=False, **kw),
            DataLoader(test_ds, shuffle=False, **kw),
            full_train.classes)
