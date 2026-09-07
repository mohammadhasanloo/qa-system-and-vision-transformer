"""CIFAR-10 loaders sized for the backbone being used."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ImageNet statistics: both backbones are pretrained on it.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224


def build_transforms(train: bool, input_size: int = INPUT_SIZE) -> transforms.Compose:
    """Resize to the backbone's input size and normalise.

    CIFAR-10 is 32x32 and both backbones expect 224x224, so the upscale is not
    optional. Augmentation is applied on the training split only.
    """
    steps = [transforms.Resize((input_size, input_size))]
    if train:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomCrop(input_size, padding=4)]
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


def cifar10_loaders(
    root: Path | str = "./data", batch_size: int = 32, input_size: int = INPUT_SIZE
) -> tuple[DataLoader, DataLoader]:
    """Training and test loaders, downloading CIFAR-10 on first use."""
    train_set = datasets.CIFAR10(
        root=str(root), train=True, download=True, transform=build_transforms(True, input_size)
    )
    test_set = datasets.CIFAR10(
        root=str(root), train=False, download=True, transform=build_transforms(False, input_size)
    )
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2),
    )
