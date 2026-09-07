"""CIFAR-10 classification with DeiT and VGG19."""

from vision.data import cifar10_loaders
from vision.models import build_deit, build_vgg19, trainable_parameters
from vision.training import History, fit, run_epoch

__all__ = [
    "History",
    "build_deit",
    "build_vgg19",
    "cifar10_loaders",
    "fit",
    "run_epoch",
    "trainable_parameters",
]
