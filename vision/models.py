"""Backbones for CIFAR-10 classification."""

from __future__ import annotations

import timm
import torch
from torch import nn
from torchvision import models

NUM_CLASSES = 10
VGG_FEATURE_DIM = 25088  # 512 x 7 x 7 after the adaptive pool
BLOCK5_CONV1 = 28  # index into vgg19.features


def build_deit(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """DeiT-Base distilled, with a fresh classification head.

    A vision transformer trained with a distillation token, which is what lets it
    work on a dataset far smaller than the ImageNet-21k scale plain ViTs need.
    """
    return timm.create_model(
        "deit_base_distilled_patch16_224", pretrained=pretrained, num_classes=num_classes
    )


def build_vgg19(
    num_classes: int = NUM_CLASSES,
    trainable_feature_block: int = BLOCK5_CONV1,
    pretrained: bool = True,
) -> nn.Module:
    """VGG19 with everything frozen except one convolutional block and the head.

    Freezing happens in two passes: every parameter off, then one block back on.
    Fine-tuning more of a pretrained backbone is not automatically better on a
    dataset this small, and leaving the whole feature extractor trainable
    overfits rather than helping.
    """
    model = models.vgg19(weights=models.VGG19_Weights.DEFAULT if pretrained else None)

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.features[trainable_feature_block].parameters():
        parameter.requires_grad = True

    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(VGG_FEATURE_DIM, 256),
        nn.ELU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )
    return model


def trainable_parameters(model: nn.Module) -> int:
    """How many parameters will actually receive gradients."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
