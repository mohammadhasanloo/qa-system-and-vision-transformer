"""Tests for the backbones, built without pretrained weights so they need no download."""

from __future__ import annotations

import torch

from vision.data import INPUT_SIZE, build_transforms
from vision.models import NUM_CLASSES, build_vgg19, trainable_parameters


def test_vgg19_outputs_one_logit_per_class():
    model = build_vgg19(pretrained=False).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE))
    assert logits.shape == (2, NUM_CLASSES)


def test_vgg19_freezes_the_backbone_except_one_block():
    """Only one convolutional block may be trainable; the rest stays fixed."""
    model = build_vgg19(pretrained=False)
    frozen = sum(p.numel() for p in model.features.parameters() if not p.requires_grad)
    trainable_in_features = sum(
        p.numel() for p in model.features.parameters() if p.requires_grad
    )
    assert frozen > 0
    assert trainable_in_features > 0
    # One convolutional block out of sixteen: a small minority of the backbone.
    assert trainable_in_features / (frozen + trainable_in_features) < 0.2


def test_classifier_head_is_always_trainable():
    model = build_vgg19(pretrained=False)
    assert all(p.requires_grad for p in model.classifier.parameters())
    assert trainable_parameters(model) > 0


def test_train_and_eval_transforms_agree_on_output_shape():
    from PIL import Image

    image = Image.new("RGB", (32, 32))
    for train in (True, False):
        assert build_transforms(train)(image).shape == (3, INPUT_SIZE, INPUT_SIZE)
