"""Training and evaluation loops."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    test_loss: list[float] = field(default_factory=list)
    test_accuracy: list[float] = field(default_factory=list)


def default_device() -> torch.device:
    """CUDA, then Apple Silicon, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """One pass over a loader. Trains when an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)

    total_loss = correct = seen = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)

    return total_loss / seen, 100.0 * correct / seen


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: torch.device | None = None,
) -> History:
    """Train for a fixed number of epochs, recording loss and accuracy each one."""
    device = device or default_device()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    history = History()

    for epoch in range(epochs):
        train_loss, train_accuracy = run_epoch(model, train_loader, loss_fn, device, optimizer)
        test_loss, test_accuracy = run_epoch(model, test_loader, loss_fn, device)

        history.train_loss.append(train_loss)
        history.train_accuracy.append(train_accuracy)
        history.test_loss.append(test_loss)
        history.test_accuracy.append(test_accuracy)

        if scheduler is not None:
            # ReduceLROnPlateau is stepped on a metric; the rest on nothing.
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_accuracy)
            else:
                scheduler.step()

        print(
            f"epoch {epoch + 1}/{epochs}  "
            f"train {train_accuracy:5.2f}% ({train_loss:.4f})  "
            f"test {test_accuracy:5.2f}% ({test_loss:.4f})"
        )

    return history
