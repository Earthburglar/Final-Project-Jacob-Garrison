"""
train.py — Unified training script for the EuroSAT land-use classification project.

Trains either the BaselineCNN or the pretrained ResNet18 model on the EuroSAT RGB
dataset. After training, saves:
  - The best model checkpoint (by validation accuracy) to results/<model>_best.pt
  - A timestamped JSON run log to results/run_<model>_<timestamp>.json
  - A training curves plot to results/<model>_training_curves.png

Usage:
    python src/train.py --model baseline
    python src/train.py --model resnet --epochs 30 --lr 0.0005
"""

import argparse
import json
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data_setup import get_dataloaders
from models import get_model


def set_random_seeds(seed: int = 42) -> None:
    """
    Seed all random number generators for reproducibility.

    Setting seeds on Python's random, NumPy, and PyTorch (both CPU and GPU) ensures
    that the weight initialization, data shuffling, and any stochastic operations
    produce the same results across runs with the same seed.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Run one full pass over the training set and return the average training loss.

    The model is set to train() mode so that BatchNorm and Dropout behave correctly
    (they behave differently during evaluation).

    Args:
        model:     The neural network being trained.
        loader:    DataLoader for the training set.
        criterion: Loss function (CrossEntropyLoss).
        optimizer: Adam optimizer.
        device:    The device (CPU or CUDA) that tensors are moved to.

    Returns:
        Average training loss over the epoch.
    """
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate the model on a dataset split (val or test) and return loss and accuracy.

    The model is set to eval() mode to disable Dropout and use running statistics in
    BatchNorm. torch.no_grad() skips gradient computation, reducing memory usage.

    Args:
        model:     The neural network to evaluate.
        loader:    DataLoader for the validation or test set.
        criterion: Loss function (CrossEntropyLoss).
        device:    The device tensors are moved to.

    Returns:
        A tuple of (average_loss, accuracy) where accuracy is in [0, 1].
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Predicted class is the index with the highest logit
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / len(loader)
    accuracy = correct / total
    return average_loss, accuracy


def save_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    val_accuracies: list[float],
    model_name: str,
    results_dir: str,
) -> None:
    """
    Save a plot of training loss, validation loss, and validation accuracy vs. epoch.

    The loss curves share a left y-axis; validation accuracy uses a right y-axis
    so both scales are readable on the same figure.

    Args:
        train_losses:    List of average training losses, one per epoch.
        val_losses:      List of average validation losses, one per epoch.
        val_accuracies:  List of validation accuracies (0–1), one per epoch.
        model_name:      "baseline" or "resnet", used in the filename.
        results_dir:     Directory where the PNG is saved.
    """
    epochs = range(1, len(train_losses) + 1)

    figure, axis_loss = plt.subplots(figsize=(10, 5))
    axis_accuracy = axis_loss.twinx()  # Second y-axis shares the x-axis

    axis_loss.plot(epochs, train_losses, label="Train Loss", color="steelblue")
    axis_loss.plot(epochs, val_losses, label="Val Loss", color="darkorange")
    axis_accuracy.plot(
        epochs,
        [acc * 100 for acc in val_accuracies],
        label="Val Accuracy (%)",
        color="green",
        linestyle="--",
    )

    axis_loss.set_xlabel("Epoch")
    axis_loss.set_ylabel("Loss")
    axis_accuracy.set_ylabel("Validation Accuracy (%)")

    # Combine legends from both axes into one box
    lines_left, labels_left = axis_loss.get_legend_handles_labels()
    lines_right, labels_right = axis_accuracy.get_legend_handles_labels()
    axis_loss.legend(lines_left + lines_right, labels_left + labels_right, loc="upper right")

    plt.title(f"Training Curves — {model_name}")
    plt.tight_layout()

    output_path = os.path.join(results_dir, f"{model_name}_training_curves.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Training curves saved to {output_path}")


def save_run_log(
    model_name: str,
    config: dict,
    history: dict,
    best_epoch: int,
    best_val_accuracy: float,
    results_dir: str,
) -> None:
    """
    Save a JSON log file capturing the full experimental record for this training run.

    This log is the primary artifact for the final report — it records every
    hyperparameter and the full loss/accuracy history so results can be reproduced
    or referenced without rerunning training.

    Args:
        model_name:        "baseline" or "resnet".
        config:            Dict of hyperparameters (epochs, batch_size, lr, model_params).
        history:           Dict with keys "train_loss", "val_loss", "val_accuracy".
        best_epoch:        Epoch (1-indexed) where best val accuracy was achieved.
        best_val_accuracy: The best validation accuracy (0–1).
        results_dir:       Directory where the JSON is saved.
    """
    timestamp = datetime.now().isoformat(timespec="seconds")
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_log = {
        "model": model_name,
        "timestamp": timestamp,
        "config": config,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
    }

    filename = f"run_{model_name}_{timestamp_file}.json"
    output_path = os.path.join(results_dir, filename)
    with open(output_path, "w") as log_file:
        json.dump(run_log, log_file, indent=4)

    print(f"Run log saved to {output_path}")


def train(args: argparse.Namespace) -> None:
    """
    Full training loop: load data, build model, train, validate, save outputs.

    Args:
        args: Parsed command-line arguments from argparse.
    """
    set_random_seeds(seed=42)

    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size
    )

    # --- Model ---
    freeze = not getattr(args, "unfreeze", False)
    model = get_model(args.model, freeze_backbone=freeze).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}  |  Trainable parameters: {num_params:,}")

    # --- Loss and optimizer ---
    # CrossEntropyLoss is the standard choice for single-label multiclass classification.
    # It combines a softmax with negative log-likelihood, so the model outputs raw logits.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    # --- Training loop ---
    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_accuracy = 0.0
    best_epoch = 1
    best_model_path = os.path.join(args.results_dir, f"{args.model}_best.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch [{epoch:>3}/{args.epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy * 100:.2f}%"
        )

        # Save the model whenever validation accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

    print(f"\nBest model: epoch {best_epoch}, val accuracy {best_val_accuracy * 100:.2f}%")
    print(f"Checkpoint saved to {best_model_path}")

    # --- Save artifacts ---
    save_training_curves(train_losses, val_losses, val_accuracies, args.model, args.results_dir)

    save_run_log(
        model_name=args.model,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "model_params": num_params,
        },
        history={
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_accuracy": val_accuracies,
        },
        best_epoch=best_epoch,
        best_val_accuracy=best_val_accuracy,
        results_dir=args.results_dir,
    )


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a model on the EuroSAT RGB dataset.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["baseline", "resnet"],
        help="Which model architecture to train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs (default: 25).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training and validation (default: 64).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for the Adam optimizer (default: 0.001).",
    )
    parser.add_argument(
        "--unfreeze",
        action="store_true",
        help="Unfreeze the ResNet backbone for full fine-tuning (ignored for baseline).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data/EuroSAT_RGB",
        help="Path to the EuroSAT_RGB directory (default: ../data/EuroSAT_RGB).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="../results",
        help="Directory for saving checkpoints, logs, and plots (default: ../results).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
