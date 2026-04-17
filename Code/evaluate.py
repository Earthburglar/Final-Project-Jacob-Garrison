"""
evaluate.py — Test-set evaluation for the EuroSAT land-use classification project.

Loads a saved model checkpoint and runs full evaluation on the held-out test set.
Produces:
  - Printed overall test accuracy and per-class classification report
  - results/<model>_confusion_matrix.png  — heatmap with class name labels
  - results/<model>_classification_report.txt — text file of the sklearn report

Usage:
    python src/evaluate.py --model baseline --model-path results/baseline_best.pt
    python src/evaluate.py --model resnet   --model-path results/resnet_best.pt
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from data_setup import get_class_names, get_dataloaders
from models import get_model


def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """
    Run the model over an entire dataset split and collect true and predicted labels.

    The model is set to eval() mode to disable Dropout and use BatchNorm running
    statistics (not batch statistics). torch.no_grad() skips gradient tracking
    since we only need forward passes here.

    Args:
        model:  Trained neural network in eval mode.
        loader: DataLoader for the test set.
        device: The device (CPU or CUDA) tensors are moved to.

    Returns:
        A tuple of (true_labels, predicted_labels), each a flat Python list of ints.
    """
    model.eval()
    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            all_true_labels.extend(labels.cpu().tolist())
            all_predicted_labels.extend(predicted.cpu().tolist())

    return all_true_labels, all_predicted_labels


def save_confusion_matrix(
    true_labels: list[int],
    predicted_labels: list[int],
    class_names: list[str],
    model_name: str,
    results_dir: str,
) -> None:
    """
    Generate and save a normalized confusion matrix heatmap as a PNG.

    Normalization (dividing each row by its sum) converts raw counts to per-class
    recall rates, making it easier to spot which classes are being confused with
    which. Labels use class names rather than numeric indices for readability.

    Args:
        true_labels:       Ground-truth class indices for the test set.
        predicted_labels:  Model-predicted class indices for the test set.
        class_names:       Ordered list of class name strings.
        model_name:        "baseline" or "resnet", used in the filename.
        results_dir:       Directory where the PNG is saved.
    """
    matrix = confusion_matrix(true_labels, predicted_labels)

    # Normalize each row to [0, 1] so the color scale reflects per-class recall
    matrix_normalized = matrix.astype(float) / matrix.sum(axis=1, keepdims=True)

    figure, axis = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        matrix_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
        vmin=0.0,
        vmax=1.0,
    )

    axis.set_xlabel("Predicted Label", fontsize=12)
    axis.set_ylabel("True Label", fontsize=12)
    axis.set_title(f"Confusion Matrix — {model_name}", fontsize=14)

    # Rotate x-axis labels so class names don't overlap
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path = os.path.join(results_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def save_classification_report(
    true_labels: list[int],
    predicted_labels: list[int],
    class_names: list[str],
    model_name: str,
    results_dir: str,
) -> str:
    """
    Generate and save sklearn's per-class classification report as a text file.

    The report includes precision, recall, F1-score, and support per class,
    plus macro and weighted averages. This is saved to disk so it can be referenced
    while writing the final report without rerunning evaluation.

    Args:
        true_labels:       Ground-truth class indices for the test set.
        predicted_labels:  Model-predicted class indices for the test set.
        class_names:       Ordered list of class name strings.
        model_name:        "baseline" or "resnet", used in the filename.
        results_dir:       Directory where the .txt file is saved.

    Returns:
        The full classification report as a string (also printed to stdout).
    """
    report = classification_report(true_labels, predicted_labels, target_names=class_names)

    output_path = os.path.join(results_dir, f"{model_name}_classification_report.txt")
    with open(output_path, "w") as report_file:
        report_file.write(f"Classification Report — {model_name}\n")
        report_file.write("=" * 60 + "\n")
        report_file.write(report)

    print(f"Classification report saved to {output_path}")
    return report


def evaluate(args: argparse.Namespace) -> None:
    """
    Load a saved model checkpoint and run full evaluation on the test set.

    Args:
        args: Parsed command-line arguments from argparse.
    """
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load model ---
    model = get_model(args.model).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Loaded checkpoint from {args.model_path}")

    # --- Load test data ---
    _, _, test_loader = get_dataloaders(data_dir=args.data_dir)
    class_names = get_class_names(args.data_dir)

    # --- Collect predictions ---
    true_labels, predicted_labels = collect_predictions(model, test_loader, device)

    # --- Overall accuracy ---
    correct = sum(t == p for t, p in zip(true_labels, predicted_labels))
    overall_accuracy = correct / len(true_labels)
    print(f"\nTest Accuracy: {overall_accuracy * 100:.2f}%")

    # --- Per-class report ---
    print("\nClassification Report:")
    print("=" * 60)
    report = save_classification_report(
        true_labels, predicted_labels, class_names, args.model, args.results_dir
    )
    print(report)

    # --- Confusion matrix ---
    save_confusion_matrix(
        true_labels, predicted_labels, class_names, args.model, args.results_dir
    )


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the EuroSAT RGB test set."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["baseline", "resnet"],
        help="Which model architecture to evaluate.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved .pt checkpoint file.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/EuroSAT_RGB",
        help="Path to the EuroSAT_RGB directory (default: data/EuroSAT_RGB).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for saving output files (default: results).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
