"""
robustness.py — Robustness evaluation for the EuroSAT land-use classification project.

Evaluates trained models against five categories of image degradation that simulate
real-world conditions relevant to autonomous aerial platforms:
    1. Gaussian noise       — sensor noise in low-light or cheap cameras
    2. Gaussian blur        — motion blur or defocus
    3. Resolution loss      — downsample then upsample, simulating low-quality sensors
    4. Rotation             — arbitrary drone heading
    5. Brightness/contrast  — varying lighting conditions throughout the day

Outputs:
    - results/robustness_summary.csv — all (model, degradation, severity, accuracy) rows
    - results/robustness_<degradation>.png — one plot per degradation showing
      accuracy vs. severity for all three models

Usage:
    python robustness.py \\
        --models baseline resnet_frozen resnet_finetuned \\
        --model-paths \\
            ../results/baseline_best.pt \\
            ../results/resnet_frozen_best.pt \\
            ../results/resnet_finetuned_best.pt
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from data_setup import IMAGENET_MEAN, IMAGENET_STD, get_dataloaders
from models import get_model


# ----------------------------------------------------------------------------
# Degradation helpers
#
# Each degradation is a function that takes a batch of normalized images (shape
# (B, 3, H, W)) and a severity parameter, and returns a degraded batch of the
# same shape and normalization. To apply pixel-space degradations we denormalize
# to [0, 1], apply the effect, clamp, and renormalize.
# ----------------------------------------------------------------------------


def _denormalize_batch(images: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of normalized images back to [0, 1] pixel space.

    Training-time transforms subtract the ImageNet mean and divide by the std,
    so pixel-level effects need to be applied in the original [0, 1] range.
    """
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return images * std + mean


def _normalize_batch(images: torch.Tensor) -> torch.Tensor:
    """Re-apply ImageNet normalization after pixel-space manipulation."""
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def apply_gaussian_noise(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Add zero-mean Gaussian noise with std sigma to [0, 1] pixel values.

    Sigma is specified as a fraction of the pixel range (0.1 = 10% noise).
    """
    pixels = _denormalize_batch(images)
    noise = torch.randn_like(pixels) * sigma
    noisy = (pixels + noise).clamp(0.0, 1.0)
    return _normalize_batch(noisy)


def apply_gaussian_blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Apply Gaussian blur with the given sigma (larger sigma = more blur).

    Kernel size is chosen as roughly 6*sigma rounded up to the next odd integer,
    which captures ~99.7% of the Gaussian mass.
    """
    kernel_size = int(2 * round(3 * sigma) + 1)
    kernel_size = max(kernel_size, 3)
    blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    return blur(images)


def apply_resolution_loss(images: torch.Tensor, downsample_factor: int) -> torch.Tensor:
    """
    Downsample then upsample to simulate a lower-resolution sensor.

    Information lost at the downsampling step cannot be recovered by the upsample,
    which is what happens with a cheap camera's upscaler.
    """
    _, _, height, width = images.shape
    low_h = max(height // downsample_factor, 1)
    low_w = max(width // downsample_factor, 1)
    downsampled = F.interpolate(
        images, size=(low_h, low_w), mode="bilinear", align_corners=False
    )
    upsampled = F.interpolate(
        downsampled, size=(height, width), mode="bilinear", align_corners=False
    )
    return upsampled


def apply_rotation(images: torch.Tensor, angle_degrees: float) -> torch.Tensor:
    """
    Rotate every image in the batch by a fixed angle (counterclockwise).

    A fixed angle range (angle, angle) forces RandomRotation to use exactly that
    angle rather than a random sample from a range.
    """
    rotation = transforms.RandomRotation(degrees=(angle_degrees, angle_degrees))
    return rotation(images)


def apply_brightness(images: torch.Tensor, factor: float) -> torch.Tensor:
    """
    Multiply pixel values by the given factor in [0, 1] space, then clamp.

    factor < 1.0 darkens, factor > 1.0 brightens.
    """
    pixels = _denormalize_batch(images)
    adjusted = (pixels * factor).clamp(0.0, 1.0)
    return _normalize_batch(adjusted)


# ----------------------------------------------------------------------------
# Degradation configuration
# ----------------------------------------------------------------------------

# Each list includes the "clean" baseline value as the first entry so the plots
# start from the clean accuracy and trend outward.
DEGRADATION_SEVERITIES: dict[str, list[float]] = {
    "gaussian_noise":  [0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
    "gaussian_blur":   [0.0, 0.5, 1.0, 2.0, 3.0, 4.0],
    "resolution_loss": [1, 2, 4, 6, 8, 12],
    "rotation":        [0, 45, 90, 135, 180, 270],
    "brightness":      [1.0, 0.75, 0.5, 0.25, 1.25, 1.5],
}

DEGRADATION_FUNCTIONS = {
    "gaussian_noise":  apply_gaussian_noise,
    "gaussian_blur":   apply_gaussian_blur,
    "resolution_loss": apply_resolution_loss,
    "rotation":        apply_rotation,
    "brightness":      apply_brightness,
}

DEGRADATION_AXIS_LABELS = {
    "gaussian_noise":  "Noise standard deviation (fraction of pixel range)",
    "gaussian_blur":   "Blur sigma",
    "resolution_loss": "Downsampling factor",
    "rotation":        "Rotation angle (degrees)",
    "brightness":      "Brightness multiplier",
}

CLEAN_SEVERITY_VALUES = {
    "gaussian_noise":  0.0,
    "gaussian_blur":   0.0,
    "resolution_loss": 1,
    "rotation":        0,
    "brightness":      1.0,
}


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------


def evaluate_under_degradation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    degradation_name: str,
    severity: float,
) -> float:
    """
    Run the model over the test set with the given degradation and return accuracy.

    When the severity equals the "clean" value for that degradation, the image is
    passed through unchanged so the reported accuracy matches the base evaluation.
    """
    degrade_fn = DEGRADATION_FUNCTIONS[degradation_name]
    is_clean = severity == CLEAN_SEVERITY_VALUES[degradation_name]

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if not is_clean:
                if degradation_name == "resolution_loss":
                    images = degrade_fn(images, int(severity))
                else:
                    images = degrade_fn(images, severity)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total


def plot_robustness_curves(
    results: dict[str, dict[str, list[tuple[float, float]]]],
    output_dir: str,
) -> None:
    """
    Save one PNG per degradation comparing all models' accuracy vs. severity.
    """
    for degradation_name, per_model_results in results.items():
        figure, axis = plt.subplots(figsize=(8, 5))

        for model_name, points in per_model_results.items():
            # Sort by severity so the line plot doesn't jump around
            sorted_points = sorted(points, key=lambda pair: pair[0])
            severities = [severity for severity, _ in sorted_points]
            accuracies = [accuracy * 100 for _, accuracy in sorted_points]
            axis.plot(severities, accuracies, marker="o", label=model_name)

        axis.set_xlabel(DEGRADATION_AXIS_LABELS[degradation_name])
        axis.set_ylabel("Test Accuracy (%)")
        axis.set_title(f"Robustness to {degradation_name.replace('_', ' ').title()}")
        axis.set_ylim(0, 100)
        axis.grid(True, alpha=0.3)
        axis.legend()

        output_path = os.path.join(output_dir, f"robustness_{degradation_name}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved {output_path}")


def save_results_csv(
    results: dict[str, dict[str, list[tuple[float, float]]]],
    output_path: str,
) -> None:
    """
    Write a long-format CSV with columns: degradation, model, severity, accuracy.
    """
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["degradation", "model", "severity", "accuracy"])
        for degradation_name, per_model_results in results.items():
            for model_name, points in per_model_results.items():
                for severity, accuracy in points:
                    writer.writerow([degradation_name, model_name, severity, f"{accuracy:.4f}"])
    print(f"Saved {output_path}")


def run_robustness_experiments(args: argparse.Namespace) -> None:
    """
    Main driver: load models, loop over degradations and severities, save outputs.
    """
    if len(args.models) != len(args.model_paths):
        raise ValueError(
            f"Number of --models ({len(args.models)}) must match number of "
            f"--model-paths ({len(args.model_paths)})."
        )

    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_loader = get_dataloaders(data_dir=args.data_dir)

    loaded_models: list[tuple[str, torch.nn.Module]] = []
    for model_name, checkpoint_path in zip(args.models, args.model_paths):
        model = get_model(model_name).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        print(f"Loaded {model_name} from {checkpoint_path}")
        loaded_models.append((model_name, model))

    results: dict[str, dict[str, list[tuple[float, float]]]] = {}

    for degradation_name, severities in DEGRADATION_SEVERITIES.items():
        print(f"\n=== Degradation: {degradation_name} ===")
        results[degradation_name] = {}

        for model_name, model in loaded_models:
            points: list[tuple[float, float]] = []
            for severity in severities:
                accuracy = evaluate_under_degradation(
                    model, test_loader, device, degradation_name, severity
                )
                points.append((severity, accuracy))
                print(
                    f"  {model_name:20s} severity={severity:<6}  accuracy={accuracy * 100:.2f}%"
                )
            results[degradation_name][model_name] = points

    csv_path = os.path.join(args.results_dir, "robustness_summary.csv")
    save_results_csv(results, csv_path)
    plot_robustness_curves(results, args.results_dir)


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate model robustness under various image degradations."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        choices=["baseline", "resnet_frozen", "resnet_finetuned"],
        help="One or more models to evaluate (space-separated).",
    )
    parser.add_argument(
        "--model-paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to saved .pt checkpoints, in the same order as --models.",
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
        help="Directory for saving CSV and plots (default: ../results).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_robustness_experiments(args)