"""
visualize.py — Grad-CAM visualizations for the EuroSAT land-use classification project.

Generates Grad-CAM heatmaps showing which regions of satellite images each model uses
to make its classification decisions. Supports visualizing one or more models side-by-side
for direct comparison, which is the primary use case for the presentation.

For each sampled image, the output figure shows:
    [Original Image] | [Model A Grad-CAM] | [Model B Grad-CAM] | ...

Sampling strategy:
    - "--sample per_class": One correctly-classified image per class (default).
    - "--sample confused": The most confused pairs from the confusion matrix.

Usage:
    # Single model
    python visualize.py --models baseline --model-paths ../results/baseline_best.pt

    # Compare all three models
    python visualize.py \\
        --models baseline resnet_frozen resnet_finetuned \\
        --model-paths ../results/baseline_best.pt ../results/resnet_frozen_best.pt ../results/resnet_finetuned_best.pt
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader

from data_setup import IMAGENET_MEAN, IMAGENET_STD, get_class_names, get_dataloaders
from models import get_model


def get_target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    """
    Return the convolutional layer to use as the Grad-CAM target for a given model.

    Grad-CAM needs a convolutional layer that produces spatial feature maps. For best
    visualizations, we use the LAST convolutional layer — by that point the features
    are highly semantic (classes) but still have enough spatial resolution to localize.

    Args:
        model:      The trained neural network.
        model_name: "baseline", "resnet_frozen", or "resnet_finetuned".

    Returns:
        The nn.Module representing the target convolutional layer.
    """
    if model_name == "baseline":
        # Last conv block of our custom CNN. block3 is a Sequential
        # whose first element is the Conv2d — Grad-CAM hooks the whole block and
        # captures the activations at its output, which is what we want.
        return model.block3[0] # The Conv2d inside block3
    elif model_name in ("resnet_frozen", "resnet_finetuned"):
        # ResNet18's last conv block is layer4. Its final Conv2d output feeds
        # directly into avg pooling and the classifier, making it the standard
        # Grad-CAM target for ResNet-family architectures.
        return model.layer4[-1]
    else:
        raise ValueError(f"No target layer defined for model '{model_name}'.")


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized tensor back to a displayable [0, 1] RGB NumPy array.

    The DataLoader applies ImageNet normalization so each channel is centered near 0.
    To display the image we need to reverse that normalization and rearrange the
    tensor from (C, H, W) to (H, W, C).

    Args:
        tensor: A normalized image tensor of shape (3, H, W).

    Returns:
        An (H, W, 3) float32 NumPy array with values in [0, 1].
    """
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    image = tensor.cpu().numpy() * std + mean
    image = np.clip(image, 0.0, 1.0)
    # (C, H, W) -> (H, W, C)
    return image.transpose(1, 2, 0).astype(np.float32)


def sample_one_per_class(
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, int]]:
    """
    Pick one image per class from the test loader (first image encountered per class).

    Grad-CAM needs a small, representative sample of images across all classes so the
    output figure covers every land-use category. Picking the first image of each
    class is simple and deterministic given a fixed random seed.

    Args:
        loader:      DataLoader for the test set.
        num_classes: Number of classes in the dataset.
        device:      Target device for the returned tensors.

    Returns:
        A list of (image_tensor, class_index) tuples, one per class, ordered by class index.
    """
    samples: dict[int, tuple[torch.Tensor, int]] = {}

    for images, labels in loader:
        for image, label in zip(images, labels):
            label_index = int(label.item())
            if label_index not in samples:
                samples[label_index] = (image.to(device), label_index)
            if len(samples) == num_classes:
                break
        if len(samples) == num_classes:
            break

    # Return samples ordered by class index for a consistent figure layout
    return [samples[i] for i in range(num_classes)]


def generate_gradcam_heatmap(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    image: torch.Tensor,
    target_class: int,
) -> np.ndarray:
    """
    Run Grad-CAM on a single image and return the resulting heatmap.

    Grad-CAM computes gradients of the target class logit with respect to the target
    layer's feature maps, then weighs the feature maps by the global average of those
    gradients to produce a 2D heatmap highlighting important spatial regions.

    Args:
        model:         The trained neural network.
        target_layer:  The convolutional layer whose activations we visualize.
        image:         A single image tensor of shape (3, H, W), already on the device.
        target_class:  The class index to compute the heatmap for.

    Returns:
        A (H, W) NumPy array with heatmap intensities in [0, 1].
    """
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    # Ensure all parameters require gradients so the backward pass can populate them.
    # This is necessary for resnet_frozen, where parameters were frozen during training.
    for param in model.parameters():
        param.requires_grad = True

    # The input tensor itself must also require gradients for Grad-CAM to work.
    input_tensor = image.unsqueeze(0).clone().detach().requires_grad_(True)

    cam = HiResCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    return grayscale_cam[0]


def build_comparison_figure(
    samples: list[tuple[torch.Tensor, int]],
    class_names: list[str],
    models_with_names: list[tuple[torch.nn.Module, str]],
    device: torch.device,
    output_path: str,
) -> None:
    """
    Generate and save a grid figure comparing Grad-CAM outputs across multiple models.

    Layout:
        Rows    = one per sampled image (one per class)
        Columns = original image, then one Grad-CAM overlay per model

    Args:
        samples:           List of (image_tensor, class_index) tuples.
        class_names:       Ordered list of class name strings.
        models_with_names: List of (model, model_name) tuples to visualize.
        device:            Target device.
        output_path:       Path where the output PNG is saved.
    """
    num_rows = len(samples)
    num_cols = 1 + len(models_with_names)  # +1 for the original image column

    figure, axes = plt.subplots(
        num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows)
    )

    # Handle edge case where there's only one row (axes is 1D instead of 2D)
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    for row_index, (image_tensor, true_class) in enumerate(samples):
        display_image = denormalize_image(image_tensor)

        # Column 0: original image
        axes[row_index, 0].imshow(display_image)
        axes[row_index, 0].set_ylabel(class_names[true_class], fontsize=11, rotation=0,
                                       ha="right", va="center", labelpad=20)
        axes[row_index, 0].set_xticks([])
        axes[row_index, 0].set_yticks([])
        if row_index == 0:
            axes[row_index, 0].set_title("Original", fontsize=12)

        # Columns 1+: one Grad-CAM overlay per model
        for model_index, (model, model_name) in enumerate(models_with_names):
            column_index = model_index + 1
            target_layer = get_target_layer(model, model_name)
            heatmap = generate_gradcam_heatmap(
                model, target_layer, image_tensor, true_class
            )
            overlay = show_cam_on_image(display_image, heatmap, use_rgb=True)

            axes[row_index, column_index].imshow(overlay)
            axes[row_index, column_index].set_xticks([])
            axes[row_index, column_index].set_yticks([])
            if row_index == 0:
                axes[row_index, column_index].set_title(model_name, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Grad-CAM comparison figure saved to {output_path}")


def visualize(args: argparse.Namespace) -> None:
    """
    Load one or more trained models and produce Grad-CAM comparison figures.

    Args:
        args: Parsed command-line arguments from argparse.
    """
    if len(args.models) != len(args.model_paths):
        raise ValueError(
            f"Number of --models ({len(args.models)}) must match number of "
            f"--model-paths ({len(args.model_paths)})."
        )

    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data and class names
    _, _, test_loader = get_dataloaders(data_dir=args.data_dir)
    class_names = get_class_names(args.data_dir)
    num_classes = len(class_names)

    # Load every requested model with its checkpoint
    models_with_names: list[tuple[torch.nn.Module, str]] = []
    for model_name, checkpoint_path in zip(args.models, args.model_paths):
        model = get_model(model_name).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        print(f"Loaded {model_name} from {checkpoint_path}")
        models_with_names.append((model, model_name))

    # Sample one image per class for the figure
    samples = sample_one_per_class(test_loader, num_classes, device)
    print(f"Sampled {len(samples)} images (one per class)")

    # Build the comparison figure
    model_tag = "_vs_".join(args.models) if len(args.models) > 1 else args.models[0]
    output_path = os.path.join(args.results_dir, f"gradcam_{model_tag}.png")
    build_comparison_figure(samples, class_names, models_with_names, device, output_path)


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM comparison figures for EuroSAT models."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        choices=["baseline", "resnet_frozen", "resnet_finetuned"],
        help="One or more models to visualize (space-separated).",
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
        help="Directory for saving the output figure (default: ../results).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize(args)