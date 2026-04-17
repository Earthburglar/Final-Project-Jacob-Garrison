"""
models.py — Model definitions for the EuroSAT land-use classification project.

Defines two architectures:
  1. BaselineCNN  — A small custom CNN trained from scratch.
  2. build_resnet — A ResNet18 pretrained on ImageNet, fine-tuned for 10-class output.

Both are accessed through the get_model() dispatcher used by train.py and evaluate.py.
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class BaselineCNN(nn.Module):
    """
    A simple 3-block convolutional neural network trained from scratch.

    Architecture overview:
        Block 1: Conv(3→32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
        Block 2: Conv(32→64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
        Block 3: Conv(64→128, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
        Classifier: Flatten → Linear(8192→256) → ReLU → Dropout(0.5) → Linear(256→10)

    Spatial dimension math (input 64×64):
        After block 1: 64/2 = 32×32
        After block 2: 32/2 = 16×16
        After block 3: 16/2 = 8×8
        Flattened: 128 channels × 8 × 8 = 8192

    Design decisions:
        - BatchNorm after each Conv stabilizes training and lets us use a higher lr.
        - MaxPool halves spatial resolution, progressively building abstract features.
        - Dropout(0.5) in the classifier reduces overfitting on the training set.
        - Three blocks give enough depth to detect spatial patterns in 64×64 images
          without the model being too large for the dataset size (~18,900 training images).
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        # Each convolutional block: Conv → BatchNorm → ReLU → MaxPool
        # padding=1 on a 3×3 kernel preserves spatial size before pooling
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64×64 → 32×32
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32×32 → 16×16
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16×16 → 8×8
        )

        # Fully connected classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),  # 8192 → 256
            nn.ReLU(),
            nn.Dropout(p=0.5),            # Regularization to prevent overfitting
            nn.Linear(256, num_classes),  # 256 → 10 class logits
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


def build_resnet(num_classes: int = 10, freeze_backbone: bool = True) -> nn.Module:
    """
    Build a ResNet18 model pretrained on ImageNet, adapted for EuroSAT classification.

    Transfer learning reuses feature representations learned from 1.2M ImageNet images.
    The backbone (all layers except the final fc) has already learned to detect edges,
    textures, and high-level visual patterns that are also useful for satellite imagery.

    Why freeze the backbone:
        Our dataset has ~18,900 training images — small compared to ImageNet's 1.2M.
        Fine-tuning all layers on a small dataset risks catastrophic forgetting: the
        pretrained weights get overwritten before the classifier head has a chance to
        converge, often hurting performance. Freezing the backbone forces only the
        classifier head to learn, which is both faster and more stable on small datasets.

    Args:
        num_classes:      Number of output classes (10 for EuroSAT).
        freeze_backbone:  If True, only the final fc layer is trainable.

    Returns:
        A modified ResNet18 nn.Module ready for training.
    """
    # Load ResNet18 with official pretrained ImageNet weights
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        # Freeze all parameters so gradients are not computed for the backbone
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully connected layer (originally 512 → 1000 for ImageNet)
    # with a new layer sized for our 10-class problem. Because this is a new nn.Linear,
    # its parameters are unfrozen by default regardless of the freeze_backbone step above.
    model.fc = nn.Linear(in_features=512, out_features=num_classes)

    return model


def get_model(model_name: str, freeze_backbone: bool = True) -> nn.Module:
    """
    Return the model corresponding to the given name string.

    Args:
        model_name: Either "baseline" (BaselineCNN) or "resnet" (pretrained ResNet18).

    Returns:
        An nn.Module instance ready to be moved to a device and trained.

    Raises:
        ValueError: If model_name is not one of the supported options.
    """
    if model_name == "baseline":
        return BaselineCNN(num_classes=10)
    elif model_name == "resnet":
        return build_resnet(num_classes=10, freeze_backbone=freeze_backbone)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose 'baseline' or 'resnet'.")
