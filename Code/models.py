"""
models.py — Model definitions for the EuroSAT land-use classification project.

Defines two architectures:
  1. BaselineCNN  — A small custom CNN trained from scratch.
  2. build_resnet — A ResNet18 pretrained on ImageNet. Can be used in frozen mode
                    (only fc trains) or fine-tuned mode (all layers train).

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
            Adaptive pooling → 1×1 spatial dims, 128 channels
            Classifier: Flatten → Linear(128→256) → ReLU → Dropout(0.5) → Linear(256→10)

        Design decisions:
            - BatchNorm after each Conv stabilizes training and lets us use a higher lr.
            - MaxPool halves spatial resolution, progressively building abstract features.
            - AdaptiveAvgPool2d makes the network resolution-agnostic: the same architecture
              works at 64×64 input (producing 8×8 feature maps) or 224×224 input (producing
              28×28 feature maps). This also reduces parameters vs. a fixed-size flatten.
            - Dropout(0.5) in the classifier reduces overfitting on the training set.
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

        # Adaptive pooling makes the classifier input size independent of the
        # input image resolution. It collapses any spatial dimension to 1x1,
        # so we always get 128 features regardless of whether the input is
        # 64x64 (producing 8x8 feature maps) or 224x224 (producing 28x28).
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),  # 128 features (one per channel) → 256
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Regularization to prevent overfitting
            nn.Linear(256, num_classes),  # 256 → 10 class logits
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


def build_resnet(num_classes: int = 10, freeze_backbone: bool = True) -> nn.Module:
    """
        Build a ResNet18 model pretrained on ImageNet, adapted for EuroSAT classification.

        Transfer learning reuses feature representations learned from 1.2M ImageNet images.
        The backbone (all layers except the final fc) has already learned to detect edges,
        textures, and high-level visual patterns that are also useful for satellite imagery.

        Frozen vs. fine-tuned tradeoffs:
            - Frozen backbone (freeze_backbone=True): Only the new final fc layer trains.
              This is fast and stable but relies entirely on ImageNet features being
              transferable to satellite imagery.
            - Fine-tuned backbone (freeze_backbone=False): All layers are trainable, so
              the pretrained features can be adapted to the new domain. This typically
              requires a lower learning rate (e.g., 1e-4) to avoid overwriting useful
              pretrained weights too aggressively, but achieves significantly higher
              accuracy when the target domain differs from ImageNet.

        Args:
            num_classes:      Number of output classes (10 for EuroSAT).
            freeze_backbone:  If True, only the final fc layer is trainable. If False,
                              all layers are trainable for full fine-tuning.

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


def get_model(model_name: str) -> nn.Module:
    """
    Return the model corresponding to the given name string.

    Model options:
        - "baseline": Custom 3-block CNN trained from scratch.
        - "resnet_frozen": Pretrained ResNet18 with backbone frozen (only fc layer trains).
        - "resnet_finetuned": Pretrained ResNet18 with all layers trainable.

    Args:
        model_name: One of "baseline", "resnet_frozen", or "resnet_finetuned".

    Returns:
        An nn.Module instance ready to be moved to a device and trained.

    Raises:
        ValueError: If model_name is not one of the supported options.
    """
    if model_name == "baseline":
        return BaselineCNN(num_classes=10)
    elif model_name == "resnet_frozen":
        return build_resnet(num_classes=10, freeze_backbone=True)
    elif model_name == "resnet_finetuned":
        return build_resnet(num_classes=10, freeze_backbone=False)
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose 'baseline', 'resnet_frozen', or 'resnet_finetuned'."
        )
