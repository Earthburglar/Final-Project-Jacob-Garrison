"""
data_setup.py — Dataset loading, transforms, and DataLoader creation for EuroSAT RGB.

This script is the entry point for all data-related operations. It loads the EuroSAT RGB
dataset from disk, defines image transforms, splits the data into train/val/test subsets,
and returns ready-to-use DataLoaders for the training pipeline.

Usage (standalone check):
    python Code/data_setup.py --data-dir data/EuroSAT_RGB
"""

import argparse

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


# ImageNet mean and std are used for normalization because both ResNet and the baseline CNN
# benefit from input scaled to a similar distribution as what ResNet was pretrained on.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms() -> transforms.Compose:
    """
    Return augmented transforms applied to the training set only.

    Augmentation (horizontal flip + small rotation) is applied only during training
    to artificially increase variety and reduce overfitting. Val/test use clean
    transforms so evaluation reflects true generalization.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),          # Explicit resize (images may already be 64x64)
        transforms.RandomHorizontalFlip(),    # Satellite imagery has no preferred orientation
        transforms.RandomRotation(15),        # Small rotation for robustness
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms() -> transforms.Compose:
    """
    Return clean transforms applied to validation and test sets.

    No augmentation is used during evaluation — we want a deterministic, unmodified
    view of the data to get an accurate measure of generalization performance.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_class_names(data_dir: str) -> list[str]:
    """
    Return the list of class names in the order used by ImageFolder.

    ImageFolder assigns class indices alphabetically, so this function loads the
    dataset briefly just to extract the class list. This order matches what the
    model outputs and what evaluate.py uses for the confusion matrix labels.

    Args:
        data_dir: Path to the EuroSAT_RGB folder containing one subfolder per class.

    Returns:
        List of class name strings, e.g. ['AnnualCrop', 'Forest', ...].
    """
    dataset = datasets.ImageFolder(root=data_dir)
    return dataset.classes


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load the EuroSAT RGB dataset and return train, val, and test DataLoaders.

    The full dataset is split 70% train / 15% val / 15% test using a fixed seed
    for reproducibility. Because torch.utils.data.random_split returns Subset
    objects that share the parent dataset's transform, we load the dataset twice:
    once with augmented transforms (for train) and once with clean transforms
    (for val and test). We then use the same index splits on both copies.

    Args:
        data_dir:    Path to the EuroSAT_RGB directory (contains 10 class subfolders).
        batch_size:  Number of images per batch.
        num_workers: Number of worker processes for parallel data loading.
        seed:        Random seed for the train/val/test split.

    Returns:
        A tuple of (train_loader, val_loader, test_loader).
    """
    # Load the dataset twice so each split can have its own transform.
    # If we used a single ImageFolder and split it, all subsets would share the same
    # transform and we couldn't apply augmentation only to training data.
    augmented_dataset = datasets.ImageFolder(root=data_dir, transform=get_train_transforms())
    clean_dataset = datasets.ImageFolder(root=data_dir, transform=get_eval_transforms())

    total_size = len(augmented_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size  # Absorb any rounding remainder

    # Generate the index split once using a fixed seed, then apply those same indices
    # to both the augmented and clean dataset copies.
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices, test_indices = random_split(
        range(total_size), [train_size, val_size, test_size], generator=generator
    )

    train_subset = Subset(augmented_dataset, train_indices.indices)
    val_subset = Subset(clean_dataset, val_indices.indices)
    test_subset = Subset(clean_dataset, test_indices.indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify data loading for EuroSAT RGB.")
    parser.add_argument("--data-dir", type=str, default="../data/EuroSAT_RGB")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    print(f"Loading dataset from: {args.data_dir}")
    train_loader, val_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size)

    print(f"Train batches : {len(train_loader):>6}  ({len(train_loader.dataset)} images)")
    print(f"Val batches   : {len(val_loader):>6}  ({len(val_loader.dataset)} images)")
    print(f"Test batches  : {len(test_loader):>6}  ({len(test_loader.dataset)} images)")
    print(f"Class names   : {get_class_names(args.data_dir)}")
