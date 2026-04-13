
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ECEN 5743 graduate deep learning final project. Compares a custom CNN baseline against a pretrained ResNet18 transfer learning model for 10-class land-use classification on the EuroSAT RGB dataset (27,000 64×64 satellite images).

## Environment

Python 3.11 with a `.venv` at the project root. Always activate before running scripts:

```bash
source .venv/bin/activate
```

## Running the Pipeline

Scripts are run in order from the project root:

```bash
# 1. Train a model (--model is required: "baseline" or "resnet")
python src/train.py --model baseline
python src/train.py --model resnet

# 2. Evaluate a saved model on the test set
python src/evaluate.py --model baseline --model-path results/baseline_best.pt
python src/evaluate.py --model resnet --model-path results/resnet_best.pt
```

Key `train.py` flags: `--epochs` (default 25), `--batch-size` (default 64), `--lr` (default 0.001), `--data-dir` (default `data/EuroSAT_RGB`), `--results-dir` (default `results`).

Key `evaluate.py` flags: `--data-dir`, `--results-dir`.

## Architecture

Four scripts in `src/`, each independently runnable:

| Script | Purpose |
|---|---|
| `data_setup.py` | `ImageFolder` loading, train/val/test split (70/15/15, seed=42), augmented transforms for train only |
| `models.py` | `BaselineCNN` (3-block Conv→BN→ReLU→Pool, then FC), `build_resnet()` (ResNet18 with frozen backbone + replaced fc), `get_model(name)` dispatcher |
| `train.py` | Unified training loop; saves best `.pt` checkpoint, timestamped JSON run log, and training curves PNG to `results/` |
| `evaluate.py` | Loads a `.pt` file, prints test accuracy + sklearn classification report, saves confusion matrix PNG and classification report `.txt` to `results/` |

**Data flow:** `data_setup.py` produces three `DataLoader`s → `train.py` consumes train/val loaders → saves `results/<model>_best.pt` + JSON log → `evaluate.py` loads the `.pt` and the test loader for final metrics.

**Transform note:** Because `random_split` returns `Subset` objects that inherit the parent dataset's transform, `data_setup.py` uses two separate `ImageFolder` instances (one augmented, one clean) rather than a single shared instance. The augmented instance is used for train; the clean instance for val and test.

## Key Conventions

- Device handling: `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` — model and batch tensors are moved to device.
- Reproducibility: `train.py` seeds `torch`, `numpy`, `random`, and `torch.cuda` at startup with seed 42.
- Loss: `CrossEntropyLoss` (single-label multiclass). Optimizer: Adam.
- Run logs are timestamped JSON: `results/run_<model>_<YYYYMMDD_HHMMSS>.json` — these are the source of truth for the final report.
- No cv2; image loading is handled entirely by torchvision.
- No Jupyter notebooks — everything is `.py` scripts.

## Dataset

`data/EuroSAT_RGB/` — gitignored, must be downloaded separately. Directory has 10 class subfolders (`AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`). Images are 64×64 RGB JPEGs.

## Outputs in `results/`

| File | Produced by |
|---|---|
| `<model>_best.pt` | `train.py` |
| `run_<model>_<timestamp>.json` | `train.py` |
| `<model>_training_curves.png` | `train.py` |
| `<model>_confusion_matrix.png` | `evaluate.py` |
| `<model>_classification_report.txt` | `evaluate.py` |