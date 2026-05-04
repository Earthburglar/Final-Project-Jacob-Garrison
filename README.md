# Final-Project-Jacob-Garrison
ECEN 5743 — Deep Learning — Final Project

# Land-Use Classification from Satellite Imagery Using CNNs

This project investigates deep learning for land-use classification from
satellite imagery and evaluates how the resulting models hold up under
real-world image degradations such as sensor noise, blur, low light, and
arbitrary flight headings. The motivation is autonomous aerial navigation
in GPS-denied environments — robustness, not just clean-data accuracy, is
a first-class concern.

Three convolutional architectures are trained on the
[EuroSAT](https://github.com/phelber/EuroSAT) dataset of 27,000 Sentinel-2
satellite images and compared across clean accuracy, interpretability,
robustness to five categories of image degradation, and inference
latency on CPU and GPU.

## Models compared

| Model                | Trainable params | Test accuracy |
|----------------------|------------------|---------------|
| Custom CNN (3 blocks)| 129,290          | 90.62%        |
| ResNet18 Frozen      | 5,130            | 91.19%        |
| ResNet18 Fine-tuned  | 11,181,642       | 98.47%        |

## Dataset

EuroSAT RGB — 27,000 RGB images derived from Sentinel-2, evenly distributed
across 10 land-cover classes: AnnualCrop, Forest, HerbaceousVegetation,
Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake.
Each image is 64 × 64 pixels at the native resolution; ground sampling
distance is 10 m per pixel. The dataset is **not** committed to this repo —
download instructions are in the Setup section below.

## Repository structure

```text
Final-Project-Jacob-Garrison/
├── Code/                     
│   ├── data_setup.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── robustness.py
│   └── benchmark.py
├── Proposal/
│   └── Final_Project_Proposal.pdf
├── Final-Presentation/
│   └── Final_Presentation_ECEN5743.pdf
├── Final-Project-Report/
│   └── Final_Project_Report_ECEN5743.pdf
├── README.md                  
├── requirements.txt           # Python dependencies (PyTorch 2.2, torchvision, etc.)
└── .gitignore
```

## Setup

### 1. Python environment

```bash
git clone https://github.com/Earthburglar/Final-Project-Jacob-Garrison.git
cd Final-Project-Jacob-Garrison
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for training but not required; all scripts
fall back to CPU automatically.

### 2. Download the EuroSAT dataset

```bash
# Recommended: torchvision's built-in download
python -c "from torchvision.datasets import EuroSAT; EuroSAT(root='./data', download=True)"
```

This produces `./data/EuroSAT_RGB/` containing one subfolder per class. The
data loader (`Code/data_setup.py`) expects this exact path.

## How to run the experiments

The scripts are designed to be run in this order. Each builds on outputs from
the earlier ones and saves checkpoints, JSON logs, CSVs, and figures to a
local `results/` directory (created automatically; gitignored).

### Step 1 — Train the three models (writes checkpoints + per-epoch JSON logs)

```bash
python Code/train.py --model baseline
python Code/train.py --model resnet_frozen
python Code/train.py --model resnet_finetuned
```

Each training run takes 15–40 minutes on a mid-range GPU. Total wall-clock
for all three runs: about 90 minutes.

### Step 2 — Evaluate clean test accuracy and generate confusion matrices

```bash
python Code/evaluate.py --model baseline
python Code/evaluate.py --model resnet_frozen
python Code/evaluate.py --model resnet_finetuned
```

Writes per-class precision/recall/F1 to `results/<model>_metrics.json` and
the confusion matrix PNG to `results/<model>_confusion_matrix.png`.

### Step 3 — Generate Grad-CAM heatmaps

```bash
python Code/visualize.py
```

Produces `results/gradcam_baseline_vs_resnet_frozen_vs_resnet_finetuned_224x224.png`,
a side-by-side heatmap comparison for one example image per class across
all three models.

### Step 4 — Robustness evaluation (five degradations × six severity levels)

```bash
python Code/robustness.py
```

Re-evaluates each trained model against test images perturbed by Gaussian
noise, Gaussian blur, resolution loss, rotation, and brightness scaling.
Writes one PNG per degradation plus a combined `robustness_summary.csv`.

### Step 5 — Inference benchmarking (CPU + GPU latency)

```bash
python Code/benchmark.py
```

Measures inference latency at batch size 1 with proper warmup and CUDA
synchronization. Writes `results/benchmark_summary.csv`.

## What each script does

| Script             | Purpose |
|--------------------|---------|
| `data_setup.py`    | Builds the train / val / test loaders. Constructs two `ImageFolder` instances over the same data — one with augmented transforms (random horizontal flip + ±15° rotation), one with clean transforms — and applies the same stratified 70 / 15 / 15 split to both. Fixed seed (42) ensures reproducibility. |
| `models.py`        | Defines the three architectures: the custom `BaselineCNN` class (three Conv-BN-ReLU-Pool blocks + adaptive pooling + FC classifier) and a `build_resnet()` factory for ResNet18 with optional backbone freezing. A `get_model(name)` dispatcher selects between them by string. |
| `train.py`         | Unified training loop. Cross-entropy loss, Adam optimizer (lr = 1e-3 for baseline / frozen, 1e-4 for fine-tuned to avoid disrupting pretrained features), 25 epochs, batch size 64. Saves best-validation-accuracy checkpoint and per-epoch metrics to JSON. |
| `evaluate.py`      | Loads a saved checkpoint, runs the test set, and computes top-1 accuracy plus per-class precision / recall / F1 via scikit-learn. Saves a row-normalized confusion matrix PNG. |
| `visualize.py`     | Generates Grad-CAM heatmaps using the `pytorch-grad-cam` library. Hooks the last convolutional layer of each model (`conv_block3` for baseline, `layer4[-1]` for ResNet variants) and produces a class-grid figure for visual comparison. |
| `robustness.py`    | Five test-time degradations (Gaussian noise, Gaussian blur, resolution loss, rotation, brightness) at multiple severity levels. Each is implemented as a torchvision transform applied to the clean test set; no model is retrained. |
| `benchmark.py`     | Inference latency and throughput benchmarking. Uses 20 warmup passes and explicit `torch.cuda.synchronize()` around each timed pass to measure kernel execution rather than asynchronous queue submission. Reports mean ± std per pass and total throughput. |

## Key results

- **Clean accuracy:** Fine-tuned ResNet wins decisively at 98.47%; the small
  custom CNN essentially ties frozen transfer learning (90.6% vs 91.2%).
- **Resolution matters:** ResNet's five spatial-reduction stages mean its
  deepest feature map is `input_size / 32`. At 64 × 64 native input that's
  only 2 × 2, which is why all experiments use 224 × 224 upsampled input.
- **Clean accuracy ≠ robustness:** the fine-tuned ResNet is the most accurate
  model on clean data but the most fragile under severe blur and resolution
  loss, where the frozen ResNet's generic ImageNet features generalize
  better.
- **Augmentation defines the robustness ceiling:** the rotation zigzag (peaks
  at 0° / 90° / 180° / 270°, drops at 45° / 135°) and the noise-induced
  collapse (>90% → ~50% at σ=0.05) trace directly to what was absent from
  the training augmentation pipeline.
- **Inference speed:** the custom CNN is ~3.8 × faster on GPU and ~1.5 × faster
  on CPU than either ResNet — a meaningful tradeoff for edge deployment.

See the final report (`Final-Project-Report/`) for full discussion.

## Reproducibility notes

- All splits use a fixed random seed of 42.
- All training runs use a fixed seed for `torch`, `numpy`, and Python's
  `random`.
- Per-epoch loss and accuracy are written to JSON in `results/` for every
  training run, so the published numbers can be cross-checked.
- Hardware used for the published numbers: NVIDIA GPU (workstation),
  PyTorch 2.2, torchvision 0.17, CUDA 12.

## Reference

Helber, P., Bischke, B., Dengel, A., and Borth, D. (2019). "EuroSAT: A Novel
Dataset and Deep Learning Benchmark for Land Use and Land Cover
Classification." *IEEE Journal of Selected Topics in Applied Earth
Observations and Remote Sensing*, 12(7), 2217–2226.

Author: Jake Garrison · ECEN 5743 — Deep Learning · Spring 2026
