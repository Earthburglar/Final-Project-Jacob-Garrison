# Final-Project-Jacob-Garrison
ECEN 5743 Final Project

# Land-Use Classification from Satellite Imagery Using Convolutional Neural Networks

## Overview
This project investigates deep learning for land-use classification from satellite imagery using the EuroSAT RGB dataset. The goal is to classify satellite images into land-use and land-cover categories such as residential, industrial, agricultural, forest, and other terrain types.

The project compares two approaches:
1. A small custom convolutional neural network (CNN) trained from scratch
2. A transfer learning model based on a pretrained ResNet architecture

The purpose of the project is to evaluate how a simple custom CNN compares to a stronger pretrained model on a clean image classification task.

## Dataset
This project uses the **EuroSAT RGB** dataset, a publicly available labeled satellite image dataset derived from Sentinel-2 imagery. The dataset contains 27,000 images across 10 classes.

Example classes include:
- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

## Project Goals
- Build a complete image classification pipeline in PyTorch
- Train and evaluate a custom CNN baseline
- Train and evaluate a pretrained ResNet transfer learning model
- Compare both models using standard classification metrics
- Analyze class-level performance with a confusion matrix

## Repository Structure
```text
.
├── Code/
├── Proposal/
├── results/
├── reports/
├── README.md
├── requirements.txt
└── .gitignore
