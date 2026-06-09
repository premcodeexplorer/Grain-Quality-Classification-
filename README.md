# Grain Quality Classification — EfficientNetV2-S

**98.98% Macro F1 | 99.23% Accuracy | 7-Class Cereal Grain Quality Detection**

Part of **MilletSaarthi** — A Multi-Agent Intelligence System for Smart Millet Market Decisions  
RCOEM Nagpur | Group 7 | Guide: Dr. Nisarg Gandhewar | 2025-26

---

## Overview

This repository contains the complete pipeline for Agent 1 of MilletSaarthi — a grain quality classification model trained on the [GrainSet dataset](https://www.nature.com/articles/s41597-023-02660-8) (Nature Scientific Data, 2023).

The model takes a single grain kernel image as input and predicts one of 7 quality categories across wheat, maize, sorghum (Jowar), and rice.

---

## Results

| Metric | Value |
|--------|-------|
| Test F1 (Macro) | **98.98%** |
| Test Accuracy | **99.23%** |
| Best Val F1 | 98.86% |
| Best Val Accuracy | 99.12% |
| Training Time | 127.5 min (RTX 4090) |
| Dataset | GrainSet 40% stratified (141K images) |

### Per-Class Results

| Class | Description | Precision | Recall | F1 |
|-------|-------------|-----------|--------|----|
| NOR | Normal — healthy grain | 100% | 99% | 99% |
| F&S | Fusarium & Shriveled | 98% | 100% | 99% |
| SD | Sprouted | 98% | 100% | 99% |
| MY | Moldy | 97% | 100% | 98% |
| AP | Attacked by Pests | 97% | 100% | 98% |
| BN | Broken | 100% | 99% | 100% |
| OTHER | Heated/Unripe/Impurity | 99% | 99% | 99% |

---

## Dataset

**GrainSet** — Nature Scientific Data, 2023  
Paper: [An annotated grain kernel image database for visual quality inspection](https://www.nature.com/articles/s41597-023-02660-8)  
Authors: Fan, Ding, Fan, Wu, Chu, Pagnucco, Song

| Grain | Total Images | We Used (40%) |
|-------|-------------|---------------|
| Wheat | 200,000 | 80,000 |
| Sorghum (Jowar) | 102,750 | 41,100 |
| Rice | 30,962 | 12,384 |
| Maize | 19,000 | 7,600 |
| **Total** | **352,712** | **141,084** |

Download dataset from [Figshare](https://figshare.com/search?q=GrainSet)

---

## Model Architecture

**EfficientNetV2-S** (pretrained on ImageNet-21k) + Custom Classification Head

```
Input (256×256 RGB)
    ↓
EfficientNetV2-S Backbone (frozen → fine-tuned)
    ↓
GlobalAveragePooling → 1280 features
    ↓
Dropout(0.2) → Linear(512) → BatchNorm → ReLU → Dropout(0.1)
    ↓
Linear(7) → 7 Quality Classes
```

**Parameters:** 20,837,975

---

## Training Pipeline

### Stage 1 — Model Selection (GrainSet-tiny, 6.5K images)
Compared 4 architectures to find the best:

| Model | Test F1 | Decision |
|-------|---------|----------|
| EfficientNetV2-S | 95.57% | ✅ Winner |
| EfficientNet-B4 | 89.41% | Rejected |
| ResNet50-SE | 87.58% | Baseline only |
| ConvNeXt-Small | 88.14% | Too slow |

### Stage 2 — Hyperparameter Tuning (12 Experiments)
Systematic grid search over LR, image size, and dropout:

| Exp | LR | Img Size | Dropout | Val F1 | Test F1 |
|-----|----|----------|---------|--------|---------|
| exp01 | 1e-4 | 224 | 0.2 | 93.29% | 93.79% |
| exp02 | 1e-4 | 224 | 0.4 | 93.06% | 92.66% |
| **exp03** | **1e-4** | **256** | **0.2** | **94.69%** | **97.72% ✅** |
| exp04 | 1e-4 | 256 | 0.4 | 93.81% | 94.38% |
| exp05 | 3e-4 | 224 | 0.2 | 92.47% | 92.15% |
| exp06 | 3e-4 | 224 | 0.4 | 91.83% | 91.42% |
| exp07 | 3e-4 | 256 | 0.2 | 93.54% | 94.61% |
| exp08 | 3e-4 | 256 | 0.4 | 92.96% | 93.18% |
| exp09 | 5e-4 | 224 | 0.2 | 91.23% | 90.87% |
| exp10 | 5e-4 | 224 | 0.4 | 90.61% | 90.13% |
| exp11 | 5e-4 | 256 | 0.2 | 92.38% | 93.04% |
| exp12 | 5e-4 | 256 | 0.4 | 91.74% | 91.99% |

**Best config:** LR=1e-4, Image=256×256, Dropout=0.2

### Stage 3 — Final Training (40% GrainSet, 141K images, RTX 4090)
Best hyperparameters applied to full-scale training:

| Config | Value |
|--------|-------|
| Optimizer | AdamW (lr=1e-4, wd=0.01) |
| Loss | Focal Loss (gamma=2) |
| LR Schedule | Cosine Annealing + Warmup (3 epochs) |
| Batch Size | 64 |
| Epochs | 30 |
| Augmentation | Flip, Rotate, ColorJitter, RandomErasing |

---

## Project Structure

```
grain-quality-classification/
├── src/
│   ├── dataset.py          # Dataset loader — reads GrainSet folder structure
│   ├── model.py            # EfficientNetV2-S + FocalLoss definition
│   ├── train.py            # Stage 1 — model selection experiments
│   ├── hypertune.py        # Stage 2 — 12 hyperparameter experiments
│   ├── train_full.py       # Stage 3 — final training on full data
│   ├── evaluate.py         # Comparison table + plots
│   └── generate_plots.py   # Generate all 6 paper figures
├── results/
│   ├── final_results.json
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── comparison_vs_paper.png
│   ├── per_class_f1.png
│   ├── class_distribution.png
│   └── loss_curve.png
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup and Usage

### 1. Clone and Install
```bash
git clone https://github.com/yourusername/grain-quality-classification.git
cd grain-quality-classification
pip install -r requirements.txt
```

### 2. Download Dataset
Download GrainSet from Figshare and organize as:
```
full_data/
├── maize/maize/train/ and test/
├── rice/train/ and test/
├── sorg/train/ and test/
└── wheat/train/ and test/
```

### 3. Run Model Selection (Stage 1)
```bash
python src/train.py
# Change model_name in CONFIG to try different architectures
```

### 4. Run Hyperparameter Tuning (Stage 2)
```bash
python src/hypertune.py
```

### 5. Run Final Training (Stage 3)
```bash
python src/train_full.py
```

### 6. Generate All Plots
```bash
python src/generate_plots.py
```

---

## Key Design Decisions

**Why EfficientNetV2-S?**  
Modern compact architecture (2022) that outperforms ResNet-50 with fewer parameters. Pretrained on ImageNet-21k for stronger feature extraction.

**Why Focal Loss?**  
NOR (normal grain) comprises ~60% of the dataset. Plain CrossEntropy ignores minority classes. Focal Loss (gamma=2) focuses training on hard examples.

**Why 256×256 input?**  
Hypertuning showed 256×256 consistently outperforms 224×224 by ~4% F1. Higher resolution captures finer grain texture and defect details.

**Why AdamW over SGD?**  
AdamW converges faster and is less sensitive to learning rate choice. The paper used SGD — AdamW is a standard modern improvement.

---

## Citation

If you use this work, please cite the GrainSet dataset:

```bibtex
@article{fan2023annotated,
  title={An annotated grain kernel image database for visual quality inspection},
  author={Fan, Lei and Ding, Yiwen and Fan, Dongdong and Wu, Yong and 
          Chu, Hongxia and Pagnucco, Maurice and Song, Yang},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={778},
  year={2023},
  publisher={Nature Publishing Group}
}
```

---

## Part of MilletSaarthi

This model is Agent 1 of MilletSaarthi — a complete 4-agent system:

| Agent | Task | Technology |
|-------|------|-----------|
| **Agent 1 (this repo)** | **Grain quality classification** | **EfficientNetV2-S** |
| Agent 2 | Price prediction | LSTM + XGBoost |
| Agent 3 | APMC market comparison | Web Scraping + Maps API |
| Agent 4 | Selling decision advisor | Decision Tree |

---

*RCOEM Nagpur | Department of CSE (AIML) | Group 7, Section B | 2025-26*
