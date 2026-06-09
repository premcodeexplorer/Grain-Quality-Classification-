# Grain Quality Classification — EfficientNetV2-S

**98.98% Macro F1 | 99.23% Accuracy | 7-Class Cereal Grain Quality Detection**

---

## Overview

A deep learning pipeline for cereal grain quality inspection trained on the [GrainSet dataset](https://www.nature.com/articles/s41597-023-02660-8) (Nature Scientific Data, 2023).

The model takes a single grain kernel image as input and predicts one of 7 quality categories across wheat, maize, sorghum, and rice.

**Input:** Single grain kernel image (PNG, RGB)  
**Output:** Quality category (NOR / F&S / SD / MY / AP / BN / OTHER)

---

## Results

| Metric | Value |
|--------|-------|
| Test F1 (Macro) | **98.98%** |
| Test Accuracy | **99.23%** |
| Best Val F1 | 98.86% |
| Best Val Accuracy | 99.12% |
| Training Time | 127.5 min |
| Dataset | GrainSet 40% stratified (141K images) |
| Hardware | NVIDIA RTX 4090 (24GB VRAM) |

### Per-Class Results

| Class | Description | Precision | Recall | F1 | Support |
|-------|-------------|-----------|--------|----|---------|
| NOR | Normal — healthy grain | 100% | 99% | 99% | 8,400 |
| F&S | Fusarium & Shriveled | 98% | 100% | 99% | 759 |
| SD | Sprouted | 98% | 100% | 99% | 980 |
| MY | Moldy | 97% | 100% | 98% | 980 |
| AP | Attacked by Pests | 97% | 100% | 98% | 980 |
| BN | Broken | 100% | 99% | 100% | 1,039 |
| OTHER | Heated / Unripe / Impurity | 99% | 99% | 99% | 969 |
| | **Macro Average** | **98%** | **100%** | **99%** | **14,107** |

---

## Dataset

**GrainSet** — Nature Scientific Data, 2023  
Paper: [An annotated grain kernel image database for visual quality inspection](https://www.nature.com/articles/s41597-023-02660-8)  
Authors: Fan, Ding, Fan, Wu, Chu, Pagnucco, Song — UNSW Sydney  
DOI: https://doi.org/10.1038/s41597-023-02660-8

| Grain | Total Images | We Used (40%) |
|-------|-------------|---------------|
| Wheat | 200,000 | 80,000 |
| Sorghum | 102,750 | 41,100 |
| Rice | 30,962 | 12,384 |
| Maize | 19,000 | 7,600 |
| **Total** | **352,712** | **141,084** |

Download from Figshare:
- Sorghum: https://doi.org/10.6084/m9.figshare.22988981.v2
- Wheat: https://doi.org/10.6084/m9.figshare.22992317.v2
- Rice: https://doi.org/10.6084/m9.figshare.22987292.v3
- Maize: https://doi.org/10.6084/m9.figshare.22987562.v2
- Tiny preview (1GB): https://doi.org/10.6084/m9.figshare.22989029.v1

---

## Model Architecture

**EfficientNetV2-S** pretrained on ImageNet-21k + Custom Classification Head

```
Input Image (256 x 256 x 3)
        ↓
EfficientNetV2-S Backbone
        ↓
GlobalAveragePooling → 1280 features
        ↓
Dropout(0.2)
        ↓
Linear(1280 → 512) → BatchNorm → ReLU
        ↓
Dropout(0.1)
        ↓
Linear(512 → 7)
        ↓
7 Quality Classes
```

**Total Parameters:** 20,837,975

---

## Training Pipeline

### Stage 1 — Model Selection (GrainSet-tiny, 6.5K images)

Compared 4 architectures under identical training conditions:

| Model | Test F1 | Test Acc | Time |
|-------|---------|----------|------|
| **EfficientNetV2-S** | **95.57%** | **96.18%** | 76.9 min |
| EfficientNet-B4 | 89.41% | 90.84% | 44.2 min |
| ResNet50-SE | 87.58% | 89.31% | 35.8 min |
| ConvNeXt-Small | 88.14% | 89.76% | 91.3 min |

**Winner: EfficientNetV2-S**

---

### Stage 2 — Hyperparameter Tuning (12 Experiments)

Systematic grid search over learning rate, image size, and dropout:

| Exp | LR | Image Size | Dropout | Val F1 | Test F1 |
|-----|----|------------|---------|--------|---------|
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

**Best config:** LR = 1e-4 | Image Size = 256×256 | Dropout = 0.2

Key finding: Image size 256 consistently outperforms 224 by ~4% F1. LR 1e-4 is most stable across all combinations.

---

### Stage 3 — Final Training (40% GrainSet, 141K images)

| Parameter | Value |
|-----------|-------|
| Model | EfficientNetV2-S |
| Optimizer | AdamW (lr=1e-4, weight_decay=0.01) |
| Loss | Focal Loss (gamma=2) |
| LR Schedule | Cosine Annealing + Warmup (3 epochs) |
| Batch Size | 64 |
| Epochs | 30 |
| Image Size | 256 × 256 |
| Augmentation | RandomFlip, RandomRotation(45), ColorJitter, RandomErasing |

Training progress:

| Epoch | Val F1 | Note |
|-------|--------|------|
| 1 | 81.7% | Warmup phase |
| 3 | 94.3% | Rapid learning |
| 8 | 96.5% | Strong convergence |
| 14 | 97.5% | New best |
| 19 | 98.3% | New best |
| 26 | 98.8% | New best |
| 29 | 98.9% | Best model saved |

---

## Project Structure

```
Grain-Quality-Classification/
├── src/
│   ├── dataset.py          # GrainSet dataset loader + stratified splits
│   ├── model.py            # EfficientNetV2-S + Focal Loss definition
│   ├── train.py            # Stage 1 — model selection experiments
│   ├── hypertune.py        # Stage 2 — 12 hyperparameter experiments
│   ├── train_full.py       # Stage 3 — final training on full data
│   ├── evaluate.py         # Comparison table + evaluation metrics
│   └── generate_plots.py   # Generate training curves + paper figures
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

## Setup

### 1. Clone and Install
```bash
git clone https://github.com/premcodeexplorer/Grain-Quality-Classification-.git
cd Grain-Quality-Classification-
pip install -r requirements.txt
```

### 2. Download Dataset
Download GrainSet from Figshare links above and organize as:
```
full_data/
├── maize/
│   └── maize/
│       ├── train/
│       │   ├── 0_NOR/
│       │   ├── 1_F&S/
│       │   └── ...
│       └── test/
├── rice/
│   ├── train/
│   └── test/
├── sorg/
│   ├── train/
│   └── test/
└── wheat/
    ├── train/
    └── test/
```

### 3. Run Stage 1 — Model Selection
```bash
python src/train.py
# Edit CONFIG in train.py to switch between models
```

### 4. Run Stage 2 — Hyperparameter Tuning
```bash
python src/hypertune.py
# Runs all 12 combinations automatically
# Results saved to results/hypertune_results/
```

### 5. Run Stage 3 — Final Training
```bash
python src/train_full.py
# Results saved to results/final_results/
```

### 6. Generate Plots
```bash
python src/generate_plots.py
# Generates all 6 figures
```

---

## Key Design Decisions

**Why EfficientNetV2-S?**
Newer architecture (2022) that outperforms ResNet-50 with fewer parameters. Pretrained on ImageNet-21k provides stronger feature initialization.

**Why Focal Loss?**
NOR (normal grain) is ~60% of the dataset creating class imbalance. Focal Loss (gamma=2) down-weights easy examples and focuses training on hard minority classes.

**Why 256×256 input?**
Hypertuning showed 256×256 consistently outperforms 224×224. Higher resolution captures finer grain texture and defect details critical for quality inspection.

**Why AdamW?**
Faster convergence and less sensitive to learning rate than SGD. Built-in weight decay decoupling provides better regularization.

---

## Citation

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