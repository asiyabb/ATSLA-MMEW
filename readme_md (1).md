# ATSLA-DFER Baseline on MMEW Dataset

This repository contains the implementation and experimental setup for establishing the official macro-expression recognition baseline using the **ATSLA-DFER** (Attention-Driven Temporal-Spatial Learning Architecture for Dynamic Facial Expression Recognition) on the **MMEW** (Micro-and-Macro Expression Warehouse) dataset.

This baseline serves as the official benchmark for comparing against the proposed micro-expression-guided transformer framework in our thesis research.

---

## 📌 Methodological Overview

* **Dataset**: MMEW Macro-Expression sequences (180 video sequences, 6 emotion classes).
* **Architecture**: Dual-backbone setup featuring:
  * **IR50** (ResNet-like deep spatial feature extraction)
  * **MobileFaceNet** (Lightweight facial landmark/feature tracking)
  * **GLAM** (Global-Local Attention Module)
  * **Patch-Based & Unique Attention** mechanisms.
* **Pre-trained Weights**: Cross-dataset pre-trained weights (`Pretrain_DFEW_MAFW.pth`) transferred from large-scale dynamic datasets (DFEW & MAFW).
* **Evaluation Protocol**: Strict Subject-Independent 5-Fold Cross-Validation.
* **Loss Function**: Class-Weighted Cross-Entropy combined with temporal consistency and distillation losses to eliminate mode collapse and handle severe class imbalance.

---

## 📁 Repository Structure

```
ATSLA_MMEW/
├── dataset.py            # MMEW video sequence dataset parser & subject tracking
├── model.py              # ATSLA-DFER model architecture & combined emotion loss
├── helper.py             # Pre-trained checkpoint loader utility
├── main.py               # 5-fold cross-validation training execution script
├── visualizer.py         # Evaluation report generator & confusion matrix plotter
├── Pretrain_DFEW_MAFW.pth # Pre-trained backbone weights
└── README.md             # Project documentation
```

---

## 🛠️ Environment Setup

### Prerequisites
* Python 3.10+
* CUDA-capable GPU (Recommended: 8GB+ VRAM)

### Installation
Ensure your active Python environment has PyTorch, Torchvision, Scikit-Learn, and Matplotlib installed:

```bash
pip install torch torchvision scikit-learn matplotlib pillow numpy
```

---

## 🚀 Running the Baseline Experiment

### 1. Training (Subject-Independent 5-Fold Cross-Validation)

Execute `main.py` to train the ATSLA model across all 5 subject-independent folds:

```bash
python main.py --gpu 0 --epochs 20 --batch-size 16 --lr 5e-5
```

#### Key Command-Line Arguments:
* `--epochs`: Number of epochs per fold (Default: `20`).
* `--batch-size`: Mini-batch size (Default: `16`).
* `--lr`: Learning rate (Default: `5e-5`).
* `--gpu`: ID of GPU device to use (Default: `'0'`).
* `--pretrain-path`: Path to `Pretrain_DFEW_MAFW.pth` (Default: `'Pretrain_DFEW_MAFW.pth'`).

---

### 2. Evaluation and Visualization

Once training finishes, run `visualizer.py` to evaluate the saved checkpoints, output detailed classification metrics, and save the aggregated normalized confusion matrix (`cm.png`):

```bash
python visualizer.py
```

---

## 📊 Baseline Performance Benchmark

Under subject-independent 5-fold cross-validation on the 180 MMEW macro-expression sequences, the ATSLA baseline achieves the following performance:

* **Weighted Average Recognition (WAR / Accuracy)**: `49.00%`
* **Unweighted Average Recognition (UAR / Recall)**: `36.00%`

### Confusion Matrix Output
The aggregated normalized confusion matrix (`cm.png`) verifies that the model successfully predicts across diverse emotion categories (e.g., happiness, sadness, surprise) while establishing an accurate baseline for comparison.

---

## 📖 Citation & References

* **ATSLA Model**: Attention-Driven Temporal-Spatial Learning Architecture for Dynamic Facial Expression Recognition.
* **MMEW Dataset**: Ben, X., et al. "Video-based facial micro-expression analysis: A survey of datasets, features and algorithms." *IEEE TPAMI*, 2021.