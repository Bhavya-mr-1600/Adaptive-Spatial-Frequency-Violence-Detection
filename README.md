# $\text{CF}^2$-EfficientNetV2-S: Violence & Abnormal Event Detection
### Official Reviewer Validation & Reproducibility Package

This repository provides the complete implementation, feature extraction pipeline, and evaluation benchmarks for the paper **"CF²-EfficientNetV2-S for Violence/Abnormal Event Detection"** submitted to *IJIES*.

---

## 📋 Table of Contents
1. [Overview](#-overview)
2. [Repository Structure](#-repository-structure)
3. [Environment Setup](#-environment-setup)
4. [Dataset Setup (1.45 GB)](#-dataset-setup-145-gb)
5. [Running the Code](#-running-the-code)
6. [Expected Outputs & Verification](#-expected-outputs--verification)
7. [Ablation & Model Specifications](#-model-specifications)

---

## 🔬 Overview

The pipeline implements an end-to-end framework for violent event recognition:
- **Backbone:** EfficientNetV2-S pretrained on ImageNet-1K.
- **Frequency Enhancer:** Fixed Sobel-kernel edge extraction to emphasize high-frequency motion artifacts.
- **Dual Attention Block:** Spatial and channel attention mechanisms.
- **Adaptive Fusion Gate & Transformer:** Gated multi-modal feature fusion refined through multi-head self-attention.
- **Loss Formulation:** Joint Cross-Entropy + Supervised Contrastive Loss ($\mathcal{L} = \mathcal{L}_{CE} + 0.3 \cdot \mathcal{L}_{Contrastive}$).
- **Validation:** Independent test-set evaluation (Softmax, SVM, KNN, Logistic Regression, AdaBoost, Naive Bayes) alongside **5-Fold Stratified Cross-Validation**.

---

## 📁 Repository Structure

```text
.
├── dataset/                     # Dataset root directory (see Setup below)
│   ├── train_data/
│   │   ├── Normal/
│   │   └── Violence/
│   └── test_data/
│       ├── Normal/
│       └── Violence/
├── run_verification.py          # Main execution script
├── requirements.txt             # Python dependencies
└── README.md                    # Reviewer documentation