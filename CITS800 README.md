# 🔬 Breast Cancer Diagnostics — ML Pipeline

A complete end-to-end machine learning pipeline for classifying breast tumors as **Malignant** or **Benign** using the [Breast Cancer Wisconsin (Diagnostic) dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) from UCI / Kaggle.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline Stages](#pipeline-stages)
- [Models](#models)
- [Results](#results)
- [Outputs](#outputs)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## Overview

This project trains and evaluates five classical ML classifiers on breast cancer diagnostic data. It covers the full ML workflow: exploratory data analysis, preprocessing, model training, cross-validation, evaluation visualizations, and inference on new samples.

---

## Dataset

| Property     | Value                              |
|--------------|------------------------------------|
| Source       | `sklearn.datasets.load_breast_cancer` |
| Samples      | 569                                |
| Features     | 30 numeric (cell nucleus measurements) |
| Classes      | Malignant (212) · Benign (357)     |
| Missing data | None                               |

Features include measurements such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension — each computed as mean, standard error, and worst value.

---

## Pipeline Stages

```
1. Load Data        → Breast Cancer Wisconsin dataset
2. EDA              → Class distribution, feature correlations, distributions
3. Preprocessing    → Train/test split (80/20), StandardScaler normalization
4. Training         → 5 models × 5-fold stratified cross-validation
5. Evaluation       → Accuracy, Precision, Recall, F1, ROC AUC
6. Visualization    → 5 plots (comparison, ROC, confusion matrices, feature importance, CV boxplots)
7. Inference        → Predict class + probabilities for a new sample
```

---

## Models

| Model                | Library                        |
|----------------------|--------------------------------|
| Logistic Regression  | `sklearn.linear_model`         |
| Random Forest        | `sklearn.ensemble`             |
| Gradient Boosting    | `sklearn.ensemble`             |
| Support Vector Machine (SVM) | `sklearn.svm`        |
| K-Nearest Neighbors (KNN) | `sklearn.neighbors`      |

All models are trained on standardized features and evaluated using **5-fold stratified cross-validation**.

---

## Results

> Metrics on the held-out test set (20% of data). Best model is selected by **ROC AUC**.

| Model               | Accuracy | Precision | Recall | F1    | ROC AUC |
|---------------------|----------|-----------|--------|-------|---------|
| Logistic Regression | —        | —         | —      | —     | —       |
| Random Forest       | —        | —         | —      | —     | —       |
| Gradient Boosting   | —        | —         | —      | —     | —       |
| SVM                 | —        | —         | —      | —     | —       |
| KNN                 | —        | —         | —      | —     | —       |

*Run the script to populate the results table above with actual values.*

---

## Outputs

The script saves the following plots to the working directory:

| File | Description |
|------|-------------|
| `eda.png` | Class distribution, top correlated features, top feature histogram |
| `model_comparison.png` | Bar chart comparing all metrics across models |
| `roc_curves.png` | ROC curves for all models with AUC scores |
| `confusion_matrices.png` | Confusion matrices for all models |
| `feature_importance.png` | Top 15 feature importances from Random Forest |
| `cv_boxplots.png` | 5-fold CV accuracy distribution per model |

---

## Installation

**Prerequisites:** Python 3.8+

```bash
# Clone the repository
git clone https://github.com/your-username/breast-cancer-ml-pipeline.git
cd breast-cancer-ml-pipeline

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`**
```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

## Usage

```bash
python breast_cancer_pipeline.py
```

The script will:
1. Print a summary table of all model metrics to the console
2. Save 5 evaluation plots as `.png` files
3. Print a detailed classification report for the best model
4. Run a sample prediction and display the predicted class with probabilities

---

## Project Structure

```
breast-cancer-ml-pipeline/
│
├── breast_cancer_pipeline.py   # Main pipeline script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
└── outputs/                    # Generated after running the script
    ├── eda.png
    ├── model_comparison.png
    ├── roc_curves.png
    ├── confusion_matrices.png
    ├── feature_importance.png
    └── cv_boxplots.png
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

- Dataset: [UCI Machine Learning Repository — Breast Cancer Wisconsin](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- Built with [scikit-learn](https://scikit-learn.org/), [matplotlib](https://matplotlib.org/), and [seaborn](https://seaborn.pydata.org/)
