# ============================================================
# Breast Cancer Diagnostics — Complete ML Pipeline
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("BREAST CANCER DIAGNOSTICS — ML PIPELINE")
print("=" * 60)

# Load dataset
cancer = load_breast_cancer()

# Create DataFrame
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target, name="target")

# Combine into single DataFrame
df = X.copy()
df["target"] = y

print("\nDataset Loaded Successfully")
print(f"Samples   : {df.shape[0]}")
print(f"Features  : {df.shape[1] - 1}")
print(f"Missing Values : {df.isnull().sum().sum()}")

# ============================================================
# 2. EDA
# ============================================================

print("\nPerforming Exploratory Data Analysis...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Breast Cancer Dataset Analysis", fontsize=14)

# Class Distribution
counts = df["target"].value_counts().sort_index()

axes[0].bar(
    ["Malignant", "Benign"],
    counts.values,
    color=["red", "green"]
)

axes[0].set_title("Class Distribution")
axes[0].set_ylabel("Count")

# Correlation
corr = df.corr(numeric_only=True)["target"] \
    .drop("target") \
    .abs() \
    .sort_values(ascending=False) \
    .head(10)

axes[1].barh(corr.index[::-1], corr.values[::-1])
axes[1].set_title("Top Correlated Features")
axes[1].set_xlabel("Correlation")

# Histogram
top_feature = corr.index[0]

for label, color, name in [
    (0, "red", "Malignant"),
    (1, "green", "Benign")
]:
    subset = df[df["target"] == label][top_feature]
    axes[2].hist(subset, bins=25, alpha=0.6,
                 color=color, label=name)

axes[2].set_title(f"Distribution of {top_feature}")
axes[2].legend()

plt.tight_layout()
plt.savefig("eda.png")
plt.show()

print("EDA plot saved as eda.png")

# ============================================================
# 3. PREPROCESSING
# ============================================================

print("\nPreprocessing Data...")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training Samples : {X_train_scaled.shape[0]}")
print(f"Testing Samples  : {X_test_scaled.shape[0]}")

# ============================================================
# 4. MODEL TRAINING
# ============================================================

print("\nTraining Models...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier()
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

print("\nModel Performance")
print("-" * 80)

for name, model in models.items():

    # Train model
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Cross Validation
    cv_scores = cross_val_score(
        model,
        X_train_scaled,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "fpr": fpr,
        "tpr": tpr,
        "cm": confusion_matrix(y_test, y_pred),
        "model": model,
        "y_pred": y_pred
    }

    print(f"{name:<22} Accuracy: {accuracy:.4f} | AUC: {auc:.4f}")

# ============================================================
# 5. ROC CURVES
# ============================================================

print("\nGenerating ROC Curves...")

plt.figure(figsize=(8, 6))

for name, result in results.items():
    plt.plot(
        result["fpr"],
        result["tpr"],
        label=f"{name} (AUC={result['roc_auc']:.3f})"
    )

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.grid(True)

plt.savefig("roc_curves.png")
plt.show()

print("ROC curve saved as roc_curves.png")

# ============================================================
# 6. CONFUSION MATRICES
# ============================================================

fig, axes = plt.subplots(1, len(models), figsize=(18, 4))
fig.suptitle("Confusion Matrices")

for ax, (name, result) in zip(axes, results.items()):

    sns.heatmap(
        result["cm"],
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax,
        cbar=False,
        xticklabels=["Malignant", "Benign"],
        yticklabels=["Malignant", "Benign"]
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices.png")
plt.show()

print("Confusion matrices saved")

# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================

rf_model = results["Random Forest"]["model"]

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=cancer.feature_names
)

feature_importance = feature_importance.sort_values().tail(15)

plt.figure(figsize=(8, 6))
feature_importance.plot(kind='barh')

plt.title("Top 15 Important Features")
plt.xlabel("Importance")
plt.grid(True)

plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

print("Feature importance graph saved")

# ============================================================
# 8. BEST MODEL
# ============================================================

best_model_name = max(results, key=lambda x: results[x]["roc_auc"])
best_result = results[best_model_name]

print("\n" + "=" * 60)
print(f"BEST MODEL : {best_model_name}")
print("=" * 60)

print(f"Accuracy  : {best_result['accuracy']:.4f}")
print(f"Precision : {best_result['precision']:.4f}")
print(f"Recall    : {best_result['recall']:.4f}")
print(f"F1 Score  : {best_result['f1']:.4f}")
print(f"ROC AUC   : {best_result['roc_auc']:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        best_result["y_pred"],
        target_names=["Malignant", "Benign"]
    )
)

# ============================================================
# 9. SAMPLE PREDICTION
# ============================================================

print("\nSample Prediction")

sample = X_test_scaled[0].reshape(1, -1)

prediction = best_result["model"].predict(sample)[0]
probability = best_result["model"].predict_proba(sample)[0]

print(
    f"Predicted Class : {cancer.target_names[prediction].upper()}"
)

print(
    f"Probability -> Malignant: {probability[0]:.3f}, "
    f"Benign: {probability[1]:.3f}"
)

print(
    f"Actual Class : "
    f"{cancer.target_names[y_test.iloc[0]].upper()}"
)

print("\nPipeline Completed Successfully")
