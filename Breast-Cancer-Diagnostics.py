# ============================================================
# Breast Cancer Diagnostics — Complete ML Pipeline
# Dataset: Breast Cancer Wisconsin (Diagnostic) — UCI / Kaggle
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────
print("=" * 60)
print("  BREAST CANCER DIAGNOSTICS — ML PIPELINE")
print("=" * 60)

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target          # 0 = Malignant, 1 = Benign

print(f"\n[1] Dataset loaded")
print(f"    Samples   : {df.shape[0]}")
print(f"    Features  : {df.shape[1] - 1}")
print(f"    Malignant : {(df.target == 0).sum()} ({(df.target==0).mean()*100:.1f}%)")
print(f"    Benign    : {(df.target == 1).sum()} ({(df.target==1).mean()*100:.1f}%)")
print(f"    Missing   : {df.isnull().sum().sum()}")


# ──────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ──────────────────────────────────────────────
print("\n[2] Exploratory Data Analysis")

# Class distribution
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("EDA — Breast Cancer Dataset", fontsize=14, fontweight="bold")

# Class bar
counts = df["target"].value_counts().sort_index()
axes[0].bar(["Malignant", "Benign"], counts.values, color=["#E24B4A", "#1D9E75"], width=0.5)
axes[0].set_title("Class Distribution")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 3, str(v), ha="center", fontweight="bold")

# Feature correlations with target (top 10)
corr = df.corr()["target"].drop("target").abs().sort_values(ascending=False).head(10)
axes[1].barh(corr.index[::-1], corr.values[::-1], color="#378ADD")
axes[1].set_title("Top 10 Features Correlated with Target")
axes[1].set_xlabel("|Pearson correlation|")

# Distribution of top feature by class
top_feat = corr.index[0]
for label, color, name in [(0, "#E24B4A", "Malignant"), (1, "#1D9E75", "Benign")]:
    subset = df[df["target"] == label][top_feat]
    axes[2].hist(subset, bins=25, alpha=0.6, color=color, label=name)
axes[2].set_title(f"Distribution: {top_feat}")
axes[2].set_xlabel("Value")
axes[2].legend()

plt.tight_layout()
plt.savefig("eda.png", dpi=150, bbox_inches="tight")
plt.show()
print("    Saved → eda.png")


# ──────────────────────────────────────────────
# 3. PREPROCESSING
# ──────────────────────────────────────────────
print("\n[3] Preprocessing")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"    Train : {X_train_s.shape[0]} samples")
print(f"    Test  : {X_test_s.shape[0]} samples")


# ──────────────────────────────────────────────
# 4. MODEL TRAINING & CROSS-VALIDATION
# ──────────────────────────────────────────────
print("\n[4] Training models with 5-fold cross-validation")

models = {
    "Logistic Regression" : LogisticRegression(max_iter=5000, random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting"   : GradientBoostingClassifier(random_state=42),
    "SVM"                 : SVC(probability=True, random_state=42),
    "KNN"                 : KNeighborsClassifier(),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print(f"\n    {'Model':<22} {'Acc':>7} {'F1':>7} {'Prec':>7} {'Recall':>7} {'AUC':>7} {'CV±std':>12}")
print("    " + "-" * 70)

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="accuracy")
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    r = {
        "accuracy"  : accuracy_score(y_test, y_pred),
        "precision" : precision_score(y_test, y_pred),
        "recall"    : recall_score(y_test, y_pred),
        "f1"        : f1_score(y_test, y_pred),
        "roc_auc"   : roc_auc_score(y_test, y_prob),
        "cv_mean"   : cv_scores.mean(),
        "cv_std"    : cv_scores.std(),
        "fpr"       : fpr,
        "tpr"       : tpr,
        "cm"        : confusion_matrix(y_test, y_pred),
        "model"     : model,
        "y_pred"    : y_pred,
    }
    results[name] = r

    print(f"    {name:<22} {r['accuracy']:>6.2%} {r['f1']:>6.2%} "
          f"{r['precision']:>6.2%} {r['recall']:>6.2%} "
          f"{r['roc_auc']:>6.2%} {r['cv_mean']:.2%}±{r['cv_std']:.2%}")


# ──────────────────────────────────────────────
# 5. EVALUATION PLOTS
# ──────────────────────────────────────────────
print("\n[5] Generating evaluation plots")

colors = {
    "Logistic Regression" : "#378ADD",
    "Random Forest"       : "#1D9E75",
    "Gradient Boosting"   : "#EF9F27",
    "SVM"                 : "#7F77DD",
    "KNN"                 : "#D4537E",
}

# ── 5a. Model comparison bar chart ──────────────
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
x = np.arange(len(models))
width = 0.15

fig, ax = plt.subplots(figsize=(14, 5))
for i, (m, label) in enumerate(zip(metrics, metric_labels)):
    vals = [results[n][m] for n in models]
    ax.bar(x + i * width, vals, width, label=label, alpha=0.85)

ax.set_xticks(x + width * 2)
ax.set_xticklabels(list(models.keys()), rotation=10, ha="right")
ax.set_ylim(0.88, 1.02)
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison", fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("    Saved → model_comparison.png")

# ── 5b. ROC curves ───────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for name, r in results.items():
    ax.plot(r["fpr"], r["tpr"], color=colors[name],
            label=f"{name} (AUC={r['roc_auc']:.3f})", lw=2)
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models", fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("    Saved → roc_curves.png")

# ── 5c. Confusion matrices ────────────────────────
fig, axes = plt.subplots(1, len(models), figsize=(18, 4))
fig.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")
class_names = ["Malignant", "Benign"]

for ax, (name, r) in zip(axes, results.items()):
    cm = r["cm"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                cbar=False, linewidths=0.5)
    ax.set_title(f"{name}\n({r['accuracy']:.2%})", fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
print("    Saved → confusion_matrices.png")

# ── 5d. Feature importance (Random Forest) ─────────
rf_model = results["Random Forest"]["model"]
fi = pd.Series(rf_model.feature_importances_, index=data.feature_names)
fi = fi.sort_values(ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(8, 6))
fi.plot(kind="barh", ax=ax, color="#378ADD", alpha=0.85)
ax.set_title("Feature Importance — Random Forest (Top 15)", fontweight="bold")
ax.set_xlabel("Importance")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("    Saved → feature_importance.png")

# ── 5e. Cross-validation box plots ───────────────
cv_data = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="accuracy")
    cv_data[name] = scores

fig, ax = plt.subplots(figsize=(10, 5))
bp = ax.boxplot(cv_data.values(), patch_artist=True,
                labels=[n.replace(" ", "\n") for n in cv_data.keys()])
for patch, color in zip(bp["boxes"], colors.values()):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel("CV Accuracy")
ax.set_title("5-Fold Cross-Validation Accuracy Distribution", fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("cv_boxplots.png", dpi=150, bbox_inches="tight")
plt.show()
print("    Saved → cv_boxplots.png")


# ──────────────────────────────────────────────
# 6. BEST MODEL — DETAILED REPORT
# ──────────────────────────────────────────────
best_name = max(results, key=lambda n: results[n]["roc_auc"])
best = results[best_name]

print("\n" + "=" * 60)
print(f"  BEST MODEL: {best_name}")
print("=" * 60)
print(f"  Accuracy  : {best['accuracy']:.4f}")
print(f"  Precision : {best['precision']:.4f}")
print(f"  Recall    : {best['recall']:.4f}")
print(f"  F1 Score  : {best['f1']:.4f}")
print(f"  ROC AUC   : {best['roc_auc']:.4f}")
print(f"  CV Mean   : {best['cv_mean']:.4f} ± {best['cv_std']:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, best["y_pred"],
                             target_names=["Malignant", "Benign"]))


# ──────────────────────────────────────────────
# 7. PREDICTION ON NEW SAMPLE
# ──────────────────────────────────────────────
print("\n[7] Sample prediction demo")
sample = X_test_s[0].reshape(1, -1)
pred   = best["model"].predict(sample)[0]
prob   = best["model"].predict_proba(sample)[0]
label  = data.target_names[pred]
print(f"    Predicted class : {label.upper()}")
print(f"    Probability     : Malignant={prob[0]:.3f}  Benign={prob[1]:.3f}")
print(f"    Actual class    : {data.target_names[y_test.iloc[0]].upper()}")

print("\n  All plots saved. Pipeline complete.")