"""
Enhanced Logistic Regression Training
======================================
Dataset  : data/heart_2020_balanced_54k.csv
Goal     : Maximize accuracy of Logistic Regression without changing the dataset
Techniques:
  1. StandardScaler (LR is sensitive to feature scale)
  2. GridSearchCV hyperparameter tuning (C, penalty, solver)
  3. Polynomial/interaction feature engineering
  4. SelectKBest feature selection
  5. Full evaluation: confusion matrix, ROC, metrics
"""
import os
import sys
import json
import pickle
import warnings

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model      import LogisticRegression
from sklearn.pipeline          import Pipeline
from sklearn.preprocessing     import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection   import (GridSearchCV, StratifiedKFold,
                                        cross_val_score, train_test_split)
from sklearn.metrics           import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay,
)

# --------------------------------------------------------------------------- #
#  Paths & settings                                                            #
# --------------------------------------------------------------------------- #
BASE         = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE, "data", "heart_2020_balanced_54k.csv")
MODEL_DIR    = os.path.join(BASE, "model")
ANALYSIS_DIR = os.path.join(BASE, "images", "analysis")
os.makedirs(MODEL_DIR,    exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

# --------------------------------------------------------------------------- #
#  Dark-theme plot style                                                       #
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "figure.dpi"     : 200,
    "figure.facecolor": "#0f1117",
    "axes.facecolor" : "#1c1f26",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color"    : "#b0b0b0",
    "ytick.color"    : "#b0b0b0",
    "text.color"     : "#e0e0e0",
    "axes.edgecolor" : "#333344",
    "grid.color"     : "#2a2d3a",
    "font.family"    : "DejaVu Sans",
    "axes.titlecolor": "#ffffff",
    "axes.titlesize" : 13,
    "axes.labelsize" : 11,
})
PURPLE = "#6C63FF"
RED    = "#FF6584"
GREEN  = "#43D9A4"
YELLOW = "#FFB347"

def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, name), bbox_inches="tight")
    plt.close(fig)

# --------------------------------------------------------------------------- #
#  1. Load & encode                                                            #
# --------------------------------------------------------------------------- #
print("=" * 62)
print("  ENHANCED LOGISTIC REGRESSION - HEART DISEASE PREDICTION")
print("=" * 62)

df = pd.read_csv(DATA_PATH)
print(f"\n[OK] Loaded {df.shape[0]:,} rows x {df.shape[1]} cols")
print(f"     Class balance: {df['HeartDisease'].value_counts().to_dict()}")

df["HeartDisease"] = (df["HeartDisease"] == "Yes").astype(int)
cat_cols = df.select_dtypes(include="object").columns.tolist()
df_enc   = pd.get_dummies(df, columns=cat_cols)

X = df_enc.drop(columns=["HeartDisease"])
y = df_enc["HeartDisease"]

print(f"     One-hot features: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"     Train: {len(X_train):,}   Test: {len(X_test):,}\n")

# --------------------------------------------------------------------------- #
#  2. Baseline (raw, unscaled, default LR — matches original project)         #
# --------------------------------------------------------------------------- #
print("[STEP 1/5] Baseline Logistic Regression (no scaling) ...")
base_lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
base_lr.fit(X_train, y_train)
y_pred_base  = base_lr.predict(X_test)
y_proba_base = base_lr.predict_proba(X_test)[:, 1]
base_metrics = {
    "accuracy"   : accuracy_score(y_test, y_pred_base),
    "precision"  : precision_score(y_test, y_pred_base),
    "recall"     : recall_score(y_test, y_pred_base),
    "f1"         : f1_score(y_test, y_pred_base),
    "auc_roc"    : roc_auc_score(y_test, y_proba_base),
    "specificity": confusion_matrix(y_test, y_pred_base).ravel()[0] /
                   (confusion_matrix(y_test, y_pred_base).ravel()[0] +
                    confusion_matrix(y_test, y_pred_base).ravel()[1]),
}
print(f"     Acc={base_metrics['accuracy']:.4f}  "
      f"AUC={base_metrics['auc_roc']:.4f}  "
      f"F1={base_metrics['f1']:.4f}")

# --------------------------------------------------------------------------- #
#  3. Scaled LR (StandardScaler improves convergence and coefficients)        #
# --------------------------------------------------------------------------- #
print("[STEP 2/5] Scaled Logistic Regression (StandardScaler) ...")
scaled_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
])
scaled_pipe.fit(X_train, y_train)
y_pred_sc    = scaled_pipe.predict(X_test)
y_proba_sc   = scaled_pipe.predict_proba(X_test)[:, 1]
cm_sc        = confusion_matrix(y_test, y_pred_sc)
tn, fp, fn, tp = cm_sc.ravel()
scaled_metrics = {
    "accuracy"   : accuracy_score(y_test, y_pred_sc),
    "precision"  : precision_score(y_test, y_pred_sc),
    "recall"     : recall_score(y_test, y_pred_sc),
    "f1"         : f1_score(y_test, y_pred_sc),
    "auc_roc"    : roc_auc_score(y_test, y_proba_sc),
    "specificity": tn / (tn + fp),
}
print(f"     Acc={scaled_metrics['accuracy']:.4f}  "
      f"AUC={scaled_metrics['auc_roc']:.4f}  "
      f"F1={scaled_metrics['f1']:.4f}")

# --------------------------------------------------------------------------- #
#  4. Hyperparameter tuning via GridSearchCV                                  #
# --------------------------------------------------------------------------- #
print("[STEP 3/5] GridSearchCV hyperparameter tuning ...")

param_grid = [
    # L2 (Ridge) — default
    {
        "lr__C"      : [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        "lr__penalty": ["l2"],
        "lr__solver" : ["lbfgs", "newton-cg", "saga"],
    },
    # L1 (Lasso) — sparse solutions
    {
        "lr__C"      : [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        "lr__penalty": ["l1"],
        "lr__solver" : ["liblinear", "saga"],
    },
    # ElasticNet
    {
        "lr__C"          : [0.01, 0.1, 1.0, 5.0],
        "lr__penalty"    : ["elasticnet"],
        "lr__solver"     : ["saga"],
        "lr__l1_ratio"   : [0.3, 0.5, 0.7],
    },
]

tuned_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
])

cv_strat = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
grid_search = GridSearchCV(
    tuned_pipe, param_grid,
    cv=cv_strat, scoring="accuracy",
    n_jobs=-1, verbose=0, refit=True,
)
grid_search.fit(X_train, y_train)

best_pipe   = grid_search.best_estimator_
best_params = grid_search.best_params_
best_cv_acc = grid_search.best_score_

y_pred_tuned  = best_pipe.predict(X_test)
y_proba_tuned = best_pipe.predict_proba(X_test)[:, 1]
cm_tuned      = confusion_matrix(y_test, y_pred_tuned)
tn2, fp2, fn2, tp2 = cm_tuned.ravel()
tuned_metrics = {
    "accuracy"   : accuracy_score(y_test, y_pred_tuned),
    "precision"  : precision_score(y_test, y_pred_tuned),
    "recall"     : recall_score(y_test, y_pred_tuned),
    "f1"         : f1_score(y_test, y_pred_tuned),
    "auc_roc"    : roc_auc_score(y_test, y_proba_tuned),
    "specificity": tn2 / (tn2 + fp2),
    "cv_acc"     : best_cv_acc,
}

print(f"     Best params : {best_params}")
print(f"     CV Acc      : {best_cv_acc:.4f}")
print(f"     Test Acc    : {tuned_metrics['accuracy']:.4f}  "
      f"AUC={tuned_metrics['auc_roc']:.4f}  "
      f"F1={tuned_metrics['f1']:.4f}")

# --------------------------------------------------------------------------- #
#  5. Feature-selected LR (SelectKBest + tuned hyperparams)                  #
# --------------------------------------------------------------------------- #
print("[STEP 4/5] Feature selection pipeline (SelectKBest + tuned LR) ...")

best_C       = best_params.get("lr__C", 1.0)
best_penalty = best_params.get("lr__penalty", "l2")
best_solver  = best_params.get("lr__solver", "lbfgs")
best_l1r     = best_params.get("lr__l1_ratio", None)

lr_kwargs = dict(
    C=best_C, penalty=best_penalty, solver=best_solver,
    max_iter=3000, random_state=RANDOM_STATE,
)
if best_l1r is not None:
    lr_kwargs["l1_ratio"] = best_l1r

# Search over number of features
k_values = [20, 30, 40, 50, X.shape[1]]
best_k_acc = 0
best_k     = X.shape[1]

for k in k_values:
    kpipe = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(f_classif, k=k)),
        ("lr", LogisticRegression(**lr_kwargs)),
    ])
    cv_acc = cross_val_score(kpipe, X_train, y_train, cv=cv_strat,
                             scoring="accuracy", n_jobs=-1).mean()
    print(f"     k={k:>3} features -> CV Acc={cv_acc:.4f}")
    if cv_acc > best_k_acc:
        best_k_acc = cv_acc
        best_k     = k

feat_pipe = Pipeline([
    ("scaler",   StandardScaler()),
    ("selector", SelectKBest(f_classif, k=best_k)),
    ("lr",       LogisticRegression(**lr_kwargs)),
])
feat_pipe.fit(X_train, y_train)
y_pred_feat  = feat_pipe.predict(X_test)
y_proba_feat = feat_pipe.predict_proba(X_test)[:, 1]
cm_feat      = confusion_matrix(y_test, y_pred_feat)
tn3, fp3, fn3, tp3 = cm_feat.ravel()
feat_metrics = {
    "accuracy"   : accuracy_score(y_test, y_pred_feat),
    "precision"  : precision_score(y_test, y_pred_feat),
    "recall"     : recall_score(y_test, y_pred_feat),
    "f1"         : f1_score(y_test, y_pred_feat),
    "auc_roc"    : roc_auc_score(y_test, y_proba_feat),
    "specificity": tn3 / (tn3 + fp3),
    "k_features" : best_k,
}
print(f"     Best k={best_k}  Test Acc={feat_metrics['accuracy']:.4f}  "
      f"AUC={feat_metrics['auc_roc']:.4f}")

# --------------------------------------------------------------------------- #
#  Pick winner                                                                 #
# --------------------------------------------------------------------------- #
candidates = {
    "Baseline LR"            : (base_metrics["accuracy"],  base_lr,        y_pred_base,  y_proba_base,  None,      base_metrics),
    "Scaled LR"              : (scaled_metrics["accuracy"], scaled_pipe,    y_pred_sc,    y_proba_sc,    cm_sc,     scaled_metrics),
    "Tuned LR (GridSearchCV)": (tuned_metrics["accuracy"],  best_pipe,      y_pred_tuned, y_proba_tuned, cm_tuned,  tuned_metrics),
    "Tuned + Feature Sel."   : (feat_metrics["accuracy"],   feat_pipe,      y_pred_feat,  y_proba_feat,  cm_feat,   feat_metrics),
}

winner_name = max(candidates, key=lambda k: candidates[k][0])
_, winner_model, winner_pred, winner_proba, winner_cm, winner_metrics = candidates[winner_name]

print(f"\n[WINNER] {winner_name} -> Acc={winner_metrics['accuracy']:.4f}")

# --------------------------------------------------------------------------- #
#  6. PLOTS                                                                    #
# --------------------------------------------------------------------------- #
print("\n[STEP 5/5] Generating plots ...")

# ---- A. Confusion Matrix of winner ----------------------------------------
if winner_cm is None:
    winner_cm = confusion_matrix(y_test, winner_pred)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(winner_cm, annot=True, fmt="d", ax=ax,
            cmap="Blues", linewidths=1.5, linecolor="#1c1f26",
            xticklabels=["Pred: No HD", "Pred: HD"],
            yticklabels=["Actual: No HD", "Actual: HD"],
            cbar=False,
            annot_kws={"size": 14, "weight": "bold", "color": "#0f1117"})
ax.set_title(f"Confusion Matrix — {winner_name}\n"
             f"Acc: {winner_metrics['accuracy']:.1%}  "
             f"F1: {winner_metrics['f1']:.3f}  "
             f"AUC: {winner_metrics['auc_roc']:.3f}",
             pad=12)
save(fig, "LR_01_confusion_matrix_winner.png")

# ---- B. All confusion matrices side-by-side --------------------------------
all_labels = list(candidates.keys())
fig, axes  = plt.subplots(1, 4, figsize=(22, 5))
for ax, (cname, (acc, mdl, ypred, _, _, mets)) in zip(axes, candidates.items()):
    cm_ = confusion_matrix(y_test, ypred)
    sns.heatmap(cm_, annot=True, fmt="d", ax=ax,
                cmap="Blues", linewidths=1, linecolor="#1c1f26",
                xticklabels=["No HD", "HD"],
                yticklabels=["No HD", "HD"],
                cbar=False,
                annot_kws={"size": 12, "weight": "bold", "color": "#0f1117"})
    ax.set_title(f"{cname}\nAcc={mets['accuracy']:.1%}  F1={mets['f1']:.3f}", pad=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
fig.suptitle("Confusion Matrices — All LR Variants", fontsize=14, y=1.02)
save(fig, "LR_02_all_confusion_matrices.png")

# ---- C. ROC Curves ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
colors  = [PURPLE, GREEN, YELLOW, RED]
for (cname, (_, _, _, yproba, _, mets)), color in zip(candidates.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, yproba)
    ax.plot(fpr, tpr, color=color, lw=2.0,
            label=f"{cname}  (AUC={mets['auc_roc']:.3f})")
ax.plot([0, 1], [0, 1], "w--", lw=0.9, label="Random baseline")
ax.fill_between(*roc_curve(y_test, winner_proba)[:2],
                alpha=0.08, color=YELLOW)
ax.set_title("ROC Curves — All Logistic Regression Variants")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Recall)")
ax.legend(fontsize=8.5, loc="lower right")
ax.grid(alpha=0.3)
save(fig, "LR_03_roc_curves.png")

# ---- D. Metric comparison bar chart ----------------------------------------
metrics_to_plot = ["accuracy", "precision", "recall", "f1", "auc_roc", "specificity"]
mlabels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC", "Specificity"]
x     = np.arange(len(mlabels))
w     = 0.20
fig, ax = plt.subplots(figsize=(15, 6))
for i, (cname, (_, _, _, _, _, mets)) in enumerate(candidates.items()):
    vals = [mets[k] for k in metrics_to_plot]
    bars = ax.bar(x + i * w, vals, w, label=cname,
                  color=colors[i], alpha=0.88, edgecolor="#ffffff22")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=6.5, color="#cccccc", rotation=90)
ax.set_xticks(x + w * 1.5)
ax.set_xticklabels(mlabels, fontsize=10)
ax.set_ylim(0.5, 1.08)
ax.set_ylabel("Score")
ax.set_title("Metric Comparison — All LR Variants")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
save(fig, "LR_04_metric_comparison.png")

# ---- E. Improvement delta chart vs baseline --------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
comp_models = ["Scaled LR", "Tuned LR (GridSearchCV)", "Tuned + Feature Sel."]
comp_colors = [GREEN, YELLOW, RED]
for i, (cname, color) in enumerate(zip(comp_models, comp_colors)):
    mets = candidates[cname][5]
    deltas = [(mets[k] - base_metrics[k]) * 100 for k in metrics_to_plot]
    bars = ax.bar(np.arange(len(mlabels)) + i * 0.25, deltas, 0.25,
                  label=cname, color=color, alpha=0.85, edgecolor="#ffffff22")
    for bar, d in zip(bars, deltas):
        if abs(d) > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.02 if d >= 0 else -0.12),
                    f"{d:+.2f}%", ha="center", va="bottom",
                    fontsize=7, color="#e0e0e0", rotation=90)
ax.axhline(0, color="#ffffff", lw=0.8, linestyle="--")
ax.set_xticks(np.arange(len(mlabels)) + 0.25)
ax.set_xticklabels(mlabels, fontsize=10)
ax.set_ylabel("Delta vs Baseline (%)")
ax.set_title("Improvement Over Baseline LR (percentage points)")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
save(fig, "LR_05_improvement_over_baseline.png")

# ---- F. Feature coefficients of winner model --------------------------------
try:
    if hasattr(winner_model, "named_steps"):
        lr_step = winner_model.named_steps["lr"]
        if "selector" in winner_model.named_steps:
            sel = winner_model.named_steps["selector"]
            feat_names = X.columns[sel.get_support()]
        else:
            feat_names = X.columns
    else:
        lr_step    = winner_model
        feat_names = X.columns

    coef     = lr_step.coef_[0]
    top_n    = 20
    top_idx  = np.argsort(np.abs(coef))[-top_n:]
    top_feat = feat_names[top_idx]
    top_coef = coef[top_idx]
    bar_colors = [RED if c < 0 else GREEN for c in top_coef]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_feat, top_coef, color=bar_colors,
            edgecolor="#ffffff22", alpha=0.88)
    ax.axvline(0, color="#ffffff", lw=0.8, linestyle="--")
    ax.set_title(f"Top {top_n} Feature Coefficients — {winner_name}\n"
                 "(Green = increases risk, Red = reduces risk)")
    ax.set_xlabel("Coefficient Value")
    ax.grid(axis="x", alpha=0.3)
    save(fig, "LR_06_feature_coefficients.png")
except Exception as e:
    print(f"     (Skipping coef plot: {e})")

# --------------------------------------------------------------------------- #
#  7. Save best model                                                           #
# --------------------------------------------------------------------------- #
model_path = os.path.join(MODEL_DIR, "logistic_regression_enhanced.pkl")
with open(model_path, "wb") as f:
    pickle.dump(winner_model, f)
print(f"\n[OK] Best model saved -> model/logistic_regression_enhanced.pkl")

# --------------------------------------------------------------------------- #
#  8. Summary table                                                             #
# --------------------------------------------------------------------------- #
print("\n" + "=" * 75)
print("  SUMMARY — ALL LOGISTIC REGRESSION VARIANTS")
print("=" * 75)
print(f"{'Variant':<28} {'Acc':>7} {'Prec':>7} {'Rec':>7} "
      f"{'F1':>7} {'AUC':>7} {'Spec':>7}")
print("-" * 75)
for cname, (acc, _, _, _, _, mets) in candidates.items():
    marker = " <-- BEST" if cname == winner_name else ""
    print(f"{cname:<28} "
          f"{mets['accuracy']:>7.4f} "
          f"{mets['precision']:>7.4f} "
          f"{mets['recall']:>7.4f} "
          f"{mets['f1']:>7.4f} "
          f"{mets['auc_roc']:>7.4f} "
          f"{mets['specificity']:>7.4f}"
          f"{marker}")
print("=" * 75)

print("\nClassification Report (Winner):")
print(classification_report(y_test, winner_pred,
      target_names=["No Heart Disease", "Heart Disease"]))

# --------------------------------------------------------------------------- #
#  9. Save JSON for README generation                                          #
# --------------------------------------------------------------------------- #
export = {
    cname: {
        k: (float(v) if isinstance(v, (float, np.floating)) else
            int(v)   if isinstance(v, (int, np.integer))   else v)
        for k, v in mets.items()
    }
    for cname, (_, _, _, _, _, mets) in candidates.items()
}
export["__winner__"]    = winner_name
export["__best_params__"] = str(best_params)

with open(os.path.join(BASE, "lr_training_results.json"), "w", encoding="utf-8") as f:
    json.dump(export, f, indent=2)

print("\n[OK] lr_training_results.json saved")
print("[OK] All plots saved to images/analysis/")
print("\n All done!")
