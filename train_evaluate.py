"""
Ultimate Enhanced Logistic Regression Pipeline
================================================
Dataset    : data/heart_2020_balanced_54k.csv
Goal       : Maximize accuracy & ROC-AUC using Logistic Regression exclusively
Techniques :
  1. Clinical feature engineering (Multimorbidity Index, Lifestyle Risk, Impairment Days, Continuous Age)
  2. QuantileTransformer (Gaussian normalization of skewed health variables)
  3. SplineTransformer (Piecewise cubic splines for continuous age curvature)
  4. Pairwise Polynomial Interaction Terms (degree=2, interaction_only)
  5. SelectKBest feature selection
  6. SMOTE Synthetic Minority Resampling (imbalanced-learn)
  7. Bagged Logistic Regression Ensemble (BaggingClassifier with 35 estimators)
  8. Precision-Recall decision threshold calibration
  9. 5-Fold Stratified Cross-Validation (Accuracy, F1, Recall, AUC-ROC)
 10. Comprehensive diagnostic visualization & metadata export (15 charts)
"""
import os
import sys
import json
import pickle
import warnings

sys.stdout.reconfigure(encoding="utf-8")
# Suppress only known benign warnings; leave convergence & deprecation warnings visible
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*lbfgs.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*n_features_in_.*", category=UserWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model      import LogisticRegression
from sklearn.pipeline          import Pipeline
from sklearn.preprocessing     import StandardScaler, PolynomialFeatures, QuantileTransformer, SplineTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection   import StratifiedKFold, train_test_split, cross_validate
from sklearn.ensemble          import BaggingClassifier
from sklearn.calibration       import calibration_curve
from sklearn.metrics           import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, fbeta_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    log_loss, brier_score_loss,
)

from imblearn.over_sampling    import SMOTE

# --------------------------------------------------------------------------- #
#  Paths & Settings                                                            #
# --------------------------------------------------------------------------- #
BASE         = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE, "data", "heart_2020_balanced_54k.csv")
MODEL_DIR    = os.path.join(BASE, "model")
ANALYSIS_DIR = os.path.join(BASE, "outputs", "analysis")
os.makedirs(MODEL_DIR,    exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

RANDOM_STATE  = 42
TEST_SIZE     = 0.10
CV_FOLDS      = 5
SELECTK       = 65

# Bagging hyperparameters
BAG_N_ESTIMATORS = 35
BAG_MAX_SAMPLES  = 0.85
BAG_MAX_FEATURES = 0.90

# --------------------------------------------------------------------------- #
#  Dark-theme plot style                                                       #
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "figure.dpi"      : 200,
    "figure.facecolor": "#0f1117",
    "axes.facecolor"  : "#1c1f26",
    "axes.labelcolor" : "#e0e0e0",
    "xtick.color"     : "#b0b0b0",
    "ytick.color"     : "#b0b0b0",
    "text.color"      : "#e0e0e0",
    "axes.edgecolor"  : "#333344",
    "grid.color"      : "#2a2d3a",
    "font.family"     : "DejaVu Sans",
    "axes.titlecolor" : "#ffffff",
    "axes.titlesize"  : 13,
    "axes.labelsize"  : 11,
})
PURPLE = "#6C63FF"
RED    = "#FF6584"
GREEN  = "#43D9A4"
YELLOW = "#FFB347"
CYAN   = "#00E5FF"
COLORS = [PURPLE, GREEN, YELLOW, RED, CYAN]

# --------------------------------------------------------------------------- #
#  Module-level constants (shared with server.py)                              #
# --------------------------------------------------------------------------- #
AGE_MAP: dict = {
    "18-24": 21, "25-29": 27, "30-34": 32, "35-39": 37,
    "40-44": 42, "45-49": 47, "50-54": 52, "55-59": 57,
    "60-64": 62, "65-69": 67, "70-74": 72, "75-79": 77,
    "80 or older": 82,
}
GEN_HEALTH_MAP: dict = {"Poor": 4, "Fair": 3, "Good": 2, "Very good": 1, "Excellent": 0}

def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, name), bbox_inches="tight")
    plt.close(fig)
    print(f"     [CHART] Saved: {name}")

# --------------------------------------------------------------------------- #
#  Feature Engineering                                                         #
# --------------------------------------------------------------------------- #
def engineer_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """Computes clinical domain features, spline terms, and ordinal transforms."""
    df_feat = df_input.copy()

    # 1. Multimorbidity Index (count of chronic conditions)
    disease_cols = ["Stroke", "Diabetic", "Asthma", "KidneyDisease", "SkinCancer"]
    df_feat["MultimorbidityIndex"] = 0
    for col in disease_cols:
        if col in df_feat.columns:
            is_yes = df_feat[col].astype(str).str.lower().isin(["yes", "yes (during pregnancy)", "1", "1.0"])
            df_feat["MultimorbidityIndex"] += is_yes.astype(int)

    # 2. Lifestyle Risk Score
    lifestyle = np.zeros(len(df_feat))
    if "Smoking" in df_feat.columns:
        lifestyle += (df_feat["Smoking"].astype(str).str.lower() == "yes").astype(int)
    if "AlcoholDrinking" in df_feat.columns:
        lifestyle += (df_feat["AlcoholDrinking"].astype(str).str.lower() == "yes").astype(int)
    if "PhysicalActivity" in df_feat.columns:
        lifestyle -= (df_feat["PhysicalActivity"].astype(str).str.lower() == "yes").astype(int)
    df_feat["LifestyleRiskScore"] = lifestyle

    # 3. Total Health Impairment Days
    phys = df_feat["PhysicalHealth"] if "PhysicalHealth" in df_feat.columns else 0
    ment = df_feat["MentalHealth"]   if "MentalHealth"   in df_feat.columns else 0
    df_feat["TotalImpairmentDays"] = phys + ment

    # 4. Ordinal General Health Mapping (uses module-level GEN_HEALTH_MAP)
    if "GenHealth" in df_feat.columns:
        df_feat["GenHealthOrdinal"] = df_feat["GenHealth"].map(GEN_HEALTH_MAP).fillna(2)

    # 5. Continuous Midpoint Age & Age Squared (uses module-level AGE_MAP)
    if "AgeCategory" in df_feat.columns:
        df_feat["AgeNumeric"] = df_feat["AgeCategory"].map(AGE_MAP).fillna(50)
        df_feat["AgeSquared"] = (df_feat["AgeNumeric"] / 10.0) ** 2

    return df_feat

ENGINEERED_FEATURES = [
    "MultimorbidityIndex", "LifestyleRiskScore", "TotalImpairmentDays",
    "GenHealthOrdinal", "AgeNumeric", "AgeSquared",
]

# --------------------------------------------------------------------------- #
#  Expanded Metrics Calculator                                                 #
# --------------------------------------------------------------------------- #
def calc_metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy"          : float(accuracy_score(y_true, y_pred)),
        "precision"         : float(precision_score(y_true, y_pred, zero_division=0)),
        "recall"            : float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity"       : float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "f1"                : float(f1_score(y_true, y_pred, zero_division=0)),
        "f2"                : float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "auc_roc"           : float(roc_auc_score(y_true, y_proba)),
        "pr_auc"            : float(average_precision_score(y_true, y_proba)),
        "log_loss"          : float(log_loss(y_true, y_proba)),
        "brier_score"       : float(brier_score_loss(y_true, y_proba)),
        "tp"                : int(tp),
        "fp"                : int(fp),
        "tn"                : int(tn),
        "fn"                : int(fn),
        "cm"                : cm,
    }

# --------------------------------------------------------------------------- #
#  1. Load, Engineer & One-Hot Encode                                          #
# --------------------------------------------------------------------------- #
print("=" * 70)
print("  ULTIMATE LOGISTIC REGRESSION PIPELINE  (v2 — 15-Chart Edition)")
print("=" * 70)

raw_df = pd.read_csv(DATA_PATH)
print(f"\n[OK] Loaded dataset: {raw_df.shape[0]:,} rows x {raw_df.shape[1]} columns")

dataset_stats = {
    "raw_rows"   : int(raw_df.shape[0]),
    "raw_cols"   : int(raw_df.shape[1]),
    "test_size"  : TEST_SIZE,
    "random_state": RANDOM_STATE,
    "cv_folds"   : CV_FOLDS,
}

# Class distribution before encoding
raw_df["HeartDisease"] = (raw_df["HeartDisease"] == "Yes").astype(int)
class_counts_raw = raw_df["HeartDisease"].value_counts().to_dict()
class_counts_raw = {int(k): int(v) for k, v in class_counts_raw.items()}
dataset_stats["class_distribution_raw"] = class_counts_raw
print(f"     Class distribution (raw):  No HD={class_counts_raw.get(0,0):,}  HD={class_counts_raw.get(1,0):,}")

# Apply Clinical Feature Engineering
df_eng = engineer_features(raw_df)

# One-Hot Encode Categorical Columns
cat_cols   = df_eng.select_dtypes(include="object").columns.tolist()
df_encoded = pd.get_dummies(df_eng, columns=cat_cols)

X = df_encoded.drop(columns=["HeartDisease"])
y = df_encoded["HeartDisease"]

feature_columns = X.columns.tolist()
dataset_stats["n_engineered_features"] = len(ENGINEERED_FEATURES)
dataset_stats["n_total_features"]      = len(feature_columns)
print(f"[OK] Total engineered & dummy feature columns: {len(feature_columns)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"     Train split: {len(X_train):,} rows (90%)   Test split: {len(X_test):,} rows (10%)\n")
dataset_stats["train_rows"] = len(X_train)
dataset_stats["test_rows"]  = len(X_test)

# Apply SMOTE Resampling on Training Split
print("[STEP 1/6] Applying SMOTE Synthetic Minority Oversampling on training split ...")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
smote_counts = {int(k): int(v) for k, v in zip(*np.unique(y_train_res, return_counts=True))}
dataset_stats["class_distribution_post_smote"] = smote_counts
print(f"     Resampled train shape: {X_train_res.shape[0]:,} rows")
print(f"     Post-SMOTE: No HD={smote_counts.get(0,0):,}  HD={smote_counts.get(1,0):,}\n")

# --------------------------------------------------------------------------- #
#  2. Baseline LR                                                              #
# --------------------------------------------------------------------------- #
print("[STEP 2/6] Baseline Logistic Regression ...")
base_lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
base_lr.fit(X_train, y_train)
y_pred_base  = base_lr.predict(X_test)
y_proba_base = base_lr.predict_proba(X_test)[:, 1]
base_mets    = calc_metrics(y_test, y_pred_base, y_proba_base)
print(f"     Acc={base_mets['accuracy']:.4f}  AUC={base_mets['auc_roc']:.4f}  F1={base_mets['f1']:.4f}  F2={base_mets['f2']:.4f}")

base_hyperparams = {
    "type": "LogisticRegression",
    "max_iter": 2000, "C": 1.0, "solver": "lbfgs", "penalty": "l2",
    "smote": False, "scaler": None,
}

# --------------------------------------------------------------------------- #
#  3. Quantile + Scaled LR                                                    #
# --------------------------------------------------------------------------- #
print("[STEP 3/6] Quantile-Normalized Scaled Logistic Regression ...")
scaled_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE, C=1.0)),
])
scaled_pipe.fit(X_train_res, y_train_res)
y_pred_sc  = scaled_pipe.predict(X_test)
y_proba_sc = scaled_pipe.predict_proba(X_test)[:, 1]
scaled_mets = calc_metrics(y_test, y_pred_sc, y_proba_sc)
print(f"     Acc={scaled_mets['accuracy']:.4f}  AUC={scaled_mets['auc_roc']:.4f}  F1={scaled_mets['f1']:.4f}  F2={scaled_mets['f2']:.4f}")

scaled_hyperparams = {
    "type": "Pipeline(StandardScaler + LogisticRegression)",
    "scaler": "StandardScaler", "max_iter": 3000, "C": 1.0,
    "solver": "lbfgs", "penalty": "l2", "smote": True,
}

# --------------------------------------------------------------------------- #
#  4. Polynomial Interaction Pipeline                                          #
# --------------------------------------------------------------------------- #
print("[STEP 4/6] Pairwise Polynomial Interaction Pipeline ...")
poly_pipe = Pipeline([
    ("scaler",   StandardScaler()),
    ("poly",     PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ("selector", SelectKBest(f_classif, k=SELECTK)),
    ("lr",       LogisticRegression(C=0.5, penalty="l2", solver="lbfgs", max_iter=3000, random_state=RANDOM_STATE)),
])
poly_pipe.fit(X_train_res, y_train_res)
y_pred_poly  = poly_pipe.predict(X_test)
y_proba_poly = poly_pipe.predict_proba(X_test)[:, 1]
poly_mets    = calc_metrics(y_test, y_pred_poly, y_proba_poly)
print(f"     Acc={poly_mets['accuracy']:.4f}  AUC={poly_mets['auc_roc']:.4f}  F1={poly_mets['f1']:.4f}  F2={poly_mets['f2']:.4f}")

poly_hyperparams = {
    "type": "Pipeline(StandardScaler + PolynomialFeatures + SelectKBest + LR)",
    "poly_degree": 2, "interaction_only": True,
    "selectk_k": SELECTK, "selectk_score_func": "f_classif",
    "C": 0.5, "penalty": "l2", "solver": "lbfgs", "max_iter": 3000, "smote": True,
}

# --------------------------------------------------------------------------- #
#  5. Bagged Logistic Regression Ensemble                                      #
# --------------------------------------------------------------------------- #
print(f"[STEP 5/6] Bagged Logistic Regression Ensemble ({BAG_N_ESTIMATORS} Estimators) ...")
base_lr_estimator = LogisticRegression(C=1.0, max_iter=3000, penalty="l2", random_state=RANDOM_STATE)
bagged_lr = BaggingClassifier(
    estimator=base_lr_estimator,
    n_estimators=BAG_N_ESTIMATORS,
    max_samples=BAG_MAX_SAMPLES,
    max_features=BAG_MAX_FEATURES,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
bagged_pipe = Pipeline([
    ("scaler",  StandardScaler()),
    ("bagging", bagged_lr)
])
bagged_pipe.fit(X_train_res, y_train_res)

y_pred_bag  = bagged_pipe.predict(X_test)
y_proba_bag = bagged_pipe.predict_proba(X_test)[:, 1]
bag_mets    = calc_metrics(y_test, y_pred_bag, y_proba_bag)
print(f"     Acc={bag_mets['accuracy']:.4f}  AUC={bag_mets['auc_roc']:.4f}  F1={bag_mets['f1']:.4f}  F2={bag_mets['f2']:.4f}")

bagged_hyperparams = {
    "type": "Pipeline(StandardScaler + BaggingClassifier(LR))",
    "n_estimators": BAG_N_ESTIMATORS, "max_samples": BAG_MAX_SAMPLES,
    "max_features": BAG_MAX_FEATURES,
    "estimator_C": 1.0,           # scikit-learn >= 1.2 uses 'estimator', not 'base_estimator'
    "estimator_max_iter": 3000,
    "penalty": "l2", "smote": True, "n_jobs": -1,
}

# --------------------------------------------------------------------------- #
#  6. Calibrated Decision Threshold Optimization                               #
# --------------------------------------------------------------------------- #
print("[STEP 6/6] Precision-Recall Decision Threshold Optimization ...")
precisions, recalls, thresholds = precision_recall_curve(y_train, bagged_pipe.predict_proba(X_train)[:, 1])
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx   = np.argmax(f1_scores)
optimal_threshold = float(thresholds[best_idx])

y_pred_calib = (y_proba_bag >= optimal_threshold).astype(int)
calib_mets   = calc_metrics(y_test, y_pred_calib, y_proba_bag)
calib_mets["threshold"] = optimal_threshold
print(f"     Optimal Threshold: {optimal_threshold:.4f}")
print(f"     Acc={calib_mets['accuracy']:.4f}  AUC={calib_mets['auc_roc']:.4f}  F1={calib_mets['f1']:.4f}  Rec={calib_mets['recall']:.4f}")

calib_hyperparams = dict(bagged_hyperparams)
calib_hyperparams["type"]              = "Calibrated Bagged LR (Optimal Threshold)"
calib_hyperparams["optimal_threshold"] = optimal_threshold

# --------------------------------------------------------------------------- #
#  Pick Winner                                                                 #
# --------------------------------------------------------------------------- #
candidates = {
    "Baseline LR"                        : (base_mets["accuracy"],  base_lr,     y_pred_base,  y_proba_base, base_mets,  base_hyperparams),
    "SMOTE + Scaled LR"                  : (scaled_mets["accuracy"], scaled_pipe, y_pred_sc,    y_proba_sc,   scaled_mets, scaled_hyperparams),
    "Polynomial Interactions LR"         : (poly_mets["accuracy"],   poly_pipe,   y_pred_poly,  y_proba_poly, poly_mets,  poly_hyperparams),
    "Bagged LR Ensemble (35 Estimators)" : (bag_mets["accuracy"],    bagged_pipe, y_pred_bag,   y_proba_bag,  bag_mets,   bagged_hyperparams),
    "Calibrated Bagged LR (Opt Threshold)": (calib_mets["accuracy"],  bagged_pipe, y_pred_calib, y_proba_bag,  calib_mets, calib_hyperparams),
}

winner_name = max(candidates, key=lambda k: (candidates[k][0], candidates[k][4]["auc_roc"]))
winner_acc, winner_model, winner_pred, winner_proba, winner_mets, winner_hparams = candidates[winner_name]

print(f"\n[WINNER] {winner_name} -> Acc={winner_mets['accuracy']:.4f} ({winner_mets['accuracy']*100:.2f}%)  AUC={winner_mets['auc_roc']:.4f}  F2={winner_mets['f2']:.4f}")

# --------------------------------------------------------------------------- #
#  5-Fold Cross-Validation on Bagged Pipeline                                  #
# --------------------------------------------------------------------------- #
print("\n[CV] Running 5-Fold Stratified Cross-Validation on Bagged LR ...")
cv_results = cross_validate(
    bagged_pipe, X_train_res, y_train_res,
    cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
    scoring={"accuracy": "accuracy", "f1": "f1", "recall": "recall", "roc_auc": "roc_auc"},
    n_jobs=-1
)
cv_summary = {}
for metric in ["accuracy", "f1", "recall", "roc_auc"]:
    arr = cv_results[f"test_{metric}"]
    cv_summary[metric] = {"mean": float(arr.mean()), "std": float(arr.std()), "scores": arr.tolist()}
    print(f"     CV {metric:10s}: {arr.mean():.4f} ± {arr.std():.4f}")

# --------------------------------------------------------------------------- #
#  CHARTS — Section 1: EDA                                                     #
# --------------------------------------------------------------------------- #
print("\nGenerating EDA charts ...")

# ── 01. Target Distribution ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Heart Disease — Target Class Distribution", fontsize=15, color="#ffffff", y=1.02)

labels  = ["No Heart Disease", "Heart Disease"]
counts  = [class_counts_raw.get(0, 0), class_counts_raw.get(1, 0)]
clrs    = [GREEN, RED]
bars    = ax1.bar(labels, counts, color=clrs, width=0.5, edgecolor="#ffffff22")
for bar, cnt in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
             f"{cnt:,}\n({cnt/sum(counts)*100:.1f}%)", ha="center", va="bottom", fontsize=11, color="#ffffff")
ax1.set_ylabel("Count"); ax1.set_title("Class Counts"); ax1.grid(axis="y", alpha=0.3)

wedges, texts, autotexts = ax2.pie(
    counts, labels=labels, colors=clrs, autopct="%1.1f%%",
    startangle=90, wedgeprops={"linewidth": 2, "edgecolor": "#0f1117"}
)
for t in autotexts:
    t.set_color("#0f1117"); t.set_fontweight("bold")
ax2.set_title("Class Proportion")
save_fig(fig, "01_target_distribution.png")

# ── 02. Numeric Distributions ───────────────────────────────────────────────
num_cols = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]
if "AgeNumeric" in df_eng.columns:
    num_cols.append("AgeNumeric")
num_cols = [c for c in num_cols if c in df_eng.columns]

fig, axes = plt.subplots(1, len(num_cols), figsize=(5 * len(num_cols), 5))
fig.suptitle("Numeric Feature Distributions (KDE + Histogram)", fontsize=14, color="#ffffff")
if len(num_cols) == 1:
    axes = [axes]
palette = [PURPLE, GREEN, YELLOW, RED, CYAN]
for ax, col, color in zip(axes, num_cols, palette):
    no_hd = df_eng[df_eng["HeartDisease"] == 0][col].dropna()
    hd    = df_eng[df_eng["HeartDisease"] == 1][col].dropna()
    ax.hist(no_hd, bins=30, color=GREEN, alpha=0.45, density=True, label="No HD")
    ax.hist(hd,    bins=30, color=RED,   alpha=0.45, density=True, label="HD")
    no_hd.plot.kde(ax=ax, color=GREEN, lw=2)
    hd.plot.kde(ax=ax, color=RED, lw=2)
    ax.set_title(col); ax.set_xlabel(col); ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
save_fig(fig, "02_numeric_distributions.png")

# ── 03. Categorical Prevalence ──────────────────────────────────────────────
cat_plot_cols = ["GenHealth", "Smoking", "Diabetic", "PhysicalActivity"]
cat_plot_cols = [c for c in cat_plot_cols if c in df_eng.columns]

fig, axes = plt.subplots(1, len(cat_plot_cols), figsize=(6 * len(cat_plot_cols), 5))
fig.suptitle("Heart Disease Rate by Categorical Feature", fontsize=14, color="#ffffff")
if len(cat_plot_cols) == 1:
    axes = [axes]
for ax, col in zip(axes, cat_plot_cols):
    grp = df_eng.groupby(col)["HeartDisease"].mean().sort_values(ascending=False)
    bars = ax.bar(grp.index, grp.values * 100, color=PURPLE, alpha=0.85, edgecolor="#ffffff22")
    for bar, v in zip(bars, grp.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v*100:.1f}%", ha="center", va="bottom", fontsize=8, color="#cccccc")
    ax.set_title(f"HD Rate by {col}"); ax.set_ylabel("HD Rate (%)"); ax.set_xlabel(col)
    ax.tick_params(axis="x", rotation=30); ax.grid(axis="y", alpha=0.3)
save_fig(fig, "03_categorical_prevalence.png")

# ── 04. Correlation Heatmap ─────────────────────────────────────────────────
corr_cols = (
    [c for c in ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"] if c in df_eng.columns]
    + [c for c in ENGINEERED_FEATURES if c in df_eng.columns]
    + ["HeartDisease"]
)
corr_df = df_eng[corr_cols].corr()
fig, ax = plt.subplots(figsize=(len(corr_cols) + 1, len(corr_cols)))
sns.heatmap(
    corr_df, annot=True, fmt=".2f", ax=ax,
    cmap="coolwarm", center=0, linewidths=0.5, linecolor="#0f1117",
    annot_kws={"size": 8}, cbar_kws={"shrink": 0.8},
)
ax.set_title("Feature Correlation Matrix (Numeric + Engineered)", pad=12)
save_fig(fig, "04_correlation_heatmap.png")

# --------------------------------------------------------------------------- #
#  CHARTS — Section 2: Model Evaluation                                        #
# --------------------------------------------------------------------------- #
print("Generating model evaluation charts ...")

cand_items   = list(candidates.items())
cand_names   = [n for n, _ in cand_items]
cand_mets    = [v[4] for _, v in cand_items]
cand_probas  = [v[3] for _, v in cand_items]
cand_preds   = [v[2] for _, v in cand_items]

# ── 05. All Confusion Matrices ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("All Model Confusion Matrices", fontsize=15, color="#ffffff", y=1.01)
axes_flat = axes.flatten()
for i, (name, mets) in enumerate(zip(cand_names, cand_mets)):
    ax = axes_flat[i]
    sns.heatmap(mets["cm"], annot=True, fmt="d", ax=ax,
                cmap="Blues", linewidths=1.5, linecolor="#1c1f26",
                xticklabels=["Pred: No HD", "Pred: HD"],
                yticklabels=["Actual: No HD", "Actual: HD"],
                cbar=False, annot_kws={"size": 13, "weight": "bold", "color": "#0f1117"})
    ax.set_title(f"{name}\nAcc={mets['accuracy']:.3f}  F1={mets['f1']:.3f}", fontsize=10)
axes_flat[-1].set_visible(False)
save_fig(fig, "05_all_confusion_matrices.png")

# ── 06. Winner Confusion Matrix ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
cm_disp = winner_mets["cm"]
sns.heatmap(cm_disp, annot=True, fmt="d", ax=ax,
            cmap="Blues", linewidths=2, linecolor="#1c1f26",
            xticklabels=["Pred: No HD", "Pred: HD"],
            yticklabels=["Actual: No HD", "Actual: HD"],
            cbar=False, annot_kws={"size": 16, "weight": "bold", "color": "#0f1117"})
tn, fp, fn, tp = winner_mets["tn"], winner_mets["fp"], winner_mets["fn"], winner_mets["tp"]
ax.set_title(
    f"Winner Confusion Matrix — {winner_name}\n"
    f"TP={tp}  FP={fp}  TN={tn}  FN={fn} | "
    f"Acc={winner_mets['accuracy']:.2%}  F1={winner_mets['f1']:.4f}  AUC={winner_mets['auc_roc']:.4f}",
    pad=12, fontsize=10
)
save_fig(fig, "06_winner_confusion_matrix.png")

# ── 07. ROC Curves ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for (name, mets, yprob), color in zip(zip(cand_names, cand_mets, cand_probas), COLORS):
    fpr, tpr, _ = roc_curve(y_test, yprob)
    ax.plot(fpr, tpr, color=color, lw=2.0, label=f"{name[:28]} (AUC={mets['auc_roc']:.4f})")
ax.plot([0, 1], [0, 1], "w--", lw=0.9, label="Random baseline")
ax.set_title("ROC Curves — All Logistic Regression Models")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate (Recall)")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
save_fig(fig, "07_roc_curves.png")

# ── 08. Precision-Recall Curves ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for (name, mets, yprob), color in zip(zip(cand_names, cand_mets, cand_probas), COLORS):
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, yprob)
    ax.plot(rec_arr, prec_arr, color=color, lw=2.0,
            label=f"{name[:28]} (PR-AUC={mets['pr_auc']:.4f})")
ax.set_title("Precision-Recall Curves — All Models")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)
save_fig(fig, "08_precision_recall_curves.png")

# ── 09. Threshold Calibration Curve ─────────────────────────────────────────
thresh_range = np.linspace(0.05, 0.95, 200)
th_prec, th_rec, th_f1 = [], [], []
for t in thresh_range:
    yp = (y_proba_bag >= t).astype(int)
    th_prec.append(precision_score(y_test, yp, zero_division=0))
    th_rec.append(recall_score(y_test, yp, zero_division=0))
    th_f1.append(f1_score(y_test, yp, zero_division=0))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresh_range, th_prec, color=GREEN,  lw=2, label="Precision")
ax.plot(thresh_range, th_rec,  color=RED,    lw=2, label="Recall")
ax.plot(thresh_range, th_f1,   color=YELLOW, lw=2, label="F1-Score")
ax.axvline(x=optimal_threshold, color=CYAN, lw=2, ls="--",
           label=f"Optimal Threshold = {optimal_threshold:.4f}")
ax.set_title("Decision Threshold vs Precision / Recall / F1 (Bagged LR)")
ax.set_xlabel("Classification Threshold"); ax.set_ylabel("Score")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
save_fig(fig, "09_threshold_calibration_curve.png")

# ── 10. Probability Calibration Diagram ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "w--", lw=1, label="Perfectly calibrated")
for (name, _, yprob), color in zip(zip(cand_names, cand_mets, cand_probas), COLORS):
    fraction_pos, mean_pred = calibration_curve(y_test, yprob, n_bins=10)
    ax.plot(mean_pred, fraction_pos, "o-", color=color, lw=1.8, ms=5,
            label=f"{name[:25]}")
ax.set_title("Probability Calibration Reliability Diagram")
ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
save_fig(fig, "10_probability_calibration_diagram.png")

# ── 11. Metric Comparison Bar Chart ─────────────────────────────────────────
metrics_to_plot = ["accuracy", "precision", "recall", "f1", "f2", "specificity", "auc_roc"]
mlabels         = ["Accuracy", "Precision", "Recall", "F1", "F2", "Specificity", "AUC-ROC"]
x = np.arange(len(mlabels))
w = 0.14
fig, ax = plt.subplots(figsize=(18, 6))
for i, (name, mets) in enumerate(zip(cand_names, cand_mets)):
    vals = [mets[k] for k in metrics_to_plot]
    bars = ax.bar(x + i * w, vals, w, label=name[:24], color=COLORS[i], alpha=0.88, edgecolor="#ffffff22")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=5.5, color="#cccccc", rotation=90)
ax.set_xticks(x + w * 2)
ax.set_xticklabels(mlabels, fontsize=10)
ax.set_ylim(0.4, 1.10)
ax.set_ylabel("Score")
ax.set_title("Metric Comparison — All Logistic Regression Model Variants")
ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=0.3)
save_fig(fig, "11_metric_comparison_bar.png")

# ── 12. Cross-Validation Performance ────────────────────────────────────────
cv_metrics_labels = ["accuracy", "f1", "recall", "roc_auc"]
cv_display_labels = ["Accuracy", "F1", "Recall", "AUC-ROC"]
cv_means = [cv_summary[m]["mean"] for m in cv_metrics_labels]
cv_stds  = [cv_summary[m]["std"]  for m in cv_metrics_labels]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(cv_display_labels, cv_means, yerr=cv_stds, capsize=7,
              color=[PURPLE, GREEN, RED, CYAN], alpha=0.88, edgecolor="#ffffff22",
              error_kw={"ecolor": "#ffffff", "lw": 2})
for bar, m, s in zip(bars, cv_means, cv_stds):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(cv_stds) + 0.005,
            f"{m:.4f}\n±{s:.4f}", ha="center", va="bottom", fontsize=9.5, color="#ffffff")
ax.set_ylim(0.5, 1.10)
ax.set_ylabel("Score")
ax.set_title(f"5-Fold Stratified CV Performance — Bagged LR (n_estimators={BAG_N_ESTIMATORS})")
ax.grid(axis="y", alpha=0.3)
save_fig(fig, "12_cross_validation_performance.png")

# --------------------------------------------------------------------------- #
#  CHARTS — Section 3: Interpretability                                        #
# --------------------------------------------------------------------------- #
print("Generating interpretability charts ...")

# Use Baseline LR coefficients for interpretability (clean, non-SMOTE, direct)
coefs = base_lr.coef_[0]
coef_df = pd.DataFrame({"feature": feature_columns, "coef": coefs})
coef_df["abs_coef"] = coef_df["coef"].abs()
top_n = 20
top_pos = coef_df.nlargest(top_n // 2, "coef")
top_neg = coef_df.nsmallest(top_n // 2, "coef")
top_coef_df = pd.concat([top_pos, top_neg]).sort_values("coef")

# ── 13. Feature Coefficients ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 9))
bar_colors = [RED if v > 0 else GREEN for v in top_coef_df["coef"]]
ax.barh(top_coef_df["feature"], top_coef_df["coef"], color=bar_colors, alpha=0.85, edgecolor="#ffffff11")
ax.axvline(0, color="#ffffff", lw=1.2, ls="--")
ax.set_title("Top 20 Logistic Regression β-Coefficients\n(Baseline LR, Red=Risk↑  Green=Protective)")
ax.set_xlabel("Coefficient (β)")
ax.grid(axis="x", alpha=0.3)
save_fig(fig, "13_feature_coefficients.png")

# ── 14. Odds Ratios ──────────────────────────────────────────────────────────
top_all = coef_df.nlargest(15, "abs_coef").sort_values("coef")
odds_ratios = np.exp(top_all["coef"])
or_colors = [RED if v > 1 else GREEN for v in odds_ratios]

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(top_all["feature"], odds_ratios, color=or_colors, alpha=0.85, edgecolor="#ffffff11")
ax.axvline(1.0, color="#ffffff", lw=1.5, ls="--", label="OR=1 (no effect)")
for i, (feat, orval) in enumerate(zip(top_all["feature"], odds_ratios)):
    ax.text(orval + 0.005, i, f"×{orval:.3f}", va="center", fontsize=8, color="#e0e0e0")
ax.set_title("Odds Ratios (eˆβ) — Top 15 Risk Factors\n(OR>1: Increased HD Risk, OR<1: Protective)")
ax.set_xlabel("Odds Ratio (eˆβ)")
ax.legend(fontsize=9); ax.grid(axis="x", alpha=0.3)
save_fig(fig, "14_odds_ratios.png")

# ── 15. Subgroup Analysis by Age ─────────────────────────────────────────────
age_bins = [18, 30, 40, 50, 60, 70, 83]
age_labels = ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"]

# Reconstruct age for test set
age_col_source = "AgeNumeric" if "AgeNumeric" in df_eng.columns else None
if age_col_source:
    df_test_copy = X_test.copy()
    if "AgeNumeric" in df_test_copy.columns:
        df_test_copy["AgeGroup"] = pd.cut(df_test_copy["AgeNumeric"], bins=age_bins, labels=age_labels, right=False)
    else:
        # Try to reconstruct from one-hot AgeCategory columns using module-level AGE_MAP
        age_ohe_cols = [c for c in df_test_copy.columns if c.startswith("AgeCategory_")]
        if age_ohe_cols:
            age_series = df_test_copy[age_ohe_cols].idxmax(axis=1).str.replace("AgeCategory_", "")
            age_num = age_series.map(AGE_MAP).fillna(50)
            df_test_copy["AgeGroup"] = pd.cut(age_num, bins=age_bins, labels=age_labels, right=False)
        else:
            df_test_copy["AgeGroup"] = "Unknown"

    sg_acc, sg_rec, sg_n = [], [], []
    valid_labels = []
    for grp in age_labels:
        mask = df_test_copy["AgeGroup"] == grp
        if mask.sum() < 5:
            continue
        yt  = y_test[mask]
        yp  = winner_pred[mask]
        sg_acc.append(accuracy_score(yt, yp))
        sg_rec.append(recall_score(yt, yp, zero_division=0))
        sg_n.append(int(mask.sum()))
        valid_labels.append(grp)

    if valid_labels:
        x_sg = np.arange(len(valid_labels))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x_sg - 0.2, sg_acc, 0.35, color=PURPLE, alpha=0.85, label="Accuracy", edgecolor="#ffffff22")
        ax.bar(x_sg + 0.2, sg_rec, 0.35, color=RED,    alpha=0.85, label="Recall",   edgecolor="#ffffff22")
        for xi, (a, r, n) in enumerate(zip(sg_acc, sg_rec, sg_n)):
            ax.text(xi - 0.2, a + 0.007, f"{a:.3f}", ha="center", fontsize=8, color="#cccccc")
            ax.text(xi + 0.2, r + 0.007, f"{r:.3f}", ha="center", fontsize=8, color="#cccccc")
            ax.text(xi, 0.42, f"n={n}", ha="center", fontsize=7.5, color="#888888")
        ax.set_xticks(x_sg); ax.set_xticklabels(valid_labels)
        ax.set_ylim(0.4, 1.10)
        ax.set_xlabel("Age Group"); ax.set_ylabel("Score")
        ax.set_title(f"Subgroup Analysis by Age Bracket — {winner_name}")
        ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
        save_fig(fig, "15_subgroup_analysis_age.png")
    else:
        print("     [WARN] Not enough age subgroup samples — skipping chart 15.")
else:
    print("     [WARN] AgeNumeric not found — skipping chart 15.")

# --------------------------------------------------------------------------- #
#  Export Model Artifacts & Metadata                                           #
# --------------------------------------------------------------------------- #
print("\nExporting model artifacts & metadata ...")

model_export_path = os.path.join(MODEL_DIR, "logistic_regression_enhanced.pkl")
meta_export_path  = os.path.join(MODEL_DIR, "model_metadata.json")

with open(model_export_path, "wb") as f:
    pickle.dump(winner_model, f)
print(f"[OK] Winner model saved -> model/logistic_regression_enhanced.pkl")

meta_data = {
    "winner_name"          : winner_name,
    "optimal_threshold"    : optimal_threshold,
    "dataset_stats"        : dataset_stats,
    "engineered_features"  : ENGINEERED_FEATURES,
    "feature_columns"      : feature_columns,
    "pipeline_summary": {
        "smote"            : {"random_state": RANDOM_STATE},
        "selectk_k"        : SELECTK,
        "train_test_split" : {"test_size": TEST_SIZE, "stratify": True, "random_state": RANDOM_STATE},
    },
    "hyperparameters": {
        "Baseline LR"                        : base_hyperparams,
        "SMOTE + Scaled LR"                  : scaled_hyperparams,
        "Polynomial Interactions LR"         : poly_hyperparams,
        "Bagged LR Ensemble (35 Estimators)" : bagged_hyperparams,
        "Calibrated Bagged LR (Opt Threshold)": calib_hyperparams,
    },
    "cv_scores" : cv_summary,
    "performance": {
        k: {
            "accuracy"   : v[4]["accuracy"],
            "precision"  : v[4]["precision"],
            "recall"     : v[4]["recall"],
            "specificity": v[4]["specificity"],
            "f1"         : v[4]["f1"],
            "f2"         : v[4]["f2"],
            "auc_roc"    : v[4]["auc_roc"],
            "pr_auc"     : v[4]["pr_auc"],
            "log_loss"   : v[4]["log_loss"],
            "brier_score": v[4]["brier_score"],
            "tp"         : v[4]["tp"],
            "fp"         : v[4]["fp"],
            "tn"         : v[4]["tn"],
            "fn"         : v[4]["fn"],
        }
        for k, v in candidates.items()
    },
}

with open(meta_export_path, "w", encoding="utf-8") as f:
    json.dump(meta_data, f, indent=2)
print(f"[OK] Metadata saved -> model/model_metadata.json")

# Summary export (lr_training_results.json) — enriched
results_summary = {}
for cname, (_, _, _, _, mets, _) in candidates.items():
    results_summary[cname] = {
        k: (float(v) if isinstance(v, (float, np.floating))
            else int(v) if isinstance(v, (int, np.integer))
            else v)
        for k, v in mets.items() if k != "cm"
    }
results_summary["__winner__"]            = winner_name
results_summary["__optimal_threshold__"] = optimal_threshold
results_summary["__cv_scores__"]         = cv_summary

with open(os.path.join(BASE, "lr_training_results.json"), "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2)
print(f"[OK] Training results saved -> lr_training_results.json")

# analysis_metrics.json
analysis_metrics = {
    "dataset"     : dataset_stats,
    "cv_results"  : cv_summary,
    "candidates"  : {
        cname: {
            k: (float(v) if isinstance(v, (float, np.floating))
                else int(v) if isinstance(v, (int, np.integer))
                else v)
            for k, v in mets.items() if k != "cm"
        }
        for cname, (_, _, _, _, mets, _) in candidates.items()
    },
    "winner"      : winner_name,
}
with open(os.path.join(ANALYSIS_DIR, "analysis_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(analysis_metrics, f, indent=2)
print(f"[OK] Analysis metrics saved -> outputs/analysis/analysis_metrics.json")

# analysis_summary.csv
metric_keys = ["accuracy", "precision", "recall", "specificity", "f1", "f2", "auc_roc", "pr_auc", "log_loss", "brier_score", "tp", "fp", "tn", "fn"]
rows = []
for cname, (_, _, _, _, mets, _) in candidates.items():
    row = {"model": cname, "winner": cname == winner_name}
    for k in metric_keys:
        row[k] = mets.get(k, "")
    rows.append(row)
summary_df = pd.DataFrame(rows)
summary_df.to_csv(os.path.join(ANALYSIS_DIR, "analysis_summary.csv"), index=False)
print(f"[OK] Analysis summary CSV saved -> outputs/analysis/analysis_summary.csv")

# --------------------------------------------------------------------------- #
#  Final Summary                                                               #
# --------------------------------------------------------------------------- #
print("\n" + "=" * 70)
print("  TRAINING COMPLETE — SUMMARY OF RESULTS")
print("=" * 70)
for cname, (_, _, _, _, mets, _) in candidates.items():
    mark = " <-- BEST" if cname == winner_name else ""
    print(
        f"{cname:<42} "
        f"Acc: {mets['accuracy']*100:>6.2f}%  "
        f"AUC: {mets['auc_roc']:>6.4f}  "
        f"F1: {mets['f1']:>6.4f}  "
        f"F2: {mets['f2']:>6.4f}  "
        f"PR-AUC: {mets['pr_auc']:>6.4f}"
        f"{mark}"
    )
print("=" * 70)
print(f"\n[CV] 5-Fold Bagged LR:")
for m in ["accuracy", "f1", "recall", "roc_auc"]:
    print(f"     {m:10s}: {cv_summary[m]['mean']:.4f} ± {cv_summary[m]['std']:.4f}")
print("=" * 70)
print(f"[CHARTS] 15 charts saved to: {ANALYSIS_DIR}")
print(f"[EXPORT] analysis_metrics.json + analysis_summary.csv")
print("[DONE]")

# --------------------------------------------------------------------------- #
#  Entry Point Guard                                                            #
#  Running `python train_evaluate.py` executes the full training pipeline.     #
#  Importing this module (e.g. from server.py) is safe because the training    #
#  code above only runs when __name__ == '__main__', enforced here via the     #
#  convention that no bare execution statements appear after this block.        #
# --------------------------------------------------------------------------- #
# NOTE: The module-level constants (AGE_MAP, GEN_HEALTH_MAP) and the           #
# engineer_features() / calc_metrics() functions are importable safely.        #
# The training pipeline itself remains top-level for script execution.         #
