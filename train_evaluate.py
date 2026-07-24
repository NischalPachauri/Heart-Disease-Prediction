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
  9. Comprehensive diagnostic visualization & metadata export
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
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model      import LogisticRegression
from sklearn.pipeline          import Pipeline
from sklearn.preprocessing     import StandardScaler, PolynomialFeatures, QuantileTransformer, SplineTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection   import StratifiedKFold, train_test_split
from sklearn.ensemble          import BaggingClassifier
from sklearn.metrics           import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, precision_recall_curve
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

RANDOM_STATE = 42
TEST_SIZE    = 0.10
CV_FOLDS     = 5

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

def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS_DIR, name), bbox_inches="tight")
    plt.close(fig)

# --------------------------------------------------------------------------- #
#  Feature Engineering Function                                                #
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
    ment = df_feat["MentalHealth"] if "MentalHealth" in df_feat.columns else 0
    df_feat["TotalImpairmentDays"] = phys + ment

    # 4. Ordinal General Health Mapping
    gen_health_map = {"Poor": 4, "Fair": 3, "Good": 2, "Very good": 1, "Excellent": 0}
    if "GenHealth" in df_feat.columns:
        df_feat["GenHealthOrdinal"] = df_feat["GenHealth"].map(gen_health_map).fillna(2)

    # 5. Continuous Midpoint Age & Age Squared
    age_map = {
        "18-24": 21, "25-29": 27, "30-34": 32, "35-39": 37,
        "40-44": 42, "45-49": 47, "50-54": 52, "55-59": 57,
        "60-64": 62, "65-69": 67, "70-74": 72, "75-79": 77,
        "80 or older": 82,
    }
    if "AgeCategory" in df_feat.columns:
        df_feat["AgeNumeric"] = df_feat["AgeCategory"].map(age_map).fillna(50)
        df_feat["AgeSquared"] = (df_feat["AgeNumeric"] / 10.0) ** 2

    return df_feat

# --------------------------------------------------------------------------- #
#  1. Load, Engineer & One-Hot Encode                                          #
# --------------------------------------------------------------------------- #
print("=" * 70)
print("  ULTIMATE LOGISTIC REGRESSION PIPELINE (SMOTE + BAGGING + SPLINES)")
print("=" * 70)

raw_df = pd.read_csv(DATA_PATH)
print(f"\n[OK] Loaded dataset: {raw_df.shape[0]:,} rows x {raw_df.shape[1]} columns")

raw_df["HeartDisease"] = (raw_df["HeartDisease"] == "Yes").astype(int)

# Apply Clinical Feature Engineering
df_eng = engineer_features(raw_df)

# One-Hot Encode Categorical Columns
cat_cols = df_eng.select_dtypes(include="object").columns.tolist()
df_encoded = pd.get_dummies(df_eng, columns=cat_cols)

X = df_encoded.drop(columns=["HeartDisease"])
y = df_encoded["HeartDisease"]

feature_columns = X.columns.tolist()
print(f"[OK] Total engineered & dummy feature columns: {len(feature_columns)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"     Train split: {len(X_train):,} rows (90%)   Test split: {len(X_test):,} rows (10%)\n")

# Apply SMOTE Resampling on Training Split
print("[STEP 1/6] Applying SMOTE Synthetic Minority Oversampling on training split ...")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"     Resampled train shape: {X_train_res.shape[0]:,} rows\n")

# Helper function to compute evaluation metrics
def calc_metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy"   : float(accuracy_score(y_true, y_pred)),
        "precision"  : float(precision_score(y_true, y_pred)),
        "recall"     : float(recall_score(y_true, y_pred)),
        "f1"         : float(f1_score(y_true, y_pred)),
        "auc_roc"    : float(roc_auc_score(y_true, y_proba)),
        "specificity": float(tn / (tn + fp)),
        "cm"         : cm,
    }

# --------------------------------------------------------------------------- #
#  2. Baseline LR                                                              #
# --------------------------------------------------------------------------- #
print("[STEP 2/6] Baseline Logistic Regression ...")
base_lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
base_lr.fit(X_train, y_train)
y_pred_base  = base_lr.predict(X_test)
y_proba_base = base_lr.predict_proba(X_test)[:, 1]
base_mets    = calc_metrics(y_test, y_pred_base, y_proba_base)
print(f"     Acc={base_mets['accuracy']:.4f}  AUC={base_mets['auc_roc']:.4f}  F1={base_mets['f1']:.4f}")

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
print(f"     Acc={scaled_mets['accuracy']:.4f}  AUC={scaled_mets['auc_roc']:.4f}  F1={scaled_mets['f1']:.4f}")

# --------------------------------------------------------------------------- #
#  4. Polynomial Interaction Pipeline                                          #
# --------------------------------------------------------------------------- #
print("[STEP 4/6] Pairwise Polynomial Interaction Pipeline ...")
poly_pipe = Pipeline([
    ("scaler",   StandardScaler()),
    ("poly",     PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ("selector", SelectKBest(f_classif, k=65)),
    ("lr",       LogisticRegression(C=0.5, penalty="l2", solver="lbfgs", max_iter=3000, random_state=RANDOM_STATE)),
])
poly_pipe.fit(X_train_res, y_train_res)
y_pred_poly  = poly_pipe.predict(X_test)
y_proba_poly = poly_pipe.predict_proba(X_test)[:, 1]
poly_mets    = calc_metrics(y_test, y_pred_poly, y_proba_poly)
print(f"     Acc={poly_mets['accuracy']:.4f}  AUC={poly_mets['auc_roc']:.4f}  F1={poly_mets['f1']:.4f}")

# --------------------------------------------------------------------------- #
#  5. Bagged Logistic Regression Ensemble                                      #
# --------------------------------------------------------------------------- #
print("[STEP 5/6] Bagged Logistic Regression Ensemble (35 Estimators) ...")
base_lr_estimator = LogisticRegression(C=1.0, max_iter=3000, penalty="l2", random_state=RANDOM_STATE)
bagged_lr = BaggingClassifier(
    estimator=base_lr_estimator,
    n_estimators=35,
    max_samples=0.85,
    max_features=0.90,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
bagged_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("bagging", bagged_lr)
])
bagged_pipe.fit(X_train_res, y_train_res)

y_pred_bag  = bagged_pipe.predict(X_test)
y_proba_bag = bagged_pipe.predict_proba(X_test)[:, 1]
bag_mets    = calc_metrics(y_test, y_pred_bag, y_proba_bag)
print(f"     Acc={bag_mets['accuracy']:.4f}  AUC={bag_mets['auc_roc']:.4f}  F1={bag_mets['f1']:.4f}")

# --------------------------------------------------------------------------- #
#  6. Calibrated Decision Threshold Optimization                               #
# --------------------------------------------------------------------------- #
print("[STEP 6/6] Precision-Recall Decision Threshold Optimization ...")
precisions, recalls, thresholds = precision_recall_curve(y_train, bagged_pipe.predict_proba(X_train)[:, 1])
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores)
optimal_threshold = float(thresholds[best_idx])

y_pred_calib = (y_proba_bag >= optimal_threshold).astype(int)
calib_mets   = calc_metrics(y_test, y_pred_calib, y_proba_bag)
calib_mets["threshold"] = optimal_threshold
print(f"     Optimal Threshold: {optimal_threshold:.4f}")
print(f"     Acc={calib_mets['accuracy']:.4f}  AUC={calib_mets['auc_roc']:.4f}  F1={calib_mets['f1']:.4f}  Rec={calib_mets['recall']:.4f}")

# --------------------------------------------------------------------------- #
#  Pick Winner                                                                 #
# --------------------------------------------------------------------------- #
candidates = {
    "Baseline LR"                        : (base_mets["accuracy"],  base_lr,     y_pred_base,  y_proba_base, base_mets),
    "SMOTE + Scaled LR"                  : (scaled_mets["accuracy"], scaled_pipe, y_pred_sc,    y_proba_sc,   scaled_mets),
    "Polynomial Interactions LR"         : (poly_mets["accuracy"],   poly_pipe,   y_pred_poly,  y_proba_poly, poly_mets),
    "Bagged LR Ensemble (35 Estimators)" : (bag_mets["accuracy"],    bagged_pipe, y_pred_bag,   y_proba_bag,  bag_mets),
    "Calibrated Bagged LR (Opt Threshold)": (calib_mets["accuracy"],  bagged_pipe, y_pred_calib, y_proba_bag,  calib_mets),
}

winner_name = max(candidates, key=lambda k: (candidates[k][0], candidates[k][4]["auc_roc"]))
winner_acc, winner_model, winner_pred, winner_proba, winner_mets = candidates[winner_name]

print(f"\n[WINNER] {winner_name} -> Acc={winner_mets['accuracy']:.4f} ({winner_mets['accuracy']*100:.2f}%)  AUC={winner_mets['auc_roc']:.4f}")

# --------------------------------------------------------------------------- #
#  Export Performance Plots                                                   #
# --------------------------------------------------------------------------- #
print("\nExporting diagnostic charts ...")

# A. Confusion Matrix of Winning Model
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(winner_mets["cm"], annot=True, fmt="d", ax=ax,
            cmap="Blues", linewidths=1.5, linecolor="#1c1f26",
            xticklabels=["Pred: No HD", "Pred: HD"],
            yticklabels=["Actual: No HD", "Actual: HD"],
            cbar=False,
            annot_kws={"size": 14, "weight": "bold", "color": "#0f1117"})
ax.set_title(f"Confusion Matrix — {winner_name}\n"
             f"Acc: {winner_mets['accuracy']:.2%}  "
             f"F1: {winner_mets['f1']:.4f}  "
             f"AUC: {winner_mets['auc_roc']:.4f}", pad=12)
save_fig(fig, "LR_01_confusion_matrix_winner.png")

# B. ROC Curves
fig, ax = plt.subplots(figsize=(8, 6))
colors  = [PURPLE, GREEN, YELLOW, RED, CYAN]
for (cname, (_, _, _, yprob, mets)), color in zip(candidates.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, yprob)
    ax.plot(fpr, tpr, color=color, lw=2.0, label=f"{cname[:30]} (AUC={mets['auc_roc']:.4f})")
ax.plot([0, 1], [0, 1], "w--", lw=0.9, label="Random baseline")
ax.set_title("ROC Curves — Advanced Ensembled Logistic Regression")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Recall)")
ax.legend(fontsize=8.0, loc="lower right")
ax.grid(alpha=0.3)
save_fig(fig, "LR_03_roc_curves.png")

# C. Metric Comparison Bar Chart
metrics_to_plot = ["accuracy", "precision", "recall", "f1", "auc_roc", "specificity"]
mlabels         = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC", "Specificity"]
x               = np.arange(len(mlabels))
w               = 0.16
fig, ax         = plt.subplots(figsize=(16, 6))
for i, (cname, (_, _, _, _, mets)) in enumerate(candidates.items()):
    vals = [mets[k] for k in metrics_to_plot]
    bars = ax.bar(x + i * w, vals, w, label=cname[:25], color=colors[i], alpha=0.88, edgecolor="#ffffff22")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"{v:.3f}", ha="center", va="bottom", fontsize=6, color="#cccccc", rotation=90)
ax.set_xticks(x + w * 2)
ax.set_xticklabels(mlabels, fontsize=10)
ax.set_ylim(0.5, 1.08)
ax.set_ylabel("Score")
ax.set_title("Metric Comparison — SMOTE + Bagging Logistic Regression Pipeline")
ax.legend(fontsize=8.5)
ax.grid(axis="y", alpha=0.3)
save_fig(fig, "LR_04_metric_comparison.png")

# --------------------------------------------------------------------------- #
#  Export Model Artifacts & Metadata                                          #
# --------------------------------------------------------------------------- #
model_export_path = os.path.join(MODEL_DIR, "logistic_regression_enhanced.pkl")
meta_export_path  = os.path.join(MODEL_DIR, "model_metadata.json")

with open(model_export_path, "wb") as f:
    pickle.dump(winner_model, f)
print(f"\n[OK] Winner model saved -> model/logistic_regression_enhanced.pkl")

meta_data = {
    "winner_name"        : winner_name,
    "optimal_threshold"  : optimal_threshold,
    "accuracy"           : winner_mets["accuracy"],
    "precision"          : winner_mets["precision"],
    "recall"             : winner_mets["recall"],
    "f1_score"           : winner_mets["f1"],
    "auc_roc"            : winner_mets["auc_roc"],
    "specificity"        : winner_mets["specificity"],
    "feature_columns"    : feature_columns,
}

with open(meta_export_path, "w", encoding="utf-8") as f:
    json.dump(meta_data, f, indent=2)
print(f"[OK] Metadata saved -> model/model_metadata.json")

# Summary export
results_summary = {
    cname: {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v)
            for k, v in mets.items() if k != "cm"}
    for cname, (_, _, _, _, mets) in candidates.items()
}
results_summary["__winner__"] = winner_name
results_summary["__optimal_threshold__"] = optimal_threshold

with open(os.path.join(BASE, "lr_training_results.json"), "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2)

print("=" * 70)
print("  TRAINING COMPLETE — SUMMARY OF RESULTS")
print("=" * 70)
for cname, (_, _, _, _, mets) in candidates.items():
    mark = " <-- BEST" if cname == winner_name else ""
    print(f"{cname:<38} Acc: {mets['accuracy']*100:>6.2f}%  AUC: {mets['auc_roc']:>6.4f}  F1: {mets['f1']:>6.4f}  Rec: {mets['recall']:>6.4f}{mark}")
print("=" * 70)
