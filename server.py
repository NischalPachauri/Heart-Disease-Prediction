"""
Heart Disease Prediction — Flask API Server
============================================
Loads the trained Logistic Regression model (.pkl) and exposes a REST
endpoint so the frontend can receive real model predictions instead of
the JS approximation.

Usage:
    pip install flask flask-cors
    python server.py

Endpoints:
    GET  /health   → {"status":"ok", "model":"...", "threshold":...}
    POST /predict  → {"probability":0.34, "percent":34.0, "prediction":0, ...}
"""

import os
import json
import logging
import pickle

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_PKL = os.path.join(BASE, "model", "logistic_regression_enhanced.pkl")
META_JSON = os.path.join(BASE, "model", "model_metadata.json")

# ── Module-level constants (mirrors train_evaluate.py) ────────────────────────
AGE_MAP: dict[str, int] = {
    "18-24": 21, "25-29": 27, "30-34": 32, "35-39": 37,
    "40-44": 42, "45-49": 47, "50-54": 52, "55-59": 57,
    "60-64": 62, "65-69": 67, "70-74": 72, "75-79": 77,
    "80 or older": 82,
}
GEN_HEALTH_MAP: dict[str, int] = {
    "Poor": 4, "Fair": 3, "Good": 2, "Very good": 1, "Excellent": 0,
}


# ── Feature engineering (mirrors train_evaluate.py:engineer_features) ─────────
def engineer_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the same feature engineering used during training."""
    df = df_input.copy()

    # 1. Multimorbidity Index
    disease_cols = ["Stroke", "Diabetic", "Asthma", "KidneyDisease", "SkinCancer"]
    df["MultimorbidityIndex"] = 0
    for col in disease_cols:
        if col in df.columns:
            is_yes = (
                df[col].astype(str).str.lower()
                .isin(["yes", "yes (during pregnancy)", "1", "1.0"])
            )
            df["MultimorbidityIndex"] += is_yes.astype(int)

    # 2. Lifestyle Risk Score
    lifestyle = np.zeros(len(df))
    if "Smoking" in df.columns:
        lifestyle += (df["Smoking"].astype(str).str.lower() == "yes").astype(int)
    if "AlcoholDrinking" in df.columns:
        lifestyle += (df["AlcoholDrinking"].astype(str).str.lower() == "yes").astype(int)
    if "PhysicalActivity" in df.columns:
        lifestyle -= (df["PhysicalActivity"].astype(str).str.lower() == "yes").astype(int)
    df["LifestyleRiskScore"] = lifestyle

    # 3. Total Health Impairment Days
    phys = df["PhysicalHealth"] if "PhysicalHealth" in df.columns else 0
    ment = df["MentalHealth"]   if "MentalHealth"   in df.columns else 0
    df["TotalImpairmentDays"] = phys + ment

    # 4. Ordinal General Health
    if "GenHealth" in df.columns:
        df["GenHealthOrdinal"] = df["GenHealth"].map(GEN_HEALTH_MAP).fillna(2)

    # 5. Continuous Age & Age²
    if "AgeCategory" in df.columns:
        df["AgeNumeric"] = df["AgeCategory"].map(AGE_MAP).fillna(50)
        df["AgeSquared"] = (df["AgeNumeric"] / 10.0) ** 2

    return df


# ── Load model artefacts ──────────────────────────────────────────────────────
log.info("Loading model …  %s", MODEL_PKL)
with open(MODEL_PKL, "rb") as _f:
    MODEL: object = pickle.load(_f)

with open(META_JSON, encoding="utf-8") as _f:
    METADATA: dict = json.load(_f)

FEATURE_COLUMNS: list[str] = METADATA["feature_columns"]
THRESHOLD: float           = METADATA["optimal_threshold"]
WINNER_NAME: str           = METADATA["winner_name"]

log.info(
    "Ready  model=%s  threshold=%.4f  features=%d",
    WINNER_NAME, THRESHOLD, len(FEATURE_COLUMNS),
)

# ── Flask application ─────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow all origins so the static index.html can call us


@app.route("/health")
def health():
    """Liveness / readiness probe."""
    return jsonify({
        "status"    : "ok",
        "model"     : WINNER_NAME,
        "threshold" : THRESHOLD,
        "features"  : len(FEATURE_COLUMNS),
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept a JSON body with the same field names used in the HTML form,
    run the trained pipeline, and return the predicted probability.

    Expected fields (all strings, matching training encoding):
        AgeCategory, Sex, BMICategory, Race, SleepTime (int),
        Stroke, AlcoholDrinking, PhysicalActivity, DiffWalking,
        Diabetic, GenHealth, Asthma, KidneyDisease, SkinCancer,
        Smoking, PhysicalHealth (int), MentalHealth (int)
    """
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    try:
        df = pd.DataFrame([payload])

        # Coerce numeric fields
        for num_col in ("PhysicalHealth", "MentalHealth", "SleepTime"):
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce").fillna(0)

        # Engineer features
        df_eng     = engineer_features(df)

        # One-hot encode categorical columns
        cat_cols   = df_eng.select_dtypes(include="object").columns.tolist()
        df_encoded = pd.get_dummies(df_eng, columns=cat_cols)

        # Align to training schema (add missing dummies as 0)
        for col in FEATURE_COLUMNS:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        X = df_encoded[FEATURE_COLUMNS].fillna(0)

        proba    = float(MODEL.predict_proba(X)[0, 1])
        pred_bin = int(proba >= THRESHOLD)

        log.info("predict  proba=%.4f  pred=%d", proba, pred_bin)

        return jsonify({
            "probability" : round(proba, 6),
            "percent"     : round(proba * 100, 1),
            "prediction"  : pred_bin,
            "threshold"   : round(THRESHOLD, 4),
            "model"       : WINNER_NAME,
        })

    except Exception as exc:
        log.exception("Prediction error: %s", exc)
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    log.info("Starting server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
