# Heart Disease Prediction — Logistic Regression

Cardiovascular risk screening tool using a Logistic Regression model trained on 54,746 CDC BRFSS survey records.
- **Model:** Logistic Regression with SMOTE resampling, Bagged ensemble, and calibrated decision threshold
- **AUC-ROC:** 0.841 &nbsp;|&nbsp; **Recall:** 86.6% &nbsp;|&nbsp; **Accuracy:** 76.7%

---

## Project Structure

```
.
├── index.html              # Frontend — open in browser or serve via Python
├── train_evaluate.py       # ML training pipeline (re-run to retrain model)
├── requirements.txt        # pip dependencies
├── Pipfile                 # Pipenv config
├── lr_training_results.json
├── LICENSE
├── data/
│   ├── heart_2020_balanced_54k.csv   # Training dataset (54,746 rows, 50/50 balanced)
│   └── heart_2020_cleaned.csv        # Full cleaned CDC dataset
├── model/
│   ├── logistic_regression_enhanced.pkl  # Trained model (loaded by app)
│   └── model_metadata.json               # Feature columns, optimal threshold, metrics
└── outputs/
    └── analysis/           # Training charts (confusion matrices, ROC curves, etc.)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the frontend (from the project root)

**Option A — open directly (no terminal needed):**
Double-click `index.html` in File Explorer.

**Option B — serve via Python:**
```bash
python -m http.server 8000
```
Then open **http://localhost:8000** in your browser.

### 3. Retrain the model (optional)

```bash
python train_evaluate.py
```
This retrains the model, saves updated `.pkl` and `model_metadata.json`, and regenerates all charts in `outputs/analysis/`.

---

## Dataset Features

| Category | Features |
|---|---|
| Demographics | Age, Sex, Race, BMI |
| Medical History | Stroke, Diabetes, Asthma, Kidney Disease, Skin Cancer |
| Lifestyle | Smoking, Alcohol, Physical Activity, Difficulty Walking |
| Health Impairment | Physical Health Days, Mental Health Days, Sleep Hours |
| General Health | Self-reported health status (Excellent → Poor) |

---

## Model Pipeline

```
Raw CSV → Feature Engineering → One-Hot Encoding → Train/Test Split (90/10)
→ SMOTE Resampling → Bagged Logistic Regression (35 estimators)
→ Threshold Calibration (0.395) → Saved model (.pkl)
```
