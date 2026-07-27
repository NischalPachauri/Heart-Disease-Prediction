# Heart Disease Prediction — Logistic Regression

Cardiovascular risk screening tool using a Logistic Regression model trained on 54,746 CDC BRFSS survey records.
- **Model:** Logistic Regression (Baseline winner / Bagged ensemble variant available)
- **AUC-ROC:** 0.841 &nbsp;|&nbsp; **Recall:** 78.4% &nbsp;|&nbsp; **Accuracy:** 76.7% &nbsp;|&nbsp; **F1-Score:** 0.771

---

## Project Structure

```
.
├── index.html              # Frontend web app (open in browser)
├── server.py               # Flask REST API server (serves real predictions)
├── train_evaluate.py       # ML training pipeline (re-run to retrain model)
├── requirements.txt        # pip dependencies
├── Pipfile                 # Pipenv environment config
├── lr_training_results.json# Exported model metrics summary
├── LICENSE
├── data/
│   └── heart_2020_balanced_54k.csv   # Training dataset (54,746 rows, 50/50 balanced)
├── model/
│   ├── logistic_regression_enhanced.pkl  # Trained scikit-learn model
│   └── model_metadata.json               # Feature columns, threshold, metadata
└── outputs/
    └── analysis/           # Training charts & evaluation artifacts
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Prediction API Server (Optional but recommended)

To serve live predictions from the trained `.pkl` model:

```bash
python server.py
```
*Runs on `http://localhost:5000`*.

### 3. Open the Frontend

**Option A — Local HTTP server (recommended):**
```bash
python -m http.server 8000
```
Then open **http://localhost:8000** in your browser.

*Note: The web app automatically detects `server.py` at `http://localhost:5000`. If `server.py` is running, it calls the live trained `.pkl` model. If not, it gracefully falls back to the client-side approximation with a clear visual badge.*

**Option B — Direct file access:**
Double-click `index.html` in File Explorer.

### 4. Retrain the Model (Optional)

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
→ Logistic Regression Baseline / Bagged Ensemble
→ Exported Model (.pkl + metadata.json)
```
