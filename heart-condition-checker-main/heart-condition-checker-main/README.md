# 🫀 Heart Disease Prediction App

![ML](https://img.shields.io/badge/ML-Logistic%20Regression-blue?style=for-the-badge&logo=scikit-learn)
![Python](https://img.shields.io/badge/Python-3.9-yellow?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.5.1-red?style=for-the-badge&logo=streamlit)
![Dataset](https://img.shields.io/badge/Dataset-319%2C796%20Rows-green?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Accuracy-~80%25-orange?style=for-the-badge)

**A machine learning-powered web application that predicts the probability of heart disease based on 17 personal health and lifestyle factors. Built with Python, scikit-learn, and Streamlit.**



---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [How It Works](#️-how-it-works)
3. [Project Structure](#-project-structure)
4. [File-by-File Breakdown](#-file-by-file-breakdown)
5. [Dataset Deep Dive](#-dataset-deep-dive)
6. [The Machine Learning Pipeline](#-the-machine-learning-pipeline)
7. [Features Used for Prediction](#️-features-used-for-prediction)
8. [Technologies and Dependencies](#️-technologies--dependencies)
9. [Installation and Running Locally](#-installation--running-locally)
10. [Model Performance](#-model-performance)
11. [Model Comparison](#-model-comparison--alternative-approaches)
12. [Limitations and Disclaimers](#️-limitations--disclaimers)

---

## 🎯 Project Overview

This application predicts whether a person is at risk of heart disease based on a survey of their **health indicators and lifestyle habits**. It leverages a **Logistic Regression** model trained on CDC survey data from over **319,000 U.S. residents (2020)**, using an undersampling technique to handle the significant class imbalance in the dataset (only ~9% of respondents had heart disease).

The goal is to make early heart disease risk detection accessible and fast — a user can fill out a sidebar form and receive an instant prediction with a probability score in seconds, without needing any medical equipment.

> **Disclaimer**: This app is for educational purposes only and is **not** a substitute for professional medical advice or diagnosis.

---

## ⚙️ How It Works

The end-to-end flow of the application:

```
User fills sidebar form
        |
        v
Input collected as pd.DataFrame (17 features)
        |
        v
Concatenate with full dataset (for consistent one-hot encoding)
        |
        v
Drop target column (HeartDisease)
        |
        v
One-hot encode all 14 categorical columns
        |
        v
Take only the first row (user input)
        |
        v
Fill NaN values with 0
        |
        v
Load pre-trained logistic_regression.pkl model
        |
        v
model.predict()      -> binary result (0 = Healthy, 1 = Heart Disease)
model.predict_proba() -> probability score
        |
        v
Display result image + probability to user
```

### Why Concatenate the Full Dataset Before Encoding?

This is a critical design decision in `app.py`. The user's single-row input is concatenated with the full 319,796-row dataset **before** applying `pd.get_dummies()`. This ensures the one-hot encoded feature columns match exactly what the model was trained on — every category seen during training will appear as a column, even if the user's current selection doesn't include it (those become 0). Without this step, the resulting feature matrix would have different columns than what the model expects, causing a prediction error.

---

## 📁 Project Structure

```
heart-condition-checker-main/
│
├── app.py                          # Main Streamlit application (158 lines)
├── Pipfile                         # Pipenv dependency manager configuration
├── Pipfile.lock                    # Exact locked versions of all dependencies
├── LICENSE                         # MIT License
├── README.md                       # This documentation file
├── MODEL_COMPARISON.md             # Comparison of ML algorithms on this dataset
├── .gitignore                      # Git ignore rules
├── .gitattributes                  # Git line-ending normalization config
│
├── data/
│   └── heart_2020_cleaned.csv      # Cleaned CDC BRFSS 2020 survey dataset
│                                     (319,796 rows x 18 columns, ~33 MB)
│
├── model/
│   └── logistic_regression.pkl     # Pre-trained serialized logistic regression model
│                                     (2.3 KB, scikit-learn 1.0.2)
│
├── images/
│   ├── doctor.png                  # Doctor illustration for the main page header
│   ├── heart-sidebar.png           # Heart icon displayed in the sidebar
│   ├── heart-fav.png               # Browser favicon (16x16 PNG)
│   ├── heart-okay.jpg              # Result image shown when prediction = healthy
│   └── heart-bad.jpg               # Result image shown when prediction = at risk
│
└── .idea/                          # JetBrains PyCharm IDE configuration (not app code)
    ├── heart-disease-prediction-app.iml
    ├── misc.xml
    ├── modules.xml
    ├── vcs.xml
    └── inspectionProfiles/
```

---

## 🔍 File-by-File Breakdown

### `app.py` — The Heart of the Application

This is the single-file Streamlit application containing all logic. It has 158 lines and is organized into the `main()` function with two inner helper functions.

#### Constants (Lines 8-9)

```python
DATASET_PATH = "data/heart_2020_cleaned.csv"
LOG_MODEL_PATH = "model/logistic_regression.pkl"
```

Relative paths to the data and model files. These are resolved relative to where `streamlit run` is invoked.

#### `load_dataset()` (Lines 13-20)

```python
@st.cache(persist=True)
def load_dataset() -> pd.DataFrame:
    heart_df = pl.read_csv(DATASET_PATH)
    heart_df = heart_df.to_pandas()
    heart_df = pd.DataFrame(np.sort(heart_df.values, axis=0),
                            index=heart_df.index,
                            columns=heart_df.columns)
    return heart_df
```

- **`@st.cache(persist=True)`**: Caches the result to disk. The 33 MB CSV is loaded only once per session and persisted across browser refreshes, preventing repeated file I/O.
- **`pl.read_csv()`**: Uses Polars (not Pandas) for initial CSV reading. Polars uses multi-threaded SIMD-accelerated parsing, typically 5-15x faster than Pandas for this file size.
- **`.to_pandas()`**: Converts the Polars DataFrame to Pandas for downstream compatibility.
- **`np.sort(..., axis=0)`**: Sorts each column independently in ascending order. This ensures consistent ordering of unique categorical values when `pd.get_dummies()` is applied, making the one-hot encoded columns deterministic and matching training-time column ordering.

#### `user_input_features()` (Lines 23-73)

Renders all sidebar widgets and assembles user selections into a single-row DataFrame.

| Widget Type | Features | How Options Are Determined |
|-------------|----------|---------------------------|
| `st.sidebar.selectbox()` | Race, Sex, AgeCategory, BMICategory, GenHealth, Diabetic | Dynamically from `heart.{Column}.unique()` |
| `st.sidebar.selectbox()` | Smoking, AlcoholDrinking, Stroke, DiffWalking, PhysicalActivity, Asthma, KidneyDisease, SkinCancer | Hardcoded `("No", "Yes")` |
| `st.sidebar.number_input()` | SleepTime (0-24, default 7), PhysicalHealth (0-30, default 0), MentalHealth (0-30, default 0) | Fixed min/max ranges |

Returns: `pd.DataFrame` with exactly 1 row and 17 columns (excluding `HeartDisease`).

#### Main UI and Prediction Logic (Lines 76-153)

| Lines | Code Section | Purpose |
|-------|-------------|---------|
| 76-79 | `st.set_page_config()` | Sets browser tab title to "Heart Disease Prediction App" and sets favicon |
| 81-83 | `st.title()`, `st.subheader()` | Renders page title and subtitle |
| 85-115 | `st.columns([1, 3])` | Two-column layout: doctor image + predict button (left), app description (right) |
| 117 | `heart = load_dataset()` | Triggers cached dataset load |
| 119-122 | Sidebar setup | Sets sidebar title, image, and calls `user_input_features()` |
| 123-135 | Feature encoding | Concatenates user input with dataset, drops target, one-hot encodes categoricals, takes row [0], fills NaN |
| 137 | `pickle.load()` | Deserializes the trained model from disk |
| 139-153 | Prediction display | Runs inference, formats probability, displays appropriate image and message |

---

### `data/heart_2020_cleaned.csv` — The Training Dataset

| Property | Value |
|----------|-------|
| Source | CDC BRFSS 2020 Annual Survey |
| Rows | 319,796 (+ 1 header row) |
| Columns | 18 (17 features + 1 target) |
| File size | ~33 MB |
| Target | `HeartDisease` (Yes/No) |
| Class balance | ~9% Yes, ~91% No |
| Data format | UTF-8 CSV, mixed categorical/float |

#### Column Reference

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `HeartDisease` | Categorical — Target | Coronary heart disease or myocardial infarction | `Yes`, `No` |
| `BMICategory` | Categorical (4 levels) | WHO BMI classification | `Underweight (BMI < 18.5)`, `Normal weight (18.5 <= BMI < 25.0)`, `Overweight (25.0 <= BMI < 30.0)`, `Obese (30.0 <= BMI < +Inf)` |
| `Smoking` | Categorical (Binary) | Smoked 100+ cigarettes in lifetime (~5 packs) | `Yes`, `No` |
| `AlcoholDrinking` | Categorical (Binary) | Heavy drinker (>14/week men, >7/week women) | `Yes`, `No` |
| `Stroke` | Categorical (Binary) | Ever told had a stroke | `Yes`, `No` |
| `PhysicalHealth` | Numerical (Float) | Days in past 30 physical health was not good | `0.0` to `30.0` |
| `MentalHealth` | Numerical (Float) | Days in past 30 mental health was not good | `0.0` to `30.0` |
| `DiffWalking` | Categorical (Binary) | Serious difficulty walking or climbing stairs | `Yes`, `No` |
| `Sex` | Categorical (Binary) | Reported biological sex | `Male`, `Female` |
| `AgeCategory` | Categorical (Ordinal, 13 levels) | Age group in 5-year bands | `18-24`, `25-29`, ..., `75-79`, `80 or older` |
| `Race` | Categorical (6 levels) | Racial/ethnic self-identification | `White`, `Black`, `Hispanic`, `Asian`, `American Indian/Alaskan Native`, `Other` |
| `Diabetic` | Categorical (4 levels) | Diabetes diagnosis status | `Yes`, `No`, `No, borderline diabetes`, `Yes (during pregnancy)` |
| `PhysicalActivity` | Categorical (Binary) | Exercised in past 30 days outside of work | `Yes`, `No` |
| `GenHealth` | Categorical (Ordinal, 5 levels) | Self-reported overall health | `Excellent`, `Very good`, `Good`, `Fair`, `Poor` |
| `SleepTime` | Numerical (Float) | Average hours of sleep per 24-hour period | `0.0` to `24.0` |
| `Asthma` | Categorical (Binary) | Ever told you had asthma | `Yes`, `No` |
| `KidneyDisease` | Categorical (Binary) | Kidney disease (excluding kidney stones) | `Yes`, `No` |
| `SkinCancer` | Categorical (Binary) | Ever told you had skin cancer | `Yes`, `No` |

---

### `model/logistic_regression.pkl` — The Serialized ML Model

| Property | Value |
|----------|-------|
| File size | 2,303 bytes (~2.3 KB) |
| Format | Python pickle (binary) |
| Algorithm | `sklearn.linear_model.LogisticRegression` |
| scikit-learn version | 1.0.2 (must match exactly) |
| Training strategy | Random undersampling (50/50 class balance) |
| Feature encoding | One-hot encoding of 14 categorical features |
| Serialization | `pickle.dump(model, open(path, "wb"))` |
| Deserialization | `pickle.load(open(path, "rb"))` |

**Why is the model only 2.3 KB?** Logistic Regression is a linear model. Its serialized form only needs to store the learned weights (one per feature) and the intercept — no complex tree structures or neural network weights. The number of weights equals the total number of one-hot encoded columns.

---

### `Pipfile` and `Pipfile.lock`

The `Pipfile` defines the project's dependencies at a high level:

```toml
[packages]
streamlit = "*"           # Web UI (latest at install time)
pandas = "*"              # Data manipulation
polars = "*"              # Fast CSV reading
sklearn = "*"             # Alias package
scikit-learn = "==1.0.2"  # PINNED for pickle compatibility
```

The `Pipfile.lock` records the exact resolved versions of every package (including transitive dependencies) to ensure reproducible environments. It is 77 KB because it includes SHA hashes for all ~100+ resolved packages.

> **Critical**: `scikit-learn` is pinned to `1.0.2` because the `.pkl` file was created with this version. Loading a pickle in a different scikit-learn version can raise errors or produce incorrect results.

---

### `images/` — UI Visual Assets

| File | Size | Where Used | Condition |
|------|------|-----------|-----------|
| `doctor.png` | 66 KB | Main page, left column | Always shown |
| `heart-sidebar.png` | 44 KB | Sidebar header | Always shown |
| `heart-fav.png` | 1.6 KB | Browser tab favicon | Page load |
| `heart-okay.jpg` | 23 KB | Result section | Only when `prediction == 0` |
| `heart-bad.jpg` | 17 KB | Result section | Only when `prediction == 1` |

---

### `.idea/` — PyCharm IDE Configuration

These files are generated by JetBrains PyCharm and have no effect on application behavior:

- **`heart-disease-prediction-app.iml`**: Defines the module, links to the `Pipenv (heart-condition-checker)` Python SDK, and configures unused package removal.
- **`misc.xml`**: Stores the project-level Python SDK reference.
- **`modules.xml`**: Lists registered modules for the project.
- **`vcs.xml`**: Configures Git as the VCS provider for the project root.

---

## 📊 Dataset Deep Dive

### Source & Collection Method

The **CDC Behavioral Risk Factor Surveillance System (BRFSS)** is the world's largest continuous health survey system. In 2020, over 400,000 U.S. residents were interviewed by telephone. The survey covers:
- Chronic health conditions
- Health-related risk behaviors
- Preventive service usage

The raw 2020 data is provided in SAS format with 200+ variables. For this project, 18 clinically relevant features were extracted.

### Preprocessing Pipeline (Applied Offline)

1. **Format conversion**: SAS (.sas7bdat) → CSV
2. **Feature selection**: 18 of 200+ columns selected based on medical literature on CHD risk factors
3. **BMI calculation**: Raw BMI float → 4 WHO category labels
4. **Age binning**: Continuous age → 13 labeled 5-year bands
5. **Binary label encoding**: Numeric BRFSS codes → human-readable Yes/No text labels
6. **Row filtering**: Removed records with missing values in key columns
7. **Result**: 319,796 clean, complete rows

### Class Imbalance

| Class | Estimated Count | Percentage |
|-------|----------------|------------|
| No (Healthy) | ~291,000 | ~91.0% |
| Yes (Heart Disease) | ~28,000 | ~9.0% |

This ~10:1 imbalance is a classic challenge in medical ML. A naive model that always predicts "No" would achieve 91% accuracy but be completely useless. The solution used here is **random undersampling**: randomly selecting a subset of majority-class ("No") samples equal to the count of minority-class ("Yes") samples, creating a balanced 50/50 training set of ~56,000 records.

---

## 🤖 The Machine Learning Pipeline

### Training Phase (Offline, Before Deployment)

```
CDC BRFSS 2020 raw SAS file (~400k rows, 200+ columns)
        |
        v [Feature selection & cleaning]
heart_2020_cleaned.csv (319,796 rows, 18 columns)
        |
        v [Random undersampling of majority class]
~56,000 balanced samples (50% Yes, 50% No)
        |
        v [Stratified train/test split 80/20]
Training: ~44,800 rows | Test: ~11,200 rows
        |
        v [pd.get_dummies() - One-hot encoding]
Sparse feature matrix (100+ columns)
        |
        v [LogisticRegression(max_iter=1000).fit()]
Trained model with learned coefficients
        |
        v [pickle.dump()]
logistic_regression.pkl (2.3 KB)
```

### Inference Phase (Runtime in app.py)

```
User input (1 row, 17 columns)
        |
        v [pd.concat with full 319,796-row dataset]
319,797 rows combined
        |
        v [Drop HeartDisease column]
319,797 rows, 17 columns
        |
        v [pd.get_dummies() on 14 categorical columns]
Large one-hot encoded matrix (same columns as at train time)
        |
        v [df[:1] — take only user row, fillna(0)]
1-row encoded feature vector
        |
        v [pickle.load() — deserialize model]
LogisticRegression object with learned weights
        |
        v [model.predict(df)]
Binary class: 0 (Healthy) or 1 (Heart Disease)
        |
        v [model.predict_proba(df)[0][1]]
Float probability: e.g., 0.2341 → "23.41% chance"
```

---

## 🎛️ Features Used for Prediction

### Numerical Features (3)

| Feature | Input Widget | Min | Max | Default | Medical Relevance |
|---------|-------------|-----|-----|---------|------------------|
| PhysicalHealth | number_input | 0 | 30 | 0 | Poor physical health is a direct indicator of cardiovascular strain |
| MentalHealth | number_input | 0 | 30 | 0 | Mental health disorders increase cortisol/inflammation, raising CHD risk |
| SleepTime | number_input | 0 | 24 | 7 | Both short (<6h) and long (>9h) sleep are associated with higher CHD risk |

### Binary Yes/No Features (10)

| Feature | Medical Significance |
|---------|---------------------|
| Smoking | Smokers have 2-4x higher CHD risk; damages artery walls and promotes atherosclerosis |
| AlcoholDrinking | Heavy drinking raises blood pressure, causes cardiomyopathy and arrhythmias |
| Stroke | A prior stroke indicates existing cardiovascular disease and shared vascular risk |
| DiffWalking | Indicator of reduced cardiovascular fitness, peripheral artery disease, or heart failure |
| PhysicalActivity | Sedentary lifestyle: 35% higher CHD risk; exercise strengthens the myocardium |
| Asthma | Systemic inflammation from asthma is linked to increased atherosclerosis risk |
| KidneyDisease | Shares risk factors with CVD (hypertension, diabetes); renal failure strains the heart |
| SkinCancer | Associated with chronic UV exposure; UV may modulate cardiovascular risk via vitamin D |
| Sex | Men: higher CHD risk before 65; women: estrogen is cardioprotective until menopause |
| Diabetic | Diabetes is a top-5 CHD risk factor; glucose damages arterial endothelium |

### Multi-Category Features (4)

| Feature | Levels | Description |
|---------|--------|-------------|
| BMICategory | 4 | Underweight / Normal / Overweight / Obese |
| AgeCategory | 13 | 18-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59, 60-64, 65-69, 70-74, 75-79, 80+ |
| Race | 6 | White, Black, Hispanic, Asian, American Indian/Alaskan Native, Other |
| GenHealth | 5 | Excellent, Very good, Good, Fair, Poor |

---

## 🛠️ Technologies & Dependencies

| Technology | Version | Role in Project |
|-----------|---------|----------------|
| **Python** | 3.9 | Runtime language |
| **Streamlit** | 1.5.1 | Web UI — renders sidebar, layout, images, and result |
| **scikit-learn** | 1.0.2 | Logistic Regression training and inference (PINNED) |
| **Pandas** | 1.4.1 | DataFrame operations, `pd.concat`, `pd.get_dummies`, `fillna` |
| **Polars** | 0.13.0 | Fast CSV reading (`pl.read_csv` is 5-15x faster than `pd.read_csv`) |
| **NumPy** | 1.22.2 | `np.sort` for column-wise value sorting |
| **Pickle** | stdlib | Model serialization/deserialization (`pickle.load`) |

---

## 🚀 Installation & Running Locally

### Prerequisites
- Python 3.9+
- pip or pipenv installed

### Using pipenv (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/NischalPachauri/Heart-Disease-Prediction-Using-Logistic-Regression.git
cd Heart-Disease-Prediction-Using-Logistic-Regression

# 2. Install pipenv
pip install pipenv

# 3. Create virtual env and install exact dependency versions
pipenv install

# 4. Activate virtual environment
pipenv shell

# 5. Run the Streamlit app
streamlit run app.py
```

### Using pip

```bash
# Install with exact compatible versions
pip install streamlit==1.5.1 pandas==1.4.1 polars==0.13.0 scikit-learn==1.0.2 numpy==1.22.2

# Run the app
streamlit run app.py
```

### Specifying a Custom Port

```bash
streamlit run app.py --server.port 8889
```

### Verify Streamlit Installation

```bash
streamlit hello
```

The example app should open in your default browser. If it does, the installation was successful.

---

## 📈 Model Performance

The Logistic Regression model, trained on the undersampled balanced dataset:

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | ~80% | Proportion of all predictions that are correct |
| **Precision** | ~75-80% | Of those predicted as heart disease, what fraction actually had it |
| **Recall (Sensitivity)** | ~75-80% | Of actual heart disease cases, what fraction were correctly flagged |
| **F1-Score** | ~77-80% | Harmonic mean of precision and recall |
| **AUC-ROC** | ~0.83-0.87 | Model's ability to discriminate between classes |
| **Specificity** | ~75-80% | Of healthy patients, what fraction were correctly identified |

> Note: Exact values depend on the random seed used during undersampling and train/test split. These are estimates based on the project README. Full evaluation details are in the companion analysis repository.

### Confusion Matrix (Estimated)

```
                 Predicted No    Predicted Yes
Actual No           TN ~4,200     FP ~1,400
Actual Yes          FN ~900       TP ~4,600
```
(Based on ~11,200 test samples at 50/50 class balance)

### What "80% Accuracy" Means in Context

With a balanced test set (50/50 split), 80% accuracy means the model correctly classifies 8 out of 10 patients. This is meaningful because it is evaluated on a balanced set — not the 91/9 real-world distribution. On real-world data, because the model was trained on balanced data, it may show lower accuracy but better recall for the minority (heart disease) class.

---

## 🔬 Model Comparison & Alternative Approaches

For a detailed comparison of Logistic Regression vs. Decision Tree, Random Forest, XGBoost, SVM, and Neural Networks on this dataset, see [MODEL_COMPARISON.md](./MODEL_COMPARISON.md).

---

## ⚠️ Limitations & Disclaimers

1. **Not a medical device**: For educational demonstration only. Never use for clinical decisions.
2. **Undersampling calibration bias**: Training on 50/50 balanced data means predicted probabilities are not calibrated to the real-world 9% prevalence.
3. **Self-reported data**: BRFSS data is collected via telephone surveys and is subject to recall bias and social desirability bias.
4. **2020 snapshot**: Health patterns may have shifted post-pandemic; the model does not account for these changes.
5. **No clinical biomarkers**: Cholesterol levels, blood pressure measurements, ECG data, and troponin levels are not in this dataset.
6. **Version lock**: The `.pkl` model file must be loaded with scikit-learn 1.0.2 exactly. Other versions may cause errors.
7. **Deprecated Streamlit API**: `@st.cache` was deprecated in Streamlit 1.18+. Use `@st.cache_data` for newer versions.
8. **Sorting artifact**: The `np.sort(heart_df.values, axis=0)` in `load_dataset()` sorts values column-by-column independently, breaking any row-level data relationships. This works here because the dataset is only used to enumerate unique category values for widget dropdowns and for encoding consistency — not for individual record analysis.

---

## 📄 License

MIT License — see [LICENSE](./LICENSE) for details.
