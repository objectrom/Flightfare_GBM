# ✈️ Flight Fare Prediction with Gradient Boosting Models

> Predicting airline base fares using Expedia flight data and ensemble machine learning models (XGBoost, LightGBM, CatBoost, Stacking)

---

## 👥 Team

| 2019202076 | 2021610006 | 2021204088 | 2021204016 | 2021204042 | 2022204048 |
|:---:|:---:|:---:|:---:|:---:|:---:|

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Pipeline](#pipeline)
4. [EDA & Feature Engineering](#eda--feature-engineering)
5. [Model Training & Comparison](#model-training--comparison)
6. [XAI (SHAP Analysis)](#xai-shap-analysis)
7. [Stacking Ensemble](#stacking-ensemble)
8. [Real Sample Test](#real-sample-test)
9. [Conclusion](#conclusion)

---

## 📌 Project Overview

Airfare pricing is influenced by dozens of variables — departure time, route distance, cabin class, remaining seats, oil prices, and macroeconomic indicators. This project uses **real-world Expedia booking data** to train and compare multiple GBM-based models for predicting **base fare (baseFare)**.

### 🗓️ Project Timeline

| Period | Task |
|--------|------|
| May 11 – May 15 | Data collection & exploration |
| May 16 – May 18 | Exploratory Data Analysis (EDA) |
| May 19 – May 31 | Feature engineering & preprocessing |
| Jun 01 – Jun 05 | Model training, evaluation & XAI |

---

## 📂 Dataset

- **Source**: [Kaggle — Expedia Flight Prices](https://www.kaggle.com/datasets/dilwong/flightprices)
- **Search date range**: Apr 16, 2022 – Oct 5, 2022 (16 major U.S. routes on Expedia)
- **Travel date range**: Apr 17, 2022 – Nov 19, 2022
- **Original size**: ~8.2M rows → **1% random sampling → 821,388 rows**
- **Features**: 27 original → **39 after engineering**

### External Data Sources

| Data | Source | Join Key |
|------|--------|----------|
| Weather (temp, humidity, wind, etc.) | Visual Crossing API | flightDate × airport |
| CPI, PPI, Unemployment Rate | FRED API | searchDate (monthly) |
| Crude Oil Price (WTI) | Business Insider | searchDate |
| Events (holidays & major events 2022) | Manual research | flightDate |

---

## 🔄 Pipeline

```
Raw Data (Kaggle)
    ↓
1% Random Sampling (821,388 rows)
    ↓
EDA & Feature Engineering (27 → 39 features)
    ↓
Preprocessing (Box-Cox transform, Target Encoding, One-Hot Encoding)
    ↓
Model Training (Linear Regression / RF / XGBoost / AdaBoost / LightGBM / CatBoost)
    ↓
Evaluation (MSE / R² / MAPE) with K-Fold Cross Validation
    ↓
XAI via SHAP
    ↓
Stacking Ensemble (Meta Model: XGBoost)
    ↓
Real Sample Test on Unseen Data
```

---

## 🔍 EDA & Feature Engineering

### Engineered Features

| Feature | Description | Source Column |
|---------|-------------|---------------|
| `departure_hour` | Departure hour (int) | `segmentsDepartureTimeRaw` |
| `hour_bin` | Time-of-day bucket (0–6 / 6–12 / 12–18 / 18–24), one-hot | `departure_hour` |
| `days_before_departure` | Days between search date and flight date | `searchDate`, `flightDate` |
| `flightMonth` | Month of departure | `flightDate` |
| `flightDayType_weekend` | Weekend flag (0/1) | `flightDate` |
| `cabin_score` | Cabin class score (basic=0 ~ first=4), target encoded | `segmentsCabinCode` |
| `stop_count` | Number of stopovers (pipe-delimited count) | `segmentsCabinCode` |
| `travelDurationMinutes` | Flight duration in minutes (from ISO 8601) | `travelDuration` |
| `airline_score` | Airline quality score | `segmentsAirlineName` |
| `totalTravelDistance` | Total route distance (computed from airport coordinates) | Airport coordinate file |

> ⚠️ `fare_per_minute` (baseFare / travelDurationMinutes) was excluded due to **target leakage**

### Target Variable Transformation

- **Target**: `baseFare` (base ticket price)
- Right-skewed distribution → **Box-Cox transformation** applied (λ ≈ 0, effectively log transform)
- Predictions are inverse-transformed back to the original scale after inference

### Outlier Removal

- IQR-based filtering applied
- 821,388 rows → **769,423 rows** after cleaning (51,965 removed)

---

## 📊 Model Training & Comparison

> **Metrics**: MSE, R², MAPE  
> **Validation**: 3–5 Fold Cross Validation  
> **Split**: Train 70% / Val 10% / Test 20%

### Overall Performance (Test Set)

| Model | Test MSE | Test R² | Test MAPE |
|-------|----------|---------|-----------|
| Linear Regression | 15,801.08 | 0.51 | 65.3% |
| Random Forest | 9,362.71 | 0.71 | 44.87% |
| XGBoost | 9,021.58 | 0.7168 | 43.46% |
| AdaBoost | 15,761.95 | 0.5271 | 58.37% |
| LightGBM | 9,216.53 | 0.71 | 36.65% |
| **CatBoost** | **8,367.27** | **0.7411** | **32.44%** |

### Key Hyperparameters

<details>
<summary>Random Forest</summary>

```
n_estimators=500, max_depth=30
oob_score=True, min_samples_split=2
random_state=42, n_jobs=-1
Grid Search | 3-Fold CV: R² = 0.7245 ± 0.0142
```
</details>

<details>
<summary>XGBoost</summary>

```
subsample=0.97, reg_lambda=1.27, reg_alpha=0.04
max_depth=8, learning_rate=0.17, gamma=0.26
colsample_bytree=0.72, min_child_weight=2
Early stopping | RandomizedSearchCV
5-Fold CV: R² = 0.7509 ± 0.0039
```
</details>

<details>
<summary>LightGBM</summary>

```
learning_rate=0.2, max_depth=10, reg_lambda=7
subsample=0.8, colsample_bytree=0.8
Early stopping (criterion: RMSE, patience=1000)
RandomizedSearchCV | 5-Fold CV: R² = 0.7833 ± 0.0024
```
</details>

<details>
<summary>CatBoost</summary>

```
learning_rate=0.2, iterations=3000, depth=10
l2_leaf_reg=3, bootstrap_type=Bernoulli
Ordered Target Encoding + Ordered Boosting
RandomizedSearchCV | 5-Fold CV: R² = 0.7550 ± 0.0045
```
</details>

---

## 🧠 XAI (SHAP Analysis)

SHAP (SHapley Additive exPlanations) was applied to all models for both global-level and instance-level interpretability.

### Top Features by SHAP Importance (across models)

| Feature | Direction of Effect |
|---------|---------------------|
| `totalTravelDistance` | ↑ Longer distance → Higher fare |
| `isBasicEconomy` | ↑ Basic economy class → Lower fare |
| `airline_score` | ↑ Higher-rated airline → Higher fare |
| `stop_count` | More stopovers → Fare varies |
| `days_before_departure` | Booking timing significantly affects fare |

---

## 🔗 Stacking Ensemble

A 2-layer heterogeneous stacking setup was used:

| Layer | Models |
|-------|--------|
| Base Models | Linear Regression, Random Forest, LightGBM, CatBoost |
| Meta Model | **XGBoost** |

Base model predictions are used as input features for the meta model, tuned separately via RandomizedSearchCV.

### Stacking Performance

| Model | MSE | R² | MAPE |
|-------|-----|-----|------|
| Linear Regression | 15,801.31 | 0.5111 | 65.30% |
| Random Forest | 10,395.75 | 0.6784 | 44.44% |
| LightGBM | 9,397.45 | 0.7093 | 38.30% |
| CatBoost | 9,402.48 | 0.7091 | 38.40% |
| **Stacking (XGBoost Meta)** | **8,460.21** | **0.7383** | **36.85%** |

---

## 🧪 Real Sample Test

### Test Setup

- **Test file**: `test_1000_final.csv` (1,000 randomly sampled holdout rows)
- **Goal**: Evaluate generalization on completely unseen data
- **Note**: Real-time fare retrieval via AviationStack / Kiwi.com API was attempted but access was restricted; a holdout sample was used instead

### CPI Domain Shift Correction

Training data is from 2022 (CPI ≈ 292.66), while inference targets 2025 conditions (CPI ≈ 321.47).  
Correction applied: `predicted_fare × (2025 CPI / 2022 CPI)`

| Model | MSE | R² | MAPE |
|-------|-----|-----|------|
| CatBoost | 65,239.74 | 0.7408 | 24.71% |
| LightGBM | 86,353.77 | 0.6570 | 42.27% |

---

## 💡 Conclusion

### Best Model: **CatBoost**

CatBoost achieved the best single-model performance — lowest MSE (8,367.27) and MAPE (32.44%). Its **Ordered Boosting** and **Ordered Target Encoding** effectively prevent data leakage and overfitting, making it well-suited for structured tabular data with categorical features.

### Key Takeaways

| Area | Summary |
|------|---------|
| Strengths | Rich external feature fusion (weather, macro indicators, events) boosted predictive power |
| Limitations | MAPE remained elevated despite MSE gains; real-time API access was limited |
| Learnings | CatBoost's ordered boosting mechanism, SHAP-based interpretation, Box-Cox transform effects |

### Limitations & Future Work

- Domain shift between 2022 training data and real-world inference requires more robust adaptation
- Unseen data showed distributional differences in several features (modality gap)
- More sophisticated domain adaptation or time-aware modeling could improve real-sample accuracy

---

## 📁 Repository Structure

```
Flightfare_GBM/
├── data/
│   ├── Final_cleaned_data.csv      # Preprocessed dataset (821,388 rows × 40 cols)
│   ├── real_sample_test.csv        # Real-world test sample
│   └── test_1000_final.csv         # Holdout test set (1,000 rows)
├── notebooks/
│   ├── EDA.ipynb
│   ├── Preprocessing.ipynb
│   ├── Model_LinearRegression.ipynb
│   ├── Model_RandomForest.ipynb
│   ├── Model_XGBoost_AdaBoost.ipynb
│   ├── Model_LightGBM.ipynb
│   ├── Model_CatBoost.ipynb
│   ├── Model_Stacking.ipynb
│   └── XAI_SHAP.ipynb
└── README.md
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4)
![LightGBM](https://img.shields.io/badge/LightGBM-2ECC71)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCD00?logoColor=black)
![SHAP](https://img.shields.io/badge/SHAP-XAI-FF6B6B)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle&logoColor=white)

---

*This project is based on 2022 Expedia flight data. Domain shift correction is recommended for any inference beyond that time period.*
