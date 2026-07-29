# Project FORESIGHT
### Production Level Retail Demand Forecasting using Machine Learning

<<<<<<< HEAD
> Demand & Inventory Intelligence — Forecast future product demand, detect stockout/overstock risks, and support business decision-making through an interactive Streamlit dashboard.
=======
> Demand & Inventory Intelligence — Forecast future product demand, detect stockout/overstock risks, and support business decision-making through an interactive dashboard.
>>>>>>> a588cb5ad198c720718877a237d45bdc53f1a35b

---

## Overview

<<<<<<< HEAD
Project FORESIGHT is an end-to-end retail demand forecasting system that ingests raw sales and inventory data, engineers time-series features, trains gradient boosting models with cross-validation, scores inventory risk per SKU, and surfaces everything through a multi-page Streamlit dashboard.
=======
Project FORESIGHT is an end-to-end retail demand forecasting system that ingests raw sales and inventory data, engineers time-series features, trains gradient boosting models, scores inventory risk per SKU, and surfaces everything through a multi-page Streamlit dashboard.
>>>>>>> a588cb5ad198c720718877a237d45bdc53f1a35b

---

## Project Structure

```
Project_FORESIGHT/
├── app/
│   ├── Home.py                  # Streamlit landing page
│   └── pages/
│       ├── 1_Dashboard.py       # KPI metrics overview
│       ├── 2_Forecast.py        # Demand forecast by SKU
│       ├── 3_Risk_Scoring.py    # Stockout / Overstock risk
│       └── 4_About.py           # Project info
├── data/
│   ├── raw/
│   │   ├── sales_daily.csv
│   │   ├── sku_master.csv
│   │   ├── calendar.csv
│   │   └── inventory_snapshots.csv
│   └── processed/
│       └── processed_data.csv
├── models/
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── best_model.pkl
│   ├── label_encoder.pkl
│   └── model_metrics.json
├── src/
│   ├── data_loader.py           # Loads and validates raw CSVs
│   ├── preprocessing.py         # Cleans, merges, and saves processed data
│   ├── feature_engineering.py   # Builds date, lag, rolling, price, inventory features
<<<<<<< HEAD
│   ├── train_model.py           # Trains models with TimeSeriesSplit + WAPE comparison
=======
│   ├── train_model.py           # Trains XGBoost & LightGBM, saves best model
>>>>>>> a588cb5ad198c720718877a237d45bdc53f1a35b
│   ├── evaluate.py              # MAE, RMSE, R2, MAPE, WAPE metrics
│   └── risk_scoring.py          # Stockout / Overstock risk scoring per SKU
├── images/
│   ├── sales_trend.png
│   ├── monthly_sales.png
│   ├── category_sales.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   └── risk_distribution.png
├── generate_images.py           # Generates all visualization PNGs
├── main.py                      # Project info & structure check
└── requirements.txt
```

---

## ML Pipeline

```
Raw Data → DataLoader → DataPreprocessor → FeatureEngineering → ModelTrainer → RiskScoring
```

| Step | Module | Description |
|---|---|---|
| 1 | `data_loader.py` | Validates and loads 4 raw CSVs |
| 2 | `preprocessing.py` | Removes duplicates, handles nulls, merges all datasets |
<<<<<<< HEAD
| 3 | `feature_engineering.py` | Date, lag (1/7/14), rolling stats, price & inventory features |
| 4 | `train_model.py` | Trains XGBoost & LightGBM with TimeSeriesSplit + WAPE comparison |
=======
| 3 | `feature_engineering.py` | Date features, lag (1/7/14), rolling stats, price & inventory features |
| 4 | `train_model.py` | Trains XGBoost & LightGBM, auto-selects best by RMSE |
>>>>>>> a588cb5ad198c720718877a237d45bdc53f1a35b
| 5 | `evaluate.py` | Computes MAE, RMSE, R² Score, MAPE, WAPE |
| 6 | `risk_scoring.py` | Scores each SKU as Low / Medium / High risk with recommendations |

---

## Features Engineered

<<<<<<< HEAD
| Group | Features |
|---|---|
| Date | year, month, week, day, day_of_week, quarter, is_weekend |
| Lag | lag_1, lag_7, lag_14 |
| Rolling | rolling_mean_7, rolling_std_7, rolling_mean_30 |
| Price | price_difference, discount_percentage |
| Inventory | inventory_gap, total_inventory |

---

## Models & Validation
=======
- **Date** — year, month, week, day, day_of_week, quarter, is_weekend
- **Lag** — lag_1, lag_7, lag_14 (units sold)
- **Rolling** — rolling_mean_7, rolling_std_7, rolling_mean_30
- **Price** — price_difference, discount_percentage
- **Inventory** — inventory_gap, total_inventory

---

## Models
>>>>>>> a588cb5ad198c720718877a237d45bdc53f1a35b

| Model | Library | Estimators | Learning Rate | Max Depth |
|---|---|---|---|---|
| XGBoost | xgboost 2.1.4 | 300 | 0.05 | 6 |
| LightGBM | lightgbm 4.6.0 | 300 | 0.05 | 6 |

<<<<<<< HEAD
**Cross-Validation — TimeSeriesSplit (5 Folds)**
- Respects temporal order — no data leakage from future to past
- Each fold trains on past data and validates on the next time window
- Mean WAPE across all 5 folds is used to select the best model

**Baseline WAPE vs Model WAPE**

After each model trains, a comparison is printed against the Seasonal Naive baseline (lag-7):

```
==================================================
  XGBOOST  —  WAPE Comparison
==================================================
  Baseline WAPE (Seasonal Naive) : 12.45%
  Model WAPE                     : 4.32%
  CV WAPE (TimeSeriesSplit)       : 5.10%
  Improvement                    : 8.13%  [BETTER]
==================================================
```

- Best model is selected by lowest **CV WAPE** and saved as `models/best_model.pkl`
- All metrics are written to `models/model_metrics.json` after every run

---

## Metrics Saved (`model_metrics.json`)

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R2 Score | Coefficient of Determination |
| MAPE (%) | Mean Absolute Percentage Error |
| WAPE (%) | Weighted Absolute Percentage Error |
| CV_WAPE (%) | Mean WAPE across 5 TimeSeriesSplit folds |
| Baseline_WAPE (%) | Seasonal Naive benchmark WAPE |
| WAPE_Improvement | Baseline WAPE − Model WAPE |
=======
- Best model is selected by lowest RMSE and saved as `models/best_model.pkl`
- All metrics are saved to `models/model_metrics.json` after every training run
>>>>>>> a588cb5ad198c720718877a237d45bdc53f1a35b

---

## Risk Scoring

Each SKU receives a `risk_score` (0–100) and a `risk_level`:

| Risk Level | Score Range | Recommendation |
|---|---|---|
| Low | 0 – 39 | Inventory Level is Healthy |
| Medium | 40 – 69 | Run Discount / Promotion |
| High | 70 – 100 | Reorder Inventory Immediately |

Risk report is saved to `outputs/risk_report.csv`.

---

## Dashboard (Streamlit)

| Page | Description |
|---|---|
| Home | Project overview and workflow |
| Dashboard | Total products, stockout & overstock KPIs |
| Forecast | Demand prediction by SKU |
| Risk Scoring | Inventory risk by type (Stockout / Overstock) |
| About | Tech stack and project info |

---

## Setup & Run

**1. Clone the repository:**
```bash
git clone https://github.com/<your-username>/Project_FORESIGHT.git
cd Project_FORESIGHT
```

**2. Create a virtual environment:**

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Run the full training pipeline:**
```bash
cd src
python train_model.py
```

**5. Generate visualizations:**
```bash
cd ..
python generate_images.py
```

**6. Launch the dashboard:**
```bash
streamlit run app/Home.py
```

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data | pandas, numpy |
| ML | scikit-learn, xgboost, lightgbm |
| Forecasting | statsmodels, prophet |
| Visualization | matplotlib, seaborn, plotly |
| Dashboard | streamlit |
| Persistence | joblib |

---

<<<<<<< HEAD
## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

> Replace the link above with your deployed Streamlit app URL.

---

## Project Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAW DATA INPUTS                          │
│   sales_daily.csv  │  sku_master.csv  │  calendar.csv  │        │
│                    inventory_snapshots.csv                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   DataLoader    │  Validates & loads 4 CSVs
                    └────────┬────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   DataPreprocessor   │  Dedup, null handling, merge
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  FeatureEngineering  │  Date, Lag, Rolling,
                  │                      │  Price, Inventory features
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │    ModelTrainer      │
                  │                      │
                  │  ┌────────────────┐  │
                  │  │ XGBoost Model  │  │  TimeSeriesSplit
                  │  └────────────────┘  │  (5 Folds)
                  │  ┌────────────────┐  │
                  │  │ LightGBM Model │  │  Baseline WAPE
                  │  └────────────────┘  │  vs Model WAPE
                  └──────────┬───────────┘
                             │
               ┌─────────────┴──────────────┐
               │                            │
               ▼                            ▼
   ┌───────────────────────┐   ┌────────────────────────┐
   │    best_model.pkl     │   │   model_metrics.json   │
   │  (lowest CV WAPE)     │   │  MAE, RMSE, R2, MAPE,  │
   └───────────┬───────────┘   │  WAPE, CV_WAPE,        │
               │               │  Baseline_WAPE         │
               │               └────────────────────────┘
               │
               ▼
      ┌─────────────────┐
      │   RiskScoring   │  Stockout & Overstock score
      │                 │  per SKU → Low / Medium / High
      └────────┬────────┘
               │
               ▼
      ┌─────────────────┐
      │ risk_report.csv │
      └────────┬────────┘
               │
               ▼
  ┌────────────────────────────┐
  │   Streamlit Dashboard      │
  │                            │
  │  Home │ Dashboard          │
  │  Forecast │ Risk Scoring   │
  │  About                     │
  └────────────────────────────┘
```

---

## Author

Developed by **Utkarsh Chaudhari**
=======
## Author

Developed by **Utkarsh Chaudhari**  
>>>>>>> a588cb5ad198c720718877a237d45bdc53f1a35b
Project Version: `1.0`
