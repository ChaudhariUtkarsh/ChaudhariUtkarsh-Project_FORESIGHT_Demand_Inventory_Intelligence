# Project FORESIGHT — Demand & Inventory Intelligence

## 1. Project Problem

Project FORESIGHT is a demand forecasting and inventory intelligence system designed to help businesses make better inventory decisions.

The system uses historical sales, product, calendar, pricing, and inventory data to:

* Forecast future product demand.
* Identify stockout and overstock risks at SKU level.
* Estimate Sales at Risk.
* Estimate Capital Locked in inventory.
* Provide actionable inventory recommendations.
* Support decision-making through a Streamlit dashboard.
* Provide forecasting/scoring functionality through a FastAPI service.

---

## 2. Dataset

The project uses the following datasets:

```text
data/raw/
├── clean/
│   ├── store_master.csv
│   ├── sku_master.csv
│   ├── customer_master.csv
│   ├── promotions.csv
│   ├── inventory_snapshot.csv
│   └── sales_transactions.csv
│
└── anomalies/
    ├── store_master.csv
    ├── sku_master.csv
    ├── customer_master.csv
    ├── promotions.csv
    ├── inventory_snapshot.csv
    ├── sales_transactions.csv
    └── sku_inventory_flags.csv
```

### Dataset Description

```text
|        Dataset           |               Purpose                          |
|--------------------------|------------------------------------------------|
| `sales_transactions.csv` | Historical transaction-level sales             |
| `sku_master.csv`         | SKU/product information, pricing and brand     |
| `store_master.csv`       | Store information and location                 |
| `customer_master.csv`    | Customer and loyalty information               |
| `promotions.csv`         | Promotion and discount information             |
| `inventory_snapshot.csv` | Inventory availability and reorder information |
| `sku_inventory_flags.csv`| Ground-truth inventory anomaly labels          |
```

### Processed Data

After preprocessing and feature engineering:

```text
data/processed/processed_data.csv
```

The processed dataset is used as the source for weekly SKU-level model preparation, risk analysis and forecasting.

The training pipeline additionally creates:

```text
data/processed/weekly_model_data.csv
```

This dataset has one row per SKU per week and uses `weekly_units_sold` as the forecasting target.

---

## 3. Installation

### Step 1 — Clone Repository

```bash
git clone https://github.com/ChaudhariUtkarsh/ChaudhariUtkarsh-Project_FORESIGHT_Demand_Inventory_Intelligence.git
cd Project_FORESIGHT_Demand_Inventory_Intelligence
```

### Step 2 — Create Virtual Environment

Windows:

```powershell
python -m venv .venv
```

Activate:

```powershell
.venv\Scripts\activate
```

### Step 3 — Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## 4. Run Commands

### Data Preprocessing and Model Training

Run:

```powershell
python src\train_model.py
```

This performs:

1. Builds a clean SKU/day table from the processed extract.
2. Aggregates daily `units_sold` to weekly SKU-level demand.
3. Creates leakage-safe weekly lag and rolling features.
4. Builds the seasonal-naive weekly baseline.
5. Runs rolling-origin cross-validation.
6. Trains XGBoost and LightGBM.
7. Compares model WAPE against the baseline.
8. Selects the lowest-CV-WAPE production model (or the baseline if ML does not win).
9. Calculates an 80% forecast interval.
10. Saves the weekly dataset, model artifacts, metadata and metrics.

---

### Forecast Check

Run:

```powershell
python src\predict.py
```

The output includes:

```text
Predicted Demand
80% Lower Bound
80% Upper Bound
80% Prediction Interval
```

---

### Risk Analysis

Run:

```powershell
python src\risk_analysis.py
```

The risk analysis calculates:

* Stockout Risk Score
* Overstock Risk Score
* Risk Level
* Primary Risk
* Recommended Action
* Sales at Risk
* Capital Locked

Output:

```text
data/risk_analysis/sku_risk_analysis.csv
```

---

### Business Insights

Run:

```powershell
python src\business_insights.py
```

This generates:

* Top 10 risky SKUs
* Total Sales at Risk
* Total Capital Locked
* Risk distribution

Output:

```text
data/business_insights/business_insights_summary.csv
```

---

### Decisioning Grid

Run:

```powershell
python src\decisioning_grid.py
```

The decisioning grid classifies SKUs into:

* Reorder Now
* Markdown / Clear
* Watch / Volatile
* Healthy

Output:

```text
data/decisioning_grid/decisioning_grid.csv
```

---

## 5. Model

Project FORESIGHT now forecasts at **weekly SKU level**, aligned with the Zidio engagement brief.

### Forecast grain

```text
Daily units_sold
        ↓
SKU + Week aggregation
        ↓
weekly_units_sold
        ↓
Weekly lag / rolling features
        ↓
XGBoost / LightGBM
```

### Forecast horizon

The dashboard and predictor support a defined **6–8 week** forecast horizon.

### Baseline

The project uses a **weekly seasonal-naive baseline with a 4-week seasonal lag**. The supplied dataset contains approximately one year of history, so a 52-week annual lag does not provide enough prior-year observations for a fair rolling backtest.

### Models

* XGBoost (`XGBRegressor`)
* LightGBM (`LGBMRegressor`)

Models are evaluated using **rolling-origin cross-validation**. Features are constructed from prior observations so future target information does not enter the model features.

---

## 6. WAPE vs Baseline

The current weekly backtest results generated from the supplied project data are:

| Model | Rolling-Origin CV WAPE |
| --- | ---: |
| Seasonal Naive Baseline | **53.60%** |
| XGBoost | **40.48%** |
| LightGBM | **40.40%** |

### Selected model

```text
Best model: LightGBM
```

LightGBM improves WAPE by **13.20 percentage points** versus the seasonal-naive baseline on rolling-origin cross-validation.

These values are saved in:

```text
models/model_metrics.json
```

The model metadata is saved in:

```text
models/model_metadata.json
```

---

## 7. Forecast Uncertainty

Project FORESIGHT provides an **80% forecast uncertainty interval** along with the predicted demand.

The forecast output contains:

```text
Predicted Demand
80% Lower Bound
80% Upper Bound
80% Prediction Interval
```

Example:

```text
Predicted Demand        : 29.12
80% Lower Bound         : 19.89
80% Upper Bound         : 38.35
80% Prediction Interval : 19.89 - 38.35
```

The uncertainty interval helps decision-makers understand the expected range around the forecast instead of relying only on a single predicted value.

---

## 8. Risk Analysis

Risk scoring is performed at SKU level.

### Stockout Risk

Days of Supply is used to calculate Stockout Risk Score.

```text
DOS <= 3 days       → 1.00
DOS <= 7 days       → 0.75
DOS <= 14 days      → 0.40
DOS > 14 days       → 0.10
```

### Overstock Risk

```text
DOS >= 90 days      → 1.00
DOS >= 60 days      → 0.75
DOS >= 30 days      → 0.40
DOS < 30 days       → 0.10
```

### Business Metrics

```text
Capital Locked
= Current Inventory × Average Unit Price
```

```text
Sales at Risk
= Average Daily Demand
× Average Unit Price
× 7
× Stockout Risk Score
```

---

## 9. Decisioning Grid

The dashboard uses four decisioning quadrants:

| Decision Quadrant | Purpose                              |
| ----------------- | ------------------------------------ |
| Reorder Now       | Immediate replenishment required     |
| Markdown / Clear  | Reduce excess inventory              |
| Watch / Volatile  | Monitor demand and inventory closely |
| Healthy           | Maintain current inventory strategy  |

This converts SKU-level risk scores into actionable business decisions.

---

## 10. Dashboard

The project provides an interactive Streamlit dashboard containing:

* Demand Forecast
* Forecast Uncertainty Interval
* SKU Risk Analysis
* ₹ Sales at Risk
* ₹ Capital Locked
* Business Insights
* Risk Distribution
* Decisioning Grid
* Model Evaluation

### Dashboard URL

```text
https://YOUR-DASHBOARD-URL.streamlit.app
```

> Replace the above placeholder with the actual public Streamlit URL after deployment.

---

## 11. API

Project FORESIGHT also provides a FastAPI scoring service.

### API URL

```text
https://YOUR-API-URL
```

> Replace the above placeholder with the actual deployed FastAPI public URL.

### Local API

To run the API locally:

```powershell
uvicorn api:app --host 0.0.0.0 --port 8000
```

Local API:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 12. Key Assumptions

* Forecasting target is `units_sold`.
* Historical sales data is representative of future demand patterns.
* A 4-week seasonal lag is used for the weekly Seasonal Naive baseline.
* Rolling-Origin Cross-Validation is used to evaluate forecasting performance over time.
* Negative demand predictions are constrained to zero.
* Days of Supply is calculated using current inventory and average daily demand.
* ₹ Sales at Risk is estimated over a 7-day exposure period.
* ₹ Capital Locked represents inventory value based on current inventory and average unit price.
* Risk scores are calculated using predefined Days-of-Supply thresholds.
* The 80% uncertainty interval represents an estimated forecast range and should be interpreted as a planning aid rather than a guarantee.

---

## 13. Project Structure

```text
Project_FORESIGHT_Demand_Inventory_Intelligence/
│
├── app/
│   ├── Home.py
│   ├── login.py
│   ├── utils.py
│   ├── style.css
│   └── pages/
│       ├── 1_Dashboard.py
│       ├── 2_Forecast.py
│       ├── 3_Risk_Scoring.py
│       └── 4_About.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── risk_analysis/
│   ├── business_insights/
│   └── decisioning_grid/
│
├── models/
│   ├── best_model.pkl
│   ├── lightgbm_model.pkl
│   ├── xgboost_model.pkl
│   ├── label_encoder.pkl
│   ├── model_metrics.json
│   └── model_metadata.json
│
├── reports/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── baseline.py
│   ├── train_model.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── risk_scoring.py
│   ├── risk_analysis.py
│   ├── business_insights.py
│   └── decisioning_grid.py
│
├── api.py
├── config.py
├── requirements.txt
├── Procfile
└── README.md
```

---

## 14. Deployment

### Streamlit Dashboard

The dashboard is deployed as a public Streamlit application.

```text
Dashboard URL:
https://chaudhariutkarsh-projectforesightdemandinventoryintelligence-b.streamlit.app/
```

### FastAPI Scoring Service

The scoring API is deployed as a public FastAPI service.

```text
API URL:
https://chaudhariutkarsh-project-foresight.onrender.com
```

Both URLs should be publicly accessible without requiring the local development environment.

---

## 15. Conclusion

Project FORESIGHT provides an end-to-end demand forecasting and inventory intelligence solution.

The system combines machine learning forecasting, uncertainty estimation, SKU-level risk scoring, business insights and decisioning recommendations.

The selected LightGBM model achieved a 40.40% Rolling-Origin CV WAPE, compared with 53.60% for the Seasonal Naive baseline, resulting in a 13.20 percentage-point improvement over the baseline.

The solution enables businesses to prioritize high-risk SKUs, monitor ₹ Sales at Risk and ₹ Capital Locked, and make data-driven inventory decisions.
