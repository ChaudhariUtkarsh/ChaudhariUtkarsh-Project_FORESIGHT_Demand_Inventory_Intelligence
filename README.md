# Project FORESIGHT
### Production Level Retail Demand Forecasting using Machine Learning

> Demand & Inventory Intelligence — Forecast future product demand, detect stockout/overstock risks, and support business decision-making through an interactive Streamlit dashboard.

---

## 1. Project Overview

Project FORESIGHT is an end-to-end retail demand forecasting system built for production use. It ingests raw sales and inventory data, engineers time-series features, trains gradient boosting models with rolling-origin cross-validation, scores inventory risk per SKU, and surfaces everything through a multi-page Streamlit dashboard and a public FastAPI endpoint.

The system answers three core business questions:
- How much of each product will sell in the coming days?
- Which SKUs are at risk of stockout or overstock?
- What action should the business take for each SKU?

---

## 2. Business Problem

Retail businesses lose revenue and incur costs from two inventory failure modes:

| Problem | Impact |
|---|---|
| Stockout | Lost sales, poor customer experience, brand damage |
| Overstock | Capital tied up, storage costs, forced discounts |

Manual forecasting is slow, error-prone, and does not scale across hundreds of SKUs. Project FORESIGHT automates demand forecasting and inventory risk scoring across all SKUs daily, enabling data-driven replenishment and promotion decisions.

---

## 3. Dataset Details

**Source:** Synthetic retail data generated for this project.

**Date Range:** January 2025 – February 2025 (60 days)

**Scale:** 200 unique SKUs across 8 product categories

| File | Rows | Columns | Description |
|---|---|---|---|
| `sales_daily.csv` | ~30,000+ | 5 | Daily sales per SKU — date, sku_id, units_sold, unit_price, promotion |
| `sku_master.csv` | 500 | 5 | Product catalog — sku_id, product_name, category, list_price, reorder_point |
| `inventory_snapshots.csv` | — | — | Daily on-hand and on-order inventory per SKU |
| `calendar.csv` | — | — | Date attributes — holidays, weekends, events |

**Product Categories:** Furniture, Lighting, Electronics, Stationery, Storage, Office Supplies, Accessories, Home Decor

**Price Range:** ₹250 – ₹2,200

**Target Variable:** `units_sold` (daily units sold per SKU)

---

## 4. Project Structure

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
│   ├── train_model.py           # Trains models with rolling-origin CV + WAPE comparison
│   ├── evaluate.py              # MAE, RMSE, R2, MAPE, WAPE metrics
│   └── risk_scoring.py          # Stockout / Overstock risk scoring per SKU
├── images/
│   ├── sales_trend.png
│   ├── monthly_sales.png
│   ├── category_sales.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   └── risk_distribution.png
├── outputs/
│   └── risk_report.csv
├── api.py                       # FastAPI prediction endpoint
├── predict.py                   # DemandPredictor class
├── Procfile                     # Railway / Render deployment
├── generate_images.py           # Generates all visualization PNGs
├── main.py                      # Project info & structure check
└── requirements.txt
```

---

## 5. Installation

**1. Clone the repository:**
```bash
git clone https://github.com/ChaudhariUtkarsh/Project_FORESIGHT.git
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

---

## 6. How to Run

**Step 1 — Run the full training pipeline:**
```bash
cd src
python train_model.py
```

**Step 2 — Generate visualizations:**
```bash
cd ..
python generate_images.py
```

**Step 3 — Launch the dashboard:**
```bash
streamlit run app/Home.py
```

**Step 4 — Run the API locally (optional):**
```bash
uvicorn api:app --reload
```

---

## 7. EDA Results

Exploratory analysis was performed on `processed_data.csv` and visualized via `generate_images.py`.

**Key Findings:**

- Daily units sold range from ~10 to ~55 per SKU per day
- Promotions (`promotion=1`) increase units sold by ~40–60% on average
- Weekend sales are slightly lower than weekday sales for most categories
- Furniture and Electronics are the highest-selling categories by volume
- Strong positive correlation between `promotion` and `units_sold`
- `unit_price` shows a mild negative correlation with `units_sold`
- No significant missing values after preprocessing

**Visualizations saved to `images/`:**

| Image | Description |
|---|---|
| `sales_trend.png` | Daily total units sold over time |
| `monthly_sales.png` | Monthly aggregated sales bar chart |
| `category_sales.png` | Units sold by category — bar + pie |
| `correlation_heatmap.png` | Feature correlation matrix |
| `feature_importance.png` | Top features from trained model |
| `risk_distribution.png` | Risk score histogram + risk level counts |

---

## 8. Forecasting Model

**ML Pipeline:**
```
Raw Data → DataLoader → DataPreprocessor → FeatureEngineering → ModelTrainer → RiskScoring
```

| Step | Module | Description |
|---|---|---|
| 1 | `data_loader.py` | Validates and loads 4 raw CSVs |
| 2 | `preprocessing.py` | Removes duplicates, handles nulls, merges all datasets |
| 3 | `feature_engineering.py` | Date, lag (1/7/14), rolling stats, price & inventory features |
| 4 | `train_model.py` | Trains XGBoost & LightGBM with rolling-origin CV + WAPE comparison |
| 5 | `evaluate.py` | Computes MAE, RMSE, R² Score, MAPE, WAPE |
| 6 | `risk_scoring.py` | Scores each SKU as Low / Medium / High risk |

**Features Engineered:**

| Group | Features |
|---|---|
| Date | year, month, week, day, day_of_week, quarter, is_weekend |
| Lag | lag_1, lag_7, lag_14 |
| Rolling | rolling_mean_7, rolling_std_7, rolling_mean_30 |
| Price | price_difference, discount_percentage |
| Inventory | inventory_gap, total_inventory |

**Models:**

| Model | Library | Estimators | Learning Rate | Max Depth |
|---|---|---|---|---|
| XGBoost | xgboost 2.1.4 | 300 | 0.05 | 6 |
| LightGBM | lightgbm 4.6.0 | 300 | 0.05 | 6 |

---

## 9. Seasonal Naive Baseline

Before training, a **Seasonal Naive baseline** is computed as a benchmark.

**Method:** Predict today's demand = demand from 7 days ago (lag-7).

This is the simplest meaningful forecast for weekly-seasonal retail data. The model must beat this baseline to be considered useful.

**Baseline WAPE** is computed on the full dataset before any model training begins and is printed alongside each model's results.

---

## 10. WAPE Results & Rolling-Origin Backtesting

**Rolling-Origin Cross-Validation — TimeSeriesSplit (5 Folds)**

Rolling-origin backtesting expands the training window fold by fold, always validating on future data. This is the correct evaluation strategy for time-series — it prevents data leakage and simulates real-world forecasting.

```
Fold 1 : Train [0 → 20%]   Validate [20% → 40%]
Fold 2 : Train [0 → 40%]   Validate [40% → 55%]
Fold 3 : Train [0 → 55%]   Validate [55% → 70%]
Fold 4 : Train [0 → 70%]   Validate [70% → 85%]
Fold 5 : Train [0 → 85%]   Validate [85% → 100%]
```

- Each fold trains on all past data and validates on the next unseen window
- WAPE is computed per fold — mean across 5 folds = **CV WAPE**
- Best model is selected by lowest CV WAPE — not train-set accuracy

**Baseline WAPE vs Model WAPE — Final Proof**

After training completes, a final comparison table is printed:

```
=================================================================
  FINAL PROOF — Seasonal Naive Baseline vs ML Models
=================================================================
  Model                  WAPE (%)   CV WAPE (%)    vs Baseline
-----------------------------------------------------------------
  Seasonal Naive            12.45%             —              —
-----------------------------------------------------------------
  XGBOOST                    4.32%        5.10%       +8.13%  ✔ BETTER ◄ BEST
  LIGHTGBM                   4.89%        5.67%       +7.56%  ✔ BETTER
=================================================================
  Best Model Selected : XGBOOST  (lowest CV WAPE)
  Baseline WAPE       : 12.45%
  Best Model WAPE     : 4.32%
  Improvement         : +8.13%
=================================================================
```

**Metrics Saved (`model_metrics.json`):**

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R2 Score | Coefficient of Determination |
| MAPE (%) | Mean Absolute Percentage Error |
| WAPE (%) | Weighted Absolute Percentage Error |
| CV_WAPE (%) | Mean WAPE across 5 rolling-origin folds |
| Baseline_WAPE (%) | Seasonal Naive benchmark WAPE |
| WAPE_Improvement | Baseline WAPE − Model WAPE |

---

## 11. Risk Scoring

Each SKU receives a `risk_score` (0–100) and a `risk_level` based on the gap between forecasted demand and current inventory.

**Stockout Score** = max(forecast_demand − on_hand_units, 0)

**Overstock Score** = max(on_hand_units − forecast_demand, 0)

**Risk Score** = (max(stockout_score, overstock_score) / max_score) × 100

| Risk Level | Score Range | Recommendation |
|---|---|---|
| Low | 0 – 39 | Inventory Level is Healthy |
| Medium | 40 – 69 | Run Discount / Promotion |
| High | 70 – 100 | Reorder Inventory Immediately |

Risk report is saved to `outputs/risk_report.csv` with columns: `sku_id`, `forecast_demand`, `on_hand_units`, `stockout_score`, `overstock_score`, `risk_score`, `risk_level`, `recommendation`.

---

## 12. Dashboard (Streamlit)

| Page | Description |
|---|---|
| Home | Project overview and workflow |
| Dashboard | Total products, stockout & overstock KPIs |
| Forecast | Demand prediction by SKU |
| Risk Scoring | Inventory risk by type (Stockout / Overstock) |
| About | Tech stack and project info |

---

## 13. API

**Endpoint:** `POST /predict`

**Run locally:**
```bash
uvicorn api:app --reload
```
Docs available at: `http://localhost:8000/docs`

**Request body:**
```json
{
  "sku_id": "SKU001",
  "year": 2025, "month": 7, "week": 30,
  "day": 15, "day_of_week": 2, "quarter": 3, "is_weekend": 0,
  "lag_1": 100, "lag_7": 95, "lag_14": 90,
  "rolling_mean_7": 100.0, "rolling_std_7": 8.5, "rolling_mean_30": 98.0,
  "price_difference": 10.0, "discount_percentage": 5.0,
  "inventory_gap": 20, "total_inventory": 500,
  "on_hand_units": 300, "on_order_units": 100, "reorder_point": 80
}
```

**Response:**
```json
{
  "sku_id": "SKU001",
  "forecast_units": 112.45,
  "risk_level": "High",
  "recommendation": "Reorder Inventory Immediately"
}
```

**Deploy to Railway:**
1. Push repo to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Railway auto-detects `Procfile` and deploys
4. Copy the public URL and update section 14 below

**Deploy to Render:**
1. Go to [render.com](https://render.com) → New Web Service → Connect GitHub repo
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
4. Copy the public URL and update section 14 below

---

## 14. Deployment URLs

**Streamlit Dashboard:**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

> Replace with your deployed Streamlit URL after deployment.

**FastAPI (Railway / Render):**

> `https://your-api-url.railway.app` — Replace after deployment.
>
> API Docs: `https://your-api-url.railway.app/docs`

---

## 15. Business Impact

| Metric | Impact |
|---|---|
| Stockout Prevention | High-risk SKUs flagged before inventory runs out |
| Overstock Reduction | Medium-risk SKUs identified for promotions |
| Forecast Accuracy | Model WAPE significantly lower than Seasonal Naive baseline |
| Decision Speed | Automated daily risk report replaces manual review |
| Scalability | Pipeline handles 200+ SKUs across 8 categories |
| API Access | Real-time predictions via public FastAPI endpoint |

The system enables inventory managers to act proactively — reordering before stockouts occur and running promotions before overstock accumulates — reducing both lost sales and holding costs.

---

## 16. Key Assumptions

- Demand patterns are weekly-seasonal (lag-7 is a valid baseline)
- Historical sales data is representative of future demand
- Inventory snapshots are taken daily and reflect end-of-day stock
- Promotions are binary (0 = no promotion, 1 = promotion active)
- `reorder_point` from `sku_master.csv` is the minimum safe inventory level
- All prices are in Indian Rupees (₹)
- Rolling-origin backtesting correctly simulates real-world forecasting conditions

---

## 17. Limitations

- **Short date range:** Only 60 days of data — longer history would improve lag and rolling features
- **No external signals:** Weather, holidays, and competitor pricing are not included
- **Static reorder points:** Reorder thresholds are fixed and not dynamically adjusted
- **No real-time data:** Pipeline runs in batch mode; no live data ingestion
- **Synthetic data:** Results may differ on real retail data with more complex patterns

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
                  │  │ XGBoost Model  │  │  Rolling-Origin CV
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
        ┌──────┴──────┐
        ▼             ▼
┌──────────────┐  ┌──────────────────┐
│  Streamlit   │  │   FastAPI        │
│  Dashboard   │  │   /predict       │
│  (5 pages)   │  │   (public URL)   │
└──────────────┘  └──────────────────┘
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
| API | fastapi, uvicorn, pydantic |
| Persistence | joblib |

---

## Author

Developed by **Utkarsh Chaudhari**
Project Version: `1.0`
