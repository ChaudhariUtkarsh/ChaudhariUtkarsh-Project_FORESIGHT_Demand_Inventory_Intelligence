# PROJECT FORESIGHT

## Demand Forecasting & Inventory Intelligence

PROJECT FORESIGHT is an end-to-end data science and analytics system for **SKU-level weekly demand forecasting and inventory risk intelligence**.

The system combines data ingestion, cleaning, feature engineering, demand forecasting, model evaluation, inventory risk scoring, reorder prioritization, markdown/clear prioritization, a Streamlit dashboard, and a FastAPI scoring service.

The project is designed around the Zidio Development Project FORESIGHT engagement requirements, which require a reproducible data pipeline, demand forecast, risk scoring, planning dashboard, deployed scoring service, and stakeholder-ready reporting.

---

# 1. Project Overview

### Project Name

**PROJECT FORESIGHT — Demand & Inventory Intelligence**

### Objective

The primary objective is to help inventory and operations teams answer:

* How much will each SKU likely sell over the next few weeks?
* Which SKUs are at risk of stockout?
* Which SKUs are overstocked?
* Which products should be reordered?
* Which products should be promoted, cleared, or monitored?
* Which forecasting model performs best against a seasonal-naive baseline?

The Zidio brief specifically requires weekly SKU-level forecasting, stockout/overstock risk scoring, quantified business impact, a usable dashboard, and a deployed scoring service.

---

# 2. Key Features

PROJECT FORESIGHT provides:

* Data ingestion and preprocessing
* Data cleaning and validation
* Sales, SKU, calendar, and inventory integration
* Weekly SKU-level demand aggregation
* Forecasting feature engineering
* 52-week Seasonal Naive baseline
* XGBoost forecasting model
* LightGBM forecasting model
* Rolling-Origin Cross-Validation
* WAPE-based model evaluation
* Production model selection
* Inventory risk scoring
* Stockout risk identification
* Overstock risk identification
* Reorder prioritization
* Markdown/Clear prioritization
* Business insight generation
* Streamlit dashboard
* FastAPI prediction service
* FastAPI inventory scoring service
* PDF reports
* Model artifacts and evaluation outputs

---

# 3. Project Structure

The following structure reflects the **actual project ZIP structure**.

```text
Project_FORESIGHT/
│
├── app/
│   ├── Home.py
│   ├── login.py
│   ├── style.css
│   ├── utils.py
│   │
│   └── pages/
│       ├── 1_Dashboard.py
│       ├── 2_Forecast.py
│       ├── 3_Risk_Scoring.py
│       └── 4_About.py
│
├── data/
│   │
│   ├── raw/
│   │   ├── sales_daily.csv
│   │   ├── sales_daily_104weeks.csv
│   │   ├── sku_master.csv
│   │   ├── calendar.csv
│   │   └── inventory_snapshots.csv
│   │
│   ├── processed/
│   │   ├── processed_data.csv
│   │   ├── processed_data_backup.csv
│   │   ├── weekly_model_data.csv
│   │   ├── weekly_model_data_backup.csv
│   │   ├── model_evaluation.csv
│   │   ├── rolling_origin_cv_results.csv
│   │   ├── inventory_risk_scores.csv
│   │   ├── reorder_priority_list.csv
│   │   └── markdown_clear_priority_list.csv
│   │
│   ├── business_insights/
│   │   └── business_insights_summary.csv
│   │
│   ├── decisioning_grid/
│   │   └── decisioning_grid.csv
│   │
│   └── risk_analysis/
│       └── sku_risk_analysis.csv
│
├── models/
│   ├── best_model.pkl
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── label_encoder.pkl
│   ├── model_metadata.json
│   └── model_metrics.json
│
├── notebooks/
│   └── EDA.ipynb
│
├── reports/
│   ├── EDA_Report.pdf
│   ├── business_insights.pdf
│   ├── business_report.pdf
│   ├── feature_importance.csv
│   ├── model_evaluation_report.pdf
│   ├── model_metrics.json
│   ├── project_report.pdf
│   └── risk_analysis_report.pdf
│
├── src/
│   ├── __init__.py
│   ├── baseline.py
│   ├── business_insights.py
│   ├── check_inventory_merge.py
│   ├── data_loader.py
│   ├── diagnose_inventory.py
│   ├── decisioning_grid.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── generate_104_week_data.py
│   ├── pipeline.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── risk_analysis.py
│   ├── risk_scoring.py
│   └── train_model.py
│
├── api.py
├── config.py
├── generate_images.py
├── main.py
├── Procfile
├── render.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 4. Dataset Structure

The project uses four official raw datasets required for the FORESIGHT engagement.

```text
data/raw/
│
├── sales_daily.csv
├── sku_master.csv
├── calendar.csv
└── inventory_snapshots.csv
```

The Zidio brief defines these four provided extracts as the core project data sources.

## Dataset Description

| Dataset                   | Purpose                                                                  |
| ------------------------- | ------------------------------------------------------------------------ |
| `sales_daily.csv`         | Daily SKU-level sales and demand history                                 |
| `sku_master.csv`          | SKU/product master information                                           |
| `calendar.csv`            | Calendar, week, month, season and promotion information                  |
| `inventory_snapshots.csv` | Inventory position, on-hand, on-order, lead time and reorder information |

### `sales_daily.csv`

Contains daily SKU-level sales information including:

* Date
* SKU ID
* Units sold
* Revenue
* Unit price
* Promotion flag

### `sku_master.csv`

Contains SKU-level master information including:

* SKU ID
* Category
* Subcategory
* Launch date
* Unit cost
* List price

### `calendar.csv`

Contains calendar information including:

* Date
* Week
* Month
* Season
* Holiday indicator
* Promotion event

### `inventory_snapshots.csv`

Contains inventory information including:

* Date
* SKU ID
* On-hand units
* On-order units
* Lead time
* Reorder point

These fields correspond to the data dictionary specified by Zidio.

---

# 5. Current Dataset Statistics

The current project ZIP contains the following raw dataset sizes:

```text
sales_daily.csv
Rows    : 146,200
Columns : 5

sku_master.csv
Rows    : 500
Columns : 5

calendar.csv
Rows    : 730
Columns : 5

inventory_snapshots.csv
Rows    : 50,000
Columns : 4
```

The processed weekly forecasting dataset is:

```text
weekly_model_data.csv

Rows          : 21,000
SKUs          : 200
Unique Weeks  : 105
Start Week    : 2024-01-01
End Week      : 2025-12-29
```

---

# 6. Data Processing Pipeline

The main reproducible pipeline is:

```text
Raw Datasets
     │
     ├── sales_daily.csv
     ├── sku_master.csv
     ├── calendar.csv
     └── inventory_snapshots.csv
             │
             ▼
       Data Loading
             │
             ▼
      Data Validation
             │
             ▼
       Data Cleaning
             │
             ▼
     Data Integration
             │
             ▼
    Feature Preparation
             │
             ▼
      Processed Dataset
             │
             ▼
data/processed/processed_data.csv
```

Run:

```bash
python src/pipeline.py
```

The pipeline is designed to ingest the four raw extracts and generate an analysis-ready processed dataset.

This matches the Zidio D1 requirement for a reproducible pipeline that ingests all four extracts and performs coded cleaning and integration.

---

# 7. Weekly Demand Forecasting

The forecasting target is:

```text
weekly_units_sold
```

The forecasting level is:

```text
SKU-level
Weekly
```

The workflow is:

```text
processed_data.csv
        │
        ▼
Weekly Aggregation
        │
        ▼
weekly_model_data.csv
        │
        ▼
Feature Engineering
        │
        ▼
Forecasting Models
```

The project supports a forecast horizon of:

```text
6–8 weeks
```

This matches the Zidio scope for weekly SKU-level demand forecasting over a defined horizon such as 6–8 weeks.

---

# 8. 52-Week Seasonal Naive Baseline

The official baseline is:

```text
52-Week Seasonal Naive
```

The baseline predicts demand using the same seasonal period from the previous year.

Conceptually:

```text
Forecast(t) = Actual Demand(t - 52 weeks)
```

Configuration used by the project:

```text
Season Length       : 52 weeks
Forecast Frequency  : Weekly
Evaluation Metric   : WAPE
```

The Zidio methodology explicitly requires a seasonal-naive baseline before trusting a more complex model.

---

# 9. Forecasting Features

The model uses the following feature groups.

## Calendar Features

```text
year
month
week
quarter
week_sin
week_cos
```

## Lag Features

```text
lag_1
lag_2
lag_3
lag_4
lag_8
lag_12
lag_13
lag_26
lag_52
```

## Rolling Features

```text
rolling_mean_4
rolling_mean_8
rolling_mean_12
rolling_std_4
```

## Seasonal Feature

```text
seasonal_lag_52
```

## Inventory Features

```text
inventory_gap
total_inventory
on_hand_units
on_order_units
reorder_point
```

## SKU Feature

```text
sku_id_enc
```

Total current model features:

```text
26
```

---

# 10. Forecasting Models

The project evaluates:

```text
1. 52-Week Seasonal Naive
2. XGBoost
3. LightGBM
```

The training script is:

```text
src/train_model.py
```

Run:

```bash
python src/train_model.py
```

The training process performs:

1. Weekly data preparation
2. History validation
3. Feature engineering
4. Seasonal Naive baseline evaluation
5. XGBoost evaluation
6. LightGBM evaluation
7. Rolling-Origin Cross-Validation
8. WAPE comparison
9. Best model selection
10. Model artifact saving

---

# 11. Rolling-Origin Cross-Validation

The project uses:

```text
Rolling-Origin Cross-Validation
```

with:

```text
CV Folds : 5
```

This is appropriate for time-series forecasting because future observations must not be used to train earlier forecasts.

The Zidio requirement explicitly states that rolling-origin cross-validation should be used instead of a random train/test split for time-series forecasting.

---

# 12. WAPE Evaluation

The primary forecasting metric is:

```text
WAPE
```

Formula:

```text
WAPE =
SUM(|Actual - Forecast|)
----------------------- × 100
     SUM(|Actual|)
```

Lower WAPE indicates better forecasting performance.

The Zidio brief defines WAPE as the primary accuracy metric for the engagement.

---

# 13. Verified Model Results

The current generated model evaluation file contains:

| Model                  |     WAPE | Beats Baseline |
| ---------------------- | -------: | -------------- |
| 52-week Seasonal Naive | 12.1746% | No             |
| XGBoost                |  8.5266% | Yes            |
| LightGBM               |  8.1089% | Yes            |

### Detailed Results

```text
52-week Seasonal Naive WAPE : 12.1746%
XGBoost WAPE                : 8.5266%
LightGBM WAPE               : 8.1089%
```

LightGBM is the current best-performing model.

```text
Production Model : LightGBM
```

---

# 14. Baseline Improvement

The verified baseline and LightGBM results are:

```text
Baseline WAPE : 12.1746%
LightGBM WAPE : 8.1089%
```

Absolute WAPE improvement:

```text
4.0657 percentage points
```

Relative WAPE improvement:

```text
33.3947%
```

Therefore:

```text
LightGBM
    ↓
8.1089% WAPE
    ↓
Better than 52-week Seasonal Naive
```

The Zidio brief requires the model to beat the seasonal-naive baseline on backtest or otherwise report honestly if the baseline wins.

---

# 15. Model Artifacts

Generated model files are stored in:

```text
models/
```

Current artifacts:

```text
best_model.pkl
xgboost_model.pkl
lightgbm_model.pkl
label_encoder.pkl
model_metadata.json
model_metrics.json
```

The current production model is:

```text
models/lightgbm_model.pkl
```

and the selected best model is stored as:

```text
models/best_model.pkl
```

---

# 16. Inventory Risk Scoring

The project converts forecast information and inventory position into actionable inventory risk.

The risk layer identifies:

```text
Stockout Risk
Overstock Risk
Overall Risk
```

The Zidio specification requires every SKU to receive a risk assessment and recommended action.

---

# 17. Stockout Risk

Stockout risk considers:

```text
Forecast Demand
+
Lead Time
+
On-Hand Inventory
+
On-Order Inventory
```

The API also calculates:

```text
Lead-Time Demand
Available Inventory
Stockout Gap
Days of Supply
```

Possible recommendations include:

```text
URGENTLY REPLENISH STOCK
PLAN REPLENISHMENT
NORMAL MONITORING
```

---

# 18. Overstock Risk

Overstock risk compares forward forecast demand with available inventory.

The system calculates:

```text
Forward Window Demand
Excess Inventory Units
Days of Supply
```

Possible recommendations include:

```text
REDUCE INVENTORY / PROMOTE
MONITOR INVENTORY
NORMAL MONITORING
```

---

# 19. Risk Outputs

Processed risk outputs are stored under:

```text
data/processed/
```

Files include:

```text
inventory_risk_scores.csv
reorder_priority_list.csv
markdown_clear_priority_list.csv
```

Additional risk analysis is stored in:

```text
data/risk_analysis/
└── sku_risk_analysis.csv
```

The decisioning grid is stored in:

```text
data/decisioning_grid/
└── decisioning_grid.csv
```

---

# 20. Inventory Decisioning

The project supports three major inventory actions:

```text
1. Reorder
2. Markdown / Clear
3. Monitor
```

The decisioning concept follows the Zidio framework:

```text
High Stockout Risk
        ↓
Reorder

High Overstock Risk
        ↓
Markdown / Clear

Low Risk
        ↓
Healthy / Monitor
```

The Zidio brief describes these decision quadrants as Reorder Now, Markdown/Clear, Watch/Volatile, and Healthy.

---

# 21. Streamlit Dashboard

The dashboard is implemented using Streamlit.

Main application:

```text
app/Home.py
```

Run:

```bash
streamlit run app/Home.py
```

Dashboard pages:

```text
app/pages/
│
├── 1_Dashboard.py
├── 2_Forecast.py
├── 3_Risk_Scoring.py
└── 4_About.py
```

### Dashboard Page 1

```text
1_Dashboard.py
```

Provides the main dashboard and project-level inventory intelligence.

### Dashboard Page 2

```text
2_Forecast.py
```

Provides:

* Weekly demand forecast
* Actual vs forecast comparison
* Model performance comparison
* Forecast information
* WAPE-based evaluation

### Dashboard Page 3

```text
3_Risk_Scoring.py
```

Provides:

* Business KPIs
* Inventory risk
* Business recommendations
* Priority actions
* Risk data download
* Risk history
* Summary

### Dashboard Page 4

```text
4_About.py
```

Provides:

* Project overview
* Model evaluation
* Production model
* Business impact
* Technology stack
* Inventory risk intelligence
* Project workflow

The Zidio dashboard acceptance criteria require category/SKU filtering, forecast vs actual, risk flags and prioritised reorder/markdown views.

---

# 22. FastAPI Scoring Service

The API is implemented in:

```text
api.py
```

There is **no `api/` directory** in the current project.

The FastAPI application object is:

```python
app
```

Run locally:

```bash
uvicorn api:app --reload
```

or:

```bash
python api.py
```

---

# 23. API Endpoints

## Root Endpoint

```text
GET /
```

Returns basic service information.

## Health Endpoint

```text
GET /health
```

Used to check service and model status.

## Forecast Endpoint

```text
POST /predict
```

Returns:

* SKU
* Forecast horizon
* Total forecast units
* Weekly forecast
* Risk level
* Risk score
* Stockout risk score
* Overstock risk score
* Primary risk
* Days of supply
* Recommendation

## Inventory Scoring Endpoint

```text
POST /score
```

Returns:

* Forecast
* Risk level
* Risk score
* Stockout risk
* Overstock risk
* Lead-time demand
* Available inventory
* Stockout gap
* Forward-window demand
* Excess inventory
* Recommendation

API documentation is available locally at:

```text
/docs
```

when the FastAPI service is running.

The Zidio D6 requirement specifies that the scoring service should return forecast and risk for a SKU or batch and handle bad input gracefully.

---

# 24. Deployment Configuration

The project contains:

```text
Procfile
render.yaml
```

The current `Procfile` uses:

```text
web: uvicorn api:app --host 0.0.0.0 --port $PORT
```

This is the correct entry point for the current `api.py` structure.

### Important

Public deployment URLs should only be added after successful deployment and verification.

Do not add an unverified or placeholder URL to this README.

The Zidio submission requires a live dashboard URL and live scoring-service URL.

---

# 25. Reports

Project reports are stored under:

```text
reports/
```

Current report files:

```text
EDA_Report.pdf
business_insights.pdf
business_report.pdf
model_evaluation_report.pdf
project_report.pdf
risk_analysis_report.pdf
```

Additional report data:

```text
feature_importance.csv
model_metrics.json
```

These reports support:

* Data quality analysis
* EDA
* Business insights
* Forecast evaluation
* Model comparison
* Inventory risk analysis
* Business reporting

---

# 26. Notebooks

The project contains:

```text
notebooks/
└── EDA.ipynb
```

The notebook is used for exploratory data analysis and investigation of demand and inventory patterns.

The Zidio D2 requirement includes data-quality findings, seasonality, trend, top movers, dead stock and business-relevant insights.

---

# 27. Source Code Modules

The `src/` directory contains the main data science workflow.

### `pipeline.py`

Runs the main data ingestion, cleaning and processing pipeline.

### `baseline.py`

Implements the Seasonal Naive baseline.

### `train_model.py`

Creates weekly model data, trains forecasting models, performs evaluation and saves model artifacts.

### `feature_engineering.py`

Creates forecasting features.

### `evaluate.py`

Supports forecasting evaluation.

### `predict.py`

Provides prediction functionality used by the API.

### `risk_scoring.py`

Implements inventory risk scoring logic.

### `risk_analysis.py`

Performs SKU-level risk analysis.

### `decisioning_grid.py`

Creates inventory decisioning outputs.

### `business_insights.py`

Generates business-oriented insights.

### `data_loader.py`

Handles data loading.

### `preprocessing.py`

Handles preprocessing tasks.

### `check_inventory_merge.py`

Used for investigating inventory merge and integration.

### `diagnose_inventory.py`

Used for inventory diagnostics.

### `generate_104_week_data.py`

Provides dataset generation functionality for the 104-week dataset workflow.

---

# 28. Reproducibility

The project is designed to run from the project root.

## Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2 — Run Data Pipeline

```bash
python src/pipeline.py
```

## Step 3 — Train Models

```bash
python src/train_model.py
```

## Step 4 — Start Streamlit Dashboard

```bash
streamlit run app/Home.py
```

## Step 5 — Start FastAPI

```bash
uvicorn api:app --reload
```

---

# 29. Generated Outputs

The pipeline and training workflow generate/update:

```text
data/processed/
├── processed_data.csv
├── weekly_model_data.csv
├── model_evaluation.csv
├── rolling_origin_cv_results.csv
├── inventory_risk_scores.csv
├── reorder_priority_list.csv
└── markdown_clear_priority_list.csv
```

Model artifacts:

```text
models/
├── best_model.pkl
├── xgboost_model.pkl
├── lightgbm_model.pkl
├── label_encoder.pkl
├── model_metadata.json
└── model_metrics.json
```

---

# 30. Technology Stack

## Programming

```text
Python
```

## Data Processing

```text
pandas
numpy
```

## Machine Learning

```text
scikit-learn
XGBoost
LightGBM
```

## Forecasting

```text
Seasonal Naive
Rolling-Origin Cross-Validation
```

## Visualization

```text
Matplotlib
Seaborn
Plotly
```

## Dashboard

```text
Streamlit
```

## API

```text
FastAPI
Uvicorn
Pydantic
```

## Model Persistence

```text
Joblib
```

## Deployment

```text
Render configuration
```

---

# 31. Requirements

The project dependencies are maintained in:

```text
requirements.txt
```

Current main dependencies include:

```text
pandas==2.2.3
numpy==2.2.0
scikit-learn==1.6.1
xgboost==2.1.4
lightgbm==4.6.0
streamlit==1.44.1
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.7.4
plotly==6.0.0
matplotlib==3.10.0
seaborn==0.13.2
joblib==1.4.2
statsmodels==0.14.4
```

Install with:

```bash
pip install -r requirements.txt
```

---

# 32. Zidio Requirement Mapping

The project maps to the required FORESIGHT deliverables as follows.

| Requirement           | Project Implementation                                      |
| --------------------- | ----------------------------------------------------------- |
| D1 Data Pipeline      | `src/pipeline.py`                                           |
| D2 Data Quality & EDA | `notebooks/EDA.ipynb`, `reports/EDA_Report.pdf`             |
| D3 Demand Forecast    | `src/train_model.py`, `src/baseline.py`, forecasting models |
| D4 Risk Scoring       | `src/risk_scoring.py`, `src/risk_analysis.py`               |
| D5 Planning Dashboard | `app/Home.py`, `app/pages/`                                 |
| D6 Scoring Service    | `api.py`                                                    |
| D7 Executive Readout  | `reports/` project/business reports                         |

The seven deliverables are defined by Zidio in Section 04 of the engagement brief.

---

# 33. Model Evaluation Summary

```text
Project:
PROJECT FORESIGHT

Forecast Level:
SKU-level Weekly Demand

Target:
weekly_units_sold

Baseline:
52-week Seasonal Naive

Season Length:
52 weeks

Validation:
Rolling-Origin Cross-Validation

CV Folds:
5

Primary Metric:
WAPE

Baseline WAPE:
12.1746%

XGBoost WAPE:
8.5266%

LightGBM WAPE:
8.1089%

Best Model:
LightGBM

Production Model:
LightGBM

Forecast Horizon:
6–8 Weeks

SKUs:
200

Historical Weeks:
105

Weekly Dataset Rows:
21,000
```

---

# 34. Business Decision Outputs

PROJECT FORESIGHT converts forecasting results into inventory decisions.

```text
Demand Forecast
      │
      ▼
Inventory Risk
      │
      ├── Stockout Risk
      │        ↓
      │     Reorder
      │
      └── Overstock Risk
               ↓
          Markdown / Clear
```

The purpose is not only to generate predictions but also to turn those predictions into actionable inventory decisions.

This follows the Zidio requirement that forecasts should support reorder, clearance and monitoring decisions.

---

# 35. Important Configuration

Current verified forecasting configuration:

```text
Season Length       : 52 weeks
Forecast Frequency  : Weekly
Forecast Horizon    : 6–8 weeks
CV Folds            : 5
Primary Metric      : WAPE
Production Model    : LightGBM
```

The 52-week seasonal baseline must remain consistent across the forecasting workflow.

---

# 36. Data Quality

The project performs data preparation and validation for:

* Date parsing
* Missing values
* Duplicate records
* Data types
* SKU consistency
* Dataset integration
* Historical coverage
* Weekly aggregation
* Inventory merge validation

The Zidio brief expects cleaning decisions to be coded and documented rather than performed manually.

---

# 37. Official Raw Input Files

The official raw input datasets are:

```text
data/raw/
│
├── sales_daily.csv
├── sku_master.csv
├── calendar.csv
└── inventory_snapshots.csv
```

The project also contains:

```text
data/raw/sales_daily_104weeks.csv
```

This file is retained in the project structure, but the official primary sales input documented for the project is:

```text
sales_daily.csv
```

---

# 38. Important Project Files

### Main Dashboard

```text
app/Home.py
```

### Forecast Page

```text
app/pages/2_Forecast.py
```

### Risk Page

```text
app/pages/3_Risk_Scoring.py
```

### API

```text
api.py
```

### Main Pipeline

```text
src/pipeline.py
```

### Model Training

```text
src/train_model.py
```

### Baseline

```text
src/baseline.py
```

### Prediction

```text
src/predict.py
```

### Risk Scoring

```text
src/risk_scoring.py
```

---

# 39. Final Workflow

```text
                    RAW DATA
                       │
                       ▼
              ┌─────────────────┐
              │ Data Pipeline   │
              │ pipeline.py     │
              └────────┬────────┘
                       │
                       ▼
              Processed Dataset
                       │
                       ▼
             Weekly Model Dataset
                       │
                       ▼
             Feature Engineering
                       │
                       ▼
             ┌─────────────────┐
             │ Model Training  │
             │ train_model.py  │
             └────────┬────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
      Seasonal      XGBoost    LightGBM
       Naive
          │           │           │
          └───────────┼───────────┘
                      ▼
             Rolling-Origin CV
                      │
                      ▼
                 WAPE Compare
                      │
                      ▼
              Best Model Selection
                      │
                      ▼
                  Forecast
                      │
                      ▼
              Inventory Risk
                      │
             ┌────────┴────────┐
             ▼                 ▼
        Stockout Risk      Overstock Risk
             │                 │
             ▼                 ▼
          Reorder        Markdown / Clear
             │                 │
             └────────┬────────┘
                      ▼
                  Dashboard
                      │
                      ▼
                    API
```

---

# 40. Current Project Status

The current project contains the core FORESIGHT workflow:

```text
Data Ingestion
      ↓
Data Cleaning
      ↓
Data Integration
      ↓
Feature Engineering
      ↓
Weekly Demand Forecasting
      ↓
52-Week Seasonal Naive Baseline
      ↓
XGBoost Evaluation
      ↓
LightGBM Evaluation
      ↓
Rolling-Origin Cross-Validation
      ↓
WAPE Comparison
      ↓
Production Model Selection
      ↓
Inventory Risk Scoring
      ↓
Reorder Prioritization
      ↓
Markdown/Clear Prioritization
      ↓
Streamlit Dashboard
      ↓
FastAPI Scoring Service
```

Current production forecasting model:

```text
LightGBM
```

Current verified WAPE:

```text
8.1089%
```

52-week Seasonal Naive baseline:

```text
12.1746%
```

Relative WAPE improvement:

```text
33.3947%
```

---

# 41. Deployment Status

```text
Streamlit Dashboard:
Local deployment supported

FastAPI Service:
Local deployment supported

Render Configuration:
Available

Public Dashboard URL:
Add only after successful deployment and verification

Public API URL:
Add only after successful deployment and verification
```

No unverified or placeholder public URLs are included.

---

# 42. Submission Checklist

Before final Zidio submission, verify:

* [ ] Git repository is accessible to the mentor
* [ ] README matches the actual project structure
* [ ] Pipeline runs successfully from raw data
* [ ] Model training runs successfully
* [ ] Seasonal Naive baseline is 52 weeks
* [ ] Rolling-Origin Cross-Validation is used
* [ ] WAPE is reported
* [ ] Production model beats the baseline
* [ ] Inventory risk scoring works
* [ ] Reorder recommendations are available
* [ ] Markdown/Clear recommendations are available
* [ ] Streamlit dashboard works
* [ ] Forecast vs Actual is available
* [ ] API works locally
* [ ] Public dashboard URL is verified
* [ ] Public scoring-service URL is verified
* [ ] Executive readout is ready
* [ ] EDA/data-quality report is ready
* [ ] 3–5 minute demo video is ready
* [ ] Submission form contains all required links

Zidio's submission requirements explicitly include the repository, live dashboard URL, live scoring-service URL, README, executive readout, EDA memo, demo video and submission form.

---

# 43. Conclusion

PROJECT FORESIGHT is an end-to-end demand forecasting and inventory intelligence solution.

The system takes raw sales, SKU, calendar and inventory data and converts it into:

```text
Demand Forecasts
       +
Forecast Evaluation
       +
Inventory Risk
       +
Reorder Decisions
       +
Markdown/Clear Decisions
       +
Dashboard
       +
API Service
```

The current verified forecasting results are:

```text
52-week Seasonal Naive : 12.1746% WAPE
XGBoost                : 8.5266% WAPE
LightGBM               : 8.1089% WAPE
```

Therefore:

```text
Best Model:
LightGBM

Production Model:
LightGBM

Relative Improvement:
33.3947%
```

The project follows the core FORESIGHT objective of turning raw business data into forecasting and inventory decisions that can be used by non-technical stakeholders.

---

## Project Information

```text
Project Name : PROJECT FORESIGHT
Domain       : Data Science & Analytics
Focus        : Demand Forecasting & Inventory Intelligence
Forecast     : Weekly SKU-level
Baseline     : 52-week Seasonal Naive
Model        : LightGBM
Metric       : WAPE
Dashboard    : Streamlit
API          : FastAPI
Deployment   : Render configuration available
```

**PROJECT FORESIGHT — Demand & Inventory Intelligence**