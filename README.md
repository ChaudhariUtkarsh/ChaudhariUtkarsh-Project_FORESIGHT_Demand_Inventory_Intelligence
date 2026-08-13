# PROJECT FORESIGHT

## Demand Forecasting & Inventory Intelligence System

PROJECT FORESIGHT is an end-to-end demand forecasting and inventory intelligence system designed to forecast SKU-level demand, evaluate forecast accuracy, identify inventory risks, and support reorder and markdown/clear decisions.

The system combines:

* Data preprocessing
* Weekly demand aggregation
* Feature engineering
* 52-week Seasonal Naive baseline
* Machine Learning forecasting
* Rolling-Origin Cross-Validation
* WAPE-based model evaluation
* Inventory risk scoring
* Reorder prioritization
* Markdown/Clear prioritization
* Streamlit dashboard
* FastAPI prediction service

---

# 1. Project Objectives

The main objectives of PROJECT FORESIGHT are:

1. Forecast weekly demand at SKU level.
2. Compare machine learning models against a 52-week Seasonal Naive baseline.
3. Evaluate forecasting performance using WAPE.
4. Identify stockout and overstock risks.
5. Prioritize SKUs for reorder actions.
6. Identify excess inventory for markdown or clearance.
7. Provide business-focused insights through a dashboard.
8. Provide forecasting and inventory scoring APIs.

---

# 2. Project Structure

```text
PROJECT_FORESIGHT/
│
├── app/
│   ├── Home.py
│   ├── About.py
│   ├── pages/
│   │   ├── 1_Forecast.py
│   │   ├── 2_Inventory_Risk.py
│   │   ├── 3_Risk_Scoring.py
│   │   ├── 4_Reorder_Priority.py
│   │   └── 5_Markdown_Clear.py
│   │
│   └── utils.py
│
├── api/
│   └── main.py
│
├── data/
│   ├── raw/
│   │   ├── sales_daily.csv
│   │   ├── sku_master.csv
│   │   ├── calendar.csv
│   │   └── inventory_snapshots.csv
│   │
│   └── processed/
│       ├── processed_data.csv
│       ├── weekly_model_data.csv
│       ├── model_evaluation.csv
│       ├── rolling_origin_cv_results.csv
│       ├── inventory_risk_scores.csv
│       ├── reorder_priority_list.csv
│       └── markdown_clear_priority_list.csv
│
├── models/
│   ├── best_model.pkl
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── model_metadata.json
│   ├── model_metrics.json
│   └── label_encoder.pkl
│
├── reports/
│   ├── model_evaluation_report.pdf
│   ├── business_report.pdf
│   └── executive_summary.pdf
│
├── src/
│   ├── pipeline.py
│   ├── baseline.py
│   ├── train_model.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── risk_scoring.py
│   ├── forecasting.py
│   └── utils.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 3. Dataset Structure

PROJECT FORESIGHT uses four official raw datasets for demand forecasting, inventory analysis, and risk scoring.

```text
data/
└── raw/
    ├── sales_daily.csv
    ├── sku_master.csv
    ├── calendar.csv
    └── inventory_snapshots.csv
```

## Dataset Description

| File                      | Description                                                                   |
| ------------------------- | ----------------------------------------------------------------------------- |
| `sales_daily.csv`         | Daily SKU-level sales and demand history                                      |
| `sku_master.csv`          | SKU master information and product-level attributes                           |
| `calendar.csv`            | Calendar and time-related information                                         |
| `inventory_snapshots.csv` | Historical inventory information used for inventory analysis and risk scoring |

### `sales_daily.csv`

Contains daily SKU-level demand and sales history.

Typical information includes:

* Date
* SKU ID
* Units sold
* Sales-related information

### `sku_master.csv`

Contains product and SKU-level master information.

Typical information includes:

* SKU ID
* Product name
* Product category
* Product attributes
* Pricing information

### `calendar.csv`

Contains date and calendar-related information used to create forecasting features.

Typical information includes:

* Date
* Week
* Month
* Year
* Seasonal information
* Calendar indicators

### `inventory_snapshots.csv`

Contains historical inventory information used for:

* Inventory coverage
* Stockout risk
* Overstock risk
* Reorder analysis
* Markdown/Clear analysis

---

# 4. Data Processing Pipeline

The main data processing pipeline is:

```text
Raw Data
   │
   ├── sales_daily.csv
   ├── sku_master.csv
   ├── calendar.csv
   └── inventory_snapshots.csv
          │
          ▼
     Data Cleaning
          │
          ▼
     Data Integration
          │
          ▼
     Feature Engineering
          │
          ▼
     Processed Dataset
          │
          ▼
data/processed/processed_data.csv
```

Run the pipeline from the project root:

```bash
python src/pipeline.py
```

The pipeline:

1. Loads the four raw datasets.
2. Parses and validates dates.
3. Handles data quality issues.
4. Integrates sales, SKU, calendar, and inventory data.
5. Creates analysis-ready features.
6. Saves the processed dataset.

Output:

```text
data/processed/processed_data.csv
```

---

# 5. Weekly Demand Forecasting

PROJECT FORESIGHT forecasts demand at the weekly SKU level.

The forecasting workflow is:

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

The current verified weekly dataset contains:

```text
Rows              : 21,000
SKUs              : 200
Unique Weeks      : 105
Start Week        : 2024-01-01
End Week          : 2025-12-29
Weeks per SKU     : 105
```

The forecasting system supports a **6–8 week forecast horizon**.

---

# 6. 52-Week Seasonal Naive Baseline

PROJECT FORESIGHT uses a **52-week Seasonal Naive** model as the official forecasting baseline.

The baseline uses demand from the same week in the previous year.

The baseline forecast is:

```text
Forecast(t) = Actual Demand(t - 52 weeks)
```

For example:

```text
Week 53 → Week 1 actual demand
Week 54 → Week 2 actual demand
Week 55 → Week 3 actual demand
```

Configuration:

```python
REQUIRED_SEASON_LENGTH = 52
```

The 52-week Seasonal Naive baseline is validated before model training.

Every SKU in the current weekly dataset has:

```text
105 weeks of history
```

Therefore, the current dataset satisfies the requirement of at least 52 weeks of history.

Baseline configuration:

```text
Baseline              : 52-week Seasonal Naive
Season Length         : 52 weeks
Forecast Frequency    : Weekly
Forecast Horizon      : 6–8 weeks
Evaluation Metric     : WAPE
```

---

# 7. Model Training

The model training workflow is:

```text
Weekly Model Data
        │
        ▼
Feature Engineering
        │
        ├── Lag Features
        ├── Rolling Features
        ├── Seasonal Features
        ├── Calendar Features
        └── Inventory Features
        │
        ▼
Rolling-Origin Cross-Validation
        │
        ├── 52-Week Seasonal Naive
        ├── XGBoost
        └── LightGBM
        │
        ▼
WAPE Comparison
        │
        ▼
Best Production Model
```

Run model training using:

```bash
python src/train_model.py
```

The training process:

1. Loads weekly model data.
2. Validates 52-week history.
3. Creates forecasting features.
4. Creates five Rolling-Origin CV folds.
5. Evaluates the 52-week Seasonal Naive baseline.
6. Evaluates XGBoost.
7. Evaluates LightGBM.
8. Compares all models using WAPE.
9. Selects the best-performing production model.
10. Saves the model artifacts and evaluation results.

---

# 8. Forecasting Features

The model uses multiple groups of forecasting features.

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

## Calendar Features

```text
year
month
week
quarter
week_sin
week_cos
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

The current training run uses **26 features**.

---

# 9. Model Evaluation

The forecasting models are evaluated using **Rolling-Origin Cross-Validation** with **WAPE** as the primary metric.

## WAPE

WAPE is calculated as:

```text
WAPE =
Sum(|Actual - Forecast|)
----------------------- × 100
     Sum(|Actual|)
```

Lower WAPE indicates better forecasting performance.

---

# 10. Final Verified Model Results

The latest verified training run produced the following results:

| Model                           |   WAPE |
| ------------------------------- | -----: |
| 52-week Seasonal Naive Baseline | 12.17% |
| XGBoost                         |  8.53% |
| LightGBM                        |  8.11% |

Detailed verified values:

```text
52-week Seasonal Naive : 12.1746%
XGBoost                : 8.5266%
LightGBM               : 8.1089%
```

LightGBM is currently the best-performing model.

Therefore:

```text
Production Model: LightGBM
```

---

# 11. Baseline vs Production Model

The current verified comparison is:

```text
52-Week Seasonal Naive
          VS
LightGBM
```

The baseline WAPE is:

```text
12.1746%
```

The LightGBM WAPE is:

```text
8.1089%
```

The relative WAPE improvement is:

```text
33.3947%
```

Therefore, LightGBM currently provides approximately:

```text
33.39% relative WAPE improvement
```

over the 52-week Seasonal Naive baseline.

### Final Evaluation Summary

```text
Primary Metric:
WAPE

Baseline:
52-week Seasonal Naive

Best ML Model:
LightGBM

Baseline WAPE:
12.1746%

LightGBM WAPE:
8.1089%

Relative Improvement:
33.3947%

Production Model:
LightGBM
```

> These values are generated from the latest verified training run. If the model is retrained on changed data or configuration, the README, dashboard, and reports should be updated using the newly generated verified metrics.

---

# 12. Rolling-Origin Cross-Validation

The model evaluation uses:

```text
N_CV_FOLDS = 5
```

The validation process follows a time-series Rolling-Origin approach.

```text
Historical Data
       │
       ▼
Training Window
       │
       ▼
Validation Window
       │
       ▼
WAPE
       │
       ▼
Expand Training Window
       │
       ▼
Next Validation Window
```

Five folds are used to evaluate model performance over different historical periods.

The detailed results are saved to:

```text
data/processed/rolling_origin_cv_results.csv
```

This file should not be manually edited.

It is generated automatically by:

```bash
python src/train_model.py
```

---

# 13. Model Evaluation Output

The model comparison results are saved to:

```text
data/processed/model_evaluation.csv
```

The file contains model-level evaluation results including:

* Model name
* CV WAPE
* Baseline WAPE
* WAPE improvement
* Relative improvement
* Whether the model beats the baseline

The current ranking is:

```text
1. LightGBM       → 8.1089%
2. XGBoost        → 8.5266%
3. Seasonal Naive → 12.1746%
```

The file is automatically generated during model training.

Run:

```bash
python src/train_model.py
```

Do not manually change the WAPE values in this file.

---

# 14. Model Metrics

The following file stores the model evaluation metrics:

```text
models/model_metrics.json
```

It contains information such as:

```text
Baseline
Baseline WAPE
ML model WAPE
WAPE improvement
Best ML model
Production model
Rolling-Origin folds
Evaluation metric
```

The current verified production model is:

```text
lightgbm
```

The current verified metrics are:

```text
Baseline WAPE : 12.17%
LightGBM WAPE : 8.11%
```

The file is generated automatically by:

```bash
python src/train_model.py
```

---

# 15. Model Metadata

The model metadata file is:

```text
models/model_metadata.json
```

It stores information about:

* Project
* Target
* Forecast frequency
* Forecast horizon
* Required season length
* Actual season length
* Baseline
* Baseline status
* Features
* Dataset size
* Dataset date range
* Number of weeks
* Number of SKUs
* Production model

Current verified metadata includes:

```text
Target:
weekly_units_sold

Forecast Frequency:
Weekly

Required Season Length:
52

Actual Season Length:
52

Baseline:
52-week Seasonal Naive

Production Model:
lightgbm

Dataset Rows:
21000

Unique Weeks:
105

Unique SKUs:
200
```

WAPE metrics are stored separately in:

```text
models/model_metrics.json
```

---

# 16. Model Artifacts

Trained model artifacts are stored under:

```text
models/
```

Important artifacts include:

```text
best_model.pkl
xgboost_model.pkl
lightgbm_model.pkl
model_metadata.json
model_metrics.json
label_encoder.pkl
```

The production model is saved as:

```text
models/best_model.pkl
```

The current production model is:

```text
LightGBM
```

---

# 17. Inventory Risk Intelligence

PROJECT FORESIGHT calculates inventory risk at SKU level.

The major risk categories are:

```text
Stockout Risk
Overstock Risk
Total Inventory Risk
```

Risk scoring considers factors such as:

* Forecast demand
* Available inventory
* Lead-time demand
* Inventory coverage
* Demand pressure
* Lead-time pressure
* Projected inventory
* Inventory position

The generated risk output is:

```text
data/processed/inventory_risk_scores.csv
```

---

# 18. Reorder Priority

The system identifies SKUs that require replenishment.

Reorder analysis considers:

* Stockout risk
* Available inventory
* Forecast demand
* Lead-time demand
* Safety stock
* Reorder requirements

The output is:

```text
data/processed/reorder_priority_list.csv
```

This output supports business users in prioritizing replenishment decisions.

---

# 19. Markdown / Clear Priority

The system identifies SKUs with excess inventory.

Markdown/Clear analysis considers:

* Overstock risk
* Excess inventory
* Inventory coverage
* Forecast demand
* Inventory position

The output is:

```text
data/processed/markdown_clear_priority_list.csv
```

The output supports decisions such as:

* Markdown
* Clearance
* Inventory reduction
* Promotional action

---

# 20. Streamlit Dashboard

PROJECT FORESIGHT includes an interactive Streamlit dashboard.

Run locally:

```bash
streamlit run app/Home.py
```

The dashboard provides:

* Executive overview
* Demand forecast
* Forecast vs actual analysis
* Inventory risk
* Risk scoring
* Reorder priority
* Markdown/Clear priority
* SKU-level analysis

Dashboard filters include:

* Category
* SKU
* Risk Level
* Reorder Priority
* Markdown/Clear Priority

---

# 21. Forecast Dashboard

The forecast dashboard supports comparison between:

```text
Actual Demand
       VS
Forecast Demand
```

The dashboard can display:

```text
Actual Demand
ML Forecast
52-Week Seasonal Naive
```

Forecasting is performed at SKU level with a:

```text
6–8 week forecast horizon
```

The dashboard should use the same production model and verified model artifacts generated during training.

---

# 22. FastAPI Service

PROJECT FORESIGHT provides a FastAPI service for forecasting and inventory risk scoring.

Run locally:

```bash
uvicorn api.main:app --reload
```

The API provides endpoints for:

* Demand prediction
* Inventory risk scoring

Example prediction request:

```json
{
    "sku_id": "SKU001",
    "forecast_weeks": 8
}
```

The API uses the trained production model stored in:

```text
models/best_model.pkl
```

---

# 23. Live Deployment

The project supports deployment of both the Streamlit dashboard and FastAPI service.

At the time of this README version, **no public deployment URL is included unless it has been verified as live and accessible**.

### Streamlit Dashboard

Run locally with:

```bash
streamlit run app/Home.py
```

### FastAPI Service

Run locally with:

```bash
uvicorn api.main:app --reload
```

> Public deployment links should only be added to this section after successful deployment and verification. Do not use placeholder URLs or unverified URLs.

---

# 24. Reports

Project reports are stored under:

```text
reports/
```

Expected reports include:

```text
model_evaluation_report.pdf
business_report.pdf
executive_summary.pdf
```

Reports cover:

* Model evaluation
* Forecast accuracy
* Baseline comparison
* Inventory risk
* Reorder recommendations
* Markdown/Clear recommendations
* Business impact

All reports should use the same verified model evaluation results.

Current verified forecasting results:

```text
Baseline WAPE : 12.1746%
LightGBM WAPE : 8.1089%
Improvement   : 33.3947%
```

---

# 25. Reproducibility

To reproduce the project from the project root:

## Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2 — Run Data Pipeline

```bash
python src/pipeline.py
```

This generates:

```text
data/processed/processed_data.csv
```

## Step 3 — Train Forecasting Models

```bash
python src/train_model.py
```

This generates or updates:

```text
data/processed/weekly_model_data.csv
data/processed/model_evaluation.csv
data/processed/rolling_origin_cv_results.csv

models/best_model.pkl
models/model_metadata.json
models/model_metrics.json
models/label_encoder.pkl
```

It also trains and saves:

```text
models/xgboost_model.pkl
models/lightgbm_model.pkl
```

## Step 4 — Run Dashboard

```bash
streamlit run app/Home.py
```

## Step 5 — Run API

```bash
uvicorn api.main:app --reload
```

---

# 26. Important Configuration

The official Seasonal Naive configuration is:

```python
REQUIRED_SEASON_LENGTH = 52
```

Rolling-Origin Cross-Validation uses:

```python
N_CV_FOLDS = 5
```

Forecast horizon:

```text
6–8 weeks
```

Primary evaluation metric:

```text
WAPE
```

These settings should remain consistent across:

```text
src/baseline.py
src/train_model.py
src/evaluation.py
Dashboard
Reports
README.md
```

---

# 27. Data Quality Checks

The project performs validation for:

* Missing dates
* Invalid dates
* Duplicate records
* Missing values
* Data type consistency
* SKU availability
* Time-series coverage
* Minimum historical coverage

The current weekly dataset contains:

```text
200 SKUs
105 weeks per SKU
21,000 rows
```

The minimum history requirement for the official baseline is:

```text
52 weeks
```

Since every SKU currently has 105 weeks of history, the dataset satisfies the 52-week baseline requirement.

---

# 28. Official Raw Dataset Structure

The official raw dataset structure is:

```text
data/raw/
│
├── sales_daily.csv
├── sku_master.csv
├── calendar.csv
└── inventory_snapshots.csv
```

The project should use these files as the official raw input datasets.

The following obsolete dataset names should not be used in the official documentation:

```text
store_master.csv
customer_master.csv
promotions.csv
sales_transactions.csv
inventory_snapshot.csv
```

Official input files:

```text
sales_daily.csv
sku_master.csv
calendar.csv
inventory_snapshots.csv
```

---

# 29. Final Model Evaluation Standard

The final project evaluation standard is:

```text
Primary Metric:
WAPE

Baseline:
52-Week Seasonal Naive

Rolling-Origin CV:
5 Folds

Best ML Model:
LightGBM

Production Model:
LightGBM

Forecast Horizon:
6–8 Weeks
```

Current verified results:

```text
Baseline WAPE : 12.1746%
XGBoost WAPE  : 8.5266%
LightGBM WAPE : 8.1089%
```

Relative LightGBM improvement:

```text
33.3947%
```

Therefore:

```text
LightGBM
    ↓
8.1089% WAPE
    ↓
Best performing model
    ↓
Production Model
```

---

# 30. Business Outputs

PROJECT FORESIGHT produces three major decision-support outputs.

## Demand Forecast

```text
SKU-level weekly demand forecast
```

## Inventory Risk

```text
Stockout Risk
Overstock Risk
Total Inventory Risk
```

## Inventory Actions

```text
Reorder Priority
Markdown/Clear Priority
```

These outputs are designed to support inventory planning and replenishment decisions.

---

# 31. Project Workflow

```text
                 RAW DATA
                     │
                     ▼
        ┌─────────────────────────┐
        │     Data Pipeline       │
        │      pipeline.py        │
        └────────────┬────────────┘
                     │
                     ▼
             Processed Data
                     │
                     ▼
          Weekly Model Dataset
                     │
                     ▼
        ┌─────────────────────────┐
        │   Feature Engineering   │
        └────────────┬────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │     Model Training      │
        │    train_model.py       │
        └────────────┬────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
  52-Week Seasonal Naive    ML Models
          │                  │
          │            ┌─────┴─────┐
          │            ▼           ▼
          │         XGBoost     LightGBM
          │            │           │
          └────────────┴─────┬─────┘
                             ▼
                    Rolling-Origin CV
                             │
                             ▼
                        WAPE Comparison
                             │
                             ▼
                     Best Production Model
                             │
                             ▼
                        Forecasting
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
             Inventory Risk     Forecast Output
                    │
               ┌────┴─────┐
               ▼          ▼
            Reorder    Markdown/
            Priority    Clear
               │          │
               └────┬─────┘
                    ▼
                Dashboard
                    │
                    ▼
                  API
```

---

# 32. Current Project Status

PROJECT FORESIGHT currently provides an end-to-end workflow covering:

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
WAPE Model Comparison
      ↓
Production Model Selection
      ↓
Inventory Risk Scoring
      ↓
Reorder Recommendations
      ↓
Markdown/Clear Recommendations
      ↓
Streamlit Dashboard
      ↓
FastAPI Service
```

The current verified production model is:

```text
LightGBM
```

with a verified WAPE of:

```text
8.1089%
```

compared with the 52-week Seasonal Naive baseline WAPE of:

```text
12.1746%
```

This represents a relative WAPE improvement of:

```text
33.3947%
```

---

# 33. Final Verified Configuration

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

XGBoost WAPE:
8.5266%

LightGBM WAPE:
8.1089%

Baseline WAPE:
12.1746%

Best ML Model:
LightGBM

Production Model:
LightGBM

Forecast Horizon:
6–8 Weeks

SKUs:
200

Historical Weeks:
105

Dataset Rows:
21,000
```

---

# 34. Important Note About Generated Results

The following files are generated automatically and should not be manually edited:

```text
data/processed/model_evaluation.csv
data/processed/rolling_origin_cv_results.csv
models/model_metrics.json
models/model_metadata.json
models/best_model.pkl
```

To regenerate the model results, run:

```bash
python src/train_model.py
```

The training script calculates:

```text
Baseline WAPE
XGBoost WAPE
LightGBM WAPE
WAPE improvement
Best production model
```

directly from the current dataset and Rolling-Origin Cross-Validation results.

Therefore, the generated files should be treated as the source of truth for the current model evaluation.

---

# 35. Quick Start

From the PROJECT_FORESIGHT root directory:

```bash
pip install -r requirements.txt
```

Then:

```bash
python src/pipeline.py
```

Then:

```bash
python src/train_model.py
```

Then start the dashboard:

```bash
streamlit run app/Home.py
```

Or start the API:

```bash
uvicorn api.main:app --reload
```

---

# 36. Conclusion

PROJECT FORESIGHT combines demand forecasting with inventory intelligence to provide SKU-level forecasting, model evaluation, inventory risk identification, and actionable replenishment and clearance recommendations.

The current verified forecasting evaluation demonstrates that LightGBM performs better than the official 52-week Seasonal Naive baseline:

```text
52-week Seasonal Naive : 12.1746% WAPE
LightGBM               : 8.1089% WAPE
Relative Improvement   : 33.3947%
```

Therefore, **LightGBM is the current production forecasting model** for PROJECT FORESIGHT.

The system provides a complete workflow from raw data ingestion to forecasting, risk analysis, inventory decisions, dashboard visualization, and API-based serving.

---

## Deployment Status

```text
Streamlit Dashboard:
Local deployment supported

FastAPI Service:
Local deployment supported

Public URLs:
Not included until successfully deployed and verified
```

No placeholder deployment URLs are used in this README.
