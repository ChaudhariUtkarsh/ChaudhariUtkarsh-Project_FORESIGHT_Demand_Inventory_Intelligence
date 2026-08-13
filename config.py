import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")


SALES_DAILY_PATH = os.path.join(RAW_DATA_DIR, "sales_daily.csv")
SKU_MASTER_PATH = os.path.join(RAW_DATA_DIR, "sku_master.csv")
CALENDAR_PATH = os.path.join(RAW_DATA_DIR, "calendar.csv")
INVENTORY_SNAPSHOTS_PATH = os.path.join(RAW_DATA_DIR, "inventory_snapshots.csv")


PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_data.csv")
WEEKLY_MODEL_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "weekly_model_data.csv")
MODEL_EVALUATION_PATH = os.path.join(PROCESSED_DATA_DIR, "model_evaluation.csv")
ROLLING_ORIGIN_CV_PATH = os.path.join(PROCESSED_DATA_DIR, "rolling_origin_cv_results.csv")
INVENTORY_RISK_OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "inventory_risk_scores.csv")
REORDER_OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "reorder_priority_list.csv")
MARKDOWN_OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "markdown_clear_priority_list.csv")


MODEL_DIR = os.path.join(BASE_DIR, "models")

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
XGBOOST_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")
LIGHTGBM_MODEL_PATH = os.path.join(MODEL_DIR, "lightgbm_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
MODEL_METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")
MODEL_METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


REPORT_DIR = os.path.join(BASE_DIR, "reports")

BUSINESS_REPORT_PATH = os.path.join(REPORT_DIR, "Business_Report.pdf")
RISK_REPORT_PATH = os.path.join(REPORT_DIR, "Risk_Report.pdf")
FORECAST_REPORT_PATH = os.path.join(REPORT_DIR, "Forecast_Report.pdf")


API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Project FORESIGHT Demand Forecasting API"
API_VERSION = "1.0.0"


STREAMLIT_PAGE_TITLE = "Project FORESIGHT Dashboard"
STREAMLIT_PAGE_ICON = " "
STREAMLIT_LAYOUT = "wide"

TARGET_COLUMN = "weekly_units_sold"
FORECAST_COLUMN = "forecast_weekly_demand"
SKU_COLUMN = "sku_id"
CATEGORY_COLUMN = "category"

RANDOM_STATE = 42
TEST_SIZE = 0.20
SEASON_LENGTH = 52

FORECAST_HORIZON_MIN_WEEKS = 6
FORECAST_HORIZON_MAX_WEEKS = 8
DAYS_PER_WEEK = 7

HIGH_RISK_THRESHOLD = 70
MEDIUM_RISK_THRESHOLD = 40

STOCKOUT_HIGH_THRESHOLD = 70
STOCKOUT_MEDIUM_THRESHOLD = 40

LOW_COVERAGE_WEEKS = 4
MEDIUM_COVERAGE_WEEKS = 8
HIGH_COVERAGE_WEEKS = 12

OVERSTOCK_HIGH_THRESHOLD = 70
OVERSTOCK_MEDIUM_THRESHOLD = 50

SAFETY_MULTIPLIER = 1.20

FEATURES = [
    "year",
    "month",
    "week",
    "quarter",
    "week_sin",
    "week_cos",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "lag_8",
    "lag_12",
    "lag_13",
    "lag_26",
    "lag_52",
    "rolling_mean_4",
    "rolling_mean_8",
    "rolling_mean_12",
    "rolling_std_4",
    "inventory_gap",
    "total_inventory",
    "on_hand_units",
    "on_order_units",
    "reorder_point",
    "sku_id_enc",
    "seasonal_lag_52"
]

DIRECTORIES = [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, REPORT_DIR]

for directory in DIRECTORIES:
    os.makedirs(directory, exist_ok=True)


def get_config():
    return {
        "base_dir": BASE_DIR,
        "data_path": PROCESSED_DATA_PATH,
        "weekly_model_data_path": WEEKLY_MODEL_DATA_PATH,
        "model_path": BEST_MODEL_PATH,
        "label_encoder": LABEL_ENCODER_PATH,
        "model_metrics": MODEL_METRICS_PATH,
        "model_metadata": MODEL_METADATA_PATH,
        "reports": REPORT_DIR,
        "api_host": API_HOST,
        "api_port": API_PORT,
        "target": TARGET_COLUMN,
        "forecast": FORECAST_COLUMN,
        "season_length": SEASON_LENGTH,
        "forecast_horizon_min_weeks": FORECAST_HORIZON_MIN_WEEKS,
        "forecast_horizon_max_weeks": FORECAST_HORIZON_MAX_WEEKS,
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
        "medium_risk_threshold": MEDIUM_RISK_THRESHOLD,
        "stockout_high_threshold": STOCKOUT_HIGH_THRESHOLD,
        "stockout_medium_threshold": STOCKOUT_MEDIUM_THRESHOLD,
        "low_coverage_weeks": LOW_COVERAGE_WEEKS,
        "medium_coverage_weeks": MEDIUM_COVERAGE_WEEKS,
        "high_coverage_weeks": HIGH_COVERAGE_WEEKS,
        "overstock_high_threshold": OVERSTOCK_HIGH_THRESHOLD,
        "overstock_medium_threshold": OVERSTOCK_MEDIUM_THRESHOLD,
        "safety_multiplier": SAFETY_MULTIPLIER,
        "days_per_week": DAYS_PER_WEEK,
        "random_state": RANDOM_STATE
    }


if __name__ == "__main__":

    print("=" * 70)
    print("PROJECT FORESIGHT CONFIGURATION")
    print("=" * 70)

    print(f"Base Directory       : {BASE_DIR}")
    print(f"Processed Dataset    : " f"{PROCESSED_DATA_PATH}")
    print(f"Weekly Model Data    : " f"{WEEKLY_MODEL_DATA_PATH}")
    print(f"Best Model           : " f"{BEST_MODEL_PATH}")
    print(f"Label Encoder        : " f"{LABEL_ENCODER_PATH}")
    print(f"Model Metrics        : " f"{MODEL_METRICS_PATH}")
    print(f"Model Metadata       : " f"{MODEL_METADATA_PATH}")
    print(f"Reports Directory    : " f"{REPORT_DIR}")
    print(f"API                  : " f"http://127.0.0.1:{API_PORT}")

    print()
    print("MODEL CONFIGURATION")
    print("-" * 70)

    print(f"Season Length        : " f"{SEASON_LENGTH} weeks")
    print(f"Forecast Horizon     : " f"{FORECAST_HORIZON_MIN_WEEKS}" f"-" f"{FORECAST_HORIZON_MAX_WEEKS} weeks")
    print(f"Days Per Week        : " f"{DAYS_PER_WEEK}")

    print()
    print("RISK CONFIGURATION")
    print("-" * 70)

    print(f"High Risk Threshold  : " f"{HIGH_RISK_THRESHOLD}")
    print(f"Medium Risk Threshold: " f"{MEDIUM_RISK_THRESHOLD}")
    print(f"Stockout High        : " f"{STOCKOUT_HIGH_THRESHOLD}")
    print(f"Stockout Medium      : " f"{STOCKOUT_MEDIUM_THRESHOLD}")
    print(f"Low Coverage         : " f"{LOW_COVERAGE_WEEKS} weeks")
    print(f"Medium Coverage      : " f"{MEDIUM_COVERAGE_WEEKS} weeks")
    print(f"High Coverage        : " f"{HIGH_COVERAGE_WEEKS} weeks")
    print(f"Overstock High       : " f"{OVERSTOCK_HIGH_THRESHOLD}")
    print(f"Overstock Medium     : " f"{OVERSTOCK_MEDIUM_THRESHOLD}")
    print(f"Safety Multiplier    : " f"{SAFETY_MULTIPLIER}")

    print()
    print("=" * 70)
    print("CONFIGURATION READY")
    print("=" * 70)