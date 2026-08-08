import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_data.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
XGBOOST_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")
LIGHTGBM_MODEL_PATH = os.path.join(MODEL_DIR, "lightgbm_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
MODEL_METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")

REPORT_DIR = os.path.join(BASE_DIR, "reports")
BUSINESS_REPORT_PATH = os.path.join(REPORT_DIR, "Business_Report.pdf")
RISK_REPORT_PATH = os.path.join(REPORT_DIR, "Risk_Report.pdf")
FORECAST_REPORT_PATH = os.path.join(REPORT_DIR, "Forecast_Report.pdf")

API_HOST = "0.0.0.0"
API_PORT = 8000

API_TITLE = ("Project FORESIGHT" "Demand Forecasting API")
API_VERSION = "1.0.0"

STREAMLIT_PAGE_TITLE = ("Project FORESIGHT Dashboard")
STREAMLIT_PAGE_ICON = " "
STREAMLIT_LAYOUT = "wide"

TARGET_COLUMN = "units_sold"
FORECAST_COLUMN = "forecast_units"
SKU_COLUMN = "sku_id"
CATEGORY_COLUMN = "category"

FEATURES = [
    "year",
    "month",
    "week",
    "day",
    "day_of_week",
    "quarter",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_30",
    "price_difference",
    "discount_percentage",
    "inventory_gap",
    "total_inventory",
    "on_hand_units",
    "on_order_units",
    "reorder_point"
]

STOCKOUT_HIGH_THRESHOLD = 50
STOCKOUT_MEDIUM_THRESHOLD = 20
OVERSTOCK_HIGH_THRESHOLD = 100
OVERSTOCK_MEDIUM_THRESHOLD = 50

RANDOM_STATE = 42
TEST_SIZE = 0.20
SEASON_LENGTH = 7

DIRECTORIES = [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, REPORT_DIR]

for directory in DIRECTORIES:
    os.makedirs(directory, exist_ok=True)

def get_config():
    return {
        "base_dir": BASE_DIR,
        "data_path": PROCESSED_DATA_PATH,
        "model_path": BEST_MODEL_PATH,
        "label_encoder": LABEL_ENCODER_PATH,
        "reports": REPORT_DIR,
        "api_host": API_HOST,
        "api_port": API_PORT,
        "target": TARGET_COLUMN,
        "forecast": FORECAST_COLUMN,
        "random_state": RANDOM_STATE
    }

if __name__ == "__main__":
    print("=" * 60)
    print("PROJECT FORESIGHT CONFIGURATION")
    print("=" * 60)
    print(f"Base Directory    : {BASE_DIR}")
    print(f"Dataset           : {PROCESSED_DATA_PATH}")
    print(f"Best Model        : {BEST_MODEL_PATH}")
    print(f"Label Encoder     : {LABEL_ENCODER_PATH}")
    print(f"Reports Directory : {REPORT_DIR}")

    print(f"API               : " f"http://127.0.0.1:{API_PORT}")
    print("=" * 60)