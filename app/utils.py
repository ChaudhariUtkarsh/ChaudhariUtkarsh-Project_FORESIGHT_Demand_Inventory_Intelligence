import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

import config


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

PROCESSED_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_data.csv")
WEEKLY_DATA_PATH = os.path.join(PROCESSED_DIR, "weekly_model_data.csv")
RISK_SCORES_PATH = os.path.join(PROCESSED_DIR, "inventory_risk_scores.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


os.makedirs(REPORT_DIR, exist_ok=True)

FEATURE_COLUMNS = [
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
    "reorder_point",
    "sku_id_enc",
]


@st.cache_data
def load_data():
    """Load processed dataset."""
    if not os.path.exists(PROCESSED_DATA_PATH):
        return pd.DataFrame()
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        df.columns = [str(col).strip() for col in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_weekly_data():
    """Load weekly model dataset."""
    if not os.path.exists(WEEKLY_DATA_PATH):
        return pd.DataFrame()
    try:
        df = pd.read_csv(WEEKLY_DATA_PATH)
        df.columns = [str(col).strip() for col in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_risk_scores():
    """
    Load CENTRAL generated inventory risk scores.

    IMPORTANT:
    This function only reads the generated risk file.
    No risk calculation is performed here.
    """

    if not os.path.exists(RISK_SCORES_PATH):
        return pd.DataFrame()
    try:
        df = pd.read_csv(RISK_SCORES_PATH)
        df.columns = [str(col).strip() for col in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_resource
def load_model():
    """Load trained forecasting model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_encoder():
    """Load SKU label encoder."""
    if not os.path.exists(ENCODER_PATH):
        return None
    try:
        return joblib.load(ENCODER_PATH)
    except Exception:
        return None


def check_resources():
    """Check availability of all major project resources."""

    return {
        "processed_data": os.path.exists(PROCESSED_DATA_PATH),
        "weekly_model_data": os.path.exists(WEEKLY_DATA_PATH),
        "risk_scores": os.path.exists(RISK_SCORES_PATH),
        "model": os.path.exists(MODEL_PATH),
        "encoder": os.path.exists(ENCODER_PATH),
        "reports": os.path.exists(REPORT_DIR),
    }


def dataset_info(df):
    """Return basic dataset information."""
    if df is None or df.empty:
        return {"Rows": 0, "Columns": 0, "Column Names": []}
    return {"Rows": len(df), "Columns": len(df.columns), "Column Names": list(df.columns),}


def predict_demand(model, encoder, input_data):
    """Predict demand using the trained model."""
    df = input_data.copy()

    if (encoder is not None and "sku_id" in df.columns):
        try:
            df["sku_id_enc"] = encoder.transform(df["sku_id"].astype(str))
        except Exception:
            df["sku_id_enc"] = 0

    if "sku_id" in df.columns:
        df = df.drop(columns=["sku_id"])
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]

    if missing_features:
        raise ValueError("Missing model features: " + ", ".join(missing_features))

    X = df[FEATURE_COLUMNS]

    prediction = model.predict(X)
    prediction = prediction.clip(min=0)
    return prediction


def calculate_revenue(forecast_units, selling_price):
    """Calculate expected revenue."""
    return (forecast_units * selling_price)

def calculate_revenue_at_risk(forecast_units, inventory, selling_price):
    """Calculate revenue at risk because of shortage."""
    shortage = max(forecast_units - inventory, 0)
    return (shortage * selling_price)

def calculate_capital_locked(inventory, forecast_units, unit_cost):
    """Calculate capital locked in excess inventory."""
    excess_inventory = max(inventory - forecast_units, 0)
    return (excess_inventory * unit_cost)

def inventory_ratio(inventory, forecast):
    """Inventory / Forecast Ratio (%)"""
    if forecast <= 0:
        return 0
    return round((inventory / forecast) * 100, 2)


def get_risk_data():
    """
    Return generated central risk scores.

    IMPORTANT:
    No risk calculation is performed here.

    The source of truth is:
    inventory_risk_scores.csv
    """
    return load_risk_scores()


def filter_risk_data(df, selected_risk="ALL"):
    """
    Filter already-generated risk scores.

    IMPORTANT:
    This function DOES NOT calculate risk.
    It only filters the risk_score column generated by the central risk engine.
    """

    if df is None or df.empty:
        return pd.DataFrame()
    selected_risk = str(selected_risk).upper()

    risk_score_column = None
    possible_columns = ["risk_score", "Risk Score", "total_risk", "Total Risk",]

    for col in possible_columns:
        if col in df.columns:
            risk_score_column = col
            break

    if risk_score_column is None:
        return df.copy()
    result = df.copy()
    result[risk_score_column] = pd.to_numeric(result[risk_score_column], errors="coerce").fillna(0)

    high_threshold = getattr(config, "HIGH_RISK_THRESHOLD", 70)
    medium_threshold = getattr(config, "MEDIUM_RISK_THRESHOLD", 40)

    if selected_risk == "HIGH":
        return result[result[risk_score_column] >= high_threshold].copy()
    elif selected_risk == "MEDIUM":
        return result[(result[risk_score_column] >= medium_threshold) & (result[risk_score_column] < high_threshold)].copy()
    elif selected_risk == "LOW":
        return result[result[risk_score_column] < medium_threshold].copy()
    return result.copy()


def risk_summary(df):
    """Generate summary from CENTRAL risk scores. No new risk score is calculated. """
    if df is None or df.empty:
        return {"total": 0, "high": 0, "medium": 0, "low": 0,}
    risk_score_column = None
    possible_columns = ["risk_score", "Risk Score", "total_risk", "Total Risk",]

    for col in possible_columns:
        if col in df.columns:
            risk_score_column = col
            break

    if risk_score_column is None:
        return {"total": len(df), "high": 0, "medium": 0, "low": 0,}
    scores = pd.to_numeric(df[risk_score_column], errors="coerce").fillna(0)
    high_threshold = getattr(config, "HIGH_RISK_THRESHOLD", 70)
    medium_threshold = getattr(config, "MEDIUM_RISK_THRESHOLD", 40)
    return {"total": len(df), "high": int((scores >= high_threshold).sum()), "medium": int(((scores >= medium_threshold) & (scores < high_threshold)).sum()), "low": int((scores < medium_threshold).sum()),}


def calculate_kpis(df):
    """Dashboard KPI summary."""
    result = {"total_products": 0, "forecast_units": 0, "inventory": 0, "expected_revenue": 0, "capital_locked": 0, "revenue_at_risk": 0, "stockout": 0, "overstock": 0,}

    if df is None or df.empty:
        return result
    result["total_products"] = len(df)

    if "forecast_units" in df.columns:
        forecast = pd.to_numeric(df["forecast_units"], errors="coerce").fillna(0)
        result["forecast_units"] = int(forecast.sum())
    else:
        forecast = pd.Series(0, index=df.index)

    if "on_hand_units" in df.columns:
        inventory = pd.to_numeric(df["on_hand_units"], errors="coerce").fillna(0)
        result["inventory"] = int(inventory.sum())
    else:
        inventory = pd.Series(0, index=df.index)

    if "selling_price" in df.columns:
        selling_price = pd.to_numeric(df["selling_price"], errors="coerce").fillna(0)
    else:
        selling_price = pd.Series(0, index=df.index)

    if "unit_cost" in df.columns:
        unit_cost = pd.to_numeric(df["unit_cost"], errors="coerce").fillna(0)
    else:
        unit_cost = pd.Series(0, index=df.index)

    result["expected_revenue"] = float((forecast * selling_price).sum())
    result["capital_locked"] = float(((inventory - forecast).clip(lower=0) * unit_cost).sum())
    result["revenue_at_risk"] = float(((forecast - inventory).clip(lower=0) * selling_price).sum())
    result["stockout"] = int((forecast > inventory).sum())
    result["overstock"] = int((inventory > forecast).sum())
    return result


def export_csv(df, filename="forecast_results.csv"):
    """Save DataFrame as CSV."""

    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, filename)
    df.to_csv(path, index=False)
    return path


def format_date(date_value):
    """Format date as DD-MM-YYYY."""
    try:
        return pd.to_datetime(date_value).strftime("%d-%m-%Y")
    except Exception:
        return str(date_value)


def format_currency(amount):
    """Format currency value."""
    try:
        return f"{float(amount):,.2f}"
    except Exception:
        return "0.00"


def format_percentage(value):
    """Format percentage."""
    try:
        return f"{float(value):.2f} %"
    except Exception:
        return "0.00 %"


def save_forecast_history(record, filename="forecast_history.csv"):
    """Save forecast history."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, filename)
    new_df = pd.DataFrame([record])
    if os.path.exists(path):
        old_df = pd.read_csv(path)
        new_df = pd.concat([old_df, new_df], ignore_index=True)
    new_df.to_csv(path, index=False)
    return path



def file_exists(path):
    """Check whether file exists."""
    return os.path.exists(path)


def safe_divide(a, b):
    """Safe division."""
    if b == 0:
        return 0
    return a / b


def get_timestamp():
    """Return current timestamp."""
    return datetime.now().strftime("%d-%m-%Y %H:%M:%S")


def show_success(message):
    """Return success response."""
    return {"status": "success", "message": message,}


def show_error(message):
    """Return error response."""
    return {"status": "error", "message": message,}


def project_info():
    """Project metadata."""
    return {
        "Project": "Project FORESIGHT",
        "Version": "1.0",
        "Framework": "Streamlit",
        "Model": "XGBoost + LightGBM",
        "Forecast": "Demand Forecasting",
        "Risk Engine": "Central Risk Scoring",
        "Risk Source": "inventory_risk_scores.csv",
    }


def get_risk_thresholds():
    """
    Return thresholds from central config.
    These values are NOT used to calculate a new risk score.
    They are only used when filtering already-generated risk scores in the dashboard.
    """
    return {"HIGH": getattr(config, "HIGH_RISK_THRESHOLD", 70), "MEDIUM": getattr(config, "MEDIUM_RISK_THRESHOLD", 40),}