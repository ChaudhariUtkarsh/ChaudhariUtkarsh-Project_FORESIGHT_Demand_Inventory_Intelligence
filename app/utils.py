import os
import joblib
import pandas as pd
import streamlit as st


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        return df
    return pd.DataFrame()


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found : {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    return model


@st.cache_resource
def load_encoder():
    if os.path.exists(ENCODER_PATH):
        encoder = joblib.load(ENCODER_PATH)
        return encoder
    return None


def check_resources():
    status = {
        "dataset": os.path.exists(DATA_PATH),
        "model": os.path.exists(MODEL_PATH),
        "encoder": os.path.exists(ENCODER_PATH),
        "reports": os.path.exists(REPORT_DIR)
    }
    return status


def dataset_info(df):
    return {"Rows": len(df), "Columns": len(df.columns), "Column Names": list(df.columns)}


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
    "sku_id_enc"
]


def predict_demand(model, encoder, input_data):
    """Predict demand using trained model."""
    df = input_data.copy()
    if encoder is not None and "sku_id" in df.columns:
        try:
            df["sku_id_enc"] = encoder.transform(df["sku_id"].astype(str))
        except Exception:
            df["sku_id_enc"] = 0
    if "sku_id" in df.columns:
        df = df.drop(columns=["sku_id"])
    X = df[FEATURE_COLUMNS]
    prediction = model.predict(X)
    prediction = prediction.clip(min=0)
    return prediction

def calculate_revenue(forecast_units, selling_price):
    """Calculate expected revenue."""
    return forecast_units * selling_price

def calculate_revenue_at_risk(forecast_units, inventory, selling_price):
    """Calculate revenue lost due to stockout."""
    shortage = max(forecast_units - inventory, 0)
    return shortage * selling_price

def calculate_capital_locked(inventory, forecast_units, unit_cost):
    """Calculate excess inventory cost."""
    excess_inventory = max(inventory - forecast_units, 0)
    return excess_inventory * unit_cost

def inventory_ratio(inventory, forecast):
    """Inventory / Forecast Ratio (%)"""
    if forecast <= 0:
        return 0
    return round((inventory / forecast) * 100, 2)


def calculate_risk(inventory, forecast, stockout_threshold=100, overstock_threshold=150):
    """Inventory Risk Classification"""
    ratio = inventory_ratio(inventory, forecast)
    if ratio < stockout_threshold:
        return ("High Stockout", 90)
    elif ratio > overstock_threshold:
        return ("Overstock", 80)
    else:
        return ("Healthy", 25)


def calculate_kpis(df):
    """Dashboard KPI Summary"""
    result = {}
    result["total_products"] = len(df)
    result["forecast_units"] = int(df["forecast_units"].sum())
    result["inventory"] = int(df["on_hand_units"].sum())
    result["expected_revenue"] = float((df["forecast_units"] * df["selling_price"]).sum())
    result["capital_locked"] = float(((df["on_hand_units"] - df["forecast_units"]).clip(lower=0) * df["unit_cost"]).sum())
    result["revenue_at_risk"] = float(((df["forecast_units"] - df["on_hand_units"]).clip(lower=0) * df["selling_price"]).sum())
    result["stockout"] = int((df["forecast_units"] > df["on_hand_units"]).sum())
    result["overstock"] = int((df["on_hand_units"] > df["forecast_units"]).sum())
    return result


import os
from datetime import datetime


def export_csv(df, filename="forecast_results.csv"):
    """Save DataFrame as CSV."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, filename)
    df.to_csv(path, index=False)
    return path

def generate_report(kpis, filename="Business_Report.txt"):
    """Generate Business Report."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, filename)
    report = f"""
--------------------------------------------------------

PROJECT FORESIGHT
Business Report

--------------------------------------------------------

Generated On :
{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}

--------------------------------------------------------

Total Products : {kpis.get("total_products",0)}
Forecast Units : {kpis.get("forecast_units",0)}
Inventory : {kpis.get("inventory",0)}
Revenue at Risk : {kpis.get("revenue_at_risk",0):,.2f}
Capital Locked : {kpis.get("capital_locked",0):,.2f}
Expected Revenue : {kpis.get("expected_revenue",0):,.2f}
Stockout Products : {kpis.get("stockout",0)}
Overstock Products : {kpis.get("overstock",0)}

--------------------------------------------------------

Business Recommendation
    1. Increase inventory for Stockout items.
    2. Reduce Overstock inventory.
    3. Review demand forecast weekly.
    4. Improve inventory planning.

--------------------------------------------------------

Generated By
Project FORESIGHT

--------------------------------------------------------
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report_path


def format_date(date_value):
    try:
        return pd.to_datetime(date_value).strftime("%d-%m-%Y")
    except:
        return str(date_value)

def format_currency(amount):
    return f"{amount:,.2f}"

def format_percentage(value):
    return f"{value:.2f} %"

def save_forecast_history(record, filename="forecast_history.csv"):
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, filename)
    df = pd.DataFrame([record])
    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)
    return path

def file_exists(path):
    return os.path.exists(path)

def safe_divide(a, b):
    if b == 0:
        return 0
    return a / b

def get_timestamp():
    return datetime.now().strftime("%d-%m-%Y %H:%M:%S")

def show_success(message):
    return {"status": "success", "message": message}

def show_error(message):
    return {"status": "error", "message": message}

def project_info():
    return {
        "Project": "Project FORESIGHT",
        "Version": "1.0",
        "Framework": "Streamlit",
        "Model": "XGBoost + LightGBM",
        "Forecast": "Demand Forecasting",
        "Risk Engine": "Inventory Intelligence"
    }