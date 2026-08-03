import os
import json
import joblib
import numpy as np
import pandas as pd


# PROJECT PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")
WEEKLY_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "weekly_model_data.csv")


# FILE VALIDATION
if not os.path.exists(BEST_MODEL_PATH):
    raise FileNotFoundError(
        f"best_model.pkl not found at: "
        f"{BEST_MODEL_PATH}. "
        f"Run src/train_model.py first."
    )


if not os.path.exists(WEEKLY_DATA_PATH):
    raise FileNotFoundError(
        f"weekly_model_data.csv not found at: "
        f"{WEEKLY_DATA_PATH}. "
        f"Run src/train_model.py first."
    )


# LOAD MODEL
best_model = joblib.load(BEST_MODEL_PATH)


# LOAD LABEL ENCODER
label_encoder = None
if os.path.exists(LABEL_ENCODER_PATH):
    label_encoder = joblib.load(LABEL_ENCODER_PATH)


# LOAD METRICS
metrics_data = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)


# LOAD METADATA
metadata = {}
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)


# FEATURES
FEATURES = metadata.get(
    "features",
    [
        "year",
        "month",
        "week",
        "quarter",
        "lag_1",
        "lag_2",
        "lag_4",
        "rolling_mean_4",
        "rolling_std_4",
        "rolling_mean_8",
        "avg_unit_price",
        "promotion_rate",
        "inventory_gap",
        "total_inventory",
        "on_hand_units",
        "on_order_units",
        "reorder_point",
        "sku_id_enc",
    ]
)


# FORECAST CONFIGURATION
SEASON_LENGTH = int(metadata.get("seasonal_naive_season_length_weeks", 4))
MIN_WEEKS = int(metadata.get("forecast_horizon_min_weeks", 6))
MAX_WEEKS = int(metadata.get("forecast_horizon_max_weeks", 8))


# LOAD WEEKLY DATA
weekly_history = pd.read_csv(WEEKLY_DATA_PATH)
weekly_history["week_start"] = pd.to_datetime(weekly_history["week_start"])
weekly_history["sku_id"] = (weekly_history["sku_id"].astype(str))


# MODEL INFORMATION
best_model_name = metrics_data.get("best_model", metadata.get("best_model", "xgboost"))


if best_model_name in metrics_data:
    uncertainty_margin = float(metrics_data[best_model_name].get("Prediction_Interval_Margin", 0.0))

else:
    uncertainty_margin = 0.0



# VALIDATE FORECAST HORIZON
def _validate_horizon(forecast_weeks):
    forecast_weeks = int(forecast_weeks)
    if not (MIN_WEEKS <= forecast_weeks <= MAX_WEEKS):
        raise ValueError(
            f"Forecast horizon must be "
            f"between {MIN_WEEKS} "
            f"and {MAX_WEEKS} weeks."
        )
    return forecast_weeks


# GET SKU HISTORY
def _sku_history(sku_id):
    sku_id = str(sku_id).strip()


    if label_encoder is not None:
        valid_skus = set(label_encoder.classes_.astype(str))

        if sku_id not in valid_skus:
            raise ValueError(
                f"SKU {sku_id} "
                f"not found in trained model."
            )


    history = weekly_history[weekly_history["sku_id"] == sku_id].sort_values("week_start").copy()
    if history.empty:
        raise ValueError(
            f"No weekly history found "
            f"for SKU {sku_id}."
        )
    return sku_id, history


# DEMAND PREDICTOR
class DemandPredictor:
    """
    Recursive weekly SKU-level predictor
    for a 6-8 week horizon.
    """

    def __init__(self):
        self.model = best_model
        self.label_encoder = (label_encoder)
        self.uncertainty_margin = (uncertainty_margin)


    # FORECAST
    def forecast(self, sku_id, forecast_weeks=6):
        forecast_weeks = (_validate_horizon(forecast_weeks))
        sku_id, history = (_sku_history(sku_id))
        if self.label_encoder is None:
            raise ValueError(
                "label_encoder.pkl "
                "not found."
            )


        encoded_sku = int(self.label_encoder.transform([sku_id])[0])
        demand_history = (history["weekly_units_sold"].astype(float).tolist())
        last = (history.iloc[-1].copy())
        last_week = pd.Timestamp(history["week_start"].max())
        rows = []


        # RECURSIVE FORECAST
        for step in range(1, forecast_weeks + 1):
            future_week = (last_week + pd.Timedelta(weeks=step))


            lag_1 = (
                demand_history[-1]
                if len(demand_history) >= 1
                else 0.0
            )

            lag_2 = (
                demand_history[-2]
                if len(demand_history) >= 2
                else lag_1
            )

            lag_4 = (
                demand_history[-4]
                if len(demand_history) >= 4
                else lag_1
            )
         
            recent4 = (
                demand_history[-4:]
                if demand_history
                else [0.0]
            )

            recent8 = (
                demand_history[-8:]
                if demand_history
                else [0.0]
            )


            avg_unit_price = float(last.get("avg_unit_price", 0) or 0)
            promotion_rate = float(last.get("promotion_rate", 0) or 0)
            reorder_point = float(last.get("reorder_point", 0) or 0)
            on_hand = float(last.get("on_hand_units", 0) or 0)
            on_order = float(last.get("on_order_units", 0) or 0)

            row = {
                "year":
                    future_week.year,
                "month":
                    future_week.month,
                "week":
                    int(future_week.isocalendar().week),
                "quarter":
                    future_week.quarter,
                "lag_1":
                    lag_1,
                "lag_2":
                    lag_2,
                "lag_4":
                    lag_4,
                "rolling_mean_4":
                    float(np.mean(recent4)),
                "rolling_std_4":
                    float(np.std(recent4)) if len(recent4) > 1 else 0.0,
                "rolling_mean_8":
                    float(np.mean(recent8)),
                "avg_unit_price":
                    avg_unit_price,
                "promotion_rate":
                    promotion_rate,
                "inventory_gap":
                    on_hand - reorder_point,
                "total_inventory":
                    on_hand + on_order,
                "on_hand_units":
                    on_hand,
                "on_order_units":
                    on_order,
                "reorder_point":
                    reorder_point,
                "sku_id_enc":
                    encoded_sku,
            }

            X = pd.DataFrame([row])[FEATURES]
            prediction = float(self.model.predict(X)[0])
            prediction = max(prediction, 0.0)


            # Prediction interval
            lower = max(prediction - self.uncertainty_margin, 0.0)
            upper = (prediction + self.uncertainty_margin)

            rows.append({
                "sku_id":
                    sku_id,
                "forecast_week":
                    step,
                "week_start":
                    future_week.strftime("%Y-%m-%d"),
                "predicted_demand":
                    round(prediction, 2),
                "lower_bound_80":
                    round(lower, 2),
                "upper_bound_80":
                    round(upper, 2),
                "on_hand_units":
                    round(on_hand, 2),
                "on_order_units":
                    round(on_order, 2),
                "reorder_point":
                    round(reorder_point, 2),
            })

            demand_history.append(prediction)


        # FORECAST DATAFRAME
        forecast_df = pd.DataFrame(rows)


        # FINAL RESULT
        return {
            "sku_id":
                sku_id,
            "forecast_horizon_weeks":
                forecast_weeks,
            "total_forecast_units":
                round(float(forecast_df["predicted_demand"].sum()), 2),
            "forecast":
                forecast_df.to_dict(orient="records"),
        }


    # BACKWARD COMPATIBLE PREDICT
    def predict(self, data):
        sku_id = str(data.get("sku_id", "")).strip()
        forecast_weeks = int(data.get("forecast_weeks", MIN_WEEKS))
        result = self.forecast(sku_id, forecast_weeks)


        return {
            "sku_id":
                result["sku_id"],

            "predicted_demand":
                result["total_forecast_units"],

            "forecast_horizon_weeks":
                result["forecast_horizon_weeks"],

            "forecast":
                result["forecast"],

            "lower_bound_80":
                round(sum(x["lower_bound_80"]for x in result["forecast"]), 2),

            "upper_bound_80":
                round(sum(x["upper_bound_80"]for x in result["forecast"]),  2),
        }