import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "processed", "weekly_model_data.csv")
MODEL_FILE = os.path.join(BASE_DIR, "models", "best_model.pkl")
METADATA_FILE = os.path.join(BASE_DIR, "models", "model_metadata.json")
ENCODER_FILE = os.path.join(BASE_DIR, "models", "label_encoder.pkl")


DEFAULT_FEATURES = [
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


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


METADATA = load_json(METADATA_FILE)
FEATURES = METADATA.get("features", DEFAULT_FEATURES)
MIN_WEEKS = int(METADATA.get("forecast_horizon_min_weeks", 6))
MAX_WEEKS = int(METADATA.get("forecast_horizon_max_weeks", 8))
SEASON_LENGTH = int(METADATA.get("season_length", 52))

if not FEATURES:
    FEATURES = DEFAULT_FEATURES


def _validate_horizon(forecast_weeks):
    try:
        forecast_weeks = int(forecast_weeks)
    except Exception:
        raise ValueError("forecast_weeks must be an integer.")
    if not (MIN_WEEKS <= forecast_weeks <= MAX_WEEKS):
        raise ValueError(f"Forecast horizon must be between " f"{MIN_WEEKS} and {MAX_WEEKS} weeks.")
    return forecast_weeks


def load_pickle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    with open(path, "rb") as file:
        return pickle.load(file)


class DemandPredictor:
    def __init__(self):
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(f"Weekly model data not found: {DATA_FILE}")
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(f"Production model not found: {MODEL_FILE}")
        self.data = pd.read_csv(DATA_FILE)
        if self.data.empty:
            raise ValueError("weekly_model_data.csv is empty.")
        if "week_start" not in self.data.columns:
            raise ValueError("week_start column is missing " "from weekly_model_data.csv.")

        self.data["week_start"] = pd.to_datetime(self.data["week_start"], errors="coerce")
        self.data = self.data.dropna(subset=["week_start"])

        if "sku_id" not in self.data.columns:
            raise ValueError("sku_id column is missing " "from weekly_model_data.csv.")
        self.data["sku_id"] = self.data["sku_id"].astype(str).str.strip()

        if "weekly_units_sold" not in self.data.columns:
            raise ValueError("weekly_units_sold column is missing.")
        self.data["weekly_units_sold"] = pd.to_numeric(self.data["weekly_units_sold"], errors="coerce").fillna(0.0)

        numeric_columns = ["on_hand_units", "on_order_units", "reorder_point", "list_price", "promotion"]

        for column in numeric_columns:
            if column in self.data.columns:
                self.data[column] = pd.to_numeric(self.data[column], errors="coerce").fillna(0.0)
        self.data = self.data.sort_values(["sku_id", "week_start"]).reset_index(drop=True)
        self.model = load_pickle(MODEL_FILE)
        self.label_encoder = None

        if os.path.exists(ENCODER_FILE):
            try:
                self.label_encoder = load_pickle(ENCODER_FILE)
            except Exception:
                self.label_encoder = None
        self.valid_skus = sorted(self.data["sku_id"].dropna().unique().tolist())
        if not self.valid_skus:
            raise ValueError("No valid SKUs found in weekly model data.")

    def _normalize_sku(self, sku_id):
        if sku_id is None:
            raise ValueError("sku_id is required.")
        sku = str(sku_id).strip()
        if sku.endswith(".0"):
            try:
                sku = str(int(float(sku)))
            except Exception:
                pass
        return sku

    def _encode_sku(self, sku_id):
        sku_id = self._normalize_sku(sku_id)
        if self.label_encoder is not None:
            try:
                classes = [str(x).strip() for x in self.label_encoder.classes_]
                if sku_id in classes:
                    original_value = self.label_encoder.classes_[classes.index(sku_id)]
                    return int(self.label_encoder.transform([original_value])[0])
            except Exception:
                pass

        try:
            numeric_skus = sorted([int(float(x)) for x in self.valid_skus])
            numeric_sku = int(float(sku_id))
            if numeric_sku in numeric_skus:
                return numeric_skus.index(numeric_sku)
        except Exception:
            pass
        return 0

    def _get_sku_history(self, sku_id):
        sku_id = self._normalize_sku(sku_id)
        history = self.data[self.data["sku_id"] == sku_id].copy()
        if history.empty:
            raise ValueError(f"SKU {sku_id} not found in trained model.")
        history = history.sort_values("week_start").reset_index(drop=True)
        return history

    def _build_feature_row(self, history, future_date, sku_id, sku_encoded, predicted_history):
        demand_history = list(history["weekly_units_sold"].astype(float).values)
        demand_history.extend(predicted_history)

        iso = future_date.isocalendar()
        year = int(future_date.year)
        month = int(future_date.month)
        week = int(iso.week)
        quarter = int(future_date.quarter)
        week_sin = float(np.sin(2 * np.pi * week / SEASON_LENGTH))
        week_cos = float(np.cos(2 * np.pi * week / SEASON_LENGTH))

        def get_lag(n):
            if len(demand_history) >= n:
                return float(demand_history[-n])
            return 0.0

        lag_1 = get_lag(1)
        lag_2 = get_lag(2)
        lag_3 = get_lag(3)
        lag_4 = get_lag(4)
        lag_8 = get_lag(8)
        lag_12 = get_lag(12)
        lag_13 = get_lag(13)
        lag_26 = get_lag(26)
        lag_52 = get_lag(52)

        def rolling_mean(window):
            if len(demand_history) == 0:
                return 0.0
            values = demand_history[-window:]
            return float(np.mean(values))

        def rolling_std(window):
            if len(demand_history) < 2:
                return 0.0
            values = demand_history[-window:]
            if len(values) < 2:
                return 0.0
            return float(np.std(values, ddof=1))

        rolling_mean_4 = rolling_mean(4)
        rolling_mean_8 = rolling_mean(8)
        rolling_mean_12 = rolling_mean(12)
        rolling_std_4 = rolling_std(4)

        latest = history.iloc[-1]
        list_price = float(latest.get("list_price", 0))
        on_hand_units = float(latest.get("on_hand_units", 0))
        on_order_units = float(latest.get("on_order_units", 0))
        reorder_point = float(latest.get("reorder_point", 0))
        total_inventory = (on_hand_units + on_order_units)
        inventory_gap = (reorder_point - total_inventory)
        seasonal_lag_52 = lag_52

        row = {
            "year": year,
            "month": month,
            "week": week,
            "quarter": quarter,
            "week_sin": week_sin,
            "week_cos": week_cos,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "lag_4": lag_4,
            "lag_8": lag_8,
            "lag_12": lag_12,
            "lag_13": lag_13,
            "lag_26": lag_26,
            "lag_52": lag_52,
            "rolling_mean_4": rolling_mean_4,
            "rolling_mean_8": rolling_mean_8,
            "rolling_mean_12": rolling_mean_12,
            "rolling_std_4": rolling_std_4,
            "inventory_gap": inventory_gap,
            "total_inventory": total_inventory,
            "on_hand_units": on_hand_units,
            "on_order_units": on_order_units,
            "reorder_point": reorder_point,
            "list_price": float(latest.get("list_price", 0)),
            "sku_id_enc": sku_encoded,
            "seasonal_lag_52": seasonal_lag_52
        }

        for feature in FEATURES:
            if feature not in row:
                row[feature] = 0.0
        return row


    def _predict_model(self, X):
        prediction = self.model.predict(X)
        if isinstance(prediction, (list, tuple, np.ndarray)):
            prediction = prediction[0]
        prediction = float(prediction)
        prediction = max(prediction, 0.0)
        return prediction

    def forecast(self, sku_id, forecast_weeks=6):
        forecast_weeks = _validate_horizon(forecast_weeks)
        sku_id = self._normalize_sku(sku_id)
        history = self._get_sku_history(sku_id)
        if len(history) < 52:
            raise ValueError(
                f"SKU {sku_id} has only "
                f"{len(history)} weeks of history. "
                f"At least 52 weeks are required."
            )

        sku_encoded = self._encode_sku(sku_id)
        last_date = pd.Timestamp(history["week_start"].max())

        predicted_history = []
        forecast_rows = []

        for step in range(1, forecast_weeks + 1):
            future_date = (last_date + pd.Timedelta(weeks=step))
            row = self._build_feature_row(history=history, future_date=future_date, sku_id=sku_id, sku_encoded=sku_encoded, predicted_history=predicted_history)
            X = pd.DataFrame([row])

            missing_features = [feature for feature in FEATURES if feature not in X.columns]
            if missing_features:
                raise KeyError(f"Missing model features: " f"{missing_features}")
            X = X[FEATURES]

            X = X.apply(pd.to_numeric, errors="coerce")
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(0.0)

            predicted_demand = (self._predict_model(X))
            lower_bound_80 = round(max(0.0, predicted_demand * 0.80), 2)
            upper_bound_80 = round(predicted_demand * 1.20, 2)
            predicted_history.append(predicted_demand)

            latest = history.iloc[-1]
            on_hand_units = float(latest.get("on_hand_units", 0))
            on_order_units = float(latest.get("on_order_units", 0))
            reorder_point = float(latest.get("reorder_point", 0))
            list_price = float(latest.get("list_price", 0))
            total_inventory = (on_hand_units + on_order_units)
            inventory_gap = (reorder_point - total_inventory)

            forecast_rows.append({
                "forecast_week": step,
                "week_start": future_date.strftime("%Y-%m-%d"),
                "sku_id": sku_id,
                "predicted_demand": round(predicted_demand, 2),
                "lower_bound_80": round(lower_bound_80, 2),
                "upper_bound_80": round(upper_bound_80, 2),
                "on_hand_units": round(on_hand_units, 2),
                "on_order_units": round(on_order_units, 2),
                "reorder_point": round(reorder_point,2)
            })

        forecast_df = pd.DataFrame(forecast_rows)
        total_forecast_units = float(forecast_df["predicted_demand"].sum())
        return {
            "status": "success",
            "sku_id": sku_id,
            "forecast_horizon_weeks": forecast_weeks,
            "total_forecast_units": round(total_forecast_units, 2),
            "forecast": forecast_df.to_dict(orient="records")
        }

    def predict(self, sku_id, forecast_weeks=6):
        result = self.forecast(sku_id=sku_id, forecast_weeks=forecast_weeks)
        return result

    def score(self, sku_id, forecast_weeks=6):
        result = self.forecast(sku_id=sku_id, forecast_weeks=forecast_weeks)
        return {
            "sku_id": result["sku_id"],
            "predicted_demand": result["total_forecast_units"],
            "forecast_horizon_weeks": result["forecast_horizon_weeks"],
            "forecast": result["forecast"],
            "lower_bound_80": round(sum(item["lower_bound_80"] for item in result["forecast"]), 2),
            "upper_bound_80": round(sum(item["upper_bound_80"] for item in result["forecast"]), 2)
        }


if __name__ == "__main__":
    print("=" * 70)
    print("PROJECT FORESIGHT — DEMAND PREDICTOR TEST")
    print("=" * 70)
    try:
        predictor = DemandPredictor()
        print("Model loaded successfully.")
        print(f"Available SKUs: " f"{len(predictor.valid_skus)}")
        test_sku = "101"
        print(f"\nTesting SKU: {test_sku}")
        result = predictor.forecast(sku_id=test_sku, forecast_weeks=6)
        print("\nForecast successful.")
        print("Total forecast units:", result["total_forecast_units"])
        print("\nForecast:")
        for row in result["forecast"]:
            print(row)
        print("\nSUCCESS: DemandPredictor working.")
    except Exception as exc:
        print("\nERROR:", type(exc).__name__)
        print("MESSAGE:", str(exc))