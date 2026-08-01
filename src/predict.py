import os
import json
import joblib
import numpy as np


# PROJECT ROOT
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# MODEL PATHS
MODEL_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")


# LOAD MODEL
if not os.path.exists(BEST_MODEL_PATH):
    raise FileNotFoundError(f"best_model.pkl not found at: {BEST_MODEL_PATH}")
best_model = joblib.load(BEST_MODEL_PATH)


# LOAD LABEL ENCODER
if os.path.exists(LABEL_ENCODER_PATH):
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
else:
    label_encoder = None


# LOAD METRICS
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        metrics_data = json.load(f)

else:
    metrics_data = {}


# GET UNCERTAINTY MARGIN
best_model_name = metrics_data.get("best_model")
if best_model_name and best_model_name in metrics_data:
    best_metrics = metrics_data[best_model_name]

else:
    best_metrics = {}


uncertainty_margin = best_metrics.get("Prediction_Interval_Margin", 0.0)
uncertainty_margin = float(uncertainty_margin)


# SIMPLE PREDICTION FUNCTION
def predict_demand(features):
    """
    Predict demand and calculate 80% prediction interval.
    """
    X = np.array(features, dtype=float).reshape(1, -1)
    prediction = best_model.predict(X)[0]
    prediction = max(float(prediction), 0.0)
    lower_bound = max(prediction - uncertainty_margin, 0.0)
    upper_bound = (prediction + uncertainty_margin)
    interval_width = (upper_bound - lower_bound)
    return {
        "Predicted Demand": round(prediction, 2),
        "80% Lower Bound": round(lower_bound, 2),
        "80% Upper Bound": round(upper_bound, 2),
        "80% Prediction Interval": f"{lower_bound:.2f} - {upper_bound:.2f}",
        "Interval Width": round(interval_width, 2)
    }


# API PREDICTOR CLASS
class DemandPredictor:
    def __init__(self):
        self.model = best_model
        self.label_encoder = label_encoder
        self.uncertainty_margin = (uncertainty_margin)

    # SKU PREDICTION
    def predict(self, data):
        try:
            sku_id = str(data["sku_id"]).strip()

            if self.label_encoder is None:
                raise ValueError("label_encoder.pkl not found.")

            if sku_id not in self.label_encoder.classes_:
                raise ValueError(
                    f"SKU {sku_id} not found in trained model. "
                    f"Valid SKU range: {self.label_encoder.classes_[0]} "
                    f"to {self.label_encoder.classes_[-1]}"
                )

            encoded_sku = self.label_encoder.transform([sku_id])[0]

            features = [
                encoded_sku,
                data["year"],
                data["month"],
                data["week"],
                data["day"],
                data["day_of_week"],
                data["quarter"],
                data["is_weekend"],
                data["lag_1"],
                data["lag_7"],
                data["lag_14"],
                data["rolling_mean_7"],
                data["rolling_std_7"],
                data["rolling_mean_30"],
                data["price_difference"],
                data["discount_percentage"],
                data["inventory_gap"],
                data["total_inventory"],
                data["on_hand_units"],
                data["on_order_units"],
                data["reorder_point"]
            ]


            if len(features) != self.model.n_features_in_:
                raise ValueError(
                    f"Feature mismatch: sending {len(features)} features, "
                    f"but model expects {self.model.n_features_in_}."
                )


            prediction = float(self.model.predict([features])[0])
            prediction = max(prediction, 0.0)

            lower_bound = max(prediction - self.uncertainty_margin, 0.0)
            upper_bound = (prediction + self.uncertainty_margin)

            return {
                "sku_id": sku_id,
                "predicted_demand": round(prediction, 2),
                "lower_bound_80": round(lower_bound, 2),
                "upper_bound_80": round(upper_bound, 2),
                "uncertainty_margin": round(self.uncertainty_margin, 2)
            }

        except Exception as e:
            raise ValueError(str(e))