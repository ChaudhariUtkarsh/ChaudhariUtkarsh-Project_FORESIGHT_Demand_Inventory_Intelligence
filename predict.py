import os
import joblib
import pandas as pd
import numpy as np


# Project Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


# Feature List
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


# Demand Predictor
class DemandPredictor:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        # Load trained model
        self.model = joblib.load(MODEL_PATH)

        # Load label encoder if available
        self.label_encoder = None

        if os.path.exists(LABEL_ENCODER_PATH):
            self.label_encoder = joblib.load(LABEL_ENCODER_PATH)


    # Prepare Input
    def prepare_input(self, data):
        input_data = {}

        # Add required numerical features
        for feature in FEATURES:
            input_data[feature] = float(data.get(feature, 0))

        # SKU Encoding
        if hasattr(self.model, "feature_names_in_"):
            model_features = list(self.model.feature_names_in_)

            if "sku_id_enc" in model_features:
                sku_id = str(data.get("sku_id", ""))

                if (self.label_encoder is not None):
                    try:
                        encoded_sku = (self.label_encoder.transform([sku_id])[0])
                    except ValueError:
                        encoded_sku = 0

                else:
                    encoded_sku = 0

                input_data["sku_id_enc"] = encoded_sku

        return pd.DataFrame([input_data])


    # Prediction
    def predict(self, data):
        X = self.prepare_input(data)

        # Keep exact model feature order
        if hasattr(self.model, "feature_names_in_"):
            model_features = list(self.model.feature_names_in_)

            for feature in model_features:
                if feature not in X.columns:
                    X[feature] = 0

            X = X[model_features]

        prediction = self.model.predict(X)[0]

        # Demand cannot be negative
        prediction = max(0, float(prediction))
        return round(prediction, 2)


# Testing
if __name__ == "__main__":
    predictor = DemandPredictor()
    sample_data = {

        "year": 2026,
        "month": 7,
        "week": 30,
        "day": 29,
        "day_of_week": 2,
        "quarter": 3,
        "is_weekend": 0,

        "lag_1": 100,
        "lag_7": 110,
        "lag_14": 105,

        "rolling_mean_7": 105,
        "rolling_std_7": 10,
        "rolling_mean_30": 108,

        "price_difference": 20,
        "discount_percentage": 10,

        "inventory_gap": 50,
        "total_inventory": 500,

        "on_hand_units": 400,
        "on_order_units": 100,
        "reorder_point": 150,

        "sku_id": "SKU001"
    }

    result = predictor.predict(sample_data)
    print(f"Predicted Demand: {result} units")