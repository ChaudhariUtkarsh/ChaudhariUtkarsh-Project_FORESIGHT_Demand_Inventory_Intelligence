import os
import json
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from data_loader import DataLoader
from preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineering
from evaluate import ModelEvaluator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


FEATURES = [
    "year", "month", "week", "day", "day_of_week", "quarter", "is_weekend",
    "lag_1", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_std_7", "rolling_mean_30",
    "price_difference", "discount_percentage",
    "inventory_gap", "total_inventory",
    "on_hand_units", "on_order_units", "reorder_point"
]

TARGET = "units_sold"
MODEL_DIR = "models"


class ModelTrainer:
    def __init__(self, df):
        self.df = df
        self.le = LabelEncoder()
        os.makedirs(MODEL_DIR, exist_ok=True)

    def prepare_data(self):
        logger.info("Preparing training data...")
        df = self.df.copy()

        if "sku_id" in df.columns:
            df["sku_id_enc"] = self.le.fit_transform(df["sku_id"].astype(str))
            joblib.dump(self.le, os.path.join(MODEL_DIR, "label_encoder.pkl"))

        features = [f for f in FEATURES if f in df.columns]
        if "sku_id_enc" in df.columns:
            features.append("sku_id_enc")

        X = df[features]
        y = df[TARGET]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):
        X_train, X_test, y_train, y_test = self.prepare_data()

        models = {
            "xgboost": XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                verbosity=0
            ),
            "lightgbm": LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                verbose=-1
            )
        }

        results = {}
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred = np.maximum(y_pred, 0)

            evaluator = ModelEvaluator(y_test, y_pred)
            metrics = evaluator.summary()
            results[name] = metrics

            model_path = os.path.join(MODEL_DIR, f"{name}_model.pkl")
            joblib.dump(model, model_path)
            logger.info(f"{name} saved at {model_path}")

        best = min(results, key=lambda m: results[m]["RMSE"])
        logger.info(f"Best model: {best} (RMSE={results[best]['RMSE']})")

        best_model = joblib.load(os.path.join(MODEL_DIR, f"{best}_model.pkl"))
        joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
        logger.info("Best model saved as best_model.pkl")

        results["best_model"] = best
        metrics_path = os.path.join(MODEL_DIR, "model_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Metrics saved at {metrics_path}")

        return results


if __name__ == "__main__":
    loader = DataLoader()
    datasets = loader.load_all()

    preprocessor = DataPreprocessor(datasets)
    processed = preprocessor.process()

    engineer = FeatureEngineering(processed)
    final_df = engineer.build_features()

    trainer = ModelTrainer(final_df)
    results = trainer.train()

    print("\nTraining Complete.")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}: {metrics}")
