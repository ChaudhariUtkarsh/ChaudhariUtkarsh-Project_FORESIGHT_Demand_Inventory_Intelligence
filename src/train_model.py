import os
import json
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
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

TARGET    = "units_sold"
MODEL_DIR = "models"


def compute_wape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return (np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))) * 100


def baseline_wape(y_true, season_length=7):
    """Seasonal Naive: predict value from `season_length` steps ago."""
    y_true  = np.array(y_true)
    y_naive = np.concatenate([
        np.full(season_length, np.nan),
        y_true[:-season_length]
    ])
    mask = ~np.isnan(y_naive)
    return compute_wape(y_true[mask], y_naive[mask])


class ModelTrainer:
    def __init__(self, df):
        self.df = df.copy()
        self.le = LabelEncoder()
        os.makedirs(MODEL_DIR, exist_ok=True)

    def prepare_xy(self):
        logger.info("Preparing features...")
        df = self.df.copy()

        if "sku_id" in df.columns:
            df["sku_id_enc"] = self.le.fit_transform(df["sku_id"].astype(str))
            joblib.dump(self.le, os.path.join(MODEL_DIR, "label_encoder.pkl"))

        features = [f for f in FEATURES if f in df.columns]
        if "sku_id_enc" in df.columns:
            features.append("sku_id_enc")

        X = df[features].values
        y = df[TARGET].values
        return X, y, features

    def cross_validate(self, model, X, y, n_splits=5):
        """TimeSeriesSplit cross-validation — returns mean WAPE across folds."""
        tscv  = TimeSeriesSplit(n_splits=n_splits)
        wapes = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model.fit(X_tr, y_tr)
            y_pred = np.maximum(model.predict(X_val), 0)
            w = compute_wape(y_val, y_pred)
            wapes.append(w)
            logger.info(f"  Fold {fold} WAPE: {w:.2f}%")

        return np.mean(wapes)

    def train(self):
        X, y, features = self.prepare_xy()

        # ── Baseline WAPE (Seasonal Naive, season=7) ──────────────────────────
        b_wape = baseline_wape(y, season_length=7)
        logger.info(f"Baseline WAPE (Seasonal Naive): {b_wape:.2f}%")

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
            logger.info(f"Training {name} with TimeSeriesSplit (5 folds)...")

            cv_wape = self.cross_validate(model, X, y, n_splits=5)

            # Final fit on full data for saving
            model.fit(X, y)
            y_pred     = np.maximum(model.predict(X), 0)
            evaluator  = ModelEvaluator(y, y_pred)
            metrics    = evaluator.summary()
            model_wape = metrics["WAPE (%)"]

            metrics["CV_WAPE (%)"]       = round(cv_wape, 2)
            metrics["Baseline_WAPE (%)"] = round(b_wape, 2)
            metrics["WAPE_Improvement"]  = round(b_wape - model_wape, 2)
            results[name] = metrics

            model_path = os.path.join(MODEL_DIR, f"{name}_model.pkl")
            joblib.dump(model, model_path)
            logger.info(f"{name} saved → {model_path}")

            # ── WAPE Comparison ───────────────────────────────────────────────
            print("\n" + "=" * 50)
            print(f"  {name.upper()}  —  WAPE Comparison")
            print("=" * 50)
            print(f"  Baseline WAPE (Seasonal Naive) : {b_wape:.2f}%")
            print(f"  Model WAPE                     : {model_wape:.2f}%")
            print(f"  CV WAPE (TimeSeriesSplit)       : {cv_wape:.2f}%")
            improvement = b_wape - model_wape
            tag = "BETTER" if improvement > 0 else "WORSE"
            print(f"  Improvement                    : {improvement:.2f}%  [{tag}]")
            print("=" * 50)

        # ── Select best model by lowest CV WAPE ───────────────────────────────
        best = min(results, key=lambda m: results[m]["CV_WAPE (%)"])
        logger.info(f"Best model: {best} (CV WAPE={results[best]['CV_WAPE (%)']:.2f}%)")

        best_model = joblib.load(os.path.join(MODEL_DIR, f"{best}_model.pkl"))
        joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
        logger.info("Best model saved as best_model.pkl")

        results["best_model"] = best
        metrics_path = os.path.join(MODEL_DIR, "model_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Metrics saved → {metrics_path}")

        return results


if __name__ == "__main__":
    loader   = DataLoader()
    datasets = loader.load_all()

    preprocessor = DataPreprocessor(datasets)
    processed    = preprocessor.process()

    engineer = FeatureEngineering(processed)
    final_df = engineer.build_features()

    trainer = ModelTrainer(final_df)
    results = trainer.train()

    print("\n\nTraining Complete.")
    print(f"Best Model : {results['best_model'].upper()}")
