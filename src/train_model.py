import os
import json
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from data_loader import DataLoader
from preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineering
from evaluate import ModelEvaluator


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
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
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def compute_wape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom  = np.sum(np.abs(y_true))
    return (np.sum(np.abs(y_true - y_pred)) / denom) * 100 if denom > 0 else 0.0


def baseline_wape(y_true, season_length=7):
    """Seasonal Naive: predict value from `season_length` steps ago."""
    y_true  = np.array(y_true)
    y_naive = np.concatenate([np.full(season_length, np.nan), y_true[:-season_length]])
    mask = ~np.isnan(y_naive)
    return compute_wape(y_true[mask], y_naive[mask])


def rolling_origin_splits(dates, n_splits=5):
    """
    Date-based rolling-origin splits.

    Divides the full date range into (n_splits + 1) equal windows.
    Each fold expands the training window by one window and validates on the immediately following window — no data leakage.

    Fold 1 : Train [window 0]            Val [window 1]
    Fold 2 : Train [window 0-1]          Val [window 2]
    Fold 3 : Train [window 0-2]          Val [window 3]
    Fold 4 : Train [window 0-3]          Val [window 4]
    Fold 5 : Train [window 0-4]          Val [window 5]

    Yields
    ------
    fold        : int
    train_mask  : boolean Series
    val_mask    : boolean Series
    train_start : date
    train_end   : date
    val_start   : date
    val_end     : date
    """
    unique_dates = np.sort(dates.unique())
    total_days   = len(unique_dates)
    window_size  = total_days // (n_splits + 1)

    for fold in range(1, n_splits + 1):
        train_end_idx = fold * window_size - 1
        val_end_idx   = min((fold + 1) * window_size - 1, total_days - 1)
        train_dates = unique_dates[:train_end_idx + 1]
        val_dates   = unique_dates[train_end_idx + 1 : val_end_idx + 1]

        if len(val_dates) == 0:
            continue

        train_mask = dates.isin(train_dates)
        val_mask   = dates.isin(val_dates)
        yield (fold, train_mask, val_mask, train_dates[0],  train_dates[-1], val_dates[0],    val_dates[-1])


class ModelTrainer:
    def __init__(self, df):
        self.df       = df.copy().sort_values("date").reset_index(drop=True)
        self.le       = LabelEncoder()
        self.features = None
        os.makedirs(MODEL_DIR, exist_ok=True)

    def prepare_xy(self):
        logger.info("Preparing features...")
        df = self.df.copy()

        if "sku_id" in df.columns:
            df["sku_id_enc"] = self.le.fit_transform(df["sku_id"].astype(str))
            joblib.dump(self.le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
            logger.info("label_encoder.pkl saved.")

        features = [f for f in FEATURES if f in df.columns]
        if "sku_id_enc" in df.columns:
            features.append("sku_id_enc")

        self.features = features
        self.df       = df          # keep encoded df for rolling-origin
        return df, features

    def rolling_origin_cv(self, model_cls, model_kwargs, n_splits=5):
        """
        Proper date-based rolling-origin cross-validation.

        - Trains on all past dates up to train_end
        - Validates on the immediately following date window
        - Prints fold-level details: date ranges + WAPE
        - Returns mean WAPE and per-fold WAPE list
        """
        df    = self.df
        dates = df["date"]
        wapes = []

        print(f"\n  {'Fold':<6} {'Train Start':<14} {'Train End':<14} "
              f"{'Val Start':<14} {'Val End':<14} {'WAPE':>8}")
        print("  " + "-" * 74)

        for fold, train_mask, val_mask, tr_s, tr_e, vl_s, vl_e in \
                rolling_origin_splits(dates, n_splits):

            X_tr = df.loc[train_mask, self.features].values
            y_tr = df.loc[train_mask, TARGET].values
            X_vl = df.loc[val_mask,   self.features].values
            y_vl = df.loc[val_mask,   TARGET].values

            if len(y_vl) == 0:
                continue

            m = model_cls(**model_kwargs)
            m.fit(X_tr, y_tr)
            y_pred = np.maximum(m.predict(X_vl), 0)

            w = compute_wape(y_vl, y_pred)
            wapes.append(w)

            print(f"  {fold:<6} "
                  f"{str(tr_s)[:10]:<14} {str(tr_e)[:10]:<14} "
                  f"{str(vl_s)[:10]:<14} {str(vl_e)[:10]:<14} "
                  f"{w:>7.2f}%")

        print("  " + "-" * 74)
        mean_wape = float(np.mean(wapes))
        print(f"  {'Mean CV WAPE':<54} {mean_wape:>7.2f}%\n")
        return mean_wape, wapes

    def train(self):
        df, features = self.prepare_xy()

        # Baseline WAPE on full sorted target
        b_wape = baseline_wape(df[TARGET].values, season_length=7)
        logger.info(f"Baseline WAPE (Seasonal Naive, lag-7): {b_wape:.2f}%")

        model_configs = {
            "xgboost": (
                XGBRegressor,
                dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                     random_state=42, verbosity=0)
            ),
            "lightgbm": (
                LGBMRegressor,
                dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                     random_state=42, verbose=-1)
            )
        }

        results = {}

        for name, (model_cls, model_kwargs) in model_configs.items():
            print("\n" + "=" * 78)
            print(f"  ROLLING-ORIGIN BACKTESTING — {name.upper()}  (5 Folds, Date-Based)")
            print("=" * 78)

            cv_wape, fold_results = self.rolling_origin_cv(model_cls, model_kwargs, n_splits=5)

            # Final fit on full data
            final_model = model_cls(**model_kwargs)
            final_model.fit(df[features].values, df[TARGET].values)
            y_pred     = np.maximum(final_model.predict(df[features].values), 0)
            evaluator  = ModelEvaluator(df[TARGET].values, y_pred)
            metrics    = evaluator.summary()
            model_wape = metrics["WAPE (%)"]

            metrics["CV_WAPE (%)"] = round(cv_wape, 2)
            metrics["Baseline_WAPE (%)"] = round(b_wape, 2)
            metrics["WAPE_Improvement"] = round(b_wape - cv_wape, 2)
            metrics["Rolling_Origin_Folds"] = fold_results
            metrics["Fold_WAPEs"] = fold_results
            results[name] = metrics


            # Calculate residuals from full-data fitted model
            residuals = df[TARGET].values - y_pred

            # Residual standard deviation
            residual_std = float(np.std(residuals, ddof=1))
            z_80 = 1.2816

            # Margin of uncertainty
            uncertainty_margin = z_80 * residual_std
            metrics["Uncertainty_Level"] = "80%"
            metrics["Residual_STD"] = round(residual_std, 4)
            metrics["Prediction_Interval_Margin"] = round(uncertainty_margin, 4)
            logger.info(f"80% Forecast Uncertainty Interval calculated | " f"Margin = ±{uncertainty_margin:.4f}")

            model_path = os.path.join(MODEL_DIR, f"{name}_model.pkl")
            joblib.dump(final_model, model_path)
            logger.info(f"{name} saved → {model_path}")

            improvement = b_wape - cv_wape
            tag = "BETTER [+]" if improvement > 0 else "WORSE [-]"
            print(f"  Baseline WAPE (Seasonal Naive) : {b_wape:.2f}%")
            print(f"  Model WAPE    (full-data fit)  : {model_wape:.2f}%")
            print(f"  CV WAPE       (rolling-origin) : {cv_wape:.2f}%")
            print(f"  Improvement vs Baseline (CV)  : {improvement:+.2f}%  [{tag}]")
            print("=" * 78)

        # Best model by lowest CV WAPE
        best = min(results, key=lambda m: results[m]["CV_WAPE (%)"])
        logger.info(f"Best model: {best.upper()} (CV WAPE = {results[best]['CV_WAPE (%)']:.2f}%)")

        best_model = joblib.load(os.path.join(MODEL_DIR, f"{best}_model.pkl"))
        joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
        logger.info("best_model.pkl saved.")

        results["best_model"] = best
        metrics_path = os.path.join(MODEL_DIR, "model_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Metrics saved → {metrics_path}")

        return results


def print_final_proof(results):
    """Final comparison table: Seasonal Naive vs all models."""
    best        = results["best_model"]
    b_wape      = results[best]["Baseline_WAPE (%)"]
    model_names = [k for k in results if k != "best_model"]

    print("\n")
    print("=" * 78)
    print("  FINAL PROOF — Seasonal Naive Baseline vs ML Models (Rolling-Origin CV)")
    print("=" * 78)
    print(f"  {'Model':<20} {'WAPE (%)':>10} {'CV WAPE (%)':>13} {'vs Baseline':>13}  {'Fold WAPEs'}")
    print("-" * 78)
    print(f"  {'Seasonal Naive':<20} {b_wape:>9.2f}%  {'—':>12}  {'—':>12}")
    print("-" * 78)

    for name in model_names:
        m           = results[name]
        model_wape  = m["WAPE (%)"]
        cv_wape     = m["CV_WAPE (%)"]
        improvement = b_wape - cv_wape
        tag         = "BETTER [+]" if improvement > 0 else "WORSE [-]"
        marker      = " << BEST" if name == best else ""
        fold_str    = ", ".join(f"{w:.2f}%" for w in m.get("Fold_WAPEs", []))
        print(f"  {name.upper():<20} {model_wape:>9.2f}%  {cv_wape:>12.2f}%  {improvement:>+12.2f}%  {tag}{marker}")
        print(f"  {'':20}  Fold WAPEs: {fold_str}")

    print("=" * 78)
    print(f"  Best Model Selected : {best.upper()}  (lowest CV WAPE)")
    print(f"  Baseline WAPE       : {b_wape:.2f}%")
    print(f"  Best Model CV WAPE  : {results[best]['CV_WAPE (%)']:.2f}%")
    print(f"  Improvement         : " f"{results[best]['WAPE_Improvement']:+.2f}%")
    print("=" * 78)


if __name__ == "__main__":
    loader   = DataLoader()
    datasets = loader.load_all()
    preprocessor = DataPreprocessor(datasets)
    processed    = preprocessor.process()
    engineer = FeatureEngineering(processed)
    final_df = engineer.build_features()
    trainer = ModelTrainer(final_df)
    results = trainer.train()

    print_final_proof(results)
    print("\nTraining Complete.")
    print(f"Best Model : {results['best_model'].upper()}")