import os
import json
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


PROCESSED_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_data.csv")
WEEKLY_DATA_PATH = os.path.join(PROCESSED_DIR, "weekly_model_data.csv")
EVALUATION_PATH = os.path.join(PROCESSED_DIR, "model_evaluation.csv")
CV_RESULTS_PATH = os.path.join(PROCESSED_DIR, "rolling_origin_cv_results.csv")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


TARGET = "weekly_units_sold"
REQUIRED_SEASON_LENGTH = 52
FORECAST_HORIZON_WEEKS = 8
N_CV_FOLDS = 5


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def compute_wape(y_true, y_pred):
    """
    Calculate Weighted Absolute Percentage Error (WAPE).

    WAPE =
        SUM(|Actual - Forecast|)
        ------------------------
        SUM(|Actual|)

    Lower WAPE is better.

    The value is calculated from the actual data.
    No WAPE value is hard-coded.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have " "the same length.")
    valid_mask = (np.isfinite(y_true) & np.isfinite(y_pred))

    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        return np.nan
    denominator = np.sum(np.abs(y_true))

    if denominator == 0:
        return np.nan
    numerator = np.sum(np.abs(y_true - y_pred))

    return (numerator / denominator) * 100


def create_weekly_model_data():
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Processed data not found:\n" f"{PROCESSED_DATA_PATH}\n\n" f"Run:\n" f"python src/pipeline.py")
    logger.info(f"Loading processed data: " f"{PROCESSED_DATA_PATH}")

    df = pd.read_csv(PROCESSED_DATA_PATH)

    if "date" not in df.columns:
        raise ValueError("date column is missing " "from processed_data.csv")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()


    if "sku_id" not in df.columns:
        raise ValueError("sku_id column is missing " "from processed_data.csv")
    df["sku_id"] = (df["sku_id"].astype(str))

    if "units_sold" not in df.columns:
        raise ValueError("units_sold column is missing " "from processed_data.csv")

    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce").fillna(0)
    df["week_start"] = (df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="D"))
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)

    weekly = (df.groupby(["sku_id", "week_start"], as_index=False).agg(weekly_units_sold=("units_sold", "sum")))

    extra_columns = ["product_name", "category", "list_price", "reorder_point", "on_hand_units", "on_order_units", "lead_time_days", "safety_level", "promotion"]
    available_columns = [column for column in extra_columns if column in df.columns]

    if available_columns:
        weekly_features = (df[["sku_id", "week_start"] + available_columns].sort_values(["sku_id", "week_start"]).groupby(["sku_id", "week_start"], as_index=False).last())
        weekly = weekly.merge(weekly_features, on=["sku_id", "week_start"], how="left")

    weekly = (weekly.sort_values(["sku_id", "week_start"]).reset_index(drop=True))

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    weekly.to_csv(WEEKLY_DATA_PATH, index=False)
    logger.info(f"Weekly model data saved: " f"{WEEKLY_DATA_PATH} | " f"shape={weekly.shape}")

    print()
    print("=" * 70)
    print("WEEKLY DATA CREATED SUCCESSFULLY")
    print("=" * 70)

    print(f"Rows          : {len(weekly):,}")
    print(f"SKUs          : " f"{weekly['sku_id'].nunique():,}")
    print(f"Start week    : " f"{weekly['week_start'].min().date()}")
    print(f"End week      : " f"{weekly['week_start'].max().date()}")
    print(f"Unique weeks  : " f"{weekly['week_start'].nunique():,}")
    print(f"Output        : " f"{WEEKLY_DATA_PATH}")

    print("=" * 70)
    return weekly


def load_weekly_data():
    if os.path.exists(WEEKLY_DATA_PATH):
        logger.info(f"Loading existing weekly data: " f"{WEEKLY_DATA_PATH}")
        weekly = pd.read_csv(WEEKLY_DATA_PATH)
        required_columns = ["week_start", "sku_id", TARGET]
        missing_columns = [column for column in required_columns if column not in weekly.columns]

        if missing_columns:
            raise ValueError("weekly_model_data.csv is missing " f"required columns: {missing_columns}")

        weekly["week_start"] = pd.to_datetime(weekly["week_start"], errors="coerce")
        weekly = weekly.dropna(subset=["week_start"]).copy()
        weekly["sku_id"] = (weekly["sku_id"].astype(str))
        weekly[TARGET] = pd.to_numeric(weekly[TARGET], errors="coerce").fillna(0)
        weekly = (weekly.sort_values(["sku_id", "week_start"]).reset_index(drop=True))
        return weekly

    return create_weekly_model_data()


def determine_season_length(df):
    sku_week_counts = (df.groupby("sku_id")["week_start"].nunique())

    if sku_week_counts.empty:
        raise RuntimeError("No SKU history available.")

    max_sku_weeks = int(sku_week_counts.max())
    min_sku_weeks = int(sku_week_counts.min())
    median_sku_weeks = float(sku_week_counts.median())
    unique_weeks = int(df["week_start"].nunique())

    print()
    print("=" * 70)
    print("DATA HISTORY VALIDATION")
    print("=" * 70)

    print(f"Start              : " f"{df['week_start'].min().date()}")
    print(f"End                : " f"{df['week_start'].max().date()}")
    print(f"Unique weeks       : " f"{unique_weeks}")
    print(f"SKUs               : " f"{df['sku_id'].nunique()}")
    print(f"Min weeks per SKU  : " f"{min_sku_weeks}")
    print(f"Median weeks/SKU   : " f"{median_sku_weeks}")
    print(f"Max weeks per SKU  : " f"{max_sku_weeks}")
    print(f"Required season    : " f"{REQUIRED_SEASON_LENGTH}")


    if min_sku_weeks < REQUIRED_SEASON_LENGTH:
        raise RuntimeError(
            "\nINSUFFICIENT HISTORY\n\n"
            f"Minimum weeks for any SKU: " f"{min_sku_weeks}\n"
            f"Required: " f"{REQUIRED_SEASON_LENGTH}\n\n"
            "The official 52-week Seasonal "
            "Naive baseline requires every "
            "SKU to have at least 52 weeks."
        )

    season_length = 52

    baseline_name = ("52-week Seasonal Naive")
    baseline_status = ("FULL_52_WEEK_BASELINE")

    print()
    print("SUCCESS:")
    print("All SKUs have at least " "52 weeks of history.")

    print("Using 52-week " "Seasonal Naive baseline.")

    print("=" * 70)

    return (season_length, baseline_name, baseline_status)


def create_features(df, season_length):
    if season_length != 52:
        raise ValueError("Project FORESIGHT requires " "season_length=52.")
    df = df.copy()

    df = (df.sort_values(["sku_id", "week_start"]).reset_index(drop=True))
    group = df.groupby("sku_id", group_keys=False)
    lag_values = [1, 2, 3, 4, 8, 12, 13, 26, 52]

    for lag in lag_values:
        df[f"lag_{lag}"] = (group[TARGET].shift(lag))

    df["seasonal_lag_52"] = (group[TARGET].shift(52))

    df["rolling_mean_4"] = (group[TARGET].transform(lambda x: x.shift(1).rolling(4, min_periods=2).mean()))
    df["rolling_mean_8"] = (group[TARGET].transform(lambda x: x.shift(1).rolling(8, min_periods=3).mean()))
    df["rolling_mean_12"] = (group[TARGET].transform(lambda x: x.shift(1).rolling(12, min_periods=4).mean()))
    df["rolling_std_4"] = (group[TARGET].transform(lambda x: x.shift(1).rolling(4, min_periods=2).std()))

    df["year"] = (df["week_start"].dt.year)
    df["month"] = (df["week_start"].dt.month)
    df["week"] = (df["week_start"].dt.isocalendar().week.astype(int))
    df["quarter"] = (df["week_start"].dt.quarter)
    df["week_sin"] = (np.sin(2 * np.pi * df["week"] / 52))
    df["week_cos"] = (np.cos(2 * np.pi * df["week"] / 52))


    if ("on_hand_units" in df.columns and "on_order_units" in df.columns):
        df["on_hand_units"] = pd.to_numeric(df["on_hand_units"], errors="coerce").fillna(0)
        df["on_order_units"] = pd.to_numeric(df["on_order_units"], errors="coerce").fillna(0)
    else:
        df["on_hand_units"] = 0
        df["on_order_units"] = 0

    df["total_inventory"] = (df["on_hand_units"] + df["on_order_units"])

    if "reorder_point" not in df.columns:
        df["reorder_point"] = 0

    df["reorder_point"] = pd.to_numeric(df["reorder_point"], errors="coerce").fillna(0)
    df["inventory_gap"] = (df["total_inventory"] - df["rolling_mean_4"])

    excluded_columns = ["sku_id", "week_start", "product_name", "category"]
    numeric_columns = [column for column in df.columns if column not in excluded_columns]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["rolling_std_4"] = (df["rolling_std_4"].fillna(0))
    encoder = LabelEncoder()

    df["sku_id_enc"] = (encoder.fit_transform(df["sku_id"].astype(str)))
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(encoder, LABEL_ENCODER_PATH)
    logger.info(f"Label encoder saved: " f"{LABEL_ENCODER_PATH}")
    return df

BASE_FEATURES = [
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
    "sku_id_enc"
]


def create_rolling_splits(dates, n_splits=5):
    unique_dates = np.array(sorted(pd.Series(dates).dropna().unique()))
    total = len(unique_dates)

    if total < n_splits + 3:
        raise RuntimeError(f"Only {total} weeks available " f"for {n_splits} CV folds.")

    validation_size = max(1, total // (n_splits + 1))
    splits = []

    for fold in range(1, n_splits + 1):
        train_end_idx = (fold * validation_size - 1)

        val_start_idx = (train_end_idx + 1)
        val_end_idx = min(val_start_idx + validation_size - 1, total - 1)

        if val_start_idx > val_end_idx:
            continue

        train_dates = (unique_dates[:train_end_idx + 1])
        val_dates = (unique_dates[val_start_idx: val_end_idx + 1])
        splits.append({"fold": fold, "train_dates": train_dates, "val_dates": val_dates})
    return splits


def seasonal_naive_predict(train_df, val_df, season_length=52):
    if season_length != 52:
        raise ValueError("Project FORESIGHT official " "Seasonal Naive baseline " "must use season_length=52.")

    lookup = (train_df.set_index(["sku_id", "week_start"])[TARGET])
    actual = []
    prediction = []

    for _, row in val_df.iterrows():
        sku = str(row["sku_id"])
        current_date = pd.Timestamp(row["week_start"])
        previous_date = (current_date - pd.Timedelta(weeks=52))
        key = (sku, previous_date)

        if key not in lookup.index:
            continue

        pred = float(lookup.loc[key])
        actual.append(float(row[TARGET]))
        prediction.append(max(pred, 0))
    return (np.asarray(actual), np.asarray(prediction))


def evaluate_baseline(df, splits, season_length=52, baseline_name="52-week Seasonal Naive"):
    if season_length != 52:
        raise ValueError("Baseline evaluation must " "use a 52-week season length.")

    all_actual = []
    all_prediction = []
    results = []

    print()
    print("=" * 70)
    print(f"{baseline_name.upper()} " "ROLLING-ORIGIN CV")
    print("=" * 70)

    for split in splits:
        fold = split["fold"]

        train_dates = (split["train_dates"])
        val_dates = (split["val_dates"])
        train_df = df[df["week_start"].isin(train_dates)]
        val_df = df[df["week_start"].isin(val_dates)]
        actual, prediction = (seasonal_naive_predict(train_df, val_df, season_length=52))

        if len(actual) == 0:
            print(f"Fold {fold}: SKIPPED")
            continue

        wape = compute_wape(actual, prediction)
        all_actual.extend(actual.tolist())
        all_prediction.extend(prediction.tolist())
        results.append({
            "model": baseline_name,
            "fold": fold,
            "train_start": str(train_dates[0])[:10],
            "train_end": str(train_dates[-1])[:10],
            "val_start": str(val_dates[0])[:10],
            "val_end": str(val_dates[-1])[:10],
            "wape": round(wape, 4),
            "validation_rows": len(actual)
        })

        print(f"Fold {fold} | " f"WAPE={wape:.2f}% | " f"Rows={len(actual):,}")

    if not all_actual:
        raise RuntimeError("52-week Seasonal Naive " "baseline could not be evaluated.")
    overall_wape = compute_wape(all_actual, all_prediction)

    print("-" * 70)
    print(f"Overall Baseline WAPE: " f"{overall_wape:.2f}%")
    print("=" * 70)

    return (overall_wape, results)


def evaluate_ml_model(df, features, model_name, model, splits):

    all_actual = []
    all_prediction = []
    results = []

    print()
    print("=" * 70)
    print(f"{model_name.upper()} " "ROLLING-ORIGIN CV")
    print("=" * 70)

    for split in splits:
        fold = split["fold"]
        train_dates = (split["train_dates"])
        val_dates = (split["val_dates"])
        train_df = (df[df["week_start"].isin(train_dates)].copy())
        val_df = (df[df["week_start"].isin(val_dates)].copy())
        train_df = train_df.dropna(subset=features + [TARGET])
        val_df = val_df.dropna(subset=features + [TARGET])

        if (train_df.empty or val_df.empty):
            print(f"Fold {fold}: SKIPPED")
            continue

        X_train = train_df[features]
        y_train = train_df[TARGET]
        X_val = val_df[features]
        y_val = val_df[TARGET]
        model.fit(X_train, y_train)
        prediction = np.maximum(model.predict(X_val), 0)
        wape = compute_wape(y_val, prediction)
        all_actual.extend(y_val.tolist())
        all_prediction.extend(prediction.tolist())

        results.append({
            "model": model_name,
            "fold": fold,
            "train_start": str(train_dates[0])[:10],
            "train_end": str(train_dates[-1])[:10],
            "val_start": str(val_dates[0])[:10],
            "val_end": str(val_dates[-1])[:10],
            "wape": round(wape, 4),
            "train_rows": len(train_df),
            "validation_rows": len(val_df)
        })

        print(f"Fold {fold} | " f"WAPE={wape:.2f}% | " f"Train={len(train_df):,} | " f"Validation={len(val_df):,}")

    if not all_actual:
        raise RuntimeError(f"No valid CV results " f"for {model_name}.")

    overall_wape = compute_wape(all_actual, all_prediction)

    print("-" * 70)
    print(f"Overall {model_name} WAPE: " f"{overall_wape:.2f}%")
    print("=" * 70)

    return (overall_wape, results)


def train():

    print()
    print("=" * 70)
    print("PROJECT FORESIGHT — " "MODEL TRAINING")
    print("=" * 70)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_weekly_data()

    print(f"\nWeekly dataset shape: " f"{df.shape}")
    print(f"Target column: " f"{TARGET}")

    (season_length, baseline_name, baseline_status) = determine_season_length(df)

    df = create_features(df, season_length)
    features = [column for column in BASE_FEATURES if column in df.columns]
    seasonal_feature = "seasonal_lag_52"
    features.append(seasonal_feature)

    print(f"\nFeatures used: " f"{len(features)}")

    df_model = df.dropna(subset=[TARGET]).copy()
    print(f"Model-ready dataset shape: " f"{df_model.shape}")

    splits = create_rolling_splits(df_model["week_start"], N_CV_FOLDS)

    print(f"\nRolling-Origin CV folds: " f"{len(splits)}")
    print("Evaluation metric: WAPE")
    print(f"Baseline: " f"{baseline_name}")
    print("Baseline Season Length: " f"{REQUIRED_SEASON_LENGTH} weeks")

    (baseline_wape, baseline_results) = evaluate_baseline(df_model, splits, season_length=52, baseline_name=baseline_name)

    logger.info(f"{baseline_name} " f"CV WAPE: " f"{baseline_wape:.2f}%")

    models = {
        "xgboost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=2.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=25,
            min_child_samples=15,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=2.0,
            objective="regression",
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
    }

    evaluation_rows = []
    cv_rows = []
    model_results = {}


    for name, model in models.items():
        (cv_wape, fold_results) = evaluate_ml_model(df_model, features, name, model, splits)
        cv_rows.extend(fold_results)

        model.fit(df_model[features], df_model[TARGET])
        model_path = os.path.join(MODEL_DIR, f"{name}_model.pkl")
        joblib.dump(model, model_path)
        improvement = (baseline_wape - cv_wape)

        if (np.isfinite(baseline_wape) and baseline_wape != 0):
            improvement_percent = (improvement / baseline_wape) * 100
        else:
            improvement_percent = 0.0
        beats_baseline = (cv_wape < baseline_wape)

        model_results[name] = {
            "CV_WAPE (%)": round( cv_wape, 4),
            "Baseline_WAPE (%)": round(baseline_wape, 4),
            "WAPE_Improvement": round(improvement, 4),
            "WAPE_Improvement (%)": round(improvement_percent, 4),
            "Beats_Baseline": bool(beats_baseline),
            "Fold_WAPEs": [round(result["wape"], 4) for result in fold_results]
        }

        evaluation_rows.append({
            "model": name,
            "cv_wape": round(cv_wape, 4),
            "baseline_wape": round(baseline_wape, 4),
            "improvement": round(improvement, 4),
            "improvement_percent": round(improvement_percent, 4),
            "beats_baseline": bool(beats_baseline)
        })

        logger.info(f"{name} | " f"CV WAPE={cv_wape:.2f}% | " f"Baseline={baseline_wape:.2f}% | " f"Improvement=" f"{improvement_percent:+.2f}% | "f"Beats baseline=" f"{beats_baseline}")

    evaluation_rows.append({
        "model": baseline_name,
        "cv_wape": round(baseline_wape, 4),
        "baseline_wape": round(baseline_wape, 4),
        "improvement": 0.0,
        "improvement_percent": 0.0,
        "beats_baseline": False
    })
    cv_rows.extend(baseline_results)

    best_ml_name = min(model_results, key=lambda name: model_results[name]["CV_WAPE (%)"])
    best_ml_wape = (model_results[best_ml_name]["CV_WAPE (%)"])

    if (np.isfinite(best_ml_wape) and np.isfinite(baseline_wape) and best_ml_wape < baseline_wape):
        production_model = (best_ml_name)
        source_model_path = os.path.join(MODEL_DIR, f"{best_ml_name}_model.pkl")
        best_model = joblib.load(source_model_path)
        joblib.dump(best_model, BEST_MODEL_PATH)
        logger.info(f"Best production model: " f"{production_model}")
    else:
        production_model = ("seasonal_naive")
        baseline_object = {"model_type": "seasonal_naive", "season_length": 52, "baseline_name": baseline_name, "status": baseline_status}
        joblib.dump(baseline_object, BEST_MODEL_PATH)
        logger.info("Best production model: " f"{production_model}")

    evaluation_df = pd.DataFrame(evaluation_rows)
    evaluation_df = (evaluation_df.sort_values("cv_wape").reset_index(drop=True))
    evaluation_df.to_csv(EVALUATION_PATH, index=False)

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(CV_RESULTS_PATH, index=False)

    metrics = {
        "project": "Project FORESIGHT",
        "target": TARGET,
        "forecast_frequency": "Weekly",
        "forecast_horizon_weeks": "6-8",
        "required_season_length": REQUIRED_SEASON_LENGTH,
        "actual_season_length": season_length,
        "baseline": baseline_name,
        "baseline_status": baseline_status,
        "evaluation_metric": "WAPE",
        "rolling_origin_folds": N_CV_FOLDS,
        "baseline_cv_wape": round(baseline_wape, 4),
        "models": model_results,
        "best_ml_model": best_ml_name,
        "best_ml_cv_wape": round(best_ml_wape, 4),
        "production_model": production_model,
        "ml_beats_baseline": bool(best_ml_wape < baseline_wape)
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    metadata = {
        "project": "Project FORESIGHT",
        "target": TARGET,
        "forecast_frequency": "Weekly",
        "forecast_horizon_weeks": "6-8",
        "required_season_length": REQUIRED_SEASON_LENGTH,
        "actual_season_length": season_length,
        "baseline": baseline_name,
        "baseline_status": baseline_status,
        "features": features,
        "dataset_rows": int(len(df_model)),
        "dataset_start": str(df_model["week_start"].min()),
        "dataset_end": str(df_model["week_start"].max()),
        "unique_weeks": int(df_model["week_start"].nunique()),
        "unique_skus": int(df_model["sku_id"].nunique()),
        "production_model": production_model
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print()
    print("=" * 70)
    print("WEEKLY SKU-LEVEL MODEL RESULTS")
    print("=" * 70)

    print(f"\nBaseline: " f"{baseline_name}")
    print(f"Baseline Season Length: " f"{season_length} weeks")
    print(f"Baseline WAPE: " f"{baseline_wape:.4f}%")

    for name, result in (model_results.items()):
        print(f"\n{name.upper()}")
        print(f"CV WAPE       : " f"{result['CV_WAPE (%)']:.4f}%")
        print(f"Baseline WAPE : " f"{result['Baseline_WAPE (%)']:.4f}%")
        print(f"Improvement   : " f"{result['WAPE_Improvement (%)']:+.4f}%")
        print(f"Beats Baseline: " f"{result['Beats_Baseline']}")
        
    print(f"\nProduction Model: " f"{production_model}")
    print("\nForecast Horizon: " "6-8 Weeks")

    print()
    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(f"\nWeekly data:\n" f"{WEEKLY_DATA_PATH}")
    print(f"\nModel evaluation:\n" f"{EVALUATION_PATH}")
    print(f"\nDetailed CV results:\n" f"{CV_RESULTS_PATH}")
    print(f"\nModel metrics:\n" f"{METRICS_PATH}")
    print(f"\nMetadata:\n" f"{METADATA_PATH}")
    print(f"\nBest model:\n" f"{BEST_MODEL_PATH}")

    print()
    print("=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    train()