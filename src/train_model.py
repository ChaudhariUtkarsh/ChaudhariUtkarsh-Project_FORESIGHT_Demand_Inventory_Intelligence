import os
import json
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

DATA_PATH = os.path.join(
    DATA_DIR,
    "processed_data.csv"
)

WEEKLY_DATA_PATH = os.path.join(
    DATA_DIR,
    "weekly_model_data.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

# Zidio requirement:
# Same period last season.
#
# Weekly data:
# 52 weeks = approximately one year.

SEASON_LENGTH = 52

N_SPLITS = 5

TARGET = "weekly_units_sold"


FEATURES = [
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
    "sku_id_enc"
]


# ============================================================
# WAPE
# ============================================================

def compute_wape(y_true, y_pred):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    denom = np.sum(
        np.abs(y_true)
    )

    if denom > 0:

        return float(
            np.sum(
                np.abs(
                    y_true - y_pred
                )
            )
            / denom
            * 100
        )

    return 0.0


# ============================================================
# PREPARE WEEKLY DATA
# ============================================================

def prepare_weekly_data():

    """
    Create one clean row per SKU/week
    from processed daily data.
    """

    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"""
Processed data not found:

{DATA_PATH}

Make sure processed_data.csv exists in
data/processed/.
"""
        )

    logger.info(
        "Loading processed data: %s",
        DATA_PATH
    )

    df = pd.read_csv(
        DATA_PATH
    )

    # --------------------------------------------------------
    # Basic cleaning
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "date",
            "sku_id"
        ]
    )

    df["sku_id"] = (
        df["sku_id"]
        .astype(str)
    )

    df["units_sold"] = (
        pd.to_numeric(
            df["units_sold"],
            errors="coerce"
        )
        .fillna(0)
        .clip(lower=0)
    )

    # --------------------------------------------------------
    # Daily aggregation
    # --------------------------------------------------------

    daily = (
        df.groupby(
            [
                "sku_id",
                "date"
            ],
            as_index=False
        )
        .agg(
            units_sold=(
                "units_sold",
                "sum"
            ),

            avg_unit_price=(
                "unit_price",
                "mean"
            ),

            promotion=(
                "promotion",
                "max"
            ),

            list_price=(
                "list_price",
                "first"
            ),

            reorder_point=(
                "reorder_point",
                "last"
            ),

            on_hand_units=(
                "on_hand_units",
                "last"
            ),

            on_order_units=(
                "on_order_units",
                "last"
            )
        )
    )

    # --------------------------------------------------------
    # Week start
    # --------------------------------------------------------

    daily["date"] = pd.to_datetime(
        daily["date"]
    )

    daily["week_start"] = (
        daily["date"]
        - pd.to_timedelta(
            daily["date"].dt.dayofweek,
            unit="D"
        )
    ).dt.normalize()

    # --------------------------------------------------------
    # Weekly aggregation
    # --------------------------------------------------------

    weekly = (
        daily
        .sort_values(
            [
                "sku_id",
                "date"
            ]
        )
        .groupby(
            [
                "sku_id",
                "week_start"
            ],
            as_index=False
        )
        .agg(

            weekly_units_sold=(
                "units_sold",
                "sum"
            ),

            avg_unit_price=(
                "avg_unit_price",
                "mean"
            ),

            promotion_rate=(
                "promotion",
                "mean"
            ),

            list_price=(
                "list_price",
                "last"
            ),

            reorder_point=(
                "reorder_point",
                "last"
            ),

            on_hand_units=(
                "on_hand_units",
                "last"
            ),

            on_order_units=(
                "on_order_units",
                "last"
            )
        )
    )

    weekly = (
        weekly
        .sort_values(
            [
                "sku_id",
                "week_start"
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    g = weekly.groupby(
        "sku_id",
        group_keys=False
    )

    weekly["lag_1"] = (
        g["weekly_units_sold"]
        .shift(1)
    )

    weekly["lag_2"] = (
        g["weekly_units_sold"]
        .shift(2)
    )

    weekly["lag_4"] = (
        g["weekly_units_sold"]
        .shift(4)
    )

    # --------------------------------------------------------
    # Rolling features
    # --------------------------------------------------------

    weekly["rolling_mean_4"] = (
        g["weekly_units_sold"]
        .transform(
            lambda s:
            s.shift(1)
            .rolling(
                4,
                min_periods=1
            )
            .mean()
        )
    )

    weekly["rolling_std_4"] = (
        g["weekly_units_sold"]
        .transform(
            lambda s:
            s.shift(1)
            .rolling(
                4,
                min_periods=2
            )
            .std()
        )
    )

    weekly["rolling_mean_8"] = (
        g["weekly_units_sold"]
        .transform(
            lambda s:
            s.shift(1)
            .rolling(
                8,
                min_periods=1
            )
            .mean()
        )
    )

    # --------------------------------------------------------
    # Previous-week operational features
    # --------------------------------------------------------

    for col in [
        "avg_unit_price",
        "promotion_rate",
        "reorder_point",
        "on_hand_units",
        "on_order_units"
    ]:

        weekly[col] = (
            g[col]
            .shift(1)
        )

    # --------------------------------------------------------
    # Inventory features
    # --------------------------------------------------------

    weekly["inventory_gap"] = (
        weekly["on_hand_units"]
        - weekly["reorder_point"]
    )

    weekly["total_inventory"] = (
        weekly["on_hand_units"]
        + weekly["on_order_units"]
    )

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    weekly["year"] = (
        weekly["week_start"]
        .dt.year
    )

    weekly["month"] = (
        weekly["week_start"]
        .dt.month
    )

    weekly["week"] = (
        weekly["week_start"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    weekly["quarter"] = (
        weekly["week_start"]
        .dt.quarter
    )

    # --------------------------------------------------------
    # Replace invalid values
    # --------------------------------------------------------

    weekly = weekly.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    # --------------------------------------------------------
    # Fill numeric values
    # --------------------------------------------------------

    for col in [
        "avg_unit_price",
        "promotion_rate",
        "reorder_point",
        "on_hand_units",
        "on_order_units",
        "inventory_gap",
        "total_inventory"
    ]:

        weekly[col] = (
            weekly[col]
            .fillna(0)
        )

    for col in [
        "lag_1",
        "lag_2",
        "lag_4",
        "rolling_mean_4",
        "rolling_std_4",
        "rolling_mean_8"
    ]:

        weekly[col] = (
            weekly[col]
            .fillna(0)
        )

    # --------------------------------------------------------
    # Save weekly model dataset
    # --------------------------------------------------------

    weekly.to_csv(
        WEEKLY_DATA_PATH,
        index=False
    )

    logger.info(
        "Weekly model data saved: %s | shape=%s",
        WEEKLY_DATA_PATH,
        weekly.shape
    )

    return weekly


# ============================================================
# ROLLING ORIGIN SPLITS
# ============================================================

def rolling_origin_splits(
    unique_dates,
    n_splits=5
):

    unique_dates = np.array(
        sorted(
            pd.to_datetime(
                unique_dates
            ).unique()
        )
    )

    if len(unique_dates) < (
        n_splits + 1
    ):

        raise ValueError(
            "Not enough weekly periods "
            "for rolling-origin CV."
        )

    val_size = max(
        1,
        len(unique_dates)
        // (n_splits + 1)
    )

    first_train_end = (
        len(unique_dates)
        - n_splits * val_size
    )

    if first_train_end < 1:

        first_train_end = 1

    for fold in range(
        n_splits
    ):

        train_end = (
            first_train_end
            + fold * val_size
        )

        val_start = train_end

        val_end = min(
            val_start + val_size,
            len(unique_dates)
        )

        if val_start >= val_end:
            continue

        train_dates = (
            unique_dates[
                :train_end
            ]
        )

        val_dates = (
            unique_dates[
                val_start:val_end
            ]
        )

        yield (
            fold + 1,
            train_dates,
            val_dates
        )


# ============================================================
# MODEL TRAINER
# ============================================================

class ModelTrainer:

    def __init__(
        self,
        df
    ):

        self.df = (
            df.copy()
            .sort_values(
                [
                    "week_start",
                    "sku_id"
                ]
            )
            .reset_index(drop=True)
        )

        self.le = LabelEncoder()

        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Prepare X and y
    # --------------------------------------------------------

    def prepare_xy(self):

        self.df["sku_id"] = (
            self.df["sku_id"]
            .astype(str)
        )

        self.df["sku_id_enc"] = (
            self.le.fit_transform(
                self.df["sku_id"]
            )
        )

        joblib.dump(
            self.le,
            os.path.join(
                MODEL_DIR,
                "label_encoder.pkl"
            )
        )

        return (
            self.df,
            FEATURES
        )

    # ========================================================
    # SEASONAL-NAIVE ROLLING CV
    # ========================================================

    def baseline_cv(
        self,
        n_splits=N_SPLITS
    ):

        all_true = []
        all_pred = []

        fold_results = []

        for (
            fold,
            train_dates,
            val_dates
        ) in rolling_origin_splits(
            self.df["week_start"],
            n_splits
        ):

            train = self.df[
                self.df["week_start"]
                .isin(train_dates)
            ].copy()

            val = self.df[
                self.df["week_start"]
                .isin(val_dates)
            ].copy()

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Same period last season:
            # current week - 52 weeks
            #
            # ONLY training data can be used.
            # This prevents validation leakage.
            # ------------------------------------------------

            train_lookup = train[
                [
                    "sku_id",
                    "week_start",
                    TARGET
                ]
            ].copy()

            train_lookup[
                "forecast_week"
            ] = (
                train_lookup["week_start"]
                + pd.Timedelta(
                    weeks=SEASON_LENGTH
                )
            )

            train_lookup = (
                train_lookup
                .rename(
                    columns={
                        TARGET:
                        "baseline_forecast"
                    }
                )
            )

            val = val.merge(
                train_lookup[
                    [
                        "sku_id",
                        "forecast_week",
                        "baseline_forecast"
                    ]
                ],
                left_on=[
                    "sku_id",
                    "week_start"
                ],
                right_on=[
                    "sku_id",
                    "forecast_week"
                ],
                how="left"
            )

            # ------------------------------------------------
            # If 52-week history does not exist,
            # do NOT use future validation values.
            #
            # Fallback = SKU training mean.
            # ------------------------------------------------

            means = (
                train
                .groupby("sku_id")[TARGET]
                .mean()
            )

            val["baseline_forecast"] = (
                val.apply(
                    lambda row:
                    row["baseline_forecast"]
                    if pd.notna(
                        row["baseline_forecast"]
                    )
                    else means.get(
                        row["sku_id"],
                        0
                    ),
                    axis=1
                )
                .clip(lower=0)
            )

            fold_wape = compute_wape(
                val[TARGET],
                val["baseline_forecast"]
            )

            fold_results.append(
                {
                    "Fold": fold,
                    "WAPE (%)":
                    round(
                        fold_wape,
                        4
                    )
                }
            )

            all_true.extend(
                val[TARGET]
                .tolist()
            )

            all_pred.extend(
                val["baseline_forecast"]
                .tolist()
            )

            logger.info(
                "Seasonal-Naive fold %d | "
                "train=%s..%s | "
                "val=%s..%s | "
                "WAPE=%.2f%%",
                fold,
                train_dates[0].date(),
                train_dates[-1].date(),
                val_dates[0].date(),
                val_dates[-1].date(),
                fold_wape
            )

        overall_wape = compute_wape(
            all_true,
            all_pred
        )

        return (
            float(overall_wape),
            fold_results
        )

    # ========================================================
    # ML ROLLING ORIGIN CV
    # ========================================================

    def rolling_origin_cv(
        self,
        model_cls,
        model_kwargs,
        n_splits=N_SPLITS
    ):

        fold_wapes = []

        for (
            fold,
            train_dates,
            val_dates
        ) in rolling_origin_splits(
            self.df["week_start"],
            n_splits
        ):

            train = self.df[
                self.df["week_start"]
                .isin(train_dates)
            ]

            val = self.df[
                self.df["week_start"]
                .isin(val_dates)
            ]

            model = model_cls(
                **model_kwargs
            )

            model.fit(
                train[FEATURES],
                train[TARGET]
            )

            pred = np.maximum(
                model.predict(
                    val[FEATURES]
                ),
                0
            )

            wape = compute_wape(
                val[TARGET],
                pred
            )

            fold_wapes.append(
                float(wape)
            )

            logger.info(
                "%s fold %d | "
                "train=%s..%s | "
                "val=%s..%s | "
                "WAPE=%.2f%%",
                model_cls.__name__,
                fold,
                train_dates[0].date(),
                train_dates[-1].date(),
                val_dates[0].date(),
                val_dates[-1].date(),
                wape
            )

        return (
            float(
                np.mean(
                    fold_wapes
                )
            ),
            fold_wapes
        )

    # ========================================================
    # TRAIN ALL MODELS
    # ========================================================

    def train(self):

        df, features = (
            self.prepare_xy()
        )

        # ----------------------------------------------------
        # Seasonal-Naive baseline
        # ----------------------------------------------------

        (
            baseline_wape,
            baseline_fold_results
        ) = self.baseline_cv()

        logger.info(
            "Seasonal-Naive baseline "
            "CV WAPE: %.2f%%",
            baseline_wape
        )

        # ----------------------------------------------------
        # Model configurations
        # ----------------------------------------------------

        model_configs = {

            "xgboost": (
                XGBRegressor,

                dict(
                    n_estimators=350,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    objective="reg:squarederror",
                    verbosity=0
                )
            ),

            "lightgbm": (
                LGBMRegressor,

                dict(
                    n_estimators=350,
                    learning_rate=0.05,
                    max_depth=6,
                    num_leaves=31,
                    random_state=42,
                    verbosity=-1
                )
            )
        }

        results = {}

        # ====================================================
        # TRAIN XGBOOST + LIGHTGBM
        # ====================================================

        for (
            name,
            (
                model_cls,
                kwargs
            )
        ) in model_configs.items():

            # ------------------------------------------------
            # Rolling-Origin CV
            # ------------------------------------------------

            (
                cv_wape,
                fold_wapes
            ) = self.rolling_origin_cv(
                model_cls,
                kwargs
            )

            # ------------------------------------------------
            # Final model on complete data
            # ------------------------------------------------

            final_model = model_cls(
                **kwargs
            )

            final_model.fit(
                df[features],
                df[TARGET]
            )

            # ------------------------------------------------
            # Residual-based uncertainty
            # ------------------------------------------------

            train_pred = np.maximum(
                final_model.predict(
                    df[features]
                ),
                0
            )

            residual_std = float(
                np.std(
                    df[TARGET].values
                    - train_pred,
                    ddof=1
                )
            )

            # 80% interval:
            # approximately 1.2816 standard deviations

            uncertainty_margin = (
                1.2816
                * residual_std
            )

            # ------------------------------------------------
            # Improvement over baseline
            # ------------------------------------------------

            improvement = (
                baseline_wape
                - cv_wape
            )

            improvement_percent = 0.0

            if baseline_wape > 0:

                improvement_percent = (
                    (
                        baseline_wape
                        - cv_wape
                    )
                    / baseline_wape
                ) * 100

            # ------------------------------------------------
            # Save results
            # ------------------------------------------------

            results[name] = {

                "CV_WAPE (%)":
                    round(
                        cv_wape,
                        2
                    ),

                "Baseline_WAPE (%)":
                    round(
                        baseline_wape,
                        2
                    ),

                "WAPE_Improvement":
                    round(
                        improvement,
                        2
                    ),

                "WAPE_Improvement (%)":
                    round(
                        improvement_percent,
                        2
                    ),

                "Beats_Baseline":
                    bool(
                        cv_wape
                        < baseline_wape
                    ),

                "Fold_WAPEs":
                    [
                        round(
                            x,
                            2
                        )
                        for x
                        in fold_wapes
                    ],

                "Prediction_Interval_Margin":
                    round(
                        uncertainty_margin,
                        4
                    ),

                "Uncertainty_Level":
                    "80%",

                "Residual_STD":
                    round(
                        residual_std,
                        4
                    )
            }

            # ------------------------------------------------
            # Save model
            # ------------------------------------------------

            joblib.dump(
                final_model,
                os.path.join(
                    MODEL_DIR,
                    f"{name}_model.pkl"
                )
            )

            logger.info(
                "%s | CV WAPE=%.2f%% | "
                "Baseline=%.2f%% | "
                "Improvement=%.2f%% | "
                "Beats baseline=%s",
                name,
                cv_wape,
                baseline_wape,
                improvement_percent,
                cv_wape < baseline_wape
            )

        # ====================================================
        # BEST MODEL
        # ====================================================

        best_ml = min(
            results,
            key=lambda name:
            results[name]["CV_WAPE (%)"]
        )

        if (
            results[best_ml]["CV_WAPE (%)"]
            < baseline_wape
        ):

            best_name = best_ml

        else:

            best_name = (
                "seasonal_naive"
            )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if best_name == "seasonal_naive":

            # Save metadata object for baseline model
            seasonal_naive_metadata = {

                "model_type":
                    "Seasonal-Naive",

                "season_length_weeks":
                    SEASON_LENGTH
            }

            joblib.dump(
                seasonal_naive_metadata,
                os.path.join(
                    MODEL_DIR,
                    "best_model.pkl"
                )
            )

        else:

            best_model = joblib.load(
                os.path.join(
                    MODEL_DIR,
                    f"{best_name}_model.pkl"
                )
            )

            joblib.dump(
                best_model,
                os.path.join(
                    MODEL_DIR,
                    "best_model.pkl"
                )
            )

        # ====================================================
        # MODEL METADATA
        # ====================================================

        metadata = {

            "target":
                TARGET,

            "features":
                features,

            "forecast_grain":
                "weekly",

            "forecast_horizon_min_weeks":
                6,

            "forecast_horizon_max_weeks":
                8,

            "seasonal_naive_method":
                "same_period_last_season",

            "seasonal_naive_season_length_weeks":
                SEASON_LENGTH,

            "rolling_origin_cv_splits":
                N_SPLITS,

            "evaluation_metric":
                "WAPE",

            "best_model":
                best_name,

            "baseline_cv_wape":
                round(
                    baseline_wape,
                    2
                )
        }

        with open(
            os.path.join(
                MODEL_DIR,
                "model_metadata.json"
            ),
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4
            )

        # ====================================================
        # ADD BASELINE TO RESULTS
        # ====================================================

        results["seasonal_naive"] = {

            "CV_WAPE (%)":
                round(
                    baseline_wape,
                    2
                ),

            "Baseline_WAPE (%)":
                round(
                    baseline_wape,
                    2
                ),

            "WAPE_Improvement":
                0.0,

            "WAPE_Improvement (%)":
                0.0,

            "Beats_Baseline":
                False,

            "Fold_WAPEs":
                [
                    item["WAPE (%)"]
                    for item
                    in baseline_fold_results
                ],

            "Prediction_Interval_Margin":
                None,

            "Uncertainty_Level":
                None,

            "Residual_STD":
                None
        }

        # ====================================================
        # SAVE JSON METRICS
        # ====================================================

        metrics_json_path = os.path.join(
            MODEL_DIR,
            "model_metrics.json"
        )

        with open(
            metrics_json_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                results,
                f,
                indent=4
            )

        # ====================================================
        # SAVE CSV METRICS
        # ====================================================

        csv_rows = []

        # Seasonal Naive
        csv_rows.append({

            "Model":
                "Seasonal-Naive",

            "WAPE (%)":
                round(
                    baseline_wape,
                    2
                ),

            "Baseline WAPE (%)":
                round(
                    baseline_wape,
                    2
                ),

            "Improvement (%)":
                0.0,

            "Beats Baseline":
                False
        })

        # ML models
        for name in [
            "xgboost",
            "lightgbm"
        ]:

            if name not in results:
                continue

            csv_rows.append({

                "Model":
                    name.upper(),

                "WAPE (%)":
                    results[name][
                        "CV_WAPE (%)"
                    ],

                "Baseline WAPE (%)":
                    results[name][
                        "Baseline_WAPE (%)"
                    ],

                "Improvement (%)":
                    results[name][
                        "WAPE_Improvement (%)"
                    ],

                "Beats Baseline":
                    results[name][
                        "Beats_Baseline"
                    ]
            })

        metrics_df = pd.DataFrame(
            csv_rows
        )

        metrics_df = (
            metrics_df
            .sort_values(
                "WAPE (%)"
            )
            .reset_index(
                drop=True
            )
        )

        metrics_csv_path = os.path.join(
            DATA_DIR,
            "model_evaluation.csv"
        )

        metrics_df.to_csv(
            metrics_csv_path,
            index=False
        )

        # ====================================================
        # SAVE DETAILED FOLD RESULTS
        # ====================================================

        fold_rows = []

        # Baseline folds
        for item in baseline_fold_results:

            fold_rows.append({

                "Model":
                    "Seasonal-Naive",

                "Fold":
                    item["Fold"],

                "WAPE (%)":
                    item["WAPE (%)"]
            })

        # ML folds
        for name in [
            "xgboost",
            "lightgbm"
        ]:

            if name not in results:
                continue

            for fold_no, fold_wape in enumerate(
                results[name]["Fold_WAPEs"],
                start=1
            ):

                fold_rows.append({

                    "Model":
                        name.upper(),

                    "Fold":
                        fold_no,

                    "WAPE (%)":
                        fold_wape
                })

        fold_df = pd.DataFrame(
            fold_rows
        )

        fold_csv_path = os.path.join(
            DATA_DIR,
            "rolling_origin_cv_results.csv"
        )

        fold_df.to_csv(
            fold_csv_path,
            index=False
        )

        # ====================================================
        # FINAL LOGGING
        # ====================================================

        logger.info(
            "Best production model: %s",
            best_name
        )

        logger.info(
            "Metrics JSON saved: %s",
            metrics_json_path
        )

        logger.info(
            "Evaluation CSV saved: %s",
            metrics_csv_path
        )

        logger.info(
            "Rolling-origin CV results saved: %s",
            fold_csv_path
        )

        return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("PROJECT FORESIGHT — MODEL TRAINING")
    print("=" * 70)

    weekly = prepare_weekly_data()

    print(
        f"\nWeekly dataset shape: {weekly.shape}"
    )

    print(
        f"Seasonal period: "
        f"{SEASON_LENGTH} weeks"
    )

    print(
        f"Rolling-Origin CV folds: "
        f"{N_SPLITS}"
    )

    print(
        "Evaluation metric: WAPE"
    )

    print(
        "Baseline: Same period last season"
    )

    trainer = ModelTrainer(
        weekly
    )

    results = trainer.train()

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("WEEKLY SKU-LEVEL MODEL RESULTS")
    print("=" * 70)

    print(
        json.dumps(
            results,
            indent=4
        )
    )

    print("\n")
    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(
        f"Weekly data:\n"
        f"{WEEKLY_DATA_PATH}"
    )

    print(
        f"\nModel evaluation:\n"
        f"{os.path.join(DATA_DIR, 'model_evaluation.csv')}"
    )

    print(
        f"\nDetailed CV results:\n"
        f"{os.path.join(DATA_DIR, 'rolling_origin_cv_results.csv')}"
    )

    print(
        f"\nModel metrics:\n"
        f"{os.path.join(MODEL_DIR, 'model_metrics.json')}"
    )

    print(
        f"\nMetadata:\n"
        f"{os.path.join(MODEL_DIR, 'model_metadata.json')}"
    )

    print(
        f"\nBest model:\n"
        f"{os.path.join(MODEL_DIR, 'best_model.pkl')}"
    )

    print("\n")
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()