import os
import json
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from baseline import SeasonalNaiveBaseline


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
    os.path.dirname(os.path.abspath(__file__))
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

SEASON_LENGTH = 4
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
                np.abs(y_true - y_pred)
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
    from the processed daily extract.
    """

    # --------------------------------------------------------
    # IMPORTANT:
    # Create data/processed directory automatically
    # --------------------------------------------------------

    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Check processed input file
    # --------------------------------------------------------

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"\nProcessed data not found:\n{DATA_PATH}\n\n"
            "Make sure processed_data.csv exists in "
            "data/processed/."
        )

    # --------------------------------------------------------
    # Load processed data
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA_PATH
    )

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

    df["sku_id"] = df[
        "sku_id"
    ].astype(str)

    df["units_sold"] = pd.to_numeric(
        df["units_sold"],
        errors="coerce"
    ).fillna(0).clip(
        lower=0
    )

    # --------------------------------------------------------
    # Aggregate multiple rows to SKU/day
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
    # Calculate week start
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
        daily.sort_values(
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

    weekly = weekly.sort_values(
        [
            "sku_id",
            "week_start"
        ]
    ).reset_index(
        drop=True
    )

    g = weekly.groupby(
        "sku_id",
        group_keys=False
    )

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

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
    # Lag exogenous variables
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
    # Handle infinite values
    # --------------------------------------------------------

    weekly = weekly.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    # --------------------------------------------------------
    # Fill missing values
    # --------------------------------------------------------

    weekly["avg_unit_price"] = (
        weekly["avg_unit_price"]
        .fillna(0)
    )

    weekly["promotion_rate"] = (
        weekly["promotion_rate"]
        .fillna(0)
    )

    weekly["reorder_point"] = (
        weekly["reorder_point"]
        .fillna(0)
    )

    weekly["on_hand_units"] = (
        weekly["on_hand_units"]
        .fillna(0)
    )

    weekly["on_order_units"] = (
        weekly["on_order_units"]
        .fillna(0)
    )

    weekly["inventory_gap"] = (
        weekly["inventory_gap"]
        .fillna(0)
    )

    weekly["total_inventory"] = (
        weekly["total_inventory"]
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
    # Save weekly data
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

    if len(unique_dates) < n_splits + 1:

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
            unique_dates[:train_end]
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

    def __init__(self, df):

        self.df = (
            df.copy()
            .sort_values(
                [
                    "week_start",
                    "sku_id"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        self.le = LabelEncoder()

        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Prepare X and Y
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

    # --------------------------------------------------------
    # Baseline CV
    # --------------------------------------------------------

    def baseline_cv(
        self,
        n_splits=N_SPLITS
    ):

        baseline = SeasonalNaiveBaseline(
            SEASON_LENGTH
        )

        all_true = []
        all_pred = []

        for (
            _,
            train_dates,
            val_dates
        ) in rolling_origin_splits(
            self.df["week_start"],
            n_splits
        ):

            train_end = train_dates[-1]

            val_start = val_dates[0]

            val_end = val_dates[-1]

            train = self.df[
                self.df["week_start"]
                <= train_end
            ]

            val = self.df[
                (
                    self.df["week_start"]
                    >= val_start
                )
                &
                (
                    self.df["week_start"]
                    <= val_end
                )
            ].copy()

            lookup = self.df[
                [
                    "sku_id",
                    "week_start",
                    TARGET
                ]
            ].copy()

            lookup["source_week"] = (
                lookup["week_start"]
                + pd.Timedelta(
                    weeks=SEASON_LENGTH
                )
            )

            lookup = lookup.rename(
                columns={
                    TARGET:
                    "baseline_forecast"
                }
            )

            val = val.merge(
                lookup[
                    [
                        "sku_id",
                        "source_week",
                        "baseline_forecast"
                    ]
                ],
                left_on=[
                    "sku_id",
                    "week_start"
                ],
                right_on=[
                    "sku_id",
                    "source_week"
                ],
                how="left"
            )

            means = (
                train.groupby(
                    "sku_id"
                )[TARGET]
                .mean()
            )

            val["baseline_forecast"] = (
                val.apply(
                    lambda r:
                    r["baseline_forecast"]
                    if pd.notna(
                        r["baseline_forecast"]
                    )
                    else means.get(
                        r["sku_id"],
                        0
                    ),
                    axis=1
                )
                .clip(lower=0)
            )

            all_true.extend(
                val[TARGET].tolist()
            )

            all_pred.extend(
                val[
                    "baseline_forecast"
                ].tolist()
            )

        return compute_wape(
            all_true,
            all_pred
        )

    # --------------------------------------------------------
    # ML Rolling CV
    # --------------------------------------------------------

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
                "%s fold %d | train=%s..%s | val=%s..%s | WAPE=%.2f%%",
                model_cls.__name__,
                fold,
                train_dates[0].date(),
                train_dates[-1].date(),
                val_dates[0].date(),
                val_dates[-1].date(),
                wape
            )

        return (
            float(np.mean(fold_wapes)),
            fold_wapes
        )

    # --------------------------------------------------------
    # Train models
    # --------------------------------------------------------

    def train(self):

        df, features = (
            self.prepare_xy()
        )

        baseline_wape_value = (
            self.baseline_cv()
        )

        logger.info(
            "Seasonal-naive baseline CV WAPE: %.2f%%",
            baseline_wape_value
        )

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

        for (
            name,
            (
                model_cls,
                kwargs
            )
        ) in model_configs.items():

            cv_wape, fold_wapes = (
                self.rolling_origin_cv(
                    model_cls,
                    kwargs
                )
            )

            final_model = model_cls(
                **kwargs
            )

            final_model.fit(
                df[features],
                df[TARGET]
            )

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

            uncertainty_margin = (
                1.2816
                * residual_std
            )

            improvement = (
                baseline_wape_value
                - cv_wape
            )

            results[name] = {

                "CV_WAPE (%)":
                    round(
                        cv_wape,
                        2
                    ),

                "Baseline_WAPE (%)":
                    round(
                        baseline_wape_value,
                        2
                    ),

                "WAPE_Improvement":
                    round(
                        improvement,
                        2
                    ),

                "Fold_WAPEs":
                    [
                        round(x, 2)
                        for x in fold_wapes
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

            joblib.dump(
                final_model,
                os.path.join(
                    MODEL_DIR,
                    f"{name}_model.pkl"
                )
            )

        # ----------------------------------------------------
        # Select best model
        # ----------------------------------------------------

        best_ml = min(
            results,
            key=lambda k:
            results[k]["CV_WAPE (%)"]
        )

        if (
            results[best_ml]["CV_WAPE (%)"]
            < baseline_wape_value
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

            joblib.dump(
                SeasonalNaiveBaseline(
                    SEASON_LENGTH
                ),
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

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

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

            "seasonal_naive_season_length_weeks":
                SEASON_LENGTH,

            "best_model":
                best_name,

            "baseline_cv_wape":
                round(
                    baseline_wape_value,
                    2
                ),

            "note":
                "Four-week seasonal-naive lag is used because the supplied history is approximately one year; a 52-week lag would not provide a fair prior-year backtest."
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

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        results["best_model"] = (
            best_name
        )

        results["seasonal_naive"] = {

            "CV_WAPE (%)":
                round(
                    baseline_wape_value,
                    2
                )
        }

        with open(
            os.path.join(
                MODEL_DIR,
                "model_metrics.json"
            ),
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                results,
                f,
                indent=4
            )

        logger.info(
            "Best production model: %s",
            best_name
        )

        logger.info(
            "Metrics saved to model_metrics.json"
        )

        return results


# ============================================================
# MAIN
# ============================================================

def main():

    weekly = (
        prepare_weekly_data()
    )

    trainer = ModelTrainer(
        weekly
    )

    results = trainer.train()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "WEEKLY SKU-LEVEL MODEL RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        json.dumps(
            results,
            indent=4
        )
    )


# RUN
if __name__ == "__main__":
    main()