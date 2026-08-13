import os
import json
import joblib
import numpy as np
import pandas as pd

import config


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

WEEKLY_DATA_PATH = os.path.join(PROCESSED_DIR, "weekly_model_data.csv")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_data.csv")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

OUTPUT_PATH = os.path.join(PROCESSED_DIR, "inventory_risk_scores.csv")
REORDER_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "reorder_priority_list.csv")
MARKDOWN_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "markdown_clear_priority_list.csv")


def numeric_value(series, default=0):
    return (pd.to_numeric(series, errors="coerce").fillna(default))

def parse_dates(df):
    df = df.copy()
    if "week_start" in df.columns:
        df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def get_model_metadata():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("model_metadata.json not found:\n" f"{METADATA_PATH}")
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    return metadata

def get_model_features():
    metadata = get_model_metadata()
    features = metadata.get("features", [])
    if not features:
        raise ValueError("No features found in " "model_metadata.json")
    return features

def enrich_inventory_information(df):
    df = df.copy()
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError("processed_data.csv not found:\n" f"{PROCESSED_DATA_PATH}")
    processed = pd.read_csv(PROCESSED_DATA_PATH)
    processed = parse_dates(processed)
    if "sku_id" not in processed.columns:
        raise ValueError("sku_id column missing from " "processed_data.csv")
    df["sku_id"] = (df["sku_id"].astype(str))
    processed["sku_id"] = (processed["sku_id"].astype(str))
    inventory_columns = ["sku_id", "lead_time_days", "safety_level", "reorder_point", "product_name", "category", "list_price"]
    available_columns = [col for col in inventory_columns if col in processed.columns]
    if "date" in processed.columns:
        processed = (processed.sort_values(["sku_id", "date"]).groupby("sku_id", as_index=False).last())
    else:
        processed = (processed.groupby("sku_id", as_index=False).last())
    inventory_info = (processed[available_columns].copy())
    merge_columns = [col for col in available_columns if (col != "sku_id" and col not in df.columns)]

    if merge_columns:
        df = df.merge(inventory_info[["sku_id"] + merge_columns], on="sku_id", how="left")

    if "lead_time_days" not in df.columns:
        df["lead_time_days"] = 14
    df["lead_time_days"] = numeric_value(df["lead_time_days"], default=14)
    df["lead_time_days"] = np.clip(df["lead_time_days"], 1, 365)

    if "reorder_point" not in df.columns:
        df["reorder_point"] = 0
    df["reorder_point"] = numeric_value(df["reorder_point"])

    if "safety_level" not in df.columns:
        df["safety_level"] = np.nan
    df["safety_level"] = pd.to_numeric(df["safety_level"], errors="coerce")
    df["safety_level"] = (df["safety_level"].fillna(df["reorder_point"]))
    zero_safety = (df["safety_level"] <= 0)
    df.loc[zero_safety, "safety_level"] = (df.loc[zero_safety, "reorder_point"] * config.SAFETY_MULTIPLIER)
    return df

def create_model_features(df):
    df = df.copy()
    df = parse_dates(df)
    required = ["sku_id", "week_start", "weekly_units_sold"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("Missing columns:\n" f"{missing}")

    df["sku_id"] = (df["sku_id"].astype(str))
    df["weekly_units_sold"] = numeric_value(df["weekly_units_sold"])
    df = (df.sort_values(["sku_id", "week_start"]).reset_index(drop=True))
    group = df.groupby("sku_id", group_keys=False)

    lag_values = [1, 2, 3, 4, 8, 12, 13, 26, 52]

    for lag in lag_values:
        df[f"lag_{lag}"] = (group["weekly_units_sold"].shift(lag))
    df["seasonal_lag_52"] = (group["weekly_units_sold"].shift(52))
    df["rolling_mean_4"] = (group["weekly_units_sold"].transform(lambda x: x.shift(1).rolling(4, min_periods=2).mean()))
    df["rolling_mean_8"] = (group["weekly_units_sold"].transform(lambda x: x.shift(1).rolling(8, min_periods=3).mean()))
    df["rolling_mean_12"] = (group["weekly_units_sold"].transform(lambda x: x.shift(1).rolling(12, min_periods=4).mean()))
    df["rolling_std_4"] = (group["weekly_units_sold"].transform(lambda x: x.shift(1).rolling(4, min_periods=2).std()))

    df["year"] = (df["week_start"].dt.year)
    df["month"] = (df["week_start"].dt.month)
    df["week"] = (df["week_start"].dt.isocalendar().week.astype(int))
    df["quarter"] = (df["week_start"].dt.quarter)
    df["week_sin"] = (np.sin(2 * np.pi * df["week"] / config.SEASON_LENGTH))
    df["week_cos"] = (np.cos(2 * np.pi * df["week"] / config.SEASON_LENGTH))

    for col in ["on_hand_units", "on_order_units", "reorder_point"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = numeric_value(df[col])
    df["total_inventory"] = (df["on_hand_units"] + df["on_order_units"])
    df["inventory_gap"] = (df["total_inventory"] - df["rolling_mean_4"])

    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError("label_encoder.pkl not found:\n" f"{LABEL_ENCODER_PATH}")
    encoder = joblib.load(LABEL_ENCODER_PATH)
    known_skus = set(encoder.classes_.astype(str))

    def encode_sku(sku):
        sku = str(sku)
        if sku in known_skus:
            return int(encoder.transform([sku])[0])
        return -1

    df["sku_id_enc"] = (df["sku_id"].astype(str).map(encode_sku))
    model_features = (get_model_features())
    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    for col in model_features:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["rolling_std_4"] = (df["rolling_std_4"].fillna(0))
    return df

def calculate_forecast_demand(df):
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError("best_model.pkl not found:\n" f"{BEST_MODEL_PATH}")
    model_object = joblib.load(BEST_MODEL_PATH)

    if (isinstance(model_object, dict) and model_object.get("model_type") == "seasonal_naive"):
        season_length = int(model_object.get("season_length", config.SEASON_LENGTH))
        if season_length != config.SEASON_LENGTH:
            raise ValueError("Project FORESIGHT requires " f"{config.SEASON_LENGTH}-week " "Seasonal Naive baseline.")
        df = (df.sort_values(["sku_id", "week_start"]).copy())
        lookup = (df.set_index(["sku_id", "week_start"])["weekly_units_sold"])
        forecasts = []

        for _, row in df.iterrows():
            sku = str(row["sku_id"])
            current_date = pd.Timestamp(row["week_start"])
            previous_date = (current_date - pd.Timedelta(weeks=config.SEASON_LENGTH))
            key = (sku, previous_date)
            if key in lookup.index:
                forecast = float(lookup.loc[key])
            else:
                forecast = float(
                    row.get("weekly_units_sold", 0))
            forecasts.append(max(forecast, 0))
        df["forecast_weekly_demand"] = forecasts
        return df

    if not isinstance(model_object, dict):
        model = model_object
        print("\nCreating model features...")
        df = create_model_features(df)
        features = (get_model_features())

        print(f"Model features required: " f"{len(features)}")
        print("\nFeatures:")

        for feature in features:
            print(f"  - {feature}")

        missing_features = [col for col in features if col not in df.columns]
        if missing_features:
            raise ValueError("Missing model features:\n" f"{missing_features}")
        model_df = df.copy()
        for col in features:
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce").fillna(0)
        X = model_df[features]

        if X.shape[1] != len(features):
            raise ValueError("Feature count mismatch.\n" f"Model features: {len(features)}\n" f"Prepared features: {X.shape[1]}")

        print("\nGenerating ML forecast...")
        prediction = model.predict(X)

        df["forecast_weekly_demand"] = np.maximum(prediction, 0)
        return df
    raise ValueError("Unknown production model format.")


def calculate_risk_score(df):
    df = df.copy()
    required = ["sku_id", "week_start", "forecast_weekly_demand", "on_hand_units", "on_order_units", "reorder_point", "lead_time_days"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError("Missing risk columns:\n" f"{missing}")

    numeric_columns = ["forecast_weekly_demand", "on_hand_units", "on_order_units", "reorder_point", "lead_time_days"]
    for col in numeric_columns:
        df[col] = numeric_value(df[col])

    if "safety_level" not in df.columns:
        df["safety_level"] = 0
    df["safety_level"] = pd.to_numeric(df["safety_level"], errors="coerce").fillna(0)
    df["safety_level"] = np.maximum(df["safety_level"], 0)
    df["available_inventory"] = (df["on_hand_units"] + df["on_order_units"])
    df["available_inventory"] = np.maximum(df["available_inventory"], 0)

    df["lead_time_demand"] = (df["forecast_weekly_demand"] * (df["lead_time_days"] / config.DAYS_PER_WEEK))
    df["projected_inventory"] = (df["available_inventory"] - df["lead_time_demand"])

    safe_demand = (df["forecast_weekly_demand"].clip(lower=0.01))
    df["inventory_coverage_weeks"] = ( df["available_inventory"] / safe_demand)
    df["inventory_coverage_weeks"] = (df["inventory_coverage_weeks"].replace([np.inf, -np.inf], 999).clip(0, 999))
    df["risk_gap"] = (df["projected_inventory"] - df["safety_level"])

    df["stockout_pressure"] = np.where(df["lead_time_demand"] > 0, np.clip(((df["lead_time_demand"] - df["available_inventory"]) / df["lead_time_demand"]) * 100, 0, 100), 0)
    df["safety_gap_pressure"] = np.where(df["safety_level"] > 0, np.clip(((df["safety_level"] - df["projected_inventory"]) / df["safety_level"]) * 100, 0, 100), 0)
    df["coverage_pressure"] = np.clip(((config.MEDIUM_COVERAGE_WEEKS - df["inventory_coverage_weeks"]) / config.MEDIUM_COVERAGE_WEEKS) * 100, 0, 100)
    df["demand_pressure"] = np.where(df["available_inventory"] > 0, np.clip((df["forecast_weekly_demand"] / df["available_inventory"]) * 100, 0, 100), 100)
    df["lead_time_pressure"] = np.clip(((df["lead_time_days"] - 7) / 30) * 100, 0, 100)
    df["reorder_point_pressure"] = np.where(df["reorder_point"] > 0, np.clip(((df["reorder_point"] - df["available_inventory"]) / df["reorder_point"]) * 100, 0, 100), 0)
    df["risk_score"] = (0.30 * df["stockout_pressure"] + 0.20 * df["coverage_pressure"] + 0.15 * df["safety_gap_pressure"] + 0.15 * df["demand_pressure"] + 0.10 * df["lead_time_pressure"] + 0.10 * df["reorder_point_pressure"])

    projected_shortage = np.maximum(-df["projected_inventory"], 0)
    shortage_ratio = np.where(df["lead_time_demand"] > 0, projected_shortage / df["lead_time_demand"], 0)
    shortage_adjustment = np.clip(shortage_ratio * 15, 0, 15)
    df["risk_score"] = (df["risk_score"] + shortage_adjustment)
    df["risk_score"] = (np.clip(df["risk_score"], 0, 100).round(2))

    df["risk_level"] = np.select([df["risk_score"] >= config.HIGH_RISK_THRESHOLD, df["risk_score"] >= config.MEDIUM_RISK_THRESHOLD], ["HIGH", "MEDIUM"], default="LOW")
    df["stockout_risk_score"] = np.clip(df["stockout_pressure"], 0, 100)

    df["stockout_risk_level"] = np.select([df["stockout_risk_score"] >= config.STOCKOUT_HIGH_THRESHOLD, df["stockout_risk_score"] >= config.STOCKOUT_MEDIUM_THRESHOLD], ["HIGH", "MEDIUM"], default="LOW")
    df["stockout_risk"] = (df["stockout_risk_level"])
    df["stockout_flag"] = np.where((df["projected_inventory"] <= 0) | (df["inventory_coverage_weeks"] <= config.LOW_COVERAGE_WEEKS), "YES", "NO")

    df["risk_reason"] = np.select(
        [
            df["stockout_flag"] == "YES",
            df["inventory_coverage_weeks"] <= config.LOW_COVERAGE_WEEKS,
            df["projected_inventory"] < df["safety_level"],
            df["lead_time_days"] >= 21,
            df["forecast_weekly_demand"] > df["available_inventory"]
        ],
        [
            "Potential stockout during lead time",
            "Low inventory coverage",
            "Inventory below safety level",
            "High supplier lead time",
            "Demand is high relative to available inventory"
        ], default="Inventory position currently adequate"
    )

    df["target_inventory_level"] = (df["lead_time_demand"] + df["safety_level"])
    df["reorder_quantity"] = np.maximum((df["target_inventory_level"] - df["available_inventory"]), 0)
    df["reorder_quantity"] = (np.ceil(df["reorder_quantity"]).astype(int))

    critical_threshold = max(config.HIGH_RISK_THRESHOLD + 10, 80)
    high_threshold = (config.HIGH_RISK_THRESHOLD)
    medium_threshold = (config.MEDIUM_RISK_THRESHOLD)
    low_reorder_threshold = max(config.MEDIUM_RISK_THRESHOLD - 15, 0)

    df["reorder_priority"] = np.select(
        [
            (df["risk_score"] >= critical_threshold) & (df["reorder_quantity"] > 0),
            (df["risk_score"] >= high_threshold) & (df["reorder_quantity"] > 0),
            (df["risk_score"] >= medium_threshold) & (df["reorder_quantity"] > 0),
            (df["risk_score"] >= low_reorder_threshold) & (df["reorder_quantity"] > 0)
        ], ["CRITICAL", "HIGH", "MEDIUM", "LOW"], default="NONE"
    )
    df["inventory_status"] = np.select(
        [
            df["projected_inventory"] <= 0,
            df["inventory_coverage_weeks"] <= config.LOW_COVERAGE_WEEKS,
            df["projected_inventory"] < df["safety_level"],
            df["inventory_coverage_weeks"] >= config.HIGH_COVERAGE_WEEKS
        ], ["STOCKOUT RISK", "LOW COVERAGE", "BELOW SAFETY LEVEL", "EXCESS INVENTORY"], default="HEALTHY"
    )

    excess_inventory = (df["available_inventory"] - (df["lead_time_demand"] + df["safety_level"]))
    df["excess_inventory_units"] = (np.maximum(excess_inventory, 0))
    target = (df["lead_time_demand"] + df["safety_level"])
    df["excess_inventory_ratio"] = np.where(target > 0, (df["excess_inventory_units"] / target) * 100, 0)
    df["excess_inventory_ratio"] = (np.clip(df["excess_inventory_ratio"],0, 100).round(2))

    coverage_excess_pressure = np.clip(((df["inventory_coverage_weeks"] - config.MEDIUM_COVERAGE_WEEKS) / config.MEDIUM_COVERAGE_WEEKS) * 100, 0, 100)
    excess_pressure = np.clip(df["excess_inventory_ratio"], 0, 100)

    low_demand_pressure = np.clip(100 - np.where(df["available_inventory"] > 0, (df["forecast_weekly_demand"] / df["available_inventory"]) * 100, 100), 0, 100)
    low_risk_pressure = np.clip(100 - df["risk_score"], 0, 100)

    df["markdown_clear_score"] = (0.40 * coverage_excess_pressure + 0.30 * excess_pressure + 0.20 * low_demand_pressure + 0.10 * low_risk_pressure)
    df["markdown_clear_score"] = np.where(df["risk_level"] == "HIGH", 0, df["markdown_clear_score"])
    df["markdown_clear_score"] = (np.clip(df["markdown_clear_score"], 0, 100).round(2))

    markdown_high_threshold = (config.OVERSTOCK_HIGH_THRESHOLD)
    markdown_medium_threshold = (config.OVERSTOCK_MEDIUM_THRESHOLD)

    df["markdown_clear_priority"] = np.select(
        [
            df["risk_level"] == "HIGH",
            (df["markdown_clear_score"] >= markdown_high_threshold) & (df["excess_inventory_units"] > 0),
            (df["markdown_clear_score"] >= markdown_medium_threshold) & (df["excess_inventory_units"] > 0),
            (df["markdown_clear_score"] >= config.MEDIUM_RISK_THRESHOLD) & (df["excess_inventory_units"] > 0)
        ], ["NONE", "HIGH", "MEDIUM", "LOW"], default="NONE"
    )

    df["markdown_clear_reason"] = np.select(
        [
            df["risk_level"] == "HIGH",
            df["inventory_coverage_weeks"]>= (config.HIGH_COVERAGE_WEEKS * 2),
            (df["inventory_coverage_weeks"] >= config.HIGH_COVERAGE_WEEKS) & (df["excess_inventory_units"] > 0),
            (df["excess_inventory_units"] > 0) & (df["forecast_weekly_demand"] < df["available_inventory"])
        ],
        [
            "High risk inventory - do not markdown",
            "Very high inventory coverage",
            "High coverage with excess inventory",
            "Inventory exceeds forecast demand"
        ], default="No significant markdown requirement"
    )

    df["markdown_discount_percent"] = np.select(
        [
            df["markdown_clear_priority"] == "HIGH",
            df["markdown_clear_priority"] == "MEDIUM",
            df["markdown_clear_priority"] == "LOW"
        ], [30, 20, 10], default=0
    )

    df["recommended_action"] = np.select(
        [
            df["reorder_priority"] == "CRITICAL",
            df["reorder_priority"] == "HIGH",
            df["reorder_priority"] == "MEDIUM",
            df["reorder_priority"] == "LOW",
            df["markdown_clear_priority"] == "HIGH",
            df["markdown_clear_priority"] == "MEDIUM",
            df["markdown_clear_priority"] == "LOW"
        ],
        [
            "URGENT REORDER",
            "URGENT REORDER",
            "MONITOR AND PLAN REORDER",
            "PLAN REORDER",
            "PRIORITISE MARKDOWN / CLEARANCE",
            "CONSIDER PROMOTIONAL PRICING",
            "CONSIDER MARKDOWN"
        ], default="NO IMMEDIATE ACTION"
    )

    df["final_inventory_decision"] = np.select(
        [
            df["reorder_priority"] == "CRITICAL",
            df["reorder_priority"] == "HIGH",
            df["reorder_priority"] == "MEDIUM",
            df["markdown_clear_priority"] == "HIGH",
            df["markdown_clear_priority"] == "MEDIUM",
            df["markdown_clear_priority"] == "LOW"
        ], ["URGENT REORDER", "REORDER", "PLAN REORDER", "CLEAR INVENTORY", "MARKDOWN", "CONSIDER MARKDOWN"],
        default="HOLD / MONITOR"
    )
    return df


def run_risk_scoring():
    print("\n")
    print("=" * 70)
    print("PROJECT FORESIGHT — " "BALANCED INVENTORY RISK SCORING")
    print("=" * 70)

    print("\n")
    print("=" * 70)
    print("CENTRAL RISK CONFIGURATION")
    print("=" * 70)

    print(f"High Risk Threshold        : " f"{config.HIGH_RISK_THRESHOLD}")
    print(f"Medium Risk Threshold      : " f"{config.MEDIUM_RISK_THRESHOLD}")
    print(f"Stockout High Threshold    : " f"{config.STOCKOUT_HIGH_THRESHOLD}")
    print(f"Stockout Medium Threshold  : " f"{config.STOCKOUT_MEDIUM_THRESHOLD}")
    print(f"Low Coverage Weeks         : " f"{config.LOW_COVERAGE_WEEKS}")
    print(f"Medium Coverage Weeks      : " f"{config.MEDIUM_COVERAGE_WEEKS}")
    print(f"High Coverage Weeks        : " f"{config.HIGH_COVERAGE_WEEKS}")
    print(f"Overstock High Threshold   : " f"{config.OVERSTOCK_HIGH_THRESHOLD}")
    print(f"Overstock Medium Threshold : " f"{config.OVERSTOCK_MEDIUM_THRESHOLD}")
    print(f"Days Per Week              : " f"{config.DAYS_PER_WEEK}")
    print(f"Safety Multiplier          : " f"{config.SAFETY_MULTIPLIER}")
    print(f"Season Length              : " f"{config.SEASON_LENGTH}")
    print("=" * 70)

    if not os.path.exists(WEEKLY_DATA_PATH):
        raise FileNotFoundError("weekly_model_data.csv not found:\n" f"{WEEKLY_DATA_PATH}")
        
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError("best_model.pkl not found:\n" f"{BEST_MODEL_PATH}")

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("model_metadata.json not found:\n" f"{METADATA_PATH}")

    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError("label_encoder.pkl not found:\n" f"{LABEL_ENCODER_PATH}")

    print("\nLoading weekly model data...")
    df = pd.read_csv(WEEKLY_DATA_PATH)
    df = parse_dates(df)
    print(f"Input rows: {len(df):,}")

    print("\nLoading inventory information...")
    df = enrich_inventory_information(df)
    print("Inventory information ready.")

    metadata = get_model_metadata()
    print("\nProduction model:")
    print(f"  {metadata.get('production_model', 'UNKNOWN')}")
    print("\nModel features:")
    model_features = (metadata.get("features", []))

    for feature in model_features:
        print(f"  - {feature}")

    print(f"\nTotal model features: " f"{len(model_features)}")
    print("\nGenerating forecast demand...")
    df = calculate_forecast_demand(df)
    print("Forecast generated successfully.")

    print("\nCalculating balanced inventory risk...")
    df = calculate_risk_score(df)
    print("Balanced risk scoring completed successfully.")

    output_columns = [
        "sku_id",
        "week_start",
        "product_name",
        "category",
        "list_price",
        "weekly_units_sold",
        "forecast_weekly_demand",
        "lead_time_days",
        "lead_time_demand",
        "on_hand_units",
        "on_order_units",
        "available_inventory",
        "reorder_point",
        "safety_level",
        "projected_inventory",
        "risk_gap",
        "inventory_coverage_weeks",
        "demand_pressure",
        "lead_time_pressure",
        "stockout_pressure",
        "stockout_risk_score",
        "stockout_risk_level",
        "stockout_risk",
        "safety_gap_pressure",
        "reorder_point_pressure",
        "risk_score",
        "risk_level",
        "stockout_flag",
        "risk_reason",
        "target_inventory_level",
        "reorder_quantity",
        "reorder_priority",
        "inventory_status",
        "excess_inventory_units",
        "excess_inventory_ratio",
        "markdown_clear_score",
        "markdown_clear_priority",
        "markdown_clear_reason",
        "markdown_discount_percent",
        "recommended_action",
        "final_inventory_decision"
    ]
    output_columns = [col for col in output_columns if col in df.columns]
    result = (df[output_columns].copy())

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    reorder_columns = [
        "sku_id",
        "week_start",
        "product_name",
        "category",
        "forecast_weekly_demand",
        "lead_time_days",
        "lead_time_demand",
        "on_hand_units",
        "on_order_units",
        "available_inventory",
        "safety_level",
        "reorder_point",
        "projected_inventory",
        "risk_score",
        "risk_level",
        "stockout_risk",
        "stockout_risk_level",
        "stockout_flag",
        "reorder_quantity",
        "reorder_priority",
        "risk_reason",
        "recommended_action"
    ]

    reorder_columns = [col for col in reorder_columns if col in result.columns]
    reorder_list = (result[reorder_columns].copy())
    reorder_list = reorder_list[(reorder_list["reorder_quantity"] > 0) | (reorder_list["reorder_priority"] != "NONE")]

    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "NONE": 4}

    reorder_list["_priority_sort"] = (reorder_list["reorder_priority"].map(priority_order).fillna(99))
    reorder_list = (reorder_list.sort_values(["_priority_sort", "risk_score", "reorder_quantity"], ascending=[True, False, False]).drop(columns=["_priority_sort"]))
    reorder_list.to_csv(REORDER_OUTPUT_PATH, index=False)

    markdown_columns = [
        "sku_id",
        "week_start",
        "product_name",
        "category",
        "forecast_weekly_demand",
        "on_hand_units",
        "on_order_units",
        "available_inventory",
        "inventory_coverage_weeks",
        "safety_level",
        "projected_inventory",
        "excess_inventory_units",
        "excess_inventory_ratio",
        "risk_score",
        "risk_level",
        "markdown_clear_score",
        "markdown_clear_priority",
        "markdown_clear_reason",
        "markdown_discount_percent",
        "final_inventory_decision"
    ]

    markdown_columns = [col for col in markdown_columns if col in result.columns]
    markdown_list = (result[markdown_columns].copy())
    markdown_list = markdown_list[(markdown_list["markdown_clear_priority"] != "NONE") & (markdown_list["excess_inventory_units"] > 0) & (markdown_list["risk_level"] != "HIGH")]

    markdown_priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}
    markdown_list["_priority_sort"] = (markdown_list["markdown_clear_priority"].map(markdown_priority_order).fillna(99))
    markdown_list = (markdown_list.sort_values(["_priority_sort", "markdown_clear_score", "excess_inventory_units"], ascending=[True, False, False]).drop(columns=["_priority_sort"]))

    markdown_list.to_csv(MARKDOWN_OUTPUT_PATH, index=False)

    print("\n")
    print("=" * 70)
    print("BALANCED RISK SCORE DISTRIBUTION")
    print("=" * 70)

    print(result["risk_score"].describe().round(2).to_string())

    print("\n")
    print("=" * 70)
    print("RISK LEVEL DISTRIBUTION")
    print("=" * 70)

    risk_counts = (result["risk_level"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0))
    print(risk_counts.to_string())
    print("\nRISK LEVEL PERCENTAGE")
    print((risk_counts / len(result) * 100).round(2).to_string())

    print("\n")
    print("=" * 70)
    print("STOCKOUT RISK SUMMARY")
    print("=" * 70)

    stockout_counts = (result["stockout_risk_level"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0))

    print(stockout_counts.to_string())
    print("\nSTOCKOUT RISK PERCENTAGE")
    print((stockout_counts / len(result) * 100).round(2).to_string())

    print("\n")
    print("=" * 70)
    print("STOCKOUT FLAG SUMMARY")
    print("=" * 70)

    print(result["stockout_flag"].value_counts().to_string())

    print("\n")
    print("=" * 70)
    print("REORDER PRIORITY SUMMARY")
    print("=" * 70)

    reorder_counts = (reorder_list["reorder_priority"].value_counts().reindex(["CRITICAL", "HIGH", "MEDIUM", "LOW"], fill_value=0))

    print(reorder_counts.to_string())
    print(f"\nTotal reorder records: " f"{len(reorder_list):,}")
    print(f"Total reorder units: " f"{reorder_list['reorder_quantity'].sum():,.0f}")

    print("\n")
    print("=" * 70)
    print("MARKDOWN / CLEAR PRIORITY SUMMARY")
    print("=" * 70)

    markdown_counts = (markdown_list["markdown_clear_priority"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0))

    print(markdown_counts.to_string())
    print(f"\nTotal markdown/clear records: " f"{len(markdown_list):,}")
    print(f"Total excess inventory units: " f"{markdown_list['excess_inventory_units'].sum():,.0f}")

    print("\n")
    print("AVERAGE RISK SCORE:")
    print(f"{result['risk_score'].mean():.2f}")

    print("\n")
    print("RISK SCORE RANGE:")
    print(f"Minimum: " f"{result['risk_score'].min():.2f}")
    print(f"Maximum: " f"{result['risk_score'].max():.2f}")

    print("\n")
    print("STOCKOUT RISK SCORE RANGE:")
    print(f"Minimum: " f"{result['stockout_risk_score'].min():.2f}")
    print(f"Maximum: " f"{result['stockout_risk_score'].max():.2f}")

    print("\n")
    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(f"\nMain output:\n" f"{OUTPUT_PATH}")
    print(f"\nReorder list:\n" f"{REORDER_OUTPUT_PATH}")
    print(f"\nMarkdown/Clear list:\n" f"{MARKDOWN_OUTPUT_PATH}")

    print("\n")
    print("=" * 70)
    print("PROJECT FORESIGHT " "BALANCED RISK SCORING FINISHED")
    print("=" * 70)

if __name__ == "__main__":
    run_risk_scoring()