import os
import logging
import numpy as np
import pandas as pd


# LOGGING CONFIGURATION
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# PROJECT PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")
RISK_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "risk_analysis")
RISK_OUTPUT_PATH = os.path.join(RISK_OUTPUT_DIR, "sku_risk_analysis.csv")


# LOAD DATA
def load_data():
    logger.info("Loading processed dataset...")
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Processed dataset not found:\n{PROCESSED_DATA_PATH}")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    logger.info(f"Processed dataset loaded successfully. Shape = {df.shape}")
    return df


# RISK ANALYSIS
def perform_risk_analysis(df):
    logger.info("Starting SKU-level risk analysis...")

    # Required columns check
    required_columns = ["sku_id", "units_sold"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Make sure numeric columns are numeric
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")

    # Current Inventory
    if "on_hand_units" in df.columns:
        df["on_hand_units"] = pd.to_numeric(df["on_hand_units"], errors="coerce").fillna(0)
        current_inventory = (df.groupby("sku_id")["on_hand_units"].last())

    elif "inventory" in df.columns:
        df["inventory"] = pd.to_numeric(df["inventory"], errors="coerce").fillna(0)
        current_inventory = (df.groupby("sku_id")["inventory"].last())

    else:
        logger.warning(
            "Inventory column not found. "
            "Current inventory will be set to 0."
        )

        current_inventory = (df.groupby("sku_id").size().astype(float) * 0)

    # Average Daily Demand
    average_daily_demand = (
        df.groupby("sku_id")["units_sold"]
        .mean()
        .fillna(0)
    )

    # Average Unit Price
    price_column = None
    possible_price_columns = ["unit_price", "selling_price", "price", "average_unit_price"]

    for col in possible_price_columns:
        if col in df.columns:
            price_column = col
            break

    if price_column:
        df[price_column] = pd.to_numeric(df[price_column], errors="coerce").fillna(0)
        average_unit_price = (df.groupby("sku_id")[price_column].mean().fillna(0))

    else:
        logger.warning(
            "Price column not found. "
            "Average Unit Price will be set to 0."
        )

        average_unit_price = (df.groupby("sku_id").size().astype(float) * 0)

    # Create SKU-level dataframe
    risk_df = pd.DataFrame({
        "Current_Inventory": current_inventory,
        "Average_Daily_Demand": average_daily_demand,
        "Average_Unit_Price": average_unit_price
    })

    risk_df = risk_df.reset_index()


    # Clean values
    risk_df["Current_Inventory"] = (pd.to_numeric(risk_df["Current_Inventory"], errors="coerce").fillna(0).clip(lower=0))
    risk_df["Average_Daily_Demand"] = (pd.to_numeric(risk_df["Average_Daily_Demand"], errors="coerce").fillna(0).clip(lower=0))
    risk_df["Average_Unit_Price"] = (pd.to_numeric(risk_df["Average_Unit_Price"], errors="coerce").fillna(0).clip(lower=0))


    # DAYS OF SUPPLY
    risk_df["Days_of_Supply"] = np.where(risk_df["Average_Daily_Demand"] > 0, risk_df["Current_Inventory"] / risk_df["Average_Daily_Demand"], np.inf)

   
    # 1. STOCKOUT RISK SCORE
    risk_df["Stockout_Risk_Score"] = np.select(
        [risk_df["Days_of_Supply"] <= 3, risk_df["Days_of_Supply"] <= 7, risk_df["Days_of_Supply"] <= 14],
        [1.00, 0.75, 0.40], default=0.10
    )

  
    # 2. OVERSTOCK RISK SCORE
    risk_df["Overstock_Risk_Score"] = np.select(
        [risk_df["Days_of_Supply"] >= 90, risk_df["Days_of_Supply"] >= 60, risk_df["Days_of_Supply"] >= 30],
        [1.00, 0.75, 0.40], default=0.10
    )

  
    # 3. CAPITAL LOCKED
    risk_df["Capital_Locked"] = (risk_df["Current_Inventory"] * risk_df["Average_Unit_Price"]).round(2)


    # 4. SALES AT RISK
    risk_df["Sales_at_Risk"] = (risk_df["Average_Daily_Demand"] * risk_df["Average_Unit_Price"] * 7 * risk_df["Stockout_Risk_Score"]).round(2)


    # 5. RISK LEVEL
    risk_df["Risk_Level"] = np.select(
        [
            ((risk_df["Stockout_Risk_Score"] >= 0.75) | (risk_df["Overstock_Risk_Score"] >= 0.75)),
            ((risk_df["Stockout_Risk_Score"] >= 0.40) | (risk_df["Overstock_Risk_Score"] >= 0.40))
        ],
        ["HIGH", "MEDIUM"], default="LOW"
    )


    # 6. PRIMARY RISK
    risk_df["Primary_Risk"] = np.select(
        [
            (risk_df["Stockout_Risk_Score"] > risk_df["Overstock_Risk_Score"]),
            (risk_df["Overstock_Risk_Score"] > risk_df["Stockout_Risk_Score"])
        ],
        ["STOCKOUT", "OVERSTOCK"], default="BALANCED"
    )


    # 7. RECOMMENDED ACTION
    risk_df["Recommended_Action"] = np.select(
        [
            ((risk_df["Primary_Risk"] == "STOCKOUT") & (risk_df["Risk_Level"] == "HIGH")),
            ((risk_df["Primary_Risk"] == "STOCKOUT") & (risk_df["Risk_Level"] == "MEDIUM")),
            ((risk_df["Primary_Risk"] == "OVERSTOCK") & (risk_df["Risk_Level"] == "HIGH")),
            ((risk_df["Primary_Risk"] == "OVERSTOCK") & (risk_df["Risk_Level"] == "MEDIUM"))
        ],
        ["URGENTLY REPLENISH STOCK", "PLAN REPLENISHMENT", "REDUCE INVENTORY / PROMOTE", "MONITOR INVENTORY"], default="NORMAL MONITORING"
    )


    # FINAL COLUMN ORDER
    risk_df = risk_df[
        [
            "sku_id",
            "Current_Inventory",
            "Average_Daily_Demand",
            "Average_Unit_Price",
            "Days_of_Supply",
            "Stockout_Risk_Score",
            "Overstock_Risk_Score",
            "Risk_Level",
            "Primary_Risk",
            "Recommended_Action",
            "Sales_at_Risk",
            "Capital_Locked"
        ]
    ]

    return risk_df


# SAVE OUTPUT
def save_risk_analysis(risk_df):
    os.makedirs(RISK_OUTPUT_DIR, exist_ok=True)
    risk_df.to_csv(RISK_OUTPUT_PATH, index=False)
    logger.info(f"Risk analysis saved {RISK_OUTPUT_PATH}")


# MAIN
if __name__ == "__main__":
    print("=" * 80)
    print("PROJECT FORESIGHT - SKU RISK ANALYSIS")
    print("=" * 80)

    try:
        df = load_data()
        risk_df = perform_risk_analysis(df)
        save_risk_analysis(risk_df)

        # DISPLAY RESULTS
        print("=" * 80)
        print("RISK ANALYSIS COMPLETED")
        print("=" * 80)

        print(
            f"\nTotal SKUs analysed : "
            f"{len(risk_df)}"
        )

        print("\nRisk Level Distribution:")
        print(risk_df["Risk_Level"].value_counts())
        print("\nPrimary Risk Distribution:")
        print(risk_df["Primary_Risk"].value_counts())
        print("\nTop 10 High-Risk SKUs:")

        high_risk = (risk_df[risk_df["Risk_Level"] == "HIGH"].sort_values("Sales_at_Risk", ascending=False).head(10))
        print(high_risk.to_string(index=False))
        print("\nOutput file:")
        print(RISK_OUTPUT_PATH)
        print("\n" + "=" * 80)

    except Exception as e:
        logger.exception("Risk analysis failed.")
        print("\nERROR:")
        print(str(e))