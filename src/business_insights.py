import os
import logging
import pandas as pd


# LOGGING CONFIGURATION
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# PROJECT PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RISK_ANALYSIS_PATH = os.path.join(BASE_DIR, "data", "risk_analysis", "sku_risk_analysis.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "business_insights")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "business_insights_summary.csv")


# LOAD RISK ANALYSIS DATA
def load_risk_data():
    logger.info("Loading SKU risk analysis data...")
    if not os.path.exists(RISK_ANALYSIS_PATH):
        logger.error(f"Risk analysis file not found: {RISK_ANALYSIS_PATH}")
        raise FileNotFoundError(f"File not found:\n{RISK_ANALYSIS_PATH}")

    df = pd.read_csv(RISK_ANALYSIS_PATH)
    logger.info(f"Risk analysis loaded successfully. Shape = {df.shape}")
    return df


# VALIDATE REQUIRED COLUMNS
def validate_columns(df):
    required_columns = [
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

    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError("Missing required columns:\n" + "\n".join(missing_columns))
    logger.info("All required risk columns are available.")


# CALCULATE BUSINESS INSIGHTS
def calculate_business_insights(df):
    logger.info("Calculating business insights...")

    # Convert numeric columns
    numeric_columns = [
        "Current_Inventory",
        "Average_Daily_Demand",
        "Average_Unit_Price",
        "Days_of_Supply",
        "Stockout_Risk_Score",
        "Overstock_Risk_Score",
        "Sales_at_Risk",
        "Capital_Locked"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove invalid rows
    df = df.dropna(subset=["sku_id", "Risk_Level", "Primary_Risk"]).copy()

    # TOTAL SALES AT RISK
    total_sales_at_risk = df["Sales_at_Risk"].sum()

    # TOTAL CAPITAL LOCKED
    total_capital_locked = df["Capital_Locked"].sum()

    # RISK LEVEL DISTRIBUTION
    risk_distribution = (df["Risk_Level"].value_counts().to_dict())

    # Make sure all standard levels exist
    high_count = risk_distribution.get("HIGH", 0)
    medium_count = risk_distribution.get("MEDIUM", 0)
    low_count = risk_distribution.get("LOW", 0)

   
    # PRIMARY RISK DISTRIBUTION
    primary_risk_distribution = (df["Primary_Risk"].value_counts().to_dict())
    stockout_count = primary_risk_distribution.get("STOCKOUT", 0)
    overstock_count = primary_risk_distribution.get("OVERSTOCK", 0)
    balanced_count = primary_risk_distribution.get("BALANCED", 0)
    risk_priority = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    df["Risk_Priority"] = (df["Risk_Level"].map(risk_priority).fillna(0))

    top_10 = (
        df.sort_values(
            by=["Risk_Priority", "Stockout_Risk_Score", "Overstock_Risk_Score", "Sales_at_Risk"],
            ascending=[False, False, False, False]
        ).head(10).copy()
    )

    # Remove helper column
    top_10 = top_10.drop(columns=["Risk_Priority"], errors="ignore")

    return (
        df,
        top_10,
        total_sales_at_risk,
        total_capital_locked,
        high_count,
        medium_count,
        low_count,
        stockout_count,
        overstock_count,
        balanced_count
    )


# SAVE BUSINESS INSIGHTS
def save_business_insights(
    top_10,
    total_sales_at_risk,
    total_capital_locked,
    high_count,
    medium_count,
    low_count,
    stockout_count,
    overstock_count,
    balanced_count
):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

 
    # Create summary rows
    summary_data = {
        "Total_SKUs_Analysed": [high_count + medium_count + low_count],
        "Total_Sales_at_Risk": [round(total_sales_at_risk, 2)],
        "Total_Capital_Locked": [round(total_capital_locked, 2)],
        "HIGH_Risk_SKUs": [high_count],
        "MEDIUM_Risk_SKUs": [medium_count],
        "LOW_Risk_SKUs": [low_count],
        "STOCKOUT_Primary_Risk_SKUs": [stockout_count],
        "OVERSTOCK_Primary_Risk_SKUs": [overstock_count],
        "BALANCED_Primary_Risk_SKUs": [balanced_count]
    }

    summary_df = pd.DataFrame(summary_data)


    # Save summary
    summary_df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Business insights summary saved → {OUTPUT_PATH}")


# PRINT BUSINESS INSIGHTS
def print_business_insights(
    df,
    top_10,
    total_sales_at_risk,
    total_capital_locked,
    high_count,
    medium_count,
    low_count,
    stockout_count,
    overstock_count,
    balanced_count
):

    print("\n")
    print("=" * 90)
    print("PROJECT FORESIGHT - BUSINESS INSIGHTS")
    print("=" * 90)


    # OVERALL SUMMARY
    print("\nOVERALL BUSINESS SUMMARY")
    print("-" * 90)
    print(f"Total SKUs Analysed     : {len(df)}")
    print(f"Total Sales at Risk     : {total_sales_at_risk:,.2f}")
    print(f"Total Capital Locked    : {total_capital_locked:,.2f}")       


    # RISK LEVEL DISTRIBUTION
    print("\nRISK LEVEL DISTRIBUTION")
    print("-" * 90)
    print(f"HIGH Risk SKUs          : {high_count}")
    print(f"MEDIUM Risk SKUs        : {medium_count}")
    print(f"LOW Risk SKUs           : {low_count}")


    # PRIMARY RISK DISTRIBUTION
    print("\nPRIMARY RISK DISTRIBUTION")
    print("-" * 90)
    print(f"STOCKOUT Risk SKUs      : {stockout_count}")
    print(f"OVERSTOCK Risk SKUs     : {overstock_count}")
    print(f"BALANCED SKUs           : {balanced_count}")

  
    # RISKY SKUs
    print("\nTOP 10 RISKY SKUs")
    print("-" * 90)

    display_columns = [
        "sku_id",
        "Current_Inventory",
        "Average_Daily_Demand",
        "Days_of_Supply",
        "Stockout_Risk_Score",
        "Overstock_Risk_Score",
        "Risk_Level",
        "Primary_Risk",
        "Recommended_Action",
        "Sales_at_Risk",
        "Capital_Locked"
    ]

    print(top_10[display_columns].to_string(index=False))
    print("\n" + "=" * 90)
    print("BUSINESS INSIGHTS COMPLETED")
    print("=" * 90)
    print("\nOutput file:")
    print(OUTPUT_PATH)
    print("=" * 90)



# MAIN
if __name__ == "__main__":
    try:
        df = load_risk_data()
        validate_columns(df)
        (df, top_10, total_sales_at_risk,  total_capital_locked, high_count, medium_count, low_count, stockout_count, overstock_count, balanced_count) = calculate_business_insights(df)
        save_business_insights( top_10, total_sales_at_risk, total_capital_locked, high_count, medium_count, low_count, stockout_count, overstock_count, balanced_count)
        print_business_insights( df, top_10, total_sales_at_risk, total_capital_locked, high_count, medium_count, low_count, stockout_count, overstock_count, balanced_count)

    except Exception as e:
        logger.exception(f"Business insights generation failed: {e}")
        raise