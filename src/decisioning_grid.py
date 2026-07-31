import os
import logging
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# PROJECT ROOT
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# INPUT / OUTPUT PATH
INPUT_FILE = os.path.join(BASE_DIR, "data", "risk_analysis", "sku_risk_analysis.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "decisioning_grid")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "decisioning_grid.csv")


# LOAD DATA
logger.info("Loading SKU risk analysis data...")
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Risk analysis file not found:\n{INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)
logger.info(f"Risk analysis loaded successfully. Shape = {df.shape}")


# REQUIRED COLUMNS
required_columns = [
    "sku_id",
    "Days_of_Supply",
    "Stockout_Risk_Score",
    "Overstock_Risk_Score",
    "Risk_Level",
    "Primary_Risk",
    "Recommended_Action",
    "Sales_at_Risk",
    "Capital_Locked"
]


missing_columns = [col for col in required_columns if col not in df.columns]


if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")


# DECISIONING GRID CLASSIFICATION
def classify_decision(row):
    stockout_score = row["Stockout_Risk_Score"]
    overstock_score = row["Overstock_Risk_Score"]
    dos = row["Days_of_Supply"]

    # 1. REORDER NOW
    if stockout_score >= 0.75:
        return "Reorder Now"


    # 2. MARKDOWN / CLEAR
    elif overstock_score >= 0.75:
        return "Markdown / Clear"


    # 3. WATCH / VOLATILE
    elif (stockout_score >= 0.40 or overstock_score >= 0.40 or row["Risk_Level"] == "MEDIUM"):
        return "Watch / Volatile"


    # 4. HEALTHY
    else:
        return "Healthy"


# Apply decisioning logic
logger.info("Classifying SKUs into decisioning quadrants...")
df["Decision_Quadrant"] = df.apply(classify_decision, axis=1)


# RECOMMENDED DECISION
def recommended_decision(quadrant):
    if quadrant == "Reorder Now":
        return "Urgently replenish stock"
    elif quadrant == "Markdown / Clear":
        return "Run discount / clearance promotion"
    elif quadrant == "Watch / Volatile":
        return "Monitor demand and inventory closely"
    else:
        return "Maintain current inventory"


df["Decision_Action"] = df["Decision_Quadrant"].apply(recommended_decision)


# SAVE RESULT
os.makedirs(OUTPUT_DIR, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
logger.info(f"Decisioning Grid saved → {OUTPUT_FILE}")


# DISPLAY SUMMARY
print("\n" + "=" * 80)
print("PROJECT FORESIGHT - DECISIONING GRID")
print("=" * 80)
print("\nDECISIONING GRID DISTRIBUTION")
print("-" * 80)
distribution = (df["Decision_Quadrant"].value_counts())


for quadrant in ["Reorder Now", "Markdown / Clear", "Watch / Volatile", "Healthy"]:
    count = distribution.get(quadrant, 0)
    print(f"{quadrant:<25}: {count} SKUs")


# TOP REORDER NOW SKU
print("\nTOP REORDER NOW SKUs")
print("-" * 80)
reorder_df = df[df["Decision_Quadrant"] == "Reorder Now"].sort_values(by="Sales_at_Risk", ascending=False)


if len(reorder_df) > 0:
    print(reorder_df[
            ["sku_id", "Days_of_Supply", "Stockout_Risk_Score", "Sales_at_Risk", "Capital_Locked", "Decision_Quadrant", "Decision_Action"]
        ].head(10).to_string(index=False)
    )

else:
    print("No SKUs classified as Reorder Now.")



# TOP MARKDOWN / CLEAR SKU
print("\nTOP MARKDOWN / CLEAR SKUs")
print("-" * 80)
markdown_df = df[df["Decision_Quadrant"] == "Markdown / Clear"].sort_values(by="Capital_Locked", ascending=False)


if len(markdown_df) > 0:

    print(markdown_df[
            ["sku_id", "Days_of_Supply", "Overstock_Risk_Score", "Sales_at_Risk", "Capital_Locked", "Decision_Quadrant", "Decision_Action"]
        ].head(10).to_string(index=False)
    )

else:
    print("No SKUs classified as Markdown / Clear.")


# FINAL OUTPUT
print("\n" + "=" * 80)
print("DECISIONING GRID COMPLETED")
print("=" * 80)
print("\nOutput file:")
print(OUTPUT_FILE)
print("=" * 80)