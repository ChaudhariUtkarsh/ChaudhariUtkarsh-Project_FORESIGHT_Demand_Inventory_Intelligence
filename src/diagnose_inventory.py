import os
import pandas as pd
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


RISK_FILE = os.path.join(PROCESSED_DIR, "inventory_risk_scores.csv")
WEEKLY_FILE = os.path.join(PROCESSED_DIR, "weekly_model_data.csv")
PROCESSED_FILE = os.path.join(PROCESSED_DIR, "processed_data.csv")


def print_header(title):
    print("\n")
    print("=" * 75)
    print(title)
    print("=" * 75)


def numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def percentage(value, total):
    if total == 0:
        return 0
    return value / total * 100


def show_column_status(df, columns):
    for col in columns:
        if col in df.columns:
            print(f"  [OK] {col:<30} " f"non-null={df[col].notna().sum():,}")
        else:
            print(f"  [MISSING] {col}")


def main():
    print("\n")
    print("=" * 75)
    print("PROJECT FORESIGHT — COMPLETE INVENTORY DIAGNOSTIC")
    print("=" * 75)

    print_header("1. SOURCE FILE CHECK")
    files = {"inventory_risk_scores.csv": RISK_FILE, "weekly_model_data.csv": WEEKLY_FILE, "processed_data.csv": PROCESSED_FILE}

    for name, path in files.items():
        if os.path.exists(path):
            size_mb = (os.path.getsize(path) / (1024 * 1024))
            print(f"[OK] {name:<35} " f"{size_mb:.2f} MB")
        else:
            print(f"[MISSING] {name:<35}")

    
    if not os.path.exists(RISK_FILE):
        raise FileNotFoundError(f"\nRisk file not found:\n{RISK_FILE}")
    df = pd.read_csv(RISK_FILE)
    print(f"\nFinal risk records: {len(df):,}")

   
    print_header("2. FINAL OUTPUT COLUMN CHECK")
    required_columns = [
        "sku_id",
        "week_start",
        "weekly_units_sold",
        "forecast_weekly_demand",
        "on_hand_units",
        "on_order_units",
        "available_inventory",
        "lead_time_days",
        "lead_time_demand",
        "safety_level",
        "reorder_point",
        "projected_inventory",
        "inventory_coverage_weeks",
        "risk_score",
        "risk_level",
        "stockout_risk",
        "reorder_quantity"
    ]
    show_column_status(df, required_columns)

    
    numeric_columns = [
        "weekly_units_sold",
        "forecast_weekly_demand",
        "on_hand_units",
        "on_order_units",
        "available_inventory",
        "lead_time_days",
        "lead_time_demand",
        "safety_level",
        "reorder_point",
        "projected_inventory",
        "inventory_coverage_weeks",
        "risk_score",
        "reorder_quantity",
        "excess_inventory_units"
    ]
    df = numeric(df, numeric_columns)

   
    print_header("3. DATASET OVERVIEW")
    print(f"Total records       : {len(df):,}")

    if "sku_id" in df.columns:
        print(f"Unique SKUs         : " f"{df['sku_id'].nunique():,}")

    if "week_start" in df.columns:
        print(f"Unique dates        : " f"{df['week_start'].nunique():,}")
        print(f"First week          : " f"{df['week_start'].min()}")
        print(f"Last week           : " f"{df['week_start'].max()}")


    print_header("4. SKU DUPLICATE CHECK")
    if "sku_id" in df.columns:
        sku_counts = (df["sku_id"].value_counts())
        duplicate_skus = (sku_counts[sku_counts > 1])

        print(f"Unique SKUs: " f"{len(sku_counts):,}")
        print(f"SKUs appearing more than once: " f"{len(duplicate_skus):,}")
        print(f"Total duplicate records: " f"{duplicate_skus.sum():,}")

        if len(duplicate_skus) > 0:
            print("\nTop duplicate SKUs:")
            print(duplicate_skus.head(20).to_string())


    print_header("5. WEEKLY MODEL DATA — INVENTORY CHECK")
    if os.path.exists(WEEKLY_FILE):
        weekly = pd.read_csv(WEEKLY_FILE)
        print(f"Rows: {len(weekly):,}")
        print(f"Columns: {len(weekly.columns):,}")

        weekly_inventory_columns = [
            "sku_id",
            "on_hand_units",
            "on_order_units",
            "available_inventory",
            "reorder_point",
            "safety_level"
        ]
        show_column_status(weekly, weekly_inventory_columns)
        weekly = numeric(weekly, weekly_inventory_columns)

        for col in ["on_hand_units", "on_order_units", "available_inventory"]:
            if col in weekly.columns:
                zero_count = (weekly[col] == 0).sum()
                print(f"\n{col}")

                print(f"  Zero: " f"{zero_count:,} " f"({percentage(zero_count, len(weekly)):.2f}%)")
                print(f"  Mean: " f"{weekly[col].mean():.2f}")
                print(f"  Median: " f"{weekly[col].median():.2f}")
                print(f"  Min: " f"{weekly[col].min():.2f}")
                print(f"  Max: " f"{weekly[col].max():.2f}")

    else:
        print("weekly_model_data.csv not found.")

 
    print_header("6. PROCESSED DATA — INVENTORY CHECK")
    if os.path.exists(PROCESSED_FILE):
        processed = pd.read_csv(PROCESSED_FILE)
        print(f"Rows: {len(processed):,}")
        print(f"Columns: {len(processed.columns):,}")

        processed_inventory_columns = [
            "sku_id",
            "on_hand_units",
            "on_order_units",
            "available_inventory",
            "reorder_point",
            "safety_level",
            "lead_time_days"
        ]

        show_column_status(processed, processed_inventory_columns)
        processed = numeric(processed, processed_inventory_columns)

        for col in ["on_hand_units", "on_order_units", "available_inventory"]:
            if col in processed.columns:
                zero_count = (processed[col] == 0).sum()
                print(f"\n{col}")

                print(f"  Zero: " f"{zero_count:,} " f"({percentage(zero_count, len(processed)):.2f}%)")
                print(f"  Mean: " f"{processed[col].mean():.2f}")
                print(f"  Median: " f"{processed[col].median():.2f}")
                print(f"  Min: " f"{processed[col].min():.2f}")
                print(f"  Max: " f"{processed[col].max():.2f}")
    else:
        print("processed_data.csv not found.")

   
    print_header("7. FINAL INVENTORY DIAGNOSTIC")
    inventory_columns = ["on_hand_units", "on_order_units", "available_inventory"]

    for col in inventory_columns:
        if col not in df.columns:
            continue

        series = df[col]
        zero = (series == 0).sum()
        positive = (series > 0).sum()
        negative = (series < 0).sum()
        print(f"\n{col}")

        print(f"  Zero       : {zero:,} " f"({percentage(zero, len(df)):.2f}%)")
        print(f"  Positive   : {positive:,} " f"({percentage(positive, len(df)):.2f}%)")
        print(f"  Negative   : {negative:,} " f"({percentage(negative, len(df)):.2f}%)")
        print(f"  Mean       : {series.mean():.2f}")
        print(f"  Median     : {series.median():.2f}")

    
    print_header("8. AVAILABLE INVENTORY CONSISTENCY")
    if all(col in df.columns for col in ["on_hand_units", "on_order_units", "available_inventory"]):

        expected_available = (df["on_hand_units"] + df["on_order_units"])
        difference = (df["available_inventory"] - expected_available)
        mismatch = (difference.abs() > 0.01)

        print(f"Expected formula:")
        print("available_inventory = " "on_hand_units + on_order_units")
        print(f"\nMismatched records: " f"{mismatch.sum():,}")
        print(f"Mismatch percentage: " f"{percentage(mismatch.sum(), len(df)):.2f}%")

        if mismatch.any():
            print("\nSample mismatches:")

            cols = ["sku_id", "on_hand_units", "on_order_units", "available_inventory"]
            print(df.loc[mismatch, cols].head(20).to_string(index=False))

   
    print_header("9. FORECAST DEMAND")
    forecast = df["forecast_weekly_demand"]
    print(forecast.describe().round(2).to_string())

    
    print_header("10. FORECAST VS AVAILABLE INVENTORY")
    df["forecast_inventory_ratio"] = np.where(df["available_inventory"] > 0, df["forecast_weekly_demand"] / df["available_inventory"], np.nan)

    forecast_higher = (df["forecast_weekly_demand"] > df["available_inventory"]).sum()
    forecast_lower = (df["forecast_weekly_demand"] <= df["available_inventory"]).sum()

    zero_inventory = (df["available_inventory"] == 0).sum()

    print(
        f"Forecast > Inventory : "
        f"{forecast_higher:,} "
        f"({percentage(forecast_higher, len(df)):.2f}%)"
    )

    print(
        f"Forecast <= Inventory: "
        f"{forecast_lower:,} "
        f"({percentage(forecast_lower, len(df)):.2f}%)"
    )

    print(
        f"Zero inventory       : "
        f"{zero_inventory:,} "
        f"({percentage(zero_inventory, len(df)):.2f}%)"
    )

    print("\nForecast / Inventory ratio:")
    print(df["forecast_inventory_ratio"].describe().round(2).to_string())

  
    print_header("11. LEAD TIME DIAGNOSTIC")
    if "lead_time_days" in df.columns:
        print(df["lead_time_days"].describe().round(2).to_string())
        print("\nLead-time buckets:")

        lead_time_buckets = pd.cut(
            df["lead_time_days"],
            bins=[-np.inf, 7, 14, 21, 30, 60, np.inf],
            labels=["<=7 days", "8-14 days", "15-21 days", "22-30 days", "31-60 days", ">60 days"]
        )

        print(lead_time_buckets.value_counts(sort=False).to_string())

    
    print_header("12. LEAD-TIME DEMAND VALIDATION")
    if all(col in df.columns for col in ["forecast_weekly_demand", "lead_time_days", "lead_time_demand"]):
        expected_lead_demand = (df["forecast_weekly_demand"] * (df["lead_time_days"] / 7))
        lead_difference = (df["lead_time_demand"] - expected_lead_demand)

        mismatch = (lead_difference.abs() > 0.01)

        print("Expected formula:")
        print("lead_time_demand = " "forecast_weekly_demand x lead_time_days / 7")

        print(f"\nMismatched records: " f"{mismatch.sum():,}")
        print(f"Mismatch percentage: " f"{percentage(mismatch.sum(), len(df)):.2f}%")


    print_header("13. INVENTORY COVERAGE")
    coverage = df["inventory_coverage_weeks"]
    print(coverage.describe().round(2).to_string())

    coverage_buckets = pd.cut(
        coverage, bins=[-np.inf, 1, 2, 4, 8, 12, 20, np.inf],
        labels=["<=1 week", "1-2 weeks", "2-4 weeks", "4-8 weeks", "8-12 weeks", "12-20 weeks", ">20 weeks"]
    )
    print("\nCoverage buckets:")

    coverage_counts = (coverage_buckets.value_counts(sort=False))
    print(coverage_counts.to_string())
    print("\nCoverage percentages:")
    print((coverage_counts / len(df) * 100).round(2).to_string())

   
    print_header("14. PROJECTED INVENTORY")
    projected = df["projected_inventory"]
    print(projected.describe().round(2).to_string())

    negative_projected = (projected <= 0).sum()
    positive_projected = (projected > 0).sum()

    print(f"\nProjected inventory <= 0: " f"{negative_projected:,}")
    print(f"Percentage: " f"{percentage(negative_projected, len(df)):.2f}%")
    print(f"\nProjected inventory > 0: " f"{positive_projected:,}")
    print(f"Percentage: " f"{percentage(positive_projected, len(df)):.2f}%")

    print_header("15. STOCKOUT DIAGNOSTIC")
    if "stockout_risk" in df.columns:
        stockout_counts = (df["stockout_risk"].value_counts())

        print(stockout_counts.to_string())
        print("\nStockout percentages:")
        print((stockout_counts / len(df) * 100).round(2).to_string())


    print_header("16. STOCKOUT LOGIC VALIDATION")
    if all(col in df.columns for col in ["projected_inventory", "inventory_coverage_weeks", "stockout_risk"]):
        expected_stockout = np.where((df["projected_inventory"] <= 0) | (df["inventory_coverage_weeks"] <= 2), "YES", "NO")
        stockout_mismatch = (df["stockout_risk"].astype(str) != expected_stockout)

        print("Expected rule:")
        print("projected_inventory <= 0 " "OR coverage <= 2 weeks → YES")
        print(f"\nMismatched records: " f"{stockout_mismatch.sum():,}")


    print_header("17. SAFETY LEVEL DIAGNOSTIC")
    if "safety_level" in df.columns:
        safety = df["safety_level"]
        print(safety.describe().round(2).to_string())
        zero_safety = (safety <= 0).sum()
        print(f"\nZero/negative safety level: " f"{zero_safety:,}")

  
    print_header("18. REORDER QUANTITY")
    reorder = df["reorder_quantity"]
    print(reorder.describe().round(2).to_string())
    reorder_needed = (reorder > 0).sum()
    no_reorder = (reorder <= 0).sum()

    print(f"\nReorder required: " f"{reorder_needed:,} " f"({percentage(reorder_needed, len(df)):.2f}%)")
    print(f"No reorder: " f"{no_reorder:,} " f"({percentage(no_reorder, len(df)):.2f}%)")
    print(f"\nTotal reorder units: " f"{reorder.sum():,.0f}")


    print_header("19. RISK SCORE")
    risk = df["risk_score"]
    print(risk.describe().round(2).to_string())

    risk_buckets = pd.cut(
        risk, bins=[-np.inf, 20, 40, 60, 70, 80, 90, 100],
        labels=["<20", "20-40", "40-60", "60-70", "70-80", "80-90", "90-100"]
    )

    risk_counts = (risk_buckets.value_counts(sort=False))
    print("\nRisk buckets:")
    print(risk_counts.to_string())

   
    print_header("20. RISK LEVEL DISTRIBUTION")
    if "risk_level" in df.columns:
        risk_level_counts = (df["risk_level"].value_counts())

        print(risk_level_counts.to_string())
        print("\nRisk level percentages")
        print((risk_level_counts / len(df) * 100).round(2).to_string())

   
    print_header("21. SKU-LEVEL INVENTORY PROBLEM")
    if "sku_id" in df.columns:
        sku_summary = (df.groupby("sku_id")
            .agg(
                records=("sku_id", "size"),
                avg_forecast=("forecast_weekly_demand", "mean"),
                avg_inventory=("available_inventory", "mean"),
                zero_inventory_records=("available_inventory", lambda x: (x == 0).sum()),
                avg_coverage=("inventory_coverage_weeks", "mean"),
                avg_risk=("risk_score", "mean")
            )
        )

        sku_summary["zero_inventory_pct"] = (sku_summary["zero_inventory_records"] / sku_summary["records"] * 100)
        sku_summary = (sku_summary.sort_values("zero_inventory_pct", ascending=False))
        print(sku_summary.head(30).round(2).to_string())

   
    print_header("22. SAMPLE ZERO-INVENTORY RECORDS")
    zero_mask = (df["available_inventory"] <= 0)
    zero_columns = [
        "sku_id",
        "week_start",
        "forecast_weekly_demand",
        "on_hand_units",
        "on_order_units",
        "available_inventory",
        "lead_time_days",
        "lead_time_demand",
        "safety_level",
        "projected_inventory",
        "inventory_coverage_weeks",
        "risk_score",
        "risk_level",
        "stockout_risk",
        "reorder_quantity"
    ]

    zero_columns = [col for col in zero_columns if col in df.columns]
    print(df.loc[zero_mask, zero_columns].head(30).to_string(index=False))

    
    print_header("23. SAMPLE POSITIVE-INVENTORY RECORDS")
    positive_mask = (df["available_inventory"] > 0)
    print(df.loc[positive_mask, zero_columns].head(20).to_string(index=False))


    print_header("24. SOURCE DATA COMPARISON")
    if (os.path.exists(WEEKLY_FILE) and os.path.exists(PROCESSED_FILE)):

        weekly = pd.read_csv(WEEKLY_FILE)
        processed = pd.read_csv(PROCESSED_FILE)

        if ("sku_id" in weekly.columns and "sku_id" in processed.columns):
            weekly_skus = set(weekly["sku_id"].astype(str))
            processed_skus = set(processed["sku_id"].astype(str))

            matched = (weekly_skus & processed_skus)
            weekly_only = (weekly_skus - processed_skus)
            processed_only = (processed_skus - weekly_skus)

            print(f"weekly_model_data SKUs : " f"{len(weekly_skus):,}")
            print(f"processed_data SKUs     : " f"{len(processed_skus):,}")
            print(f"Matched SKUs            : " f"{len(matched):,}")
            print(f"Weekly-only SKUs        : " f"{len(weekly_only):,}")
            print(f"Processed-only SKUs     : " f"{len(processed_only):,}")

            if weekly_only:
                print("\nSample weekly-only SKUs:")
                print(list(weekly_only)[:20])

            if processed_only:
                print("\nSample processed-only SKUs:")
                print(list(processed_only)[:20])


    print_header("25. DATA QUALITY WARNINGS")
    warnings = []
    total = len(df)
    if total == 0:
        warnings.append("Dataset is empty.")

    zero_inventory = (df["available_inventory"] <= 0).sum()
    zero_inventory_pct = percentage(zero_inventory, total)

    if zero_inventory_pct >= 90:
        warnings.append(f"{zero_inventory_pct:.2f}% " "records have zero/negative available inventory.")

    projected_negative = (df["projected_inventory"] <= 0).sum()
    projected_negative_pct = percentage(projected_negative, total)

    if projected_negative_pct >= 90:
        warnings.append(f"{projected_negative_pct:.2f}% " "records have projected inventory <= 0.")

    low_coverage = (df["inventory_coverage_weeks"] <= 1).sum()
    low_coverage_pct = percentage(low_coverage, total)

    if low_coverage_pct >= 80:
        warnings.append(f"{low_coverage_pct:.2f}% " "records have <=1 week inventory coverage.")

    if "risk_level" in df.columns:
        high_risk = (df["risk_level"] == "HIGH").sum()
        high_risk_pct = percentage(high_risk, total)

        if high_risk_pct >= 80:
            warnings.append(f"{high_risk_pct:.2f}% " "records are HIGH risk.")

    if warnings:
        for i, warning in enumerate(warnings, start=1):
            print(f"\nWARNING {i}: {warning}")
    else:
        print("No major data quality warnings detected.")

    
    print_header("26. DIAGNOSTIC CONCLUSION")
    print("\nThe diagnostic does NOT artificially change")
    print("risk percentages.")

    print("\nIt validates:")
    print("  1. Forecast demand")
    print("  2. On-hand inventory")
    print("  3. On-order inventory")
    print("  4. Available inventory")
    print("  5. SKU merge consistency")
    print("  6. Lead-time demand")
    print("  7. Inventory coverage")
    print("  8. Projected inventory")
    print("  9. Stockout logic")
    print(" 10. Reorder quantity")
    print(" 11. Risk distribution")

    print("\nIMPORTANT:")
    print("If the source data genuinely contains very low")
    print("inventory compared with forecast demand, then")
    print("a high stockout percentage is a REAL business")
    print("signal and should not be artificially reduced.")

    print("\n")
    print("=" * 75)
    print("PROJECT FORESIGHT DIAGNOSTIC FINISHED")
    print("=" * 75)


if __name__ == "__main__":
    main()