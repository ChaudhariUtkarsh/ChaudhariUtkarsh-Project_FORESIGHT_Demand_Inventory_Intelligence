import os
import pandas as pd
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


FILES = {
    "inventory_snapshots": os.path.join( RAW_DIR, "inventory_snapshots.csv"),
    "processed_data": os.path.join(PROCESSED_DIR, "processed_data.csv"),
    "weekly_model_data": os.path.join(PROCESSED_DIR, "weekly_model_data.csv")
}


def inspect_file(name, path):
    print("\n")
    print("=" * 75)
    print(f"{name.upper()}")
    print("=" * 75)

    if not os.path.exists(path):
        print(f"[MISSING] {path}")
        return None

    df = pd.read_csv(path)
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")
    print("\nColumns:")

    for col in df.columns:
        print(f"  - {col}")
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    return df


def main():
    print("\n")
    print("=" * 75)
    print("PROJECT FORESIGHT — INVENTORY MERGE INVESTIGATION")
    print("=" * 75)

    inventory = inspect_file("inventory_snapshots", FILES["inventory_snapshots"])
    processed = inspect_file("processed_data", FILES["processed_data"])
    weekly = inspect_file("weekly_model_data", FILES["weekly_model_data"])

   
    if inventory is not None:
        print("\n")
        print("=" * 75)
        print("INVENTORY SNAPSHOT ANALYSIS")
        print("=" * 75)
        print("\nInventory columns:")
        print(inventory.columns.tolist())

        if "sku_id" in inventory.columns:
            print(f"\nUnique inventory SKUs: " f"{inventory['sku_id'].nunique():,}")


        date_candidates = ["date", "snapshot_date", "week_start", "inventory_date", "timestamp"]
        inventory_date_col = None
        for col in date_candidates:
            if col in inventory.columns:
                inventory_date_col = col
                break

        if inventory_date_col:
            inventory[inventory_date_col] = pd.to_datetime(inventory[inventory_date_col], errors="coerce")

            print(f"\nInventory date column: " f"{inventory_date_col}")
            print(f"Unique inventory dates: " f"{inventory[inventory_date_col].nunique():,}")
            print(f"First inventory date: " f"{inventory[inventory_date_col].min()}")
            print(f"Last inventory date: " f"{inventory[inventory_date_col].max()}")
            print("\nInventory dates sample:")
            print(inventory[inventory_date_col].drop_duplicates().sort_values().head(20).to_string(index=False))

  
    if weekly is not None:
        print("\n")
        print("=" * 75)
        print("WEEKLY MODEL DATA ANALYSIS")
        print("=" * 75)

        weekly["week_start"] = pd.to_datetime(weekly["week_start"], errors="coerce")
        print(f"\nWeekly unique SKUs: " f"{weekly['sku_id'].nunique():,}")
        print(f"Weekly unique dates: "  f"{weekly['week_start'].nunique():,}")
        print(f"Weekly first date: " f"{weekly['week_start'].min()}")
        print(f"Weekly last date: " f"{weekly['week_start'].max()}")
       
        inventory_cols = ["on_hand_units", "on_order_units", "reorder_point", "safety_level"]
        print("\nWeekly inventory null counts:")

        for col in inventory_cols:
            if col in weekly.columns:
                null_count = weekly[col].isna().sum()
                print(f"{col:25} " f"{null_count:,}")


    if inventory is not None and weekly is not None:
        if inventory_date_col is not None:
            print("\n")
            print("=" * 75)
            print("DATE MATCH INVESTIGATION")
            print("=" * 75)

            inventory_dates = set(inventory[inventory_date_col].dropna().dt.normalize().unique())
            weekly_dates = set(weekly["week_start"].dropna().dt.normalize().unique())
            matched_dates = (inventory_dates & weekly_dates)
            weekly_only_dates = (weekly_dates - inventory_dates)
            inventory_only_dates = (inventory_dates - weekly_dates)

            print(f"\nInventory dates : " f"{len(inventory_dates):,}")
            print(f"Weekly dates    : " f"{len(weekly_dates):,}")
            print(f"Matched dates   : " f"{len(matched_dates):,}")
            print(f"Weekly-only dates: " f"{len(weekly_only_dates):,}")
            print(f"Inventory-only dates: " f"{len(inventory_only_dates):,}")

            if weekly_only_dates:
                print("\nSample weekly-only dates:")
                for d in sorted(weekly_only_dates)[:20]:
                    print(d)

            if inventory_only_dates:
                print("\nSample inventory-only dates:")
                for d in sorted(inventory_only_dates)[:20]:
                    print(d)


    if inventory is not None and weekly is not None:
        if ("sku_id" in inventory.columns and inventory_date_col is not None):
            print("\n")
            print("=" * 75)
            print("SKU + DATE MATCH INVESTIGATION")
            print("=" * 75)

            inv_key = pd.DataFrame({"sku_id": inventory["sku_id"], "match_date": inventory[inventory_date_col].dt.normalize()}).drop_duplicates()
            weekly_key = pd.DataFrame({"sku_id": weekly["sku_id"], "match_date": weekly["week_start"].dt.normalize()}).drop_duplicates()
            merged = weekly_key.merge(inv_key, on=["sku_id", "match_date"], how="left", indicator=True)
            total_keys = len(merged)
            matched_keys = (merged["_merge"].eq("both").sum())
            missing_keys = (merged["_merge"].eq("left_only").sum())

            print(f"\nWeekly SKU-date keys : " f"{total_keys:,}")
            print(f"Matched SKU-date keys: " f"{matched_keys:,}")
            print(f"Missing SKU-date keys : " f"{missing_keys:,}")
            print(f"Match percentage      : " f"{matched_keys / total_keys * 100:.2f}%")
            print(f"Missing percentage    : " f"{missing_keys / total_keys * 100:.2f}%")
           
            missing = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
            print("\nSample missing SKU-date combinations:")
            print(missing.head(30).to_string(index=False))

   
    print("\n")
    print("=" * 75)
    print("INVESTIGATION COMPLETE")
    print("=" * 75)
    print("\nIMPORTANT:")
    print("Do NOT modify risk thresholds yet.")
    print("Do NOT replace missing inventory with 0.")
    print("First fix SKU + date inventory matching.")
    print("=" * 75)


if __name__ == "__main__":
    main()