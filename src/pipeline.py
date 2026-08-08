import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


SALES_FILE = os.path.join(RAW_DIR, "sales_daily_104weeks.csv")
SKU_FILE = os.path.join(RAW_DIR, "sku_master.csv")
CALENDAR_FILE = os.path.join(RAW_DIR, "calendar.csv")
INVENTORY_FILE = os.path.join(RAW_DIR, "inventory_snapshots.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "processed_data.csv")


def parse_date_column(df, column_name, file_name):
    """
    Robust date parser.
    Handles common formats such as:
        13-01-2025
        13/01/2025
        2025-01-13
        01/13/2025
        mixed date formats
    Invalid dates are removed.
    """

    if column_name not in df.columns:
        raise ValueError(f"'{column_name}' column missing in {file_name}")
    original_count = len(df)
    raw_dates = (df[column_name].astype(str).str.strip())
    parsed = pd.to_datetime(raw_dates, format="mixed", dayfirst=True, errors="coerce")
    missing_mask = parsed.isna()

    if missing_mask.any():
        parsed_iso = pd.to_datetime(raw_dates[missing_mask], format="ISO8601", errors="coerce")
        parsed.loc[missing_mask] = parsed_iso
    missing_mask = parsed.isna()

    if missing_mask.any():
        common_formats = [
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%m-%d-%Y",
            "%m/%d/%Y",
            "%d-%m-%y",
            "%d/%m/%y",
            "%Y-%m-%d %H:%M:%S",
            "%d-%m-%Y %H:%M:%S",
            "%d/%m/%Y %H:%M:%S"
        ]
        for fmt in common_formats:
            if not parsed.isna().any():
                break
            mask = parsed.isna()
            try:
                attempt = pd.to_datetime(raw_dates[mask], format=fmt, errors="coerce")
                parsed.loc[mask] = (attempt)
            except Exception:
                continue
    df[column_name] = parsed
    invalid_count = (df[column_name].isna().sum())
    valid_count = (original_count - invalid_count)

    if invalid_count > 0:
        print(f"WARNING: {invalid_count} invalid dates found " f"in {file_name}. Removing those rows.")
        df = df.dropna(subset=[column_name]).copy()
    print(f"{file_name}: parsed " f"{valid_count}/{original_count} " f"dates successfully.")

    if valid_count > 0:
        print(f"{file_name}: date range = " f"{df[column_name].min().date()} " f"to " f"{df[column_name].max().date()}")
    return df


def load_data():
    print("\n[1/5] Loading raw datasets...")
    required_files = [SALES_FILE, SKU_FILE, CALENDAR_FILE, INVENTORY_FILE]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"\nMissing file:\n{file_path}")

    sales = pd.read_csv(SALES_FILE)
    sku = pd.read_csv(SKU_FILE)
    calendar = pd.read_csv(CALENDAR_FILE)
    inventory = pd.read_csv(INVENTORY_FILE)

    print(f"sales_daily: {sales.shape}")
    print(f"sku_master: {sku.shape}")
    print(f"calendar: {calendar.shape}")
    print(f"inventory_snapshots: {inventory.shape}")

    return (sales, sku, calendar, inventory)


def clean_data(sales, sku, calendar, inventory):
    print("\n[2/5] Cleaning data...")

    sales = parse_date_column(sales, "date", "sales_daily_104weeks.csv")
    calendar = parse_date_column(calendar, "date", "calendar.csv")
    inventory = parse_date_column(inventory, "date", "inventory_snapshots.csv")

    sales = sales.drop_duplicates()
    sku = sku.drop_duplicates(subset=["sku_id"])
    calendar = calendar.drop_duplicates(subset=["date"])
    inventory = inventory.drop_duplicates(subset=["date", "sku_id"])

    sales = sales.dropna(subset=["date", "sku_id"])
    calendar = calendar.dropna(subset=["date"])
    inventory = inventory.dropna(subset=["date", "sku_id"])

    sales_numeric_columns = ["units_sold", "revenue", "unit_price", "promo_flag"]

    for col in sales_numeric_columns:
        if col in sales.columns:
            sales[col] = pd.to_numeric(sales[col], errors="coerce")

   
    inventory_numeric_columns = ["on_hand_units", "on_order_units", "lead_time_days", "reorder_point"]

    for col in inventory_numeric_columns:
        if col in inventory.columns:
            inventory[col] = pd.to_numeric(inventory[col], errors="coerce")

  
    if "units_sold" in sales.columns:
        sales["units_sold"] = (sales["units_sold"].fillna(0).clip(lower=0))

    if "revenue" in sales.columns:
        sales["revenue"] = (sales["revenue"].fillna(0).clip(lower=0))

    if "unit_price" in sales.columns:
        sales["unit_price"] = (sales["unit_price"].fillna(0).clip(lower=0))

    if "promo_flag" in sales.columns:
        sales["promo_flag"] = (sales["promo_flag"].fillna(0).astype(int))

   
    for col in inventory_numeric_columns:
        if col in inventory.columns:
            inventory[col] = (inventory[col].fillna(0).clip(lower=0))

    sales["sku_id"] = (sales["sku_id"].astype(str).str.strip())
    sku["sku_id"] = (sku["sku_id"].astype(str).str.strip())
    inventory["sku_id"] = (inventory["sku_id"].astype(str).str.strip())

    sales = sales.sort_values(["sku_id", "date"]).reset_index(drop=True)
    calendar = calendar.sort_values(["date"]).reset_index(drop=True)
    inventory = inventory.sort_values(["sku_id", "date"]).reset_index(drop=True)

    return (sales, sku, calendar, inventory)


def create_analysis_dataset(sales, sku, calendar, inventory):
    print("\n[3/5] Creating analysis-ready dataset...")

    df = sales.merge(sku, on="sku_id", how="left")
    df = df.merge(calendar, on="date", how="left")
    df = df.dropna(subset=["date", "sku_id"]).copy()
    inventory = inventory.dropna(subset=["date", "sku_id"]).copy()

    df = df.sort_values(["date", "sku_id"]).reset_index(drop=True)
    inventory = inventory.sort_values(["date", "sku_id"]).reset_index(drop=True)

    df = pd.merge_asof(df, inventory, on="date", by="sku_id", direction="backward")
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)
    print(f"Analysis-ready dataset shape: {df.shape}")
    return df


def create_features(df):
    print("\n[4/5] Creating features...")
    df = df.sort_values(["sku_id", "date"]).copy()

    df["year"] = (df["date"].dt.year)
    df["month_num"] = (df["date"].dt.month)
    df["week_num"] = (df["date"].dt.isocalendar().week.astype(int))
    df["day_of_week"] = (df["date"].dt.dayofweek)

    df["week_start"] = (df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="D"))

    df["lag_1"] = (df.groupby("sku_id")["units_sold"].shift(1))
    df["lag_7"] = (df.groupby("sku_id")["units_sold"].shift(7))

    df["rolling_mean_7"] = (df.groupby("sku_id")["units_sold"].transform(lambda x: x.shift(1).rolling(7).mean()))
    df["rolling_mean_28"] = (df.groupby("sku_id")["units_sold"].transform(lambda x: x.shift(1).rolling(28).mean()))

    feature_columns = ["lag_1", "lag_7", "rolling_mean_7", "rolling_mean_28"]
    for col in feature_columns:
        if col in df.columns:
            df[col] = (df[col].fillna(0))
    return df


def save_data(df):
    print("\n[5/5] Saving processed dataset...")
    df.to_csv(OUTPUT_FILE, index=False)

    min_date = df["date"].min()
    max_date = df["date"].max()

    unique_days = (df["date"].nunique())
    unique_weeks = (df["week_start"].nunique())
    unique_skus = (df["sku_id"].nunique())
    weeks_per_sku = (df.groupby("sku_id")["week_start"].nunique())

    print()
    print("----------------------------------")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("----------------------------------")

    print("\nOutput file:")
    print(OUTPUT_FILE)
    print(f"\nFinal dataset shape: {df.shape}")
    print("\nDATA RANGE")

    print(f"Start date       : {min_date.date()}")
    print(f"End date         : {max_date.date()}")
    print(f"Unique days      : {unique_days}")
    print(f"Unique weeks     : {unique_weeks}")
    print(f"Unique SKUs      : {unique_skus}")
    print(f"Min SKU weeks    : {weeks_per_sku.min()}")
    print(f"Median SKU weeks : {weeks_per_sku.median()}")
    print(f"Max SKU weeks    : {weeks_per_sku.max()}")
    print(f"SKUs >= 52 weeks : " f"{(weeks_per_sku >= 52).sum()}")
    print(f"SKUs < 52 weeks  : " f"{(weeks_per_sku < 52).sum()}")


def run_pipeline():
    sales, sku, calendar, inventory = (load_data())
    (sales,sku, calendar, inventory) = clean_data(sales, sku, calendar, inventory)
    df = create_analysis_dataset(sales, sku, calendar, inventory)
    df = create_features(df)
    save_data(df)


if __name__ == "__main__":
    run_pipeline()