import os
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
SALES_FILE = os.path.join(RAW_DIR, "sales_daily.csv")
OUTPUT_FILE = os.path.join(RAW_DIR, "sales_daily_104weeks.csv")


print("\nLoading existing sales data...")
df = pd.read_csv(SALES_FILE)
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["date", "sku_id"]).copy()
df["sku_id"] = (df["sku_id"].astype(str))

print(f"Existing rows : {len(df)}")
print(f"Existing start: {df['date'].min().date()}")
print(f"Existing end  : {df['date'].max().date()}")
print(f"SKUs          : {df['sku_id'].nunique()}")

START_DATE = pd.Timestamp("2024-01-01")
END_DATE = pd.Timestamp("2025-12-31")
dates = pd.date_range(START_DATE, END_DATE, freq="D")
skus = sorted(df["sku_id"].unique())


if "units_sold" not in df.columns:
    raise ValueError("units_sold column missing from sales_daily.csv")

df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce").fillna(0)
sku_base = (df.groupby("sku_id")["units_sold"].median().clip(lower=1))


if "unit_price" in df.columns:
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    sku_price = (df.groupby("sku_id")["unit_price"].median().fillna(0))
else:
    sku_price = pd.Series(0, index=skus)

rng = np.random.default_rng(42)
rows = []
print(f"\nGenerating {len(skus)} SKUs " f"for {len(dates)} days...")


for sku in skus:
    base = float(sku_base.get(sku, 10))
    price = float(sku_price.get(sku, 0))
    sku_factor = rng.uniform(0.85, 1.15)

    for date in dates:
        weekday_factor = {
            0: 1.00,   
            1: 1.05,
            2: 1.08,
            3: 1.12,
            4: 1.18,
            5: 1.25,
            6: 0.92   
        }[date.weekday()]

        yearly_factor = (1.0 + 0.18 * np.sin(2 * np.pi * date.dayofyear / 365.25))

        days_from_start = (date - START_DATE).days
        trend_factor = (1.0 + 0.0004 * days_from_start)

        noise = rng.normal(1.0, 0.10)

        promotion = int(rng.random() < 0.08)
        promotion_factor = (1.20 if promotion == 1 else 1.0)

        demand = (base * sku_factor * weekday_factor * yearly_factor * trend_factor * promotion_factor * noise)
        units_sold = max(0, int(round(demand)))

        rows.append({"date": date, "sku_id": sku, "units_sold": units_sold, "unit_price": price, "promotion": promotion})


generated = pd.DataFrame(rows)
generated.to_csv(OUTPUT_FILE, index=False)

print()
print("=" * 65)
print("104-WEEK SALES DATA GENERATED SUCCESSFULLY")
print("=" * 65)

print(f"Output : {OUTPUT_FILE}")
print(f"Rows   : {len(generated):,}")
print(f"Start  : {generated['date'].min().date()}")
print(f"End    : {generated['date'].max().date()}")
print(f"Days   : {generated['date'].nunique()}")
print(f"SKUs   : {generated['sku_id'].nunique()}")