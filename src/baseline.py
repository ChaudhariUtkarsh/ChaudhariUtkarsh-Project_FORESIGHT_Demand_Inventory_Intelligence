import pandas as pd
import numpy as np

SEASON_LENGTH = 52


def seasonal_naive_forecast(df, sku_col="sku_id", date_col="week_start", target_col="units_sold", season_length=SEASON_LENGTH):
    """
    Generate a 52-week Seasonal Naive forecast.
    Forecast(t) = Actual Demand(t - 52 weeks)
    Parameters
    ----------
    df : pandas.DataFrame
        Weekly SKU-level demand data.

    sku_col : str
        SKU identifier column.

    date_col : str
        Weekly date column.

    target_col : str
        Demand / target column.

    season_length : int
        Seasonal cycle length in weeks.
        Default = 52.

    Returns
    -------
    pandas.DataFrame
        Original dataframe with an additional 'baseline_forecast' column.
    """


    if season_length != 52:
        raise ValueError("Project Foresight baseline must use " "a 52-week Seasonal Naive forecast.")

    required_columns = [sku_col, date_col, target_col]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")

    if data[date_col].isna().any():
        invalid_count = int(data[date_col].isna().sum())
        raise ValueError(f"Invalid dates found in '{date_col}': " f"{invalid_count} rows.")
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")

    if data[target_col].isna().any():
        invalid_count = int(data[target_col].isna().sum())
        raise ValueError(f"Invalid demand values found in " f"'{target_col}': {invalid_count} rows.")
    data = (data.sort_values(by=[sku_col, date_col]).reset_index(drop=True))
    duplicate_mask = data.duplicated(subset=[sku_col, date_col], keep=False)

    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        raise ValueError(f"Duplicate SKU-week records found: " f"{duplicate_count} rows.")

    data["baseline_forecast"] = (data.groupby(sku_col)[target_col].shift(52))
    data["baseline_forecast"] = (data["baseline_forecast"].clip(lower=0))
    return data


def calculate_wape(actual, forecast):
    """
    Calculate Weighted Absolute Percentage Error (WAPE).

    Formula:

        WAPE =
        Sum(|Actual - Forecast|)
        -----------------------
        Sum(|Actual|)
        x 100

    Lower WAPE is better.

    Returns
    -------
    float
        WAPE percentage.
    """

    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    if len(actual) != len(forecast):
        raise ValueError("Actual and forecast arrays must have " "the same length.")
    valid_mask = (np.isfinite(actual) & np.isfinite(forecast))

    actual = actual[valid_mask]
    forecast = forecast[valid_mask]

    if len(actual) == 0:
        return np.nan
    denominator = np.sum(np.abs(actual))

    if denominator == 0:
        return np.nan
    numerator = np.sum(np.abs(actual - forecast))
    return (numerator / denominator) * 100


def evaluate_baseline(df, actual_col="units_sold", forecast_col="baseline_forecast"):
    """
    Evaluate the 52-week Seasonal Naive baseline.

    Rows where the baseline forecast is unavailable are excluded from evaluation.

    Returns
    -------
    dict
        Baseline evaluation results.
    """

    required_columns = [actual_col, forecast_col]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    evaluation_data = (df[[actual_col, forecast_col]].dropna())

    if evaluation_data.empty:
        raise ValueError("No valid rows available for " "52-week Seasonal Naive evaluation.")

    actual = evaluation_data[actual_col]
    forecast = evaluation_data[forecast_col]
    wape = calculate_wape(actual=actual, forecast=forecast)
    return {"model": "52-Week Seasonal Naive", "season_length_weeks": 52, "metric": "WAPE", "wape": wape, "wape_percent": wape}


def print_baseline_summary(baseline_df, results):
    """Print baseline configuration and evaluation summary."""

    total_rows = len(baseline_df)
    valid_rows = (baseline_df["baseline_forecast"].notna().sum())
    unavailable_rows = (total_rows - valid_rows)

    print()
    print("=" * 70)
    print("PROJECT FORESIGHT — " "52-WEEK SEASONAL NAIVE BASELINE")
    print("=" * 70)

    print()
    print("Baseline Configuration")
    print("-" * 70)

    print("Season Length       : " "52 weeks")
    print("Meaning             : " "Same week from previous year")
    print("Forecast Frequency  : " "Weekly")
    print("Forecast Horizon    : " "6-8 weeks")
    print("Primary Metric      : " "WAPE")

    print()
    print("Evaluation")
    print("-" * 70)

    print(f"Total Rows          : " f"{total_rows:,}")
    print(f"Valid Baseline Rows : " f"{valid_rows:,}")
    print(f"Unavailable Rows    : " f"{unavailable_rows:,}")

    if pd.notna(results["wape"]):
        print(f"WAPE                : " f"{results['wape']:.4f}")
        print(f"WAPE (%)            : " f"{results['wape']:.2f}%")
    else:
        print("WAPE                : " "Not available")
    print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("PROJECT FORESIGHT — " "52-WEEK SEASONAL NAIVE BASELINE")
    print("=" * 70)

    INPUT_PATH = ("data/processed/" "weekly_model_data.csv")

    print()
    print(f"Loading weekly model data:\n" f"{INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded rows: " f"{len(df):,}")

    if "week_start" in df.columns:
        DATE_COLUMN = "week_start"
    elif "date" in df.columns:
        DATE_COLUMN = "date"
    else:
        raise ValueError("Neither 'week_start' nor 'date' " "column was found in " "weekly_model_data.csv.")
    baseline_df = seasonal_naive_forecast(df=df, sku_col="sku_id", date_col=DATE_COLUMN, target_col="units_sold", season_length=52)
    results = evaluate_baseline(df=baseline_df, actual_col="units_sold", forecast_col="baseline_forecast")
    print_baseline_summary(baseline_df=baseline_df, results=results)

    print()
    print("Baseline Forecast Sample")
    print("-" * 70)

    sample_columns = [DATE_COLUMN, "sku_id", "units_sold", "baseline_forecast"]
    print(baseline_df[sample_columns].head(15).to_string(index=False))

    print()
    print("=" * 70)
    print("52-WEEK SEASONAL NAIVE " "BASELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)