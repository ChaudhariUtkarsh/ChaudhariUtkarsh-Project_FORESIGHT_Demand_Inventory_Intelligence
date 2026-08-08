import pandas as pd
import numpy as np


class SeasonalNaiveBaseline:
    """Weekly seasonal-naive baseline for SKU-level demand.
    The Zidio brief requires a seasonal-naive baseline. The supplied
    project contains roughly one year of history, so a 4-week seasonal
    lag is used as an operational seasonal reference; a 52-week annual
    lag would not have enough prior-year observations for a fair backtest.
    """

    def __init__(self, season_length=4):
        self.season_length = int(season_length)

    def predict(self, data, date_col="week_start", target_col="weekly_units_sold", group_col="sku_id"):
        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values([group_col, date_col]).reset_index(drop=True)
        df["baseline_forecast"] = (df.groupby(group_col)[target_col].shift(self.season_length))
        fallback = df.groupby(group_col)[target_col].transform("mean")
        df["baseline_forecast"] = df["baseline_forecast"].fillna(fallback).clip(lower=0)
        return df

    def forecast_future(self, history, periods=6):
        history = [max(float(x), 0.0) for x in history]
        if len(history) < self.season_length:
            raise ValueError(f"Need at least {self.season_length} weeks of history for seasonal-naive forecasting.")
        values = history.copy()
        predictions = []
        for _ in range(int(periods)):
            pred = max(values[-self.season_length], 0.0)
            predictions.append(pred)
            values.append(pred)
        return predictions