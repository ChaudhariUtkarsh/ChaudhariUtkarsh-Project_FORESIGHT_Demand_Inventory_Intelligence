import pandas as pd

class SeasonalNaiveBaseline:
    def __init__(self, season_length=7):
        """
        season_length = 7  -> Weekly Seasonality
        season_length = 30 -> Monthly Seasonality
        """
        self.season_length = season_length

    def predict(self, data: pd.DataFrame, date_col="date", target_col="units_sold"):
        """
        Parameters
        ----------
        data : DataFrame
        date_col : Date Column
        target_col : Target Column

        Returns
        -------
        DataFrame
        """

        df = data.copy()

        # Convert to datetime
        df[date_col] = pd.to_datetime(df[date_col])

        # Sort data
        df = df.sort_values(date_col)

        # Seasonal Naive Forecast
        df["baseline_forecast"] = df[target_col].shift(self.season_length)

        # Fill first season values
        df["baseline_forecast"] = (df["baseline_forecast"].fillna(df[target_col].mean()))

        return df

    def forecast_future(self, history, periods=7):
        """
        Forecast future values using last season.
        """

        history = list(history)

        if len(history) < self.season_length:
            raise ValueError("Not enough history for seasonal baseline.")

        future = history[-self.season_length:]

        prediction = []

        while len(prediction) < periods:
            prediction.extend(future)

        return prediction[:periods]


if __name__ == "__main__":

    # Sample Data
    data = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=30),
        "units_sold": [
            10,12,15,18,20,22,25,
            11,13,16,19,21,23,27,
            12,14,17,20,22,24,28,
            13,15,18,21,23,25,29,
            30,32
        ]
    })

    baseline = SeasonalNaiveBaseline(season_length=7)
    result = baseline.predict(data)
    print(result.tail(10))

    future = baseline.forecast_future(result["units_sold"], periods=7)
    print("\nNext 7 Days Forecast")
    print(future)