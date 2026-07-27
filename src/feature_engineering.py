import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


class FeatureEngineering:
    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def create_date_features(self):
        logger.info("Creating Date Features...")
        self.df["year"] = self.df["date"].dt.year
        self.df["month"] = self.df["date"].dt.month
        self.df["week"] = self.df["date"].dt.isocalendar().week.astype(int)
        self.df["day"] = self.df["date"].dt.day
        self.df["day_of_week"] = self.df["date"].dt.dayofweek
        self.df["quarter"] = self.df["date"].dt.quarter
        self.df["is_weekend"] = (
            self.df["day_of_week"] >= 5
        ).astype(int)

    def create_lag_features(self):
        logger.info("Creating Lag Features...")
        self.df = self.df.sort_values(
            ["sku_id", "date"]
        )

        self.df["lag_1"] = (
            self.df.groupby("sku_id")["units_sold"]
            .shift(1)
        )

        self.df["lag_7"] = (
            self.df.groupby("sku_id")["units_sold"]
            .shift(7)
        )

        self.df["lag_14"] = (
            self.df.groupby("sku_id")["units_sold"]
            .shift(14)
        )

    def create_rolling_features(self):
        logger.info("Creating Rolling Features...")
        self.df["rolling_mean_7"] = (
            self.df.groupby("sku_id")["units_sold"]
            .transform(
                lambda x:
                x.rolling(7).mean()
            )
        )

        self.df["rolling_std_7"] = (
            self.df.groupby("sku_id")["units_sold"]
            .transform(
                lambda x:
                x.rolling(7).std()
            )
        )

        self.df["rolling_mean_30"] = (
            self.df.groupby("sku_id")["units_sold"]
            .transform(
                lambda x:
                x.rolling(30).mean()
            )
        )

    def create_price_features(self):
        logger.info("Creating Price Features...")
        if "unit_price" in self.df.columns:
            self.df["price_difference"] = (
                self.df["list_price"]
                - self.df["unit_price"]
            )

            self.df["discount_percentage"] = (
                (
                    self.df["list_price"]
                    - self.df["unit_price"]
                )
                /
                self.df["list_price"]
            ) * 100

    def create_inventory_features(self):
        logger.info("Creating Inventory Features...")
        self.df["inventory_gap"] = (
            self.df["on_hand_units"]
            -
            self.df["reorder_point"]
        )

        self.df["total_inventory"] = (
            self.df["on_hand_units"]
            +
            self.df["on_order_units"]
        )

    def fill_missing_values(self):
        logger.info("Handling Missing Values...")
        self.df.fillna(0, inplace=True)

    def remove_invalid_rows(self):
        logger.info("Removing Invalid Records...")
        self.df = self.df[
            self.df["units_sold"] >= 0
        ]

    def build_features(self):
        logger.info("Starting Feature Engineering...")
        self.create_date_features()
        self.create_lag_features()
        self.create_rolling_features()
        self.create_price_features()
        self.create_inventory_features()
        self.remove_invalid_rows()
        self.fill_missing_values()
        logger.info("Feature Engineering Completed.")
        return self.df

if __name__ == "__main__":
    from data_loader import DataLoader
    from preprocessing import DataPreprocessor
    loader = DataLoader()
    datasets = loader.load_all()
    preprocessor = DataPreprocessor(datasets)
    processed = preprocessor.process()
    engineer = FeatureEngineering(processed)
    final_df = engineer.build_features()
    print(final_df.head())
    print("\nFinal Shape :", final_df.shape)