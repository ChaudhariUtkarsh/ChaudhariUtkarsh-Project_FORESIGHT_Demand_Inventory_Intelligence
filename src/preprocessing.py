import os
import logging
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, datasets):
        self.sales = datasets["sales"]
        self.sku = datasets["sku"]
        self.calendar = datasets["calendar"]
        self.inventory = datasets["inventory"]

    def remove_duplicates(self):
        logger.info("Removing duplicate records...")
        self.sales = self.sales.drop_duplicates()
        self.sku = self.sku.drop_duplicates()
        self.calendar = self.calendar.drop_duplicates()
        self.inventory = self.inventory.drop_duplicates()

    def handle_missing_values(self):
        logger.info("Handling missing values...")
        self.sales = self.sales.fillna(0)
        self.calendar = self.calendar.fillna("Unknown")
        self.inventory = self.inventory.fillna(0)
        self.sku = self.sku.fillna("Unknown")

    def convert_datatypes(self):
        logger.info("Converting data types...")
        if "date" in self.sales.columns:
            self.sales["date"] = pd.to_datetime(self.sales["date"], dayfirst=True)

        if "date" in self.calendar.columns:
            self.calendar["date"] = pd.to_datetime(self.calendar["date"], dayfirst=True)

        if "date" in self.inventory.columns:
            self.inventory["date"] = pd.to_datetime(self.inventory["date"], dayfirst=True)

        if "launch_date" in self.sku.columns:
            self.sku["launch_date"] = pd.to_datetime(self.sku["launch_date"])

    def merge_datasets(self):
        logger.info("Merging datasets...")
        merged = self.sales.merge(self.sku, on="sku_id", how="left")
        merged = merged.merge(self.calendar, on="date", how="left")
        merged = merged.merge(self.inventory, on=["date", "sku_id"], how="left")
        logger.info(f"Merged Shape : {merged.shape}")
        return merged

    def save_processed_data(self, dataframe, output_path=None):
        if output_path is None:
            output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, "processed_data.csv")
        dataframe.to_csv(output_file, index=False)
        logger.info(f"Processed data saved at : {output_file}")

    def process(self):
        logger.info("Starting preprocessing pipeline...")
        self.remove_duplicates()
        self.handle_missing_values()
        self.convert_datatypes()
        processed_df = self.merge_datasets()
        self.save_processed_data(processed_df)
        logger.info("Preprocessing completed successfully.")
        return processed_df