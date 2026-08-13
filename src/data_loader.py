import os
import logging
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class DataLoader:
    def __init__(self, data_path=None):
        if data_path is None:
            data_path = os.path.join(BASE_DIR, "data", "raw")
        self.data_path = data_path
        self.files = {"sales": "sales_daily.csv", "sku": "sku_master.csv", "calendar": "calendar.csv", "inventory": "inventory_snapshots.csv"}

    def validate_files(self):
        logger.info("Validating dataset files...")
        missing_files = []
        for file_name in self.files.values():
            file_path = os.path.join(self.data_path, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_name)
        if missing_files:
            raise FileNotFoundError(f"Missing Files : {missing_files}")
        logger.info("All dataset files found.")

    def load_csv(self, filename):
        file_path = os.path.join(self.data_path, filename)
        try:
            df = pd.read_csv(file_path)
            logger.info(f"{filename} Loaded Successfully " f"Shape = {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to load {filename}")
            raise e

    def load_all(self):
        self.validate_files()
        datasets = {
            "sales": self.load_csv(self.files["sales"]),
            "sku": self.load_csv(self.files["sku"]),
            "calendar": self.load_csv(self.files["calendar"]),
            "inventory": self.load_csv(self.files["inventory"])
        }
        logger.info("All datasets loaded successfully.")
        return datasets


if __name__ == "__main__":
    loader = DataLoader()
    data = loader.load_all()
    print("\nLoaded Datasets\n")
    for name, df in data.items():
        print("=" * 60)
        print(name.upper())
        print(df.head())
        print(df.shape)