import os
import logging
import numpy as np
import pandas as pd


# LOGGING CONFIGURATION
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class RiskScoring:
    def __init__(self, dataframe):
        self.df = dataframe.copy()

    # VALIDATE DATA
    def validate_data(self):
        required_columns = ["sku_id", "forecast_demand", "on_hand_units"]
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.df["forecast_demand"] = pd.to_numeric(self.df["forecast_demand"], errors="coerce").fillna(0).clip(lower=0)
        self.df["on_hand_units"] = pd.to_numeric(self.df["on_hand_units"], errors="coerce").fillna(0).clip(lower=0)


    # DAYS OF SUPPLY
    def calculate_days_of_supply(self):
        logger.info("Calculating Days of Supply...")
        average_daily_demand = (self.df["forecast_demand"] / 7)
        self.df["average_daily_demand"] = (average_daily_demand.round(2))
        self.df["days_of_supply"] = np.where(self.df["average_daily_demand"] > 0, self.df["on_hand_units"] / self.df["average_daily_demand"], np.inf)
        self.df["days_of_supply"] = self.df["days_of_supply"].replace([np.inf, -np.inf], np.nan)

    # STOCKOUT RISK
    def calculate_stockout_risk(self):
        logger.info("Calculating Stockout Risk...")
        self.df["stockout_risk_score"] = np.select([self.df["days_of_supply"] <= 3, self.df["days_of_supply"] <= 7, self.df["days_of_supply"] <= 14], [1.00, 0.75, 0.40], default=0.10)

    # OVERSTOCK RISK
    def calculate_overstock_risk(self):
        logger.info("Calculating Overstock Risk...")
        self.df["overstock_risk_score"] = np.select([self.df["days_of_supply"] >= 90, self.df["days_of_supply"] >= 60, self.df["days_of_supply"] >= 30], [1.00, 0.75, 0.40], default=0.10)

    # RISK LEVEL
    def calculate_total_risk(self):
        logger.info("Calculating Final Risk Score...")
        self.df["risk_score"] = (self.df[["stockout_risk_score", "overstock_risk_score"]].max(axis=1) * 100).round(2)

    # RISK LEVEL
    def assign_risk_level(self):
        logger.info("Assigning Risk Levels...")
        self.df["risk_level"] = np.select(
            [
                ((self.df["stockout_risk_score"] >= 0.75) | (self.df["overstock_risk_score"] >= 0.75)),
                ((self.df["stockout_risk_score"] >= 0.40) | (self.df["overstock_risk_score"] >= 0.40))
            ],
            ["HIGH", "MEDIUM"], default="LOW"
        )

    # PRIMARY RISK
    def assign_primary_risk(self):
        logger.info("Assigning Primary Risk...")
        self.df["primary_risk"] = np.select(
            [
                self.df["stockout_risk_score"] > self.df["overstock_risk_score"],
                self.df["overstock_risk_score"] > self.df["stockout_risk_score"] 
            ],
            ["STOCKOUT", "OVERSTOCK"], default="BALANCED"
        )

    # RECOMMENDATION
    def generate_recommendation(self):
        logger.info("Generating Recommendations...")
        self.df["recommendation"] = np.select(
            [
                ((self.df["primary_risk"] == "STOCKOUT") & (self.df["risk_level"] == "HIGH")),
                ((self.df["primary_risk"] == "STOCKOUT") & (self.df["risk_level"] == "MEDIUM")),
                ((self.df["primary_risk"] == "OVERSTOCK") & (self.df["risk_level"] == "HIGH")),
                ((self.df["primary_risk"] == "OVERSTOCK") & (self.df["risk_level"] == "MEDIUM"))
            ],
            ["URGENTLY REPLENISH STOCK", "PLAN REPLENISHMENT", "REDUCE INVENTORY / PROMOTE", "MONITOR INVENTORY"], default="NORMAL MONITORING"
        )

    # SAVE OUTPUT
    def save_output(self, output_folder="data/risk_analysis", filename="sku_risk_scoring.csv"):
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, filename)
        self.df.to_csv(output_path, index=False)
        logger.info(f"Risk Report Saved : {output_path}")

    # RUN PIPELINE
    def run(self):
        logger.info("Starting Risk Scoring Pipeline...")
        self.validate_data()
        self.calculate_days_of_supply()
        self.calculate_stockout_risk()
        self.calculate_overstock_risk()
        self.calculate_total_risk()
        self.assign_risk_level()
        self.assign_primary_risk()
        self.generate_recommendation()
        self.save_output()
        logger.info("Risk Scoring Completed Successfully.")
        return self.df


# TESTING
if __name__ == "__main__":
    sample = pd.DataFrame({
        "sku_id": [ "SKU001", "SKU002", "SKU003", "SKU004"],
        "forecast_demand": [200, 80, 150, 100],
        "on_hand_units": [120, 250, 140, 100]
    })

    scorer = RiskScoring(sample)
    result = scorer.run()

    print("\n" + "=" * 80)
    print("PROJECT FORESIGHT - RISK SCORING")
    print("=" * 80)
    print(
        result[
            [
                "sku_id",
                "forecast_demand",
                "on_hand_units",
                "average_daily_demand",
                "days_of_supply",
                "stockout_risk_score",
                "overstock_risk_score",
                "risk_score",
                "risk_level",
                "primary_risk",
                "recommendation"
            ]
        ].to_string(index=False)
    )
    print("=" * 80)