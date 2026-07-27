import os
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

class RiskScoring:
    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def calculate_stockout_risk(self):
        logger.info("Calculating Stockout Risk...")
        self.df["stockout_risk"] = (
            self.df["forecast_demand"] -
            self.df["on_hand_units"]
        )

        self.df["stockout_score"] = (
            self.df["stockout_risk"]
            .clip(lower=0)
        )

    def calculate_overstock_risk(self):
        logger.info("Calculating Overstock Risk...")
        self.df["overstock_risk"] = (
            self.df["on_hand_units"] -
            self.df["forecast_demand"]
        )

        self.df["overstock_score"] = (
            self.df["overstock_risk"]
            .clip(lower=0)
        )

    def calculate_total_risk(self):
        logger.info("Calculating Final Risk Score...")
        max_score = max(
            self.df["stockout_score"].max(),
            self.df["overstock_score"].max(),
            1
        )

        self.df["risk_score"] = ((self.df[["stockout_score", "overstock_score"]].max(axis=1)/ max_score) * 100).round(2)

    def assign_risk_level(self):
        logger.info("Assigning Risk Levels...")
        def risk(score):
            if score >= 70:
                return "High"
            elif score >= 40:
                return "Medium"
            return "Low"

        self.df["risk_level"] = (
            self.df["risk_score"]
            .apply(risk)
        )

    def generate_recommendation(self):
        logger.info("Generating Recommendations...")
        recommendations = []
        for _, row in self.df.iterrows():
            if row["stockout_score"] > 0:
                recommendations.append("Reorder Inventory Immediately")
            elif row["overstock_score"] > 0:
                recommendations.append("Run Discount / Promotion")
            else:
                recommendations.append("Inventory Level is Healthy")
        self.df["recommendation"] = recommendations

    def save_output(
        self,
        output_folder="outputs",
        filename="risk_report.csv"
    ):
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(
            output_folder,
            filename
        )

        self.df.to_csv(
            output_path,
            index=False
        )

        logger.info(
            f"Risk Report Saved : {output_path}"
        )

    def run(self):
        logger.info("Starting Risk Scoring Pipeline...")
        self.calculate_stockout_risk()
        self.calculate_overstock_risk()
        self.calculate_total_risk()
        self.assign_risk_level()
        self.generate_recommendation()
        self.save_output()
        logger.info("Risk Scoring Completed Successfully.")
        return self.df


# Testing
if __name__ == "__main__":
    sample = pd.DataFrame({
        "sku_id": [101,102,103,104],
        "forecast_demand": [200,80,150,100],
        "on_hand_units": [120,250,140,100]
    })

    scorer = RiskScoring(sample)
    result = scorer.run()
    print(result)