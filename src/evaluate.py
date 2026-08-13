import logging
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SEASON_LENGTH = 52


class ModelEvaluator:
    def __init__(self, y_true, y_pred):
        self.y_true = np.asarray(y_true, dtype=float)
        self.y_pred = np.asarray(y_pred, dtype=float)
        if len(self.y_true) != len(self.y_pred):
            raise ValueError("y_true and y_pred must have the same length.")

    def mae(self):
        return mean_absolute_error(self.y_true, self.y_pred)

    def rmse(self):
        return np.sqrt(
            mean_squared_error(self.y_true, self.y_pred))

    def r2(self):
        return r2_score(self.y_true, self.y_pred)

    def mape(self):
        denominator = np.where(self.y_true == 0, 1, self.y_true)
        return np.mean(np.abs((self.y_true - self.y_pred) / denominator)) * 100

    def wape(self):
        numerator = np.sum(np.abs(self.y_true - self.y_pred))
        denominator = np.sum(np.abs(self.y_true))
        if denominator == 0:
            return 0.0
        return (numerator / denominator) * 100

    def summary(self):
        logger.info("Model Evaluation")
        results = {
            "MAE": round(self.mae(), 4),
            "RMSE": round(self.rmse(), 4),
            "R2 Score": round(self.r2(), 4),
            "MAPE (%)": round(self.mape(), 2),
            "WAPE (%)": round(self.wape(), 2)
        }

        print("\n")
        print("=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        for metric, value in results.items():
            print(f"{metric:<15}: {value}")
        print("=" * 60)
        return results


    @staticmethod
    def compare(model_name, baseline_metrics, model_metrics, baseline_name="52-Week Seasonal Naive"):
        print("\n")
        print("=" * 70)
        print("52-WEEK SEASONAL NAIVE BASELINE vs MODEL")
        print("=" * 70)

        print(f"Baseline Model  : {baseline_name}")
        print(f"Season Length   : {SEASON_LENGTH} weeks")
        print()

        print(f"Baseline MAE    : " f"{baseline_metrics['MAE']}")
        print(f"{model_name} MAE : " f"{model_metrics['MAE']}")
        print()

        print(f"Baseline RMSE   : " f"{baseline_metrics['RMSE']}")
        print(f"{model_name} RMSE: " f"{model_metrics['RMSE']}")
        print()

        print(f"Baseline MAPE   : " f"{baseline_metrics['MAPE (%)']} %")
        print(f"{model_name} MAPE: " f"{model_metrics['MAPE (%)']} %")
        print()

        baseline_wape = baseline_metrics["WAPE (%)"]
        model_wape = model_metrics["WAPE (%)"]
        print(f"Baseline WAPE   : " f"{baseline_wape} %")
        print(f"{model_name} WAPE: " f"{model_wape} %")

        if baseline_wape != 0:
            improvement = ((baseline_wape - model_wape) / baseline_wape) * 100
        else:
            improvement = 0.0

        print()
        print("WAPE Improvement: " f"{improvement:.2f}%")

        if model_wape < baseline_wape:
            print()
            print(f"{model_name} beats the " f"52-week Seasonal Naive baseline.")
        elif model_wape > baseline_wape:
            print()
            print("52-week Seasonal Naive " "baseline performs better than " f"{model_name}.")
        else:
            print()
            print("Model and baseline have " "the same WAPE.")

        print("=" * 70)

        return {
            "baseline_name": baseline_name,
            "season_length_weeks": SEASON_LENGTH,
            "baseline_wape": baseline_wape,
            "model_name": model_name,
            "model_wape": model_wape,
            "wape_improvement_percent": round(improvement, 2)
        }

    def plot_predictions(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.y_true, label="Actual")
        plt.plot(self.y_pred, label="Predicted")
        plt.title("Actual vs Predicted")
        plt.xlabel("Samples")
        plt.ylabel("Demand")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    np.random.seed(10)
    actual = np.random.randint(50, 500, 100)
    predicted = (actual + np.random.randint(-30, 30, 100))
    evaluator = ModelEvaluator(actual, predicted)
    model_metrics = evaluator.summary()
    evaluator.plot_predictions()
    baseline_metrics = {"MAE": 0.0, "RMSE": 0.0, "MAPE (%)": 0.0, "WAPE (%)": 0.0}