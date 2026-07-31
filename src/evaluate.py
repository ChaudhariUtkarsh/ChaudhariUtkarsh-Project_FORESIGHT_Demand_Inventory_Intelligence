import logging
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)

    def mae(self):
        return mean_absolute_error(self.y_true, self.y_pred)

    def rmse(self):
        return np.sqrt(mean_squared_error(self.y_true, self.y_pred))

    def r2(self):
        return r2_score(self.y_true, self.y_pred)

    def mape(self):
        y_true = np.where(self.y_true == 0, 1, self.y_true)
        return np.mean(np.abs((y_true - self.y_pred) / y_true)) * 100

    def wape(self):
        numerator   = np.sum(np.abs(self.y_true - self.y_pred))
        denominator = np.sum(np.abs(self.y_true))
        if denominator == 0:
            return 0.0
        return (numerator / denominator) * 100

    def summary(self):
        logger.info("Model Evaluation")
        results = {
            "MAE":      round(self.mae(),  4),
            "RMSE":     round(self.rmse(), 4),
            "R2 Score": round(self.r2(),   4),
            "MAPE (%)": round(self.mape(), 2),
            "WAPE (%)": round(self.wape(), 2)
        }
        print("\n")
        print("=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)
        for metric, value in results.items():
            print(f"{metric:<15}: {value}")
        print("=" * 50)
        return results

    @staticmethod
    def compare(model_name, baseline_metrics, model_metrics):
        print("\n")
        print("=" * 60)
        print("BASELINE vs MODEL COMPARISON")
        print("=" * 60)
        print(f"Baseline MAE  : {baseline_metrics['MAE']}")
        print(f"{model_name} MAE : {model_metrics['MAE']}")
        print()
        print(f"Baseline RMSE : {baseline_metrics['RMSE']}")
        print(f"{model_name} RMSE: {model_metrics['RMSE']}")
        print()
        print(f"Baseline MAPE : {baseline_metrics['MAPE (%)']} %")
        print(f"{model_name} MAPE: {model_metrics['MAPE (%)']} %")
        print()
        print(f"Baseline WAPE : {baseline_metrics['WAPE (%)']} %")
        print(f"{model_name} WAPE: {model_metrics['WAPE (%)']} %")
        improvement = baseline_metrics["WAPE (%)"] - model_metrics["WAPE (%)"]
        print("\nImprovement :", round(improvement, 2), "%")
        if improvement > 0:
            print(f"{model_name} beats Seasonal Naive Baseline")
        else:
            print("Baseline performs better than the model")
        print("=" * 60)

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
    actual    = np.random.randint(50, 500, 100)
    predicted = actual + np.random.randint(-30, 30, 100)
    evaluator = ModelEvaluator(actual, predicted)
    evaluator.summary()
    evaluator.plot_predictions()
