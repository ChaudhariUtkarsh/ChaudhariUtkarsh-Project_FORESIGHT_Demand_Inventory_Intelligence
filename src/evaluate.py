import logging
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)


# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# Model Evaluator
class ModelEvaluator:
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)

    def mae(self):
        return mean_absolute_error(self.y_true, self.y_pred)

    def rmse(self):
        return np.sqrt(
            mean_squared_error(self.y_true, self.y_pred))

    def r2(self):
        return r2_score(self.y_true, self.y_pred)

    def mape(self):
        y_true = np.where(self.y_true == 0, 1, self.y_true)
        return np.mean(np.abs((y_true - self.y_pred) / y_true)) * 100

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
        print("=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)

        for metric, value in results.items():
            print(f"{metric:<15}: {value}")
            print("=" * 50)
            return results

    def plot_predictions(self):
        plt.figure(figsize=(12,6))
        plt.plot(self.y_true, label="Actual")
        plt.plot(self.y_pred, label="Predicted")
        plt.title("Actual vs Predicted")
        plt.xlabel("Samples")
        plt.ylabel("Demand")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Testing
if __name__ == "__main__":
    np.random.seed(10)
    actual = np.random.randint(50, 500, 100)
    predicted = actual + np.random.randint(-30, 30, 100)
    evaluator = ModelEvaluator(actual, predicted)
    evaluator.summary()
    evaluator.plot_predictions()
