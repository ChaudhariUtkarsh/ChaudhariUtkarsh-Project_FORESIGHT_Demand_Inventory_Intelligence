import os
import json
import joblib
import numpy as np


# PROJECT ROOT
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# MODEL PATHS
MODEL_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")


# HEADER
print("=" * 70)
print("PROJECT FORESIGHT - FORECAST")
print("=" * 70)
print("\nModel directory:")
print(MODEL_DIR)


# CHECK MODELS DIRECTORY
if not os.path.exists(MODEL_DIR):
    print("\n[ERROR] models folder does not exist.")
    print("Expected location:")
    print(MODEL_DIR)
    raise SystemExit(1)


# LOAD BEST MODEL
if os.path.exists(BEST_MODEL_PATH):
    best_model = joblib.load(BEST_MODEL_PATH)
    print("\n[OK] best_model.pkl loaded successfully")
    print("Model type:", type(best_model).__name__)

else:
    print("\n[ERROR] best_model.pkl not found")
    print("Expected file:")
    print(BEST_MODEL_PATH)
    raise SystemExit(1)


# LOAD LABEL ENCODER
if os.path.exists(LABEL_ENCODER_PATH):
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print("\n[OK] label_encoder.pkl loaded successfully")
    print("Encoder type:", type(label_encoder).__name__)

    try:
        print("Number of classes:", len(label_encoder.classes_))
    except Exception:
        pass

else:
    print("\n[WARNING] label_encoder.pkl not found")
    label_encoder = None


# LOAD MODEL METRICS
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        metrics_data = json.load(f)
    print("\n[OK] model_metrics.json loaded successfully")

else:
    print("\n[WARNING] model_metrics.json not found")
    metrics_data = {}


# GET BEST MODEL METRICS
best_model_name = metrics_data.get("best_model")

if best_model_name and best_model_name in metrics_data:
    best_metrics = metrics_data[best_model_name]

else:
    best_metrics = {}


#UNCERTAINTY MARGIN
uncertainty_margin = best_metrics.get("Prediction_Interval_Margin")

if uncertainty_margin is None:
    print(
        "\n[WARNING] 80% uncertainty margin not found "
        "in model_metrics.json."
    )
    print(
        "Please run train_model.py again after adding "
        "the uncertainty calculation."
    )

    uncertainty_margin = 0.0

else:
    uncertainty_margin = float(uncertainty_margin)
    print(
        f"\n[OK] 80% uncertainty margin loaded: "
        f"+/- {uncertainty_margin:.2f}"
    )


# PREDICTION FUNCTION
def predict_demand(features):
    """
    Predict demand and calculate 80% prediction interval.
    Parameters
    ----------
    features : list or numpy array
        Model input features.

    Returns
    -------
    dict
        Predicted demand, lower bound, upper bound, and prediction interval.
    """

    # Convert input to numpy array
    X = np.array(features, dtype=float).reshape(1, -1)

    # Model prediction
    prediction = best_model.predict(X)[0]

    # Demand cannot be negative
    prediction = max(float(prediction), 0.0)

    #UNCERTAINTY INTERVAL
    lower_bound = max(prediction - uncertainty_margin, 0.0)
    upper_bound = (prediction + uncertainty_margin)
    interval_width = (upper_bound - lower_bound)

    return {
        "Predicted Demand": round(prediction, 2),
        "80% Lower Bound": round(lower_bound, 2),
        "80% Upper Bound": round(upper_bound, 2),
        "80% Prediction Interval": (f"{lower_bound:.2f} - {upper_bound:.2f}"), "Interval Width": round(interval_width, 2)
    }


# EXAMPLE PREDICTION
try:
    n_features = int(getattr(best_model, "n_features_in_", 0))

except Exception:
    n_features = 0


if n_features > 0:
    # Demo input only.
    example_features = np.zeros(n_features)
    result = predict_demand(example_features)

    print("\n" + "=" * 70)
    print("FORECAST RESULT")
    print("=" * 70)

    print(
        f"Predicted Demand        : "
        f"{result['Predicted Demand']}"
    )

    print(
        f"80% Lower Bound         : "
        f"{result['80% Lower Bound']}"
    )

    print(
        f"80% Upper Bound         : "
        f"{result['80% Upper Bound']}"
    )

    print(
        f"80% Prediction Interval : "
        f"{result['80% Prediction Interval']}"
    )

    print(
        f"Interval Width          : "
        f"{result['Interval Width']}"
    )

    print("=" * 70)


# SHOW MODEL FILES
print("\nFiles inside models folder:")

for file_name in os.listdir(MODEL_DIR):
    file_path = os.path.join(MODEL_DIR, file_name)

    if os.path.isfile(file_path):
        file_size = os.path.getsize(file_path)
        print(f"  - {file_name} " f"({file_size / 1024:.2f} KB)")


print("\n" + "=" * 70)
print("FORECAST CHECK COMPLETED")
print("=" * 70)