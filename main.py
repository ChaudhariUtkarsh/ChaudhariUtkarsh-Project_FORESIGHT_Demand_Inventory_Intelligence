from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from predict import DemandPredictor


app = FastAPI(title="Project FORESIGHT API", description="AI-Powered Demand Forecasting & Inventory Intelligence API", version="1.0.0")

try:
    predictor = DemandPredictor()
except Exception as e:
    predictor = None
    print(f"Model loading warning: {e}")


class PredictionRequest(BaseModel):
    year: int
    month: int
    week: int
    day: int
    day_of_week: int
    quarter: int
    is_weekend: int
    lag_1: float
    lag_7: float
    lag_14: float
    rolling_mean_7: float
    rolling_std_7: float
    rolling_mean_30: float
    price_difference: float
    discount_percentage: float
    inventory_gap: float
    total_inventory: float
    on_hand_units: float
    on_order_units: float
    reorder_point: float
    sku_id: Optional[str] = None


@app.get("/")
def home():
    return {"project": "Project FORESIGHT", "status": "running", "message": "Demand Forecasting API is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": predictor is not None}

@app.post("/predict")
def predict(request: PredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Prediction model is not loaded.")
    try:
        input_data = request.model_dump()
        prediction = predictor.predict(input_data)
        return {"status": "success", "sku_id": request.sku_id, "predicted_units": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)