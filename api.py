
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.predict import DemandPredictor

#App
app = FastAPI(
    title="Project FORESIGHT API",
    description="Retail Demand Forecasting — SKU-level prediction endpoint",
    version="1.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

#Load predictor once at startup
try:
    predictor = DemandPredictor()
except FileNotFoundError as e:
    predictor = None
    print(f"WARNING: {e} — run train_model.py first.")


#Request Schema
class PredictRequest(BaseModel):
    sku_id:              str   = "SKU001"
    year:                int   = 2025
    month:               int   = 7
    week:                int   = 30
    day:                 int   = 15
    day_of_week:         int   = 2
    quarter:             int   = 3
    is_weekend:          int   = 0
    lag_1:               float = 100.0
    lag_7:               float = 95.0
    lag_14:              float = 90.0
    rolling_mean_7:      float = 100.0
    rolling_std_7:       float = 8.5
    rolling_mean_30:     float = 98.0
    price_difference:    float = 10.0
    discount_percentage: float = 5.0
    inventory_gap:       float = 20.0
    total_inventory:     float = 500.0
    on_hand_units:       float = 300.0
    on_order_units:      float = 100.0
    reorder_point:       float = 80.0


#Response Schema
class PredictResponse(BaseModel):
    sku_id:           str
    forecast_units:   float
    risk_level:       str
    recommendation:   str


#Helpers
def get_risk(forecast: float, reorder_point: float) -> tuple[str, str]:
    ratio = forecast / max(reorder_point, 1)
    if ratio >= 1.5:
        return "High",   "Reorder Inventory Immediately"
    elif ratio >= 0.9:
        return "Medium", "Run Discount / Promotion"
    else:
        return "Low",    "Inventory Level is Healthy"


#Routes 
@app.get("/")
def root():
    return {"project": "Project FORESIGHT", "version": "1.0", "status":  "running", "docs":    "/docs"}


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": predictor is not None
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run src/train_model.py first.")

    data = request.model_dump()
    forecast = predictor.predict(data["sku_id"])
    risk, recommendation = get_risk(forecast, request.reorder_point)

    return PredictResponse(sku_id         = request.sku_id, forecast_units = forecast, risk_level     = risk, recommendation = recommendation)