from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.predict import DemandPredictor
import os


# FASTAPI APPLICATION
app = FastAPI(
    title="Project FORESIGHT API",
    description=("Weekly SKU-level Demand Forecasting " "and Inventory Intelligence API"), version="3.0"
)


# CORS
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# LOAD MODEL
try:
    predictor = DemandPredictor()
    print("SUCCESS: DemandPredictor loaded successfully.")

except Exception as exc:
    predictor = None
    print(f"WARNING: DemandPredictor could not be loaded: {exc}")


# REQUEST MODELS
class PredictRequest(BaseModel):
    sku_id: str = Field("101", description="SKU identifier", examples=["101"])
    forecast_weeks: int = Field(6, ge=6, le=8, description="Forecast horizon: 6-8 weeks", examples=[6])


class ScoreRequest(BaseModel):
    sku_id: str = Field(..., description="SKU identifier")
    forecast_weeks: int = Field(default=6, ge=6, le=8, description="Forecast horizon: 6-8 weeks")
    lead_time_weeks: int = Field(default=1, ge=1, le=8, description="Supplier lead time in weeks")


# ROOT ENDPOINT
@app.get("/")
def root():
    return {
        "project": "Project FORESIGHT",
        "version": "3.0",
        "service": "Demand Forecasting & Inventory Intelligence",
        "forecast_grain": "weekly SKU-level",
        "forecast_horizon": "6-8 weeks",
        "status": "running",
        "health": "/health",
        "docs": "/docs",
        "prediction_endpoint": "/predict",
        "scoring_endpoint": "/score"
    }


# HEALTH CHECK
@app.get("/health")
def health():
    model_status = ("loaded" if predictor is not None else "not_loaded")
    return {"status": "healthy", "service": "Project FORESIGHT Scoring API", "model_status": model_status}


# HELPER FUNCTION
def calculate_risk(forecast, lead_time_weeks=1):

    if not forecast:
        return {"risk_level": "Unknown", "risk_score": 0, "recommendation": "No forecast data available"}
    total_forecast = sum(float( week.get( "predicted_demand",  0 ) )for week in forecast )
    lead_time_demand = sum(float(week.get("predicted_demand", 0))for week in forecast[:lead_time_weeks])
    first_week = forecast[0]
    on_hand = float(first_week.get("on_hand_units", 0))
    on_order = float(first_week.get("on_order_units", 0))
    available_inventory = (on_hand + on_order)
    stockout_gap = max(lead_time_demand - available_inventory, 0)
    forward_demand = total_forecast
    excess_inventory = max(on_hand - forward_demand, 0)


    if stockout_gap > 0:
        risk_level = "High Stockout"
        risk_score = min(100, int((stockout_gap / max(lead_time_demand, 1)) * 100))
        recommendation = ("Review replenishment immediately")


    elif (forward_demand > 0 and on_hand > forward_demand * 1.5):
        risk_level = "High Overstock"
        risk_score = min(100, int((excess_inventory / max(on_hand, 1)) * 100))
        recommendation = ("Review markdown / clearance")


    else:
        risk_level = "Healthy"
        risk_score = 10
        recommendation = ("Continue weekly monitoring")


    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "lead_time_demand": round(lead_time_demand, 2),
        "available_inventory": round(available_inventory, 2),
        "stockout_gap_units": round(stockout_gap, 2),
        "forward_window_demand": round(forward_demand, 2),
        "excess_inventory_units": round(excess_inventory, 2),
        "recommendation": recommendation
    }


# PREDICTION ENDPOINT
@app.post("/predict")
def predict(request: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail=("Prediction model is not loaded. " "Run src/train_model.py first."))


    try:
        result = predictor.forecast(request.sku_id, request.forecast_weeks)
        risk_result = calculate_risk(result["forecast"], lead_time_weeks=1)


        return {
            "status": "success",
            "sku_id": result["sku_id"],
            "forecast_horizon_weeks": result["forecast_horizon_weeks"],
            "total_forecast_units": result["total_forecast_units"],
            "forecast": result["forecast"],
            "risk_level": risk_result["risk_level"],
            "risk_score": risk_result["risk_score"],
            "recommendation": risk_result["recommendation"]
        }


    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


    except Exception as exc:
        raise HTTPException(status_code=500,detail=str(exc))


# SCORING ENDPOINT
@app.post("/score")
def score(request: ScoreRequest):

    if predictor is None:
        raise HTTPException(status_code=503, detail=("Prediction model is not loaded. " "Run src/train_model.py first."))


    try:
        result = predictor.forecast(request.sku_id, request.forecast_weeks)
        risk_result = calculate_risk(result["forecast"], lead_time_weeks=request.lead_time_weeks)
        return {

            "status": "success",
            "service": ("Project FORESIGHT " "Inventory Scoring Service"),
            "sku_id": result["sku_id"],
            "forecast_horizon_weeks": result["forecast_horizon_weeks"],
            "total_forecast_units": round(float(result["total_forecast_units"]), 2),
            "lead_time_weeks": (request.lead_time_weeks),
            "risk_level": risk_result["risk_level"],
            "risk_score": risk_result["risk_score"],
            "lead_time_demand": risk_result["lead_time_demand"],
            "available_inventory": risk_result["available_inventory"],
            "stockout_gap_units": risk_result["stockout_gap_units"],
            "forward_window_demand": risk_result["forward_window_demand"],
            "excess_inventory_units": risk_result["excess_inventory_units"],
            "recommendation": risk_result["recommendation"],
            "forecast": result["forecast"]
        }


    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# RUN LOCALLY
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)