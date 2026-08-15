import os
import math
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.predict import DemandPredictor


app = FastAPI(title="Project FORESIGHT API", description=("Weekly SKU-level Demand Forecasting " "and Inventory Intelligence API"), version="3.0")


app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
try:
    predictor = DemandPredictor()
    print("SUCCESS: DemandPredictor loaded successfully.")
except Exception as exc:
    predictor = None
    print(f"WARNING: DemandPredictor could not be loaded: {exc}")


class PredictRequest(BaseModel):
    sku_id: int = Field(default=101, description="SKU identifier", examples=[101])
    forecast_weeks: int = Field(default=6, ge=1, le=12)


class ScoreRequest(BaseModel):
    sku_id: int = Field(default=101, description="SKU identifier", examples=[101])
    forecast_weeks: int = Field(default=6, ge=6, le=8)
    lead_time_weeks: int = Field(default=1, ge=1, le=8)


@app.get("/")
def root():
    return {
        "project": "Project FORESIGHT",
        "version": "3.0",
        "service": "Demand Forecasting & Inventory Intelligence",
        "forecast_grain": "weekly SKU-level",
        "forecast_horizon": "6-8 weeks",
        "status": "running",
        "model_status": ("loaded" if predictor is not None else "not_loaded"),
        "health": "/health",
        "docs": "/docs",
        "prediction_endpoint": "/predict",
        "scoring_endpoint": "/score"
    }


@app.get("/health")
def health():
    if predictor is not None:
        model_status = "loaded"
        status = "healthy"
    else:
        model_status = "not_loaded"
        status = "degraded"
    return {"status": status, "service": "Project FORESIGHT Scoring API", "model_status": model_status}


def normalize_forecast_result(result):
    """Converts different forecast result formats into a JSON-compatible list of dictionaries."""
    if result is None:
        return []
    if hasattr(result, "to_dict"):
        try:
            return result.to_dict(orient="records")
        except TypeError:
            result = result.to_dict()
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        if "forecast" in result:
            return result["forecast"]
        return result
    return result


def calculate_risk(forecast, lead_time_weeks, sku_id):
    if not forecast:
        return {
            "risk_level": "UNKNOWN",
            "risk_score": 0,
            "stockout_risk_score": 0,
            "overstock_risk_score": 0,
            "primary_risk": "UNKNOWN",
            "average_daily_demand": 0,
            "days_of_supply": None,
            "lead_time_demand": 0,
            "available_inventory": 0,
            "stockout_gap_units": 0,
            "forward_window_demand": 0,
            "excess_inventory_units": 0,
            "recommendation": "No forecast data available"
        }

    total_forecast = sum(
        float(week.get("predicted_demand", 0)) for week in forecast if isinstance(week, dict))

    lead_time_weeks = min(lead_time_weeks, len(forecast))
    lead_time_demand = sum(float(week.get("predicted_demand", 0)) for week in forecast[:lead_time_weeks] if isinstance(week, dict))
    first_week = forecast[0]

    if not isinstance(first_week, dict):
        first_week = {}
    on_hand = float(first_week.get("on_hand_units", 0))
    on_order = float(first_week.get("on_order_units", 0))
    available_inventory = (on_hand + on_order)

    forecast_days = max(len(forecast) * 7, 1)
    average_daily_demand = (total_forecast / forecast_days)

    if average_daily_demand > 0:
        days_of_supply = (on_hand / average_daily_demand)
    else:
        days_of_supply = float("inf")

    if days_of_supply <= 3:
        stockout_risk_score = 1.00
    elif days_of_supply <= 7:
        stockout_risk_score = 0.75
    elif days_of_supply <= 14:
        stockout_risk_score = 0.40
    else:
        stockout_risk_score = 0.10

    if days_of_supply >= 90:
        overstock_risk_score = 1.00
    elif days_of_supply >= 60:
        overstock_risk_score = 0.75
    elif days_of_supply >= 30:
        overstock_risk_score = 0.40
    else:
        overstock_risk_score = 0.10

    if (stockout_risk_score >= 0.75 or overstock_risk_score >= 0.75):
        risk_level = "HIGH"
    elif (stockout_risk_score >= 0.40 or overstock_risk_score >= 0.40):
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    if stockout_risk_score > overstock_risk_score:
        primary_risk = "STOCKOUT"
    elif overstock_risk_score > stockout_risk_score:
        primary_risk = "OVERSTOCK"
    else:
        primary_risk = "BALANCED"

    risk_score = round(
        max(stockout_risk_score, overstock_risk_score) * 100, 2)

    stockout_gap = max(lead_time_demand - available_inventory, 0)
    excess_inventory = max(on_hand - total_forecast, 0)

    sku_id_normalized = str(sku_id).strip()
    sku_history = predictor.data[predictor.data["sku_id"].astype(str).str.strip() == sku_id_normalized]

    if sku_history.empty:
        list_price = 0.0
    else:
        list_price = pd.to_numeric(sku_history.iloc[-1].get("list_price", 0), errors="coerce")
        if pd.isna(list_price):
            list_price = 0.0
        else:
            list_price = float(list_price)
    sales_at_risk = (stockout_gap * list_price)
    locked_capital = (excess_inventory * list_price)

    if (primary_risk == "STOCKOUT" and risk_level == "HIGH"):
        recommendation = ("URGENTLY REPLENISH STOCK")
    elif (primary_risk == "STOCKOUT" and risk_level == "MEDIUM"):
        recommendation = ("PLAN REPLENISHMENT")
    elif (primary_risk == "OVERSTOCK" and risk_level == "HIGH"):
        recommendation = ("REDUCE INVENTORY / PROMOTE")
    elif (primary_risk == "OVERSTOCK" and risk_level == "MEDIUM"):
        recommendation = ("MONITOR INVENTORY")
    else:
        recommendation = ("NORMAL MONITORING")

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "stockout_risk_score": (stockout_risk_score),
        "overstock_risk_score": (overstock_risk_score),
        "primary_risk": primary_risk,
        "average_daily_demand": round(average_daily_demand, 2),
        "days_of_supply": (round(days_of_supply, 2) if math.isfinite(days_of_supply) else None),
        "lead_time_demand": round(lead_time_demand, 2),
        "available_inventory": round(available_inventory, 2),
        "stockout_gap_units": round(stockout_gap, 2),
        "forward_window_demand": round(total_forecast, 2),
        "excess_inventory_units": round(excess_inventory, 2),
        "list_price": round(list_price, 2),
        "sales_at_risk": round(sales_at_risk, 2),
        "locked_capital": round(locked_capital, 2),
        "recommendation": recommendation
    }


@app.post("/predict")
def predict(request: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail=("Prediction model is not loaded."))
    try:
        result = predictor.forecast(sku_id=request.sku_id, forecast_weeks=request.forecast_weeks)
        if isinstance(result, dict):
            response = result.copy()
            if "forecast" in response:
                response["forecast"] = (normalize_forecast_result(response["forecast"]))
            return {"status": "success", **response}
        forecast = normalize_forecast_result(result)
        return {"status": "success", "sku_id": request.sku_id, "forecast_weeks": request.forecast_weeks, "forecast": forecast}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        print("=" * 80)
        print("PREDICTION ERROR")
        print("=" * 80)
        print(f"ERROR TYPE: {type(exc).__name__}")
        print(f"ERROR MESSAGE: {str(exc)}")
        print("=" * 80)
        raise HTTPException(status_code=500, detail=(f"Prediction failed: " f"{type(exc).__name__}: " f"{str(exc)}"))


@app.post("/score")
def score(request: ScoreRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail=("Prediction model is not loaded. " "Run src/train_model.py first."))
    try:
        result = predictor.forecast(sku_id=request.sku_id, forecast_weeks=request.forecast_weeks)
        if isinstance(result, dict):
            sku_id = result.get("sku_id", request.sku_id)
            forecast_horizon_weeks = result.get("forecast_horizon_weeks", request.forecast_weeks)
            forecast = normalize_forecast_result(result.get("forecast", []))
            total_forecast_units = result.get("total_forecast_units", sum(float(week.get("predicted_demand", 0)) for week in forecast if isinstance(week, dict)))
        else:
            sku_id = request.sku_id
            forecast_horizon_weeks = (request.forecast_weeks)
            forecast = normalize_forecast_result(result)
            total_forecast_units = sum(float(week.get("predicted_demand", 0)) for week in forecast if isinstance(week, dict))
        risk_result = calculate_risk(forecast=forecast, lead_time_weeks=request.lead_time_weeks, sku_id=request.sku_id)

        return {
            "status": "success",
            "service": ("Project FORESIGHT " "Inventory Scoring Service"),
            "sku_id": sku_id,
            "forecast_horizon_weeks": (forecast_horizon_weeks),
            "total_forecast_units": round(float(total_forecast_units), 2),
            "lead_time_weeks": (request.lead_time_weeks),
            "risk_level": (risk_result["risk_level"]),
            "risk_score": (risk_result["risk_score"]),
            "stockout_risk_score": (risk_result["stockout_risk_score"]),
            "overstock_risk_score": (risk_result["overstock_risk_score"]),
            "primary_risk": (risk_result["primary_risk"]),
            "average_daily_demand": (risk_result["average_daily_demand"]),
            "days_of_supply": (risk_result["days_of_supply"]),
            "lead_time_demand": (risk_result["lead_time_demand"]),
            "available_inventory": (risk_result["available_inventory"]),
            "stockout_gap_units": (risk_result["stockout_gap_units"]),
            "forward_window_demand": (risk_result["forward_window_demand"]),
            "excess_inventory_units": (risk_result["excess_inventory_units"]),
            "list_price": risk_result["list_price"],
            "sales_at_risk": risk_result["sales_at_risk"],
            "locked_capital": risk_result["locked_capital"],
            "recommendation": (risk_result["recommendation"]),
            "forecast": forecast
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        print("=" * 80)
        print("SCORING ERROR")
        print("=" * 80)
        print(f"ERROR TYPE: {type(exc).__name__}")
        print(f"ERROR MESSAGE: {str(exc)}")
        print("=" * 80)
        raise HTTPException(status_code=500, detail=(f"Scoring failed: " f"{type(exc).__name__}: " f"{str(exc)}"))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)