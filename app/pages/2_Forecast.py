import os
import sys
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.predict import DemandPredictor


st.set_page_config(page_title="Demand Forecast", page_icon=" ", layout="wide")
st.title("Weekly Demand Forecast")
st.markdown("SKU-level demand forecast for the next **6-8 weeks**.")
st.markdown("---")


@st.cache_resource
def load_predictor():
    return DemandPredictor()
try:
    predictor = load_predictor()
except Exception as exc:
    st.error(str(exc))
    st.stop()


@st.cache_data
def load_weekly_history():
    path = os.path.join(PROJECT_ROOT, "data", "processed", "weekly_model_data.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    df = df.dropna(subset=["week_start"])
    return df
history = load_weekly_history()

if history.empty:
    st.warning("Weekly model data is not available. " "Run `python src/train_model.py` first.")
    st.stop()


required_columns = ["sku_id", "week_start", "units_sold"]
missing_columns = [col for col in required_columns if col not in history.columns]

if missing_columns:
    st.error(f"Missing columns in weekly_model_data.csv: " f"{missing_columns}")
    st.stop()
skus = sorted(history["sku_id"].astype(str).unique())


st.sidebar.header("Forecast Settings")
sku_id = st.sidebar.selectbox("Select SKU", skus)
forecast_weeks = st.sidebar.slider("Forecast Horizon (Weeks)", min_value=6, max_value=8, value=6, step=1)
st.sidebar.info("Zidio scope: weekly SKU-level forecasting " "over a defined 6-8 week horizon.")


def calculate_wape(actual, predicted):
    comparison = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()

    if comparison.empty:
        return None
    denominator = (comparison["actual"].abs().sum())

    if denominator == 0:
        return None

    error = (comparison["actual"] - comparison["predicted"]).abs().sum()
    return (error / denominator) * 100


def create_seasonal_naive(df, season_length=52):
    result = df.copy()
    result = result.sort_values(["sku_id", "week_start"]).reset_index(drop=True)
    result["seasonal_naive"] = (result.groupby("sku_id")["units_sold"].shift(season_length))
    return result

baseline_history = create_seasonal_naive(history, season_length=52)
baseline_valid = baseline_history.dropna(subset=["units_sold", "seasonal_naive"])
baseline_wape = calculate_wape(baseline_valid["units_sold"], baseline_valid["seasonal_naive"])


def load_model_metrics():
    possible_files = [
        os.path.join(PROJECT_ROOT, "data", "processed", "model_evaluation.csv"),
        os.path.join(PROJECT_ROOT, "data", "processed", "model_metrics.csv"),
        os.path.join(PROJECT_ROOT, "artifacts", "model_evaluation.csv"),
        os.path.join(PROJECT_ROOT, "artifacts", "model_metrics.csv")]

    for path in possible_files:
        if os.path.exists(path):
            try:
                metrics = pd.read_csv(path)
                return metrics
            except Exception:
                pass
    return pd.DataFrame()
model_metrics = load_model_metrics()


def get_model_wape(metrics, model_name):
    if metrics.empty:
        return None
    columns_lower = {col.lower(): col for col in metrics.columns}

    model_column = None
    wape_column = None

    for col in metrics.columns:
        if col.lower() in ["model", "model_name", "algorithm"]:
            model_column = col
            break

    for col in metrics.columns:
        if "wape" in col.lower():
            wape_column = col
            break

    if (model_column is None or wape_column is None):
        return None
    rows = metrics[metrics[model_column].astype(str).str.lower().str.contains(model_name.lower(), na=False)]

    if rows.empty:
        return None
    value = rows.iloc[0][wape_column]

    try:
        return float(str(value).replace("%", "").strip())
    except Exception:
        return None


xgb_wape = get_model_wape(model_metrics, "XGBoost")
lgb_wape = get_model_wape(model_metrics, "LightGBM")


st.sidebar.markdown("---")
st.sidebar.subheader("Model WAPE")
if baseline_wape is not None:
    st.sidebar.metric("Seasonal-Naive WAPE", f"{baseline_wape:.2f}%")

if xgb_wape is not None:
    st.sidebar.metric("XGBoost WAPE", f"{xgb_wape:.2f}%")

if lgb_wape is not None:
    st.sidebar.metric("LightGBM WAPE", f"{lgb_wape:.2f}%")


if st.button("Generate Weekly Forecast", type="primary"):
    try:
        result = predictor.forecast(sku_id, forecast_weeks)
        forecast_df = pd.DataFrame(result["forecast"])
        forecast_df["week_start"] = pd.to_datetime(forecast_df["week_start"])
        st.session_state["forecast_result"] = result
        st.session_state["forecast_df"] = forecast_df
    except Exception as exc:
        st.error(f"Forecast failed: {exc}")


if "forecast_df" not in st.session_state:
    st.info("Select an SKU and click " "**Generate Weekly Forecast**.")
    st.stop()

forecast_df = st.session_state["forecast_df"]
result = st.session_state["forecast_result"]
st.success(f"Weekly forecast generated for {sku_id}.")


c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Forecast Horizon", f"{forecast_weeks} Weeks")

with c2:
    st.metric("Total Forecast", f"{result['total_forecast_units']:,.0f} Units")

with c3:
    st.metric("Avg Weekly Demand", f"{forecast_df['predicted_demand'].mean():,.1f} Units")

with c4:
    st.metric("Peak Week", f"{forecast_df['predicted_demand'].max():,.1f} Units")


sku_history = history[history["sku_id"].astype(str) == str(sku_id)].copy()
sku_history = sku_history.sort_values("week_start")
last_week = sku_history["week_start"].max()

future_weeks = pd.date_range(start=last_week + pd.Timedelta(weeks=1), periods=forecast_weeks, freq="7D")
demand_lookup = dict(zip(sku_history["week_start"], sku_history["units_sold"]))
baseline_rows = []


for i, future_week in enumerate(future_weeks, start=1):
    previous_year_week = (future_week - pd.Timedelta(weeks=52))
    baseline_value = demand_lookup.get(previous_year_week, None)
    baseline_rows.append({"forecast_week": i, "week_start": future_week, "seasonal_naive_demand": baseline_value})
future_baseline = pd.DataFrame(baseline_rows)
forecast_df = forecast_df.merge(future_baseline, on=["forecast_week", "week_start"], how="left")
st.markdown("---")


st.subheader("Weekly SKU-Level Forecast")
show_columns = ["sku_id", "forecast_week", "week_start", "predicted_demand", "seasonal_naive_demand", "lower_bound_80", "upper_bound_80"]
available_columns = [col for col in show_columns if col in forecast_df.columns]

show_df = forecast_df[available_columns].copy()
show_df = show_df.rename(
    columns={
        "sku_id": "SKU",
        "forecast_week": "Forecast Week",
        "week_start": "Week Start",
        "predicted_demand": "ML Forecast Units",
        "seasonal_naive_demand": "Seasonal-Naive Units",
        "lower_bound_80": "80% Lower",
        "upper_bound_80": "80% Upper"
    }
)
st.dataframe(show_df, use_container_width=True, hide_index=True)


st.subheader("ML Forecast vs Seasonal-Naive Baseline")
fig = go.Figure()

fig.add_trace(go.Scatter(x=forecast_df["week_start"], y=forecast_df["upper_bound_80"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=forecast_df["week_start"], y=forecast_df["lower_bound_80"], mode="lines", line=dict(width=0), fill="tonexty", name="80% Prediction Interval"))
fig.add_trace(go.Scatter(x=forecast_df["week_start"], y=forecast_df["predicted_demand"], mode="lines+markers", name="ML Forecast"))
fig.add_trace(go.Scatter(x=forecast_df["week_start"], y=forecast_df["seasonal_naive_demand"], mode="lines+markers", name="Seasonal-Naive Baseline"))
fig.update_layout(title=(f"{sku_id} — " f"{forecast_weeks}-Week Forecast"), xaxis_title="Week", yaxis_title="Units", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)


st.markdown("---")
st.header("Model Performance Comparison")
st.markdown(
    """
    **WAPE (Weighted Absolute Percentage Error)** is used
    to compare forecasting performance.
    Lower WAPE = Better forecasting performance.
    """
)


comparison_rows = []
if baseline_wape is not None:
    comparison_rows.append({"Model": "Seasonal-Naive", "WAPE (%)": baseline_wape})

if xgb_wape is not None:
    comparison_rows.append({"Model": "XGBoost", "WAPE (%)": xgb_wape})

if lgb_wape is not None:
    comparison_rows.append({"Model": "LightGBM", "WAPE (%)": lgb_wape})

comparison_df = pd.DataFrame(comparison_rows)


if comparison_df.empty:
    st.warning(
        "Model evaluation metrics were not found. "
        "Run model evaluation first and save "
        "the results as model_evaluation.csv "
        "or model_metrics.csv."
    )

else:
    comparison_df = comparison_df.sort_values("WAPE (%)").reset_index(drop=True)
    comparison_df["WAPE (%)"] = comparison_df["WAPE (%)"].round(2)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)


    best_model = comparison_df.iloc[0]
    st.success(
        f"Best model: **{best_model['Model']}** "
        f"with WAPE = "
        f"**{best_model['WAPE (%)']:.2f}%**"
    )


    if baseline_wape is not None:
        st.subheader("Improvement over Seasonal-Naive Baseline")
        improvement_rows = []

        if xgb_wape is not None:
            xgb_improvement = ((baseline_wape - xgb_wape) / baseline_wape) * 100
            improvement_rows.append({"Model": "XGBoost", "Baseline WAPE (%)": baseline_wape, "Model WAPE (%)": xgb_wape, "Improvement (%)": xgb_improvement})

        if lgb_wape is not None:
            lgb_improvement = ((baseline_wape - lgb_wape) / baseline_wape) * 100
            improvement_rows.append({"Model": "LightGBM", "Baseline WAPE (%)": baseline_wape, "Model WAPE (%)": lgb_wape, "Improvement (%)": lgb_improvement})

        if improvement_rows:
            improvement_df = pd.DataFrame(improvement_rows)
            improvement_df = (improvement_df.round(2))
            st.dataframe(improvement_df, use_container_width=True, hide_index=True)
            st.subheader("Zidio Baseline Acceptance Check")

            if xgb_wape is not None:
                if xgb_wape < baseline_wape:
                    st.success(f"XGBoost beats Seasonal-Naive " f"baseline: " f"{xgb_wape:.2f}% " f"< " f"{baseline_wape:.2f}% WAPE")
                else:
                    st.warning("XGBoost does not beat the " "Seasonal-Naive baseline.")

            if lgb_wape is not None:
                if lgb_wape < baseline_wape:
                    st.success(f"LightGBM beats Seasonal-Naive " f"baseline: " f"{lgb_wape:.2f}% " f"< " f"{baseline_wape:.2f}% WAPE")
                else:
                    st.warning("LightGBM does not beat the " "Seasonal-Naive baseline.")


st.markdown("---")
st.subheader("Forecast vs Available Inventory")
forecast_total = float(forecast_df["predicted_demand"].sum())

on_hand = float(forecast_df["on_hand_units"].iloc[0])
on_order = float(forecast_df["on_order_units"].iloc[0])
available = (on_hand + on_order)

inv_col1, inv_col2, inv_col3, inv_col4 = st.columns(4)

with inv_col1:
    st.metric("On Hand", f"{on_hand:,.0f}")

with inv_col2:
    st.metric("On Order", f"{on_order:,.0f}")

with inv_col3:
    st.metric("Available", f"{available:,.0f}")

with inv_col4:
    st.metric("6-8 Week Demand", f"{forecast_total:,.0f}")

if forecast_total > available:
    st.error("Forecast demand is above current " "available inventory — review replenishment.")

elif available > forecast_total * 1.5:
    st.warning("Inventory is materially above forecast " "demand — review overstock risk.")

else:
    st.success("Inventory position is broadly aligned " "with forecast demand.")


st.markdown("---")
st.subheader("Download Forecast")
csv = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Weekly Forecast CSV", data=csv, file_name=(f"{sku_id}_weekly_forecast_" f"{forecast_weeks}w.csv"), mime="text/csv")


report = f"""
PROJECT FORESIGHT — Weekly Demand Forecast

SKU: {sku_id}
Forecast Horizon: {forecast_weeks} weeks
Total ML Forecast Demand: {forecast_total:,.2f} units
On Hand: {on_hand:,.2f}
On Order: {on_order:,.2f}
Available Inventory: {available:,.2f}

Seasonal-Naive:
Same period last season

Seasonal Period:
52 weeks

WAPE:
Lower WAPE indicates better performance.

Zidio alignment:
    - Weekly SKU-level forecast
    - 6-8 week defined horizon
    - Seasonal-Naive baseline
    - Same period last season
    - XGBoost vs baseline comparison
    - LightGBM vs baseline comparison
    - WAPE evaluation
    - 80% forecast interval

Generated by Project FORESIGHT
"""


st.download_button("Download Forecast Report", data=report, file_name=(f"{sku_id}_weekly_forecast_report.txt"), mime="text/plain")


st.markdown("---")
st.caption("Project FORESIGHT | Weekly SKU-Level " "Demand Forecasting | Seasonal-Naive Baseline")
