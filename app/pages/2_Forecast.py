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
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df


history = load_weekly_history()
if history.empty:
    st.warning("Weekly model data is not available. Run `python src/train_model.py` first.")
    st.stop()

skus = sorted(history["sku_id"].astype(str).unique())

st.sidebar.header("Forecast Settings")
sku_id = st.sidebar.selectbox("Select SKU", skus)
forecast_weeks = st.sidebar.slider("Forecast Horizon (Weeks)", min_value=6, max_value=8, value=6, step=1,)
st.sidebar.info("Zidio scope: weekly SKU-level forecasting over a defined horizon (e.g. 6–8 weeks).")

if st.button("Generate Weekly Forecast", type="primary"):
    try:
        result = predictor.forecast(sku_id, forecast_weeks)
        forecast_df = pd.DataFrame(result["forecast"])
        st.session_state["forecast_result"] = result
        st.session_state["forecast_df"] = forecast_df
    except Exception as exc:
        st.error(f"Forecast failed: {exc}")

if "forecast_df" not in st.session_state:
    st.info("Select an SKU and click **Generate Weekly Forecast**.")
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

st.markdown("---")
st.subheader("Weekly SKU-Level Forecast")
show_df = forecast_df[[
    "sku_id", "forecast_week", "week_start",
    "predicted_demand", "lower_bound_80", "upper_bound_80"
]].copy()
show_df.columns = [
    "SKU", "Forecast Week", "Week Start",
    "Forecast Units", "80% Lower", "80% Upper"
]
st.dataframe(show_df, use_container_width=True, hide_index=True)

st.subheader("Forecast with 80% Prediction Interval")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=forecast_df["week_start"],
    y=forecast_df["upper_bound_80"],
    mode="lines",
    line=dict(width=0),
    showlegend=False,
    hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=forecast_df["week_start"],
    y=forecast_df["lower_bound_80"],
    mode="lines",
    line=dict(width=0),
    fill="tonexty",
    name="80% interval",
))
fig.add_trace(go.Scatter(
    x=forecast_df["week_start"],
    y=forecast_df["predicted_demand"],
    mode="lines+markers",
    name="Weekly Forecast",
))
fig.update_layout(
    title=f"{sku_id} — {forecast_weeks}-Week Demand Forecast",
    xaxis_title="Week",
    yaxis_title="Units",
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Forecast vs Available Inventory")
forecast_total = float(forecast_df["predicted_demand"].sum())
on_hand = float(forecast_df["on_hand_units"].iloc[0])
on_order = float(forecast_df["on_order_units"].iloc[0])
available = on_hand + on_order

inv_col1, inv_col2, inv_col3, inv_col4 = st.columns(4)
with inv_col1:
    st.metric("On Hand", f"{on_hand:,.0f}")
with inv_col2:
    st.metric("On Order", f"{on_order:,.0f}")
with inv_col3:
    st.metric("Available", f"{available:,.0f}")
with inv_col4:
    st.metric("6–8 Week Demand", f"{forecast_total:,.0f}")

if forecast_total > available:
    st.error("Forecast demand is above current available inventory — review replenishment.")
elif available > forecast_total * 1.5:
    st.warning("Inventory is materially above forecast demand — review overstock risk.")
else:
    st.success("Inventory position is broadly aligned with forecast demand.")

st.markdown("---")
st.subheader("Download Forecast")
csv = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Weekly Forecast CSV", data=csv, file_name=f"{sku_id}_weekly_forecast_{forecast_weeks}w.csv", mime="text/csv",)

report = f"""PROJECT FORESIGHT — Weekly Demand Forecast\n\nSKU: {sku_id}\nForecast Horizon: {forecast_weeks} weeks\nTotal Forecast Demand: {forecast_total:,.2f} units\nOn Hand: {on_hand:,.2f}\nOn Order: {on_order:,.2f}\nAvailable Inventory: {available:,.2f}\n\nZidio alignment:\n- Weekly SKU-level forecast\n- 6–8 week defined horizon\n- 80% forecast interval\n\nGenerated by Project FORESIGHT\n"""

st.download_button(
    "Download Forecast Report",
    data=report, file_name=f"{sku_id}_weekly_forecast_report.txt", mime="text/plain",)

st.markdown("---")
st.caption("Project FORESIGHT | Weekly SKU-Level Demand Forecasting")
