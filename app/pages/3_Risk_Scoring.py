import os
import sys

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Risk Scoring | Project FORESIGHT", page_icon=" ", layout="wide", initial_sidebar_state="expanded")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import config

if not hasattr(config, "HIGH_RISK_THRESHOLD"):
    st.error("HIGH_RISK_THRESHOLD is missing in config.py")
    st.stop()

if not hasattr(config, "MEDIUM_RISK_THRESHOLD"):
    st.error("MEDIUM_RISK_THRESHOLD is missing in config.py")
    st.stop()

HIGH_RISK_THRESHOLD = float(config.HIGH_RISK_THRESHOLD)
MEDIUM_RISK_THRESHOLD = float(config.MEDIUM_RISK_THRESHOLD)


if MEDIUM_RISK_THRESHOLD >= HIGH_RISK_THRESHOLD:
    st.error(
        "Invalid risk thresholds in config.py. "
        "MEDIUM_RISK_THRESHOLD must be lower than "
        "HIGH_RISK_THRESHOLD."
    )
    st.stop()


def load_css():
    css_path = os.path.join(PROJECT_DIR, "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as file:
            st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)
load_css()


st.title("Inventory Risk Scoring")
st.markdown("""Identify **Stockout Risk**, **Overstock Risk** and generate business recommendations using demand forecasts and inventory position.""")
st.markdown("---")


st.sidebar.header("Risk Analysis")
st.sidebar.success("Inventory Intelligence")
st.sidebar.markdown("---")


PROCESSED_DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "processed_data.csv")
WEEKLY_DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "weekly_model_data.csv")


@st.cache_data
def load_data():
    if os.path.exists(PROCESSED_DATA_PATH):
        try:
            df = pd.read_csv(PROCESSED_DATA_PATH)
            return df, "processed_data.csv"
        except Exception:
            pass
    if os.path.exists(WEEKLY_DATA_PATH):
        try:
            df = pd.read_csv(WEEKLY_DATA_PATH)
            return df, "weekly_model_data.csv"
        except Exception:
            pass
    return pd.DataFrame(), None
df, data_source = load_data()


if df.empty:
    st.warning("Processed dataset not found. Using demonstration data.")
    df = pd.DataFrame({
        "sku_id": ["SKU001", "SKU002", "SKU003", "SKU004", "SKU005"],
        "category": ["Electronics", "Electronics", "Furniture", "Furniture", "Grocery"],
        "forecast_units": [180, 75, 120, 210, 95],
        "on_hand_units": [70, 180, 150, 80, 220],
        "on_order_units": [20, 20, 30, 10, 0],
        "selling_price": [ 450, 900, 180, 250, 60],
        "unit_cost": [320, 650, 120, 170, 35 ],
        "lead_time_weeks": [2, 2, 3, 2, 1]
    })
    data_source = "Demo Data"

df.columns = [str(column).strip() for column in df.columns]
required_columns = ["sku_id", "category", "on_hand_units"]
missing_columns = [column for column in required_columns if column not in df.columns]


if missing_columns:
    st.error(f"""Required columns are missing: {missing_columns} Available columns: {df.columns.tolist()}""")
    st.stop()

if "on_order_units" not in df.columns:
    df["on_order_units"] = 0

if "selling_price" not in df.columns:
    df["selling_price"] = 0

if "unit_cost" not in df.columns:
    df["unit_cost"] = (pd.to_numeric(df["selling_price"], errors="coerce").fillna(0) * 0.70)

if "lead_time_weeks" not in df.columns:
    df["lead_time_weeks"] = 2

forecast_candidates = ["forecast_units", "predicted_units", "forecast", "demand_forecast", "units_sold"]
forecast_column = None

for column in forecast_candidates:
    if column in df.columns:
        forecast_column = column
        break

if forecast_column is None:
    df["forecast_units"] = 0
    forecast_column = "forecast_units"


numeric_columns = ["on_hand_units", "on_order_units", "selling_price", "unit_cost", "lead_time_weeks", forecast_column]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
df[forecast_column] = df[forecast_column].clip(lower=0)


st.sidebar.header("Forecast Risk Settings")
lead_time_weeks = st.sidebar.slider("Lead Time (Weeks)", min_value=1, max_value=8, value=2)
forward_window_weeks = st.sidebar.slider("Forward Demand Window (Weeks)", min_value=2, max_value=8, value=4)
safety_stock = st.sidebar.number_input("Safety Stock (Units)", min_value=0, value=0, step=10)
st.sidebar.markdown("---")

st.sidebar.header("Risk Filter")
risk_options = ["ALL", "HIGH", "MEDIUM", "LOW"]
selected_risk = st.sidebar.selectbox("Select Risk Level", risk_options, index=0)

st.sidebar.caption(f"High Risk ≥ {HIGH_RISK_THRESHOLD:.0f}%")
st.sidebar.caption(f"Medium Risk ≥ {MEDIUM_RISK_THRESHOLD:.0f}% " f"and < {HIGH_RISK_THRESHOLD:.0f}%")
st.sidebar.caption(f"Low Risk < {MEDIUM_RISK_THRESHOLD:.0f}%")
st.sidebar.markdown("---")

categories = sorted(df["category"].dropna().astype(str).unique().tolist())
selected_category = st.sidebar.selectbox("Category", ["All"] + categories)


filtered_df = df.copy()
if selected_category != "All":
    filtered_df = filtered_df[filtered_df["category"].astype(str) == selected_category].copy()

filtered_df["Inventory Position"] = (filtered_df["on_hand_units"] + filtered_df["on_order_units"])
filtered_df["Forecast Demand"] = (filtered_df[forecast_column])
filtered_df["Stockout Gap"] = (filtered_df["Forecast Demand"] - filtered_df["Inventory Position"]).clip(lower=0)
filtered_df["Overstock Gap"] = (filtered_df["on_hand_units"] - filtered_df["Forecast Demand"]).clip(lower=0)

filtered_df["Stockout Risk"] = (filtered_df["Stockout Gap"] / filtered_df["Forecast Demand"].clip(lower=1)) * 100
filtered_df["Overstock Risk"] = (filtered_df["Overstock Gap"] / filtered_df["Forecast Demand"].clip(lower=1)) * 100


if "risk_score" in filtered_df.columns:
    existing_risk_score = pd.to_numeric(filtered_df["risk_score"], errors="coerce")
    calculated_risk_score = filtered_df[["Stockout Risk", "Overstock Risk"]].max(axis=1)
    filtered_df["risk_score"] = (existing_risk_score.fillna(calculated_risk_score))
else:
    filtered_df["risk_score"] = filtered_df[["Stockout Risk", "Overstock Risk"]].max(axis=1)

filtered_df["risk_score"] = (filtered_df["risk_score"].clip(lower=0, upper=100))


def classify_risk(score):
    if score >= HIGH_RISK_THRESHOLD:
        return "HIGH"
    elif score >= MEDIUM_RISK_THRESHOLD:
        return "MEDIUM"
    else:
        return "LOW"

filtered_df["Risk Level"] = (filtered_df["risk_score"].apply(classify_risk))


if selected_risk == "HIGH":
    filtered_df = filtered_df[filtered_df["risk_score"] >= HIGH_RISK_THRESHOLD].copy()
elif selected_risk == "MEDIUM":
    filtered_df = filtered_df[(filtered_df["risk_score"] >= MEDIUM_RISK_THRESHOLD) & (filtered_df["risk_score"] < HIGH_RISK_THRESHOLD)].copy()
elif selected_risk == "LOW":
    filtered_df = filtered_df[filtered_df["risk_score"] < MEDIUM_RISK_THRESHOLD].copy()
else:
    filtered_df = filtered_df.copy()

if filtered_df.empty:
    st.warning(f"No records found for Risk Level: {selected_risk}")
    st.info(
        f"""
        Current configuration:
        HIGH ≥ {HIGH_RISK_THRESHOLD:.0f}%
        MEDIUM ≥ {MEDIUM_RISK_THRESHOLD:.0f}% and < {HIGH_RISK_THRESHOLD:.0f}%
        LOW < {MEDIUM_RISK_THRESHOLD:.0f}%
        """
    )
    st.stop()


sku_list = sorted(filtered_df["sku_id"].astype(str).unique().tolist())
selected_sku = st.sidebar.selectbox("Select SKU", sku_list)
sku_rows = filtered_df[filtered_df["sku_id"].astype(str) == selected_sku]

if sku_rows.empty:
    st.warning("No SKU available for selected filters.")
    st.stop()
sku_data = sku_rows.iloc[0]


st.subheader("Selected Product")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("SKU", str(sku_data["sku_id"]))

with col2:
    st.metric("Category", str(sku_data["category"]))

with col3:
    st.metric("On-Hand Units", f"{sku_data['on_hand_units']:,.0f}")

with col4:
    st.metric("On-Order Units", f"{sku_data['on_order_units']:,.0f}")
st.markdown("---")


inventory = float(sku_data["on_hand_units"])
on_order = float(sku_data["on_order_units"])
selling_price = float(sku_data["selling_price"])
unit_cost = float(sku_data["unit_cost"])
total_forecast = float(sku_data[forecast_column])


weekly_columns = []
possible_week_columns = {
    1: ["week_1_forecast", "forecast_week_1", "forecast_w1", "w1_forecast"],
    2: ["week_2_forecast", "forecast_week_2", "forecast_w2", "w2_forecast"],
    3: ["week_3_forecast", "forecast_week_3", "forecast_w3", "w3_forecast"],
    4: ["week_4_forecast", "forecast_week_4", "forecast_w4", "w4_forecast"],
    5: ["week_5_forecast", "forecast_week_5", "forecast_w5", "w5_forecast"],
    6: ["week_6_forecast", "forecast_week_6", "forecast_w6", "w6_forecast"],
    7: ["week_7_forecast", "forecast_week_7", "forecast_w7", "w7_forecast"],
    8: ["week_8_forecast", "forecast_week_8", "forecast_w8", "w8_forecast"]
}

for week_number in range(1, 9):
    found_column = None
    for column in possible_week_columns[week_number]:
        if column in filtered_df.columns:
            found_column = column
            break
    if found_column:
        weekly_columns.append(found_column)

if len(weekly_columns) >= 2:
    weekly_forecast = []
    for column in weekly_columns[:8]:
        value = pd.to_numeric(sku_data[column], errors="coerce")
        if pd.isna(value):
            value = 0
        weekly_forecast.append(float(value))
    weekly_forecast = np.array(weekly_forecast, dtype=float)
else:
    weekly_value = (total_forecast / 8 if total_forecast > 0 else 0)
    weekly_forecast = np.repeat(weekly_value, 8)


if len(weekly_forecast) < 8:
    weekly_forecast = np.pad(weekly_forecast, (0, 8 - len(weekly_forecast)), mode="edge")
weekly_forecast = (weekly_forecast[:8])

forecast_table = pd.DataFrame({"Week": ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6", "Week 7", "Week 8"], "Forecast Demand": weekly_forecast})
forecast_table["Forecast Demand"] = forecast_table["Forecast Demand"].round(2)

lead_time_periods = min(int(lead_time_weeks), len(weekly_forecast))
lead_time_demand = float(np.sum(weekly_forecast[:lead_time_periods]))

inventory_position = (inventory + on_order)
required_inventory = (lead_time_demand + float(safety_stock))

stockout_gap = max(required_inventory - inventory_position, 0)
stockout_ratio = (stockout_gap / max(lead_time_demand, 1))
stockout_score = min(stockout_ratio * 100, 100)

forward_periods = min(int(forward_window_weeks), len(weekly_forecast))
forward_window_demand = float(np.sum(weekly_forecast[:forward_periods]))

overstock_gap = max(inventory - forward_window_demand, 0)
overstock_ratio = (overstock_gap / max(forward_window_demand, 1))
overstock_score = min(overstock_ratio * 100, 100)

risk_score = max(stockout_score, overstock_score)
risk_score = min(risk_score, 100)
risk_level = classify_risk(risk_score)


if (stockout_score >= HIGH_RISK_THRESHOLD and overstock_score < HIGH_RISK_THRESHOLD):
    quadrant = "REORDER NOW"
    recommendation = ("Raise a replenishment order " "before stock runs out.")
elif (overstock_score >= HIGH_RISK_THRESHOLD and stockout_score < HIGH_RISK_THRESHOLD):
    quadrant = "MARKDOWN / CLEAR"
    recommendation = ("Promote or discount inventory " "to free up capital.")
elif (stockout_score >= HIGH_RISK_THRESHOLD and overstock_score >= HIGH_RISK_THRESHOLD):
    quadrant = "WATCH / VOLATILE"
    recommendation = ("Investigate demand volatility " "and review manually.")
elif risk_level == "MEDIUM":
    quadrant = "MONITOR"
    recommendation = ("Monitor inventory and demand closely. " "Review replenishment and forecast regularly.")
else:
    quadrant = "HEALTHY"
    recommendation = ("No immediate action needed. " "Continue normal monitoring.")


stockout_value_at_stake = (stockout_gap * selling_price)
overstock_value_at_stake = (overstock_gap * unit_cost)
value_at_stake = max(stockout_value_at_stake, overstock_value_at_stake)

st.header("Business KPIs")
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Stockout Risk", f"{stockout_score:.1f}%")

with c2:
    st.metric("Overstock Risk", f"{overstock_score:.1f}%")

with c3:
    st.metric("Lead-Time Demand", f"{lead_time_demand:,.0f}")

with c4:
    st.metric("Inventory Position", f"{inventory_position:,.0f}")

with c5:
    st.metric("Risk Score", f"{risk_score:.1f}%")
st.markdown("---")

st.header("Inventory Decision")

d1, d2, d3 = st.columns(3)

with d1:
    st.metric("Risk Level", risk_level)

with d2:
    st.metric("Decision", quadrant)

with d3:
    st.metric("Value at Stake", f"₹{value_at_stake:,.0f}")
st.markdown("---")


st.subheader("6-8 Week Weekly Demand Forecast")
fig_forecast = px.bar(forecast_table, x="Week", y="Forecast Demand", text_auto=".0f", title="Weekly SKU Demand Forecast")
fig_forecast.update_layout(yaxis_title="Demand Units", xaxis_title="Forecast Week", height=450)

st.plotly_chart(fig_forecast, use_container_width=True)
st.dataframe(forecast_table, use_container_width=True, hide_index=True)
st.markdown("---")

st.subheader("Risk Indicator")
gauge_df = pd.DataFrame({"Risk Type": ["Stockout", "Overstock"], "Risk Score": [stockout_score, overstock_score]})
gauge = px.bar(gauge_df, x="Risk Type", y="Risk Score", text_auto=".1f", title="Stockout vs Overstock Risk")

gauge.update_layout(yaxis_title="Risk Score (%)", xaxis_title="Risk Type", yaxis_range=[0, 100], height=400)
st.plotly_chart(gauge, use_container_width=True)
st.markdown("---")

st.subheader("Risk Distribution")
risk_distribution = (filtered_df["Risk Level"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0).reset_index())
risk_distribution.columns = ["Risk Level", "Count"]
risk_fig = px.pie(risk_distribution, names="Risk Level", values="Count", title="Inventory Risk Distribution")
st.plotly_chart(risk_fig, use_container_width=True)
st.markdown("---")


st.subheader("Risk Summary")
risk_summary = filtered_df[["sku_id", "category", "risk_score", "Risk Level", "Stockout Risk", "Overstock Risk", "Inventory Position"]].copy()
risk_summary.columns = ["SKU", "Category", "Risk Score", "Risk Level", "Stockout Risk %", "Overstock Risk %", "Inventory Position"]
risk_summary = risk_summary.sort_values("Risk Score", ascending=False)
st.dataframe(risk_summary, use_container_width=True, hide_index=True)
st.markdown("---")


st.subheader("Forecast vs Inventory")
compare = pd.DataFrame({"Metric": ["Lead-Time Demand", "On-Hand", "On-Order", "Inventory Position"], "Units": [lead_time_demand, inventory, on_order, inventory_position]})
compare_fig = px.bar(compare, x="Metric", y="Units", text_auto=".0f", title="Forecast Demand vs Inventory Position")
compare_fig.update_layout( height=450)
st.plotly_chart(compare_fig, use_container_width=True)
st.markdown("---")


st.subheader("Risk Gap Analysis")
gap_df = pd.DataFrame({"Risk Metric": ["Stockout Gap", "Overstock Gap"], "Units": [stockout_gap, overstock_gap]})
gap_fig = px.bar(gap_df, x="Risk Metric", y="Units", text_auto=".0f", title="Inventory Risk Gap")
gap_fig.update_layout(height=400)
st.plotly_chart(gap_fig, use_container_width=True)
st.markdown("---")


st.subheader("Selected SKU Details")
details = pd.DataFrame({
    "SKU": [sku_data["sku_id"]],
    "Category": [sku_data["category"]],
    "Risk Score": [risk_score],
    "Risk Level": [risk_level],
    "Lead-Time Demand": [lead_time_demand],
    "On-Hand": [inventory],
    "On-Order": [on_order],
    "Inventory Position": [inventory_position],
    "Forward Window Demand": [forward_window_demand],
    "Stockout Gap": [stockout_gap],
    "Overstock Gap": [overstock_gap],
    "Stockout Risk %": [stockout_score],
    "Overstock Risk %": [overstock_score],
    "Decision": [quadrant],
    "Value at Stake": [value_at_stake]
})
st.dataframe(details, use_container_width=True, hide_index=True)
st.markdown("---")


st.header("Business Recommendation")
if quadrant == "REORDER NOW":
    st.error(
        f"""
        ### REORDER NOW
        **Stockout Risk:** {stockout_score:.1f}%
        **Lead-Time Demand:** {lead_time_demand:,.0f} units
        **Inventory Position:** {inventory_position:,.0f} units
        **Shortage:** {stockout_gap:,.0f} units
        **Value at Stake:** ₹{value_at_stake:,.0f}
        **Recommended Action:**
        Raise a replenishment order before stock runs out.
            - Increase reorder quantity
            - Review supplier lead time
            - Maintain safety stock
            - Monitor demand closely
        """
    )
elif quadrant == "MARKDOWN / CLEAR":
    st.warning(
        f"""
        ### MARKDOWN / CLEAR
        **Overstock Risk:** {overstock_score:.1f}%
        **Excess Units:** {overstock_gap:,.0f}
        **Capital at Risk:** ₹{value_at_stake:,.0f}
        **Recommended Action:**
        Promote or discount inventory.
            - Launch discount campaign
            - Bundle slow-moving products
            - Reduce future procurement
            - Improve inventory turnover
        """
    )
elif quadrant == "WATCH / VOLATILE":
    st.warning(
        f"""
        ### WATCH / VOLATILE
        **Stockout Risk:** {stockout_score:.1f}%
        **Overstock Risk:** {overstock_score:.1f}%
        **Recommended Action:**
        Investigate demand volatility and review manually.
            - Monitor demand trend
            - Review forecast accuracy
            - Check inventory movement
            - Reassess procurement decision
        """
    )
elif risk_level == "MEDIUM":
    st.warning(
        f"""
        ### MEDIUM RISK — MONITOR
        **Risk Score:** {risk_score:.1f}%
        **Stockout Risk:** {stockout_score:.1f}%
        **Overstock Risk:** {overstock_score:.1f}%
        **Recommended Action:**
        Monitor the SKU closely.
            - Review demand forecast
            - Monitor inventory movement
            - Check upcoming replenishment
            - Reassess risk weekly
        """
    )
else:
    st.success(
        f"""
        ### HEALTHY INVENTORY
        **Risk Score:** {risk_score:.1f}%
        **Stockout Risk:** {stockout_score:.1f}%
        **Overstock Risk:** {overstock_score:.1f}%
        **Recommended Action:**
        No immediate action required.
            - Continue normal monitoring
            - Maintain reorder schedule
            - Review inventory weekly
        """
    )
st.markdown("---")


st.header("Priority Action")
if quadrant == "REORDER NOW":
    priority_units = stockout_gap
    priority_action = ("Raise replenishment order")
elif quadrant == "MARKDOWN / CLEAR":
    priority_units = overstock_gap
    priority_action = ("Markdown / Clear inventory")
elif quadrant == "WATCH / VOLATILE":
    priority_units = 0
    priority_action = ("Manual review required")
elif risk_level == "MEDIUM":
    priority_units = 0
    priority_action = ("Monitor inventory closely")
else:
    priority_units = 0
    priority_action = ("No immediate action")

priority_df = pd.DataFrame({
    "SKU": [sku_data["sku_id"]],
    "Category": [sku_data["category"]],
    "Risk Score": [risk_score],
    "Risk Level": [risk_level],
     "Decision": [quadrant],
    "Priority Units": [priority_units],
     "Value at Stake": [value_at_stake],
    "Recommended Action": [priority_action]
})
st.dataframe(priority_df, use_container_width=True, hide_index=True)
st.markdown("---")


report = f"""
PROJECT FORESIGHT
Inventory Risk Report

========================================

SKU ID {sku_data['sku_id']}
Category {sku_data['category']}
Risk Level {risk_level}
Risk Score {risk_score:.2f}%
Decision Quadrant {quadrant}

========================================

FORECAST HORIZON
8 Weeks

Lead Time {lead_time_weeks} Weeks
Forward Window {forward_window_weeks} Weeks
Safety Stock {safety_stock} Units

========================================

LEAD-TIME DEMAND {lead_time_demand:.2f} Units
ON-HAND INVENTORY {inventory:.2f} Units
ON-ORDER INVENTORY {on_order:.2f} Units
INVENTORY POSITION {inventory_position:.2f} Units

========================================

FORWARD WINDOW DEMAND {forward_window_demand:.2f} Units
STOCKOUT GAP {stockout_gap:.2f} Units
OVERSTOCK GAP {overstock_gap:.2f} Units

========================================

STOCKOUT RISK {stockout_score:.2f}%
OVERSTOCK RISK {overstock_score:.2f}%
RISK SCORE {risk_score:.2f}%
RISK LEVEL {risk_level}
DECISION {quadrant}

========================================

STOCKOUT VALUE AT STAKE {stockout_value_at_stake:,.2f}
OVERSTOCK VALUE AT STAKE {overstock_value_at_stake:,.2f}
TOTAL VALUE AT STAKE {value_at_stake:,.2f}

========================================

RECOMMENDED ACTION {recommendation}

========================================

WEEKLY FORECAST
"""


for i, value in enumerate(weekly_forecast, start=1):
    report += (f"\nWeek {i}: " f"{value:.2f} Units")
st.download_button(label="Download Risk Report", data=report, file_name=f"Risk_Report_{selected_sku}.txt", mime="text/plain")
st.markdown("---")


st.header("Download Risk Data")
risk_export = pd.DataFrame({
    "SKU": [sku_data["sku_id"]],
    "Category": [sku_data["category"]],
    "Risk_Score": [risk_score],
    "Risk_Level": [risk_level],
    "Lead_Time_Weeks": [lead_time_weeks],
    "Forward_Window_Weeks": [forward_window_weeks],
    "Lead_Time_Demand": [lead_time_demand],
    "On_Hand_Units": [inventory],
    "On_Order_Units": [on_order],
    "Inventory_Position": [inventory_position],
    "Forward_Window_Demand": [forward_window_demand],
    "Stockout_Gap": [stockout_gap],
    "Overstock_Gap": [overstock_gap],
    "Stockout_Risk_Percent": [stockout_score],
     "Overstock_Risk_Percent": [overstock_score],
    "Decision": [quadrant],
    "Stockout_Value_at_Stake": [stockout_value_at_stake],
    "Overstock_Value_at_Stake": [overstock_value_at_stake],
    "Total_Value_at_Stake": [value_at_stake],
    "Recommended_Action": [recommendation]
})


csv_data = risk_export.to_csv(index=False).encode("utf-8")
st.download_button(label="Download Risk Analysis CSV", data=csv_data, file_name="risk_analysis.csv", mime="text/csv")
st.markdown("---")


st.header("Risk History")
history = pd.DataFrame({
    "SKU": [sku_data["sku_id"]],
    "Forecast Horizon": ["8 Weeks"],
    "Lead-Time Demand": [lead_time_demand],
    "Inventory Position": [inventory_position],
    "Risk Score": [risk_score],
    "Stockout Risk": [stockout_score],
    "Overstock Risk": [overstock_score],
    "Risk Level": [risk_level],
    "Decision": [quadrant],
    "Value at Stake": [value_at_stake]
})
st.dataframe(history, use_container_width=True, hide_index=True)
st.markdown("---")


st.header("Summary")
st.info(
f"""
    **SKU:** {sku_data['sku_id']}
    **Category:** {sku_data['category']}
    **Forecast Horizon:** 8 Weeks
    **Lead-Time Demand:** {lead_time_demand:,.0f} Units
    **On-Hand:** {inventory:,.0f} Units
    **On-Order:** {on_order:,.0f} Units
    **Inventory Position:** {inventory_position:,.0f} Units
    **Forward Window Demand:** {forward_window_demand:,.0f} Units
    **Stockout Risk:** {stockout_score:.1f}%
    **Overstock Risk:** {overstock_score:.1f}%
    **Risk Score:** {risk_score:.1f}%
    **Risk Level:** {risk_level}
    **Decision:** {quadrant}
    **Value at Stake:** ₹{value_at_stake:,.0f}
""")
st.markdown("---")


st.caption(f"""
    Project FORESIGHT |
    AI-Powered Demand Forecasting & Inventory Risk Management |
    High Risk ≥ {HIGH_RISK_THRESHOLD:.0f}% |
    Medium Risk ≥ {MEDIUM_RISK_THRESHOLD:.0f}% |
    Low Risk < {MEDIUM_RISK_THRESHOLD:.0f}%
"""
)