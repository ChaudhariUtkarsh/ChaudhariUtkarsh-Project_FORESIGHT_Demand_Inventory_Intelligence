import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os


st.set_page_config(page_title="Risk Scoring | Project FORESIGHT", page_icon=" ", layout="wide", initial_sidebar_state="expanded")


def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css()


st.title("Inventory Risk Scoring")
st.markdown(
    """
    Identify **Stockout Risk**, **Overstock Risk** and
    generate business recommendations using weekly
    demand forecasts and inventory position.
    """
)
st.markdown("---")

st.sidebar.header("Risk Analysis")
st.sidebar.markdown("---")
st.sidebar.success("Inventory Intelligence")
st.sidebar.markdown("---")


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
PROCESSED_DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "processed_data.csv")
WEEKLY_DATA_PATH = os.path.join(PROJECT_DIR, "data", "processed", "weekly_model_data.csv")

df = pd.DataFrame()
data_source = None


if os.path.exists(PROCESSED_DATA_PATH):
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        data_source = "processed_data.csv"
    except Exception as e:
        st.error(f"Unable to read processed_data.csv: {e}")
elif os.path.exists(WEEKLY_DATA_PATH):
    try:
        df = pd.read_csv(WEEKLY_DATA_PATH)
        data_source = "weekly_model_data.csv"
    except Exception as e:
        st.error(f"Unable to read weekly_model_data.csv: {e}")


if df.empty:
    st.warning("Processed dataset not found. " "Using demonstration data.")
    df = pd.DataFrame({
        "sku_id": ["SKU001", "SKU002", "SKU003", "SKU004", "SKU005"],
        "category": ["Electronics", "Electronics", "Furniture", "Furniture", "Grocery"],
        "forecast_units": [180, 75, 120, 210, 95],
        "on_hand_units": [70, 180, 150, 80, 220],
        "on_order_units": [20, 20, 30, 10, 0],
        "selling_price": [450, 900, 180, 250, 60],
        "unit_cost": [320, 650, 120, 170, 35],
        "lead_time_weeks": [2, 2, 3, 2, 1]
    })
    data_source = "Demo Data"


df.columns = [str(col).strip() for col in df.columns]

required_columns = ["sku_id", "category", "on_hand_units"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"""Required columns are missing: {missing_columns} Available columns: {df.columns.tolist()} """)
    st.stop()


if "on_order_units" not in df.columns:
    df["on_order_units"] = 0

if "unit_cost" not in df.columns:
    if "selling_price" in df.columns:
        df["unit_cost"] = (pd.to_numeric(df["selling_price"], errors="coerce").fillna(0) * 0.70)
    else:
        df["unit_cost"] = 0

if "selling_price" not in df.columns:
    df["selling_price"] = (df["unit_cost"] * 1.30)

if "lead_time_weeks" not in df.columns:
    df["lead_time_weeks"] = 2

numeric_columns = ["on_hand_units", "on_order_units", "unit_cost", "selling_price", "lead_time_weeks"]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)


forecast_candidates = ["forecast_units", "predicted_units", "forecast", "demand_forecast", "units_sold"]
forecast_column = None

for col in forecast_candidates:
    if col in df.columns:
        forecast_column = col
        break

if forecast_column is None:
    if "units_sold" in df.columns:
        df["forecast_units"] = (pd.to_numeric(df["units_sold"], errors="coerce").fillna(0))
    else:
        df["forecast_units"] = 0
    forecast_column = "forecast_units"

df[forecast_column] = pd.to_numeric(df[forecast_column], errors="coerce").fillna(0)


st.sidebar.header("Forecast Risk Settings")
lead_time_weeks = st.sidebar.slider("Lead Time (Weeks)", min_value=1, max_value=8, value=2)
forward_window_weeks = st.sidebar.slider("Forward Demand Window (Weeks)", min_value=2, max_value=8, value=4)
safety_stock = st.sidebar.number_input("Safety Stock (Units)", min_value=0, value=0, step=10)
st.sidebar.markdown("---")

st.sidebar.header("Risk Threshold")
stockout_threshold = st.sidebar.slider("Stockout Risk Threshold (%)", min_value=10, max_value=100, value=50)
overstock_threshold = st.sidebar.slider("Overstock Risk Threshold (%)", min_value=10, max_value=100, value=50)
st.sidebar.markdown("---")


categories = sorted(df["category"].dropna().astype(str).unique().tolist())
category = st.sidebar.selectbox("Category", ["All"] + categories)
filtered_df = df.copy()


if category != "All":
    filtered_df = filtered_df[filtered_df["category"].astype(str) == category]

sku_list = sorted(filtered_df["sku_id"].astype(str).unique().tolist())

if not sku_list:
    st.warning("No SKU available for selected category.")
    st.stop()

selected_sku = st.sidebar.selectbox("Select SKU", sku_list)
sku_data = filtered_df[filtered_df["sku_id"].astype(str) == selected_sku].iloc[0]


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


for i in range(1, 9):
    possible_columns = [f"week_{i}_forecast", f"week{i}_forecast", f"forecast_week_{i}", f"week_{i}", f"forecast_w{i}", f"w{i}_forecast"]
    found = None
    for col in possible_columns:
        if col in df.columns:
            found = col
            break
    if found:
        weekly_columns.append(found)


if len(weekly_columns) >= 2:
    weekly_forecast = []
    for col in weekly_columns[:8]:
        value = pd.to_numeric(sku_data[col], errors="coerce")
        if pd.isna(value):
            value = 0
        weekly_forecast.append(float(value))
    weekly_forecast = np.array(weekly_forecast, dtype=float)
else:
    weekly_value = (total_forecast / 8 if total_forecast > 0 else 0)
    weekly_forecast = np.repeat(weekly_value, 8)


if len(weekly_forecast) < 8:
    weekly_forecast = np.pad( weekly_forecast, (0, 8 - len(weekly_forecast)), mode="edge")
weekly_forecast = (weekly_forecast[:8])

forecast_table = pd.DataFrame({
    "Week": ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6", "Week 7", "Week 8" ],
    "Forecast Demand": (weekly_forecast)
})
forecast_table["Forecast Demand"] = forecast_table["Forecast Demand"].round(2)

lead_time_periods = min(int(lead_time_weeks), len(weekly_forecast))
lead_time_demand = float(np.sum(weekly_forecast[:lead_time_periods]))

inventory_position = (inventory + on_order)
required_inventory = (lead_time_demand + float(safety_stock))

stockout_gap = max(required_inventory - inventory_position, 0)
stockout_ratio = (stockout_gap / max(lead_time_demand, 1))
stockout_score = min(stockout_ratio * 100, 100)

forward_periods = min(int(forward_window_weeks),len(weekly_forecast))
forward_window_demand = float(np.sum(weekly_forecast[:forward_periods]))

overstock_gap = max(inventory - forward_window_demand, 0)
overstock_ratio = (overstock_gap / max(forward_window_demand, 1))
overstock_score = min(overstock_ratio * 100, 100)


if (stockout_score >= stockout_threshold and overstock_score < overstock_threshold):
    risk_level = "High Stockout"
    quadrant = "REORDER NOW"
    risk_score = stockout_score
    recommendation = ("Raise a replenishment order " "before stock runs out.")
elif (stockout_score < stockout_threshold and overstock_score >= overstock_threshold):
    risk_level = "Overstock"
    quadrant = "MARKDOWN / CLEAR"
    risk_score = overstock_score
    recommendation = ("Promote or discount inventory " "to free up capital.")
elif (stockout_score >= stockout_threshold and overstock_score >= overstock_threshold):
    risk_level = "Volatile"
    quadrant = "WATCH / VOLATILE"
    risk_score = max(stockout_score, overstock_score)
    recommendation = ("Investigate demand volatility " "and review manually.")
else:
    risk_level = "Healthy"
    quadrant = "HEALTHY"
    risk_score = 25
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
    st.metric("Value at Stake", f"{value_at_stake:,.0f}")
st.markdown("---")


st.header("Inventory Decision")
d1, d2, d3 = st.columns(3)

with d1:
    st.metric("Risk Level", risk_level)

with d2:
    st.metric("Decision", quadrant)

with d3:
    st.metric("Risk Score", f"{risk_score:.1f}%")
st.markdown("---")

st.subheader("6-8 Week Weekly Demand Forecast")
fig_forecast = px.bar(forecast_table, x="Week", y="Forecast Demand", text_auto=".0f", title=("Weekly SKU Demand Forecast"))
fig_forecast.update_layout(yaxis_title="Demand Units", xaxis_title="Forecast Week", height=450)
st.plotly_chart(fig_forecast, use_container_width=True)
st.dataframe(forecast_table, use_container_width=True, hide_index=True)
st.markdown("---")


st.subheader("Risk Indicator")
gauge_df = pd.DataFrame({"Risk Type": ["Stockout", "Overstock"], "Risk Score": [stockout_score, overstock_score]})

gauge = px.bar(gauge_df, x="Risk Type", y="Risk Score", text_auto=".1f", title=("Stockout vs Overstock Risk"))
gauge.update_layout(yaxis_title="Risk Score (%)", xaxis_title="Risk Type", yaxis_range=[0, 100], height=400)
st.plotly_chart(gauge, use_container_width=True)
st.markdown("---")

st.subheader("Risk Distribution")
risk_summary = filtered_df.copy()
risk_summary["Inventory Position"] = (risk_summary["on_hand_units"] + risk_summary["on_order_units"])
risk_summary["Lead Time Demand"] = (risk_summary[forecast_column])
risk_summary["Forward Demand"] = (risk_summary[forecast_column])
risk_summary["Stockout Gap"] = (risk_summary["Lead Time Demand"] - risk_summary["Inventory Position"]).clip(lower=0)
risk_summary["Overstock Gap"] = (risk_summary["on_hand_units"] - risk_summary["Forward Demand"]).clip(lower=0)
risk_summary["Stockout Risk"] = (risk_summary["Stockout Gap"] / risk_summary["Lead Time Demand"].clip(lower=1)) * 100
risk_summary["Overstock Risk"] = (risk_summary["Overstock Gap"] / risk_summary["Forward Demand"].clip(lower=1)) * 100


def classify_risk(row):
    stockout = row["Stockout Risk"]
    overstock = row["Overstock Risk"]
    if (stockout >= stockout_threshold and overstock < overstock_threshold):
        return "REORDER NOW"
    elif (stockout < stockout_threshold and overstock >= overstock_threshold):
        return "MARKDOWN / CLEAR"
    elif (stockout >= stockout_threshold and overstock >= overstock_threshold):
        return "WATCH / VOLATILE"
    return "HEALTHY"

risk_summary["Risk"] = (risk_summary.apply(classify_risk, axis=1))
risk_count = (risk_summary["Risk"].value_counts().reset_index())
risk_count.columns = ["Risk", "Count"]
risk_fig = px.pie(risk_count, names="Risk", values="Count", title=("Inventory Risk Distribution"))
st.plotly_chart(risk_fig, use_container_width=True)
st.markdown("---")


st.subheader("Forecast vs Inventory")
compare = pd.DataFrame({
    "Metric": ["Lead-Time Demand", "On-Hand", "On-Order", "Inventory Position"],
    "Units": [lead_time_demand, inventory, on_order, inventory_position]
})

compare_fig = px.bar(compare, x="Metric", y="Units", text_auto=".0f", title=("Forecast Demand vs Inventory Position"))
compare_fig.update_layout( height=450)
st.plotly_chart(compare_fig, use_container_width=True)
st.markdown("---")


st.subheader("Risk Gap Analysis")
gap_df = pd.DataFrame({"Risk Metric": ["Stockout Gap", "Overstock Gap"], "Units": [stockout_gap, overstock_gap]})
gap_fig = px.bar(gap_df, x="Risk Metric", y="Units", text_auto=".0f", title=("Inventory Risk Gap"))
gap_fig.update_layout(height=400)
st.plotly_chart(gap_fig, use_container_width=True)
st.markdown("---")


st.subheader("Selected SKU Details")
details = pd.DataFrame({
    "SKU": [sku_data["sku_id"]],
    "Category": [sku_data["category"]],
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
        ###REORDER NOW
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
        ###MARKDOWN / CLEAR
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
        ###WATCH / VOLATILE
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

else:
    st.success(
        f"""
        ###HEALTHY INVENTORY
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
else:
    priority_units = 0
    priority_action = ("No immediate action")

priority_df = pd.DataFrame({
    "SKU": [sku_data["sku_id"]],
    "Category": [sku_data["category"]],
    "Decision": [quadrant],
    "Priority Units": [priority_units],
    "Value at Stake": [value_at_stake],
    "Recommended Action": [priority_action]
})
st.dataframe(priority_df, use_container_width=True, hide_index=True)
st.markdown("---")


st.header("Risk Report")
report = f"""
----------------------------------------
PROJECT FORESIGHT
INVENTORY RISK REPORT
----------------------------------------

SKU ID {sku_data['sku_id']}
Category {sku_data['category']}

----------------------------------------

FORECAST HORIZON
8 Weeks
Lead Time {lead_time_weeks} Weeks
Forward Window {forward_window_weeks} Weeks
Safety Stock {safety_stock} Units

----------------------------------------

LEAD-TIME DEMAND {lead_time_demand:.2f} Units
ON-HAND INVENTORY {inventory:.2f} Units
ON-ORDER INVENTORY {on_order:.2f} Units
INVENTORY POSITION {inventory_position:.2f} Units

----------------------------------------

FORWARD WINDOW DEMAND {forward_window_demand:.2f} Units
STOCKOUT GAP {stockout_gap:.2f} Units
OVERSTOCK GAP {overstock_gap:.2f} Units

----------------------------------------

STOCKOUT RISK {stockout_score:.2f}%
OVERSTOCK RISK {overstock_score:.2f}%
RISK LEVEL {risk_level}
DECISION QUADRANT {quadrant}

----------------------------------------

STOCKOUT VALUE AT STAKE {stockout_value_at_stake:,.2f}
OVERSTOCK VALUE AT STAKE {overstock_value_at_stake:,.2f}
TOTAL VALUE AT STAKE {value_at_stake:,.2f}

----------------------------------------

RECOMMENDED ACTION
{recommendation}

----------------------------------------

WEEKLY FORECAST
"""

for i, value in enumerate(weekly_forecast, start=1):
    report += (f"\nWeek {i}: " f"{value:.2f} Units")

report += """
----------------------------------------
Generated by Project FORESIGHT
AI-Powered Demand & Inventory Intelligence
----------------------------------------
"""

st.download_button(label="Download Risk Report", data=report, file_name=(f"Risk_Report_{selected_sku}.txt"), mime="text/plain")
st.markdown("---")


st.header("Download Risk Data")
risk_export = pd.DataFrame({
    "SKU": [sku_data["sku_id"]],
    "Category": [sku_data["category"]],
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
    "Risk_Level": [risk_level],
    "Decision": [quadrant],
    "Stockout_Value_at_Stake": [stockout_value_at_stake],
    "Overstock_Value_at_Stake": [overstock_value_at_stake],
    "Total_Value_at_Stake": [value_at_stake],
    "Recommended_Action": [recommendation]
})
csv_data = (risk_export.to_csv(index=False).encode("utf-8"))
st.download_button(label="Download Risk Analysis CSV", data=csv_data, file_name="risk_analysis.csv", mime="text/csv")
st.markdown("---")


st.header("Risk History")
history = pd.DataFrame({
    "SKU": [sku_data["sku_id"]],
    "Forecast Horizon": ["8 Weeks"],
    "Lead-Time Demand": [lead_time_demand],
    "Inventory Position": [inventory_position],
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
    **Forecast Horizon:** 6-8 Weeks
    **Lead-Time Demand:** {lead_time_demand:,.0f} Units
    **On-Hand:** {inventory:,.0f} Units
    **On-Order:** {on_order:,.0f} Units
    **Inventory Position:** {inventory_position:,.0f} Units
    **Forward Window Demand:** {forward_window_demand:,.0f} Units
    **Stockout Risk:** {stockout_score:.1f}%
    **Overstock Risk:** {overstock_score:.1f}%
    **Decision:** {quadrant}
    **Value at Stake:** {value_at_stake:,.0f}
    """
)
st.markdown("---")
st.caption("Project FORESIGHT | " "AI-Powered Demand Forecasting & Inventory Risk Management")