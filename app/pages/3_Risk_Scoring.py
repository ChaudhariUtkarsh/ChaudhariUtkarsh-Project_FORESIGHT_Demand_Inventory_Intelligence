import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# Page Configuration
st.set_page_config(page_title="Risk Scoring", page_icon=" ", layout="wide", initial_sidebar_state="expanded")


# Title
st.title("Inventory Risk Scoring")
st.markdown("""Identify Stockout Risk, Overstock Risk and generate business recommendations using forecasted inventory.""")
st.markdown("---")


# Sidebar
st.sidebar.header("Risk Analysis")
st.sidebar.markdown("---")
st.sidebar.success("Inventory Intelligence")
st.sidebar.markdown("---")
dark_mode = st.sidebar.toggle("Dark Mode", value=False)
st.sidebar.markdown("---")


# Dataset Loading
DATA_PATH = r"E:\Zidio Development\Project_FORESIGHT\data\processed/processed_data.csv"
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)

else:
    st.warning("Processed Dataset not found.")
    df = pd.DataFrame({
        "sku_id":["SKU001", "SKU002", "SKU003", "SKU004", "SKU005"],
        "category":["Electronics", "Electronics", "Furniture", "Furniture", "Grocery"],
        "forecast_units":[180, 75, 120, 210, 95],
        "on_hand_units":[70, 180, 150, 80, 220],
        "selling_price":[450, 900, 180, 250, 60],
        "unit_cost":[320, 650, 120, 170, 35]
    })


# Risk Threshold Settings
st.sidebar.header("Risk Threshold")
stockout_threshold = st.sidebar.slider("Stockout Threshold (%)", min_value=50, max_value=150, value=100)
overstock_threshold = st.sidebar.slider("Overstock Threshold (%)", min_value=100, max_value=300, value=150)
st.sidebar.markdown("---")


# Category Filter
category = st.sidebar.selectbox("Category", ["All"] + sorted(df["category"].unique().tolist()))
filtered_df = df.copy()

if category != "All":
    filtered_df = filtered_df[filtered_df["category"] == category]


# SKU Selection
sku_list = sorted(filtered_df["sku_id"].unique().tolist())
selected_sku = st.sidebar.selectbox("Select SKU", sku_list)
sku_data = filtered_df[filtered_df["sku_id"] == selected_sku].iloc[0]


# Display Selected SKU
st.subheader("Selected Product")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("SKU", sku_data["sku_id"])

with col2:
    st.metric("Category", sku_data["category"])
    
with col3:
    st.metric("Forecast Units", int(sku_data["forecast_units"]))

st.markdown("---")


# Risk Scoring Logic
forecast = float(sku_data["forecast_units"])
inventory = float(sku_data["on_hand_units"])
selling_price = float(sku_data["selling_price"])
unit_cost = float(sku_data["unit_cost"])

# Inventory Ratio
inventory_ratio = (inventory / forecast) * 100 if forecast > 0 else 0

# Risk Classification
if inventory_ratio < stockout_threshold:
    risk_level = "High Stockout"
    risk_score = 90

elif inventory_ratio > overstock_threshold:
    risk_level = "Overstock"
    risk_score = 80

else:
    risk_level = "Healthy"
    risk_score = 25

# Revenue at Risk
revenue_at_risk = 0

if risk_level == "High Stockout":
    shortage = max(forecast - inventory, 0)
    revenue_at_risk = shortage * selling_price

# Capital Locked
capital_locked = 0

if risk_level == "Overstock":
    excess = max(inventory - (forecast * 1.20), 0)
    capital_locked = excess * unit_cost


# KPI Cards
st.header("Business KPIs")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Risk Score", f"{risk_score}%")

with c2:
    st.metric("Revenue at Risk", f"{revenue_at_risk:,.2f}")

with c3:
    st.metric("Capital Locked", f"{capital_locked:,.2f}")

with c4:
    st.metric("Risk Level", risk_level)

st.markdown("---")


# Risk Gauge
st.subheader("Risk Indicator")
gauge_df = pd.DataFrame({"Category":["Risk"], "Score":[risk_score]})
gauge = px.bar(gauge_df, x="Category", y="Score", text_auto=True, title="Inventory Risk Score")
gauge.update_layout(yaxis_title="Risk Score", height=400)
st.plotly_chart(gauge, use_container_width=True)
st.markdown("---")


# Risk Distribution
st.subheader("Risk Distribution")
risk_summary = filtered_df.copy()
risk_summary["Inventory Ratio"] = (risk_summary["on_hand_units"] / risk_summary["forecast_units"]) * 100
risk_summary["Risk"] = np.where(risk_summary["Inventory Ratio"] < stockout_threshold, "High Stockout", np.where(risk_summary["Inventory Ratio"] > overstock_threshold, "Overstock", "Healthy"))
risk_count = (risk_summary["Risk"].value_counts().reset_index())
risk_count.columns = ["Risk", "Count"]
risk_fig = px.pie(risk_count, names="Risk", values="Count", title="Inventory Risk Distribution")
st.plotly_chart(risk_fig, use_container_width=True)
st.markdown("---")


# Inventory vs Forecast
st.subheader("Forecast vs Inventory")
compare = pd.DataFrame({"Metric":["Forecast", "Inventory"], "Units":[forecast, inventory]})
compare_fig = px.bar(compare, x="Metric", y="Units", text_auto=True, title="Forecast vs Current Inventory")
st.plotly_chart(compare_fig, use_container_width=True)
st.markdown("---")


# SKU Information
st.subheader("Selected SKU Details")
details = pd.DataFrame({
    "SKU":[sku_data["sku_id"]],
    "Category":[sku_data["category"]],
    "Forecast Units":[forecast],
    "Current Inventory":[inventory],
    "Risk Level":[risk_level],
    "Risk Score":[risk_score]
})

st.dataframe(details, use_container_width=True)
st.markdown("---")


# Business Recommendation
st.header("Business Recommendation")
if risk_level == "High Stockout":
    st.error("""
        ### High Stockout Risk Recommended Actions
        1. Reorder inventory immediately
        2. Increase safety stock
        3. Contact supplier
        4. Monitor daily demand
        5. Avoid stockout losses
    """)

elif risk_level == "Overstock":
    st.warning("""
        ### Overstock Risk Recommended Actions
        1. Reduce future procurement
        2. Launch discount campaign
        3. Bundle slow-moving products
        4. Improve inventory turnover
    """)

else:
    st.success("""
        ### Healthy Inventory Recommended Actions
        1. Continue current inventory strategy
        2. Weekly inventory monitoring
        3. Maintain reorder schedule
    """)

st.markdown("---")


# Risk Report
st.header("Risk Report")
report = f"""
-----------------------------------------

PROJECT FORESIGHT
Inventory Risk Report

-----------------------------------------

SKU ID
{sku_data['sku_id']}

Category
{sku_data['category']}

-----------------------------------------

Forecast Units
{forecast}

Current Inventory
{inventory}

Inventory Ratio
{inventory_ratio:.2f} %

-----------------------------------------

Risk Level
{risk_level}

Risk Score
{risk_score} %

-----------------------------------------

Revenue at Risk {revenue_at_risk:,.2f}
Capital Locked {capital_locked:,.2f}

-----------------------------------------

Business Recommendation

"""
if risk_level == "High Stockout":
    report += """
        Increase inventory immediately.
        Increase reorder quantity.
        Maintain higher safety stock.
    """

elif risk_level == "Overstock":
    report += """
        Reduce inventory.
        Launch promotions.
        Decrease procurement.
    """

else:
    report += """
        Inventory level is healthy.
        Continue normal monitoring.
    """

report += """
-----------------------------------------

Generated by
Project FORESIGHT

-----------------------------------------
"""

st.download_button(label="Download Risk Report", data=report, file_name="Risk_Report.txt", mime="text/plain")
st.markdown("---")


# Download CSV
st.header("Download Risk Data")

risk_export = pd.DataFrame({
    "SKU":[sku_data["sku_id"]],
    "Category":[sku_data["category"]],
    "Forecast Units":[forecast],
    "Current Inventory":[inventory],
    "Risk Level":[risk_level],
    "Risk Score":[risk_score],
    "Revenue at Risk":[revenue_at_risk],
    "Capital Locked":[capital_locked]
})

csv = risk_export.to_csv(index=False).encode("utf-8")
st.download_button(label="Download CSV", data=csv, file_name="risk_analysis.csv", mime="text/csv")
st.markdown("---")


# Risk History
st.header("Risk History")

history = pd.DataFrame({
    "SKU":[sku_data["sku_id"]],
    "Forecast":[forecast],
    "Inventory":[inventory],
    "Risk Score":[risk_score],
    "Risk":[risk_level]
})

st.dataframe(history, use_container_width=True)
st.markdown("---")


# Summary
st.header("Summary")
st.success(f"""
    SKU : {sku_data['sku_id']}
    Forecast : {forecast:.0f} Units
    Inventory : {inventory:.0f} Units
    Risk Level : {risk_level}
    Revenue at Risk : {revenue_at_risk:,.2f}
    Capital Locked : {capital_locked:,.2f}
""")

st.markdown("---")


# Footer
st.caption("Project FORESIGHT | AI-Powered Demand Forecasting & Inventory Risk Management")