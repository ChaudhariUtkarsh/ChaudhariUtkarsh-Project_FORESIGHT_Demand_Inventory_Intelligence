import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os


# PAGE CONFIGURATION
st.set_page_config(page_title="Project FORESIGHT | Dashboard", page_icon=" ", layout="wide", initial_sidebar_state="expanded")


# PROJECT PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RISK_DIR = os.path.join(DATA_DIR, "risk_analysis")
DECISION_DIR = os.path.join(DATA_DIR, "decisioning_grid")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_data.csv")
WEEKLY_DATA_PATH = os.path.join(PROCESSED_DIR, "weekly_model_data.csv")
RISK_DATA_PATH = os.path.join(RISK_DIR, "sku_risk_analysis.csv")
DECISION_GRID_PATH = os.path.join(DECISION_DIR, "decisioning_grid.csv")


STYLE_PATH = os.path.join(BASE_DIR, "app", "style.css")


# LOAD CSS
def load_css():
    if os.path.exists(STYLE_PATH):
        with open(STYLE_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"style.css not found at: {STYLE_PATH}")
load_css()


# HELPER FUNCTIONS
def numeric_column(df, column, default=0):
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def first_existing_column(df, columns):
    for col in columns:
        if col in df.columns:
            return col
    return None


# LOAD DATA
processed_df = pd.DataFrame()
weekly_df = pd.DataFrame()
risk_df = pd.DataFrame()
decision_df = pd.DataFrame()


# Processed Data
if os.path.exists(PROCESSED_DATA_PATH):
    try:
        processed_df = pd.read_csv(PROCESSED_DATA_PATH)
    except Exception as e:
        st.error(f"Error loading processed_data.csv: {e}")

else:
    st.warning("processed_data.csv not found.")


# Weekly Model Data
if os.path.exists(WEEKLY_DATA_PATH):
    try:
        weekly_df = pd.read_csv(WEEKLY_DATA_PATH)
    except Exception as e:
        st.warning(f"Could not load weekly_model_data.csv: {e}")


# Risk Data
if os.path.exists(RISK_DATA_PATH):
    try:
        risk_df = pd.read_csv(RISK_DATA_PATH)
    except Exception as e:
        st.warning(f"Could not load sku_risk_analysis.csv: {e}")


# Decisioning Data
if os.path.exists(DECISION_GRID_PATH):
    try:
        decision_df = pd.read_csv(DECISION_GRID_PATH)
    except Exception as e:
        st.warning(f"Could not load decisioning_grid.csv: {e}")


# BASIC VALIDATION
if risk_df.empty and processed_df.empty:
    st.error(
        """
        No dashboard data found.
        Please make sure these files exist:
        data/processed/processed_data.csv
        data/risk_analysis/sku_risk_analysis.csv
        """
    )
    st.stop()


# PREPARE PROCESSED DATA
if not processed_df.empty:
    if "date" in processed_df.columns:
        processed_df["date"] = pd.to_datetime(processed_df["date"], errors="coerce")
        processed_df = processed_df.dropna(subset=["date"])
        processed_df = processed_df.sort_values("date")


# ADD CATEGORY TO RISK DATA
if (not processed_df.empty and "sku_id" in processed_df.columns):
    if "category" in processed_df.columns:
        category_df = (processed_df[["sku_id", "category"]].dropna(subset=["sku_id"]).drop_duplicates(subset=["sku_id"], keep="last"))
        if ("category" not in risk_df.columns and not risk_df.empty and "sku_id" in risk_df.columns):
            risk_df = risk_df.merge(category_df, on="sku_id", how="left")

        if ("category" not in decision_df.columns and not decision_df.empty and "sku_id" in decision_df.columns):
            decision_df = decision_df.merge(category_df, on="sku_id", how="left")


if risk_df.empty and not processed_df.empty:
    risk_df = (processed_df.groupby("sku_id", as_index=False).tail(1).copy())


# NORMALIZE SKU ID
if "sku_id" in risk_df.columns:
    risk_df["sku_id"] = (risk_df["sku_id"].astype(str).str.strip())

if (not processed_df.empty and "sku_id" in processed_df.columns):
    processed_df["sku_id"] = (processed_df["sku_id"].astype(str).str.strip())


# CREATE FORECAST UNITS
if "forecast_units" not in risk_df.columns:
    if "Forecast_Units" in risk_df.columns:
        risk_df["forecast_units"] = (risk_df["Forecast_Units"])

    elif "forecast" in risk_df.columns:
        risk_df["forecast_units"] = (risk_df["forecast"])
    elif "Average_Daily_Demand" in risk_df.columns:
        risk_df["forecast_units"] = (numeric_column(risk_df, "Average_Daily_Demand") * 7)

    else:
        risk_df["forecast_units"] = 0

risk_df["forecast_units"] = numeric_column(risk_df, "forecast_units")


# INVENTORY COLUMNS
if "on_hand_units" not in risk_df.columns:
    if "Current_Inventory" in risk_df.columns:
        risk_df["on_hand_units"] = (risk_df["Current_Inventory"])

    elif "inventory" in risk_df.columns:
        risk_df["on_hand_units"] = (risk_df["inventory"])

    else:
        risk_df["on_hand_units"] = 0


if "on_order_units" not in risk_df.columns:
    if "On_Order_Units" in risk_df.columns:
        risk_df["on_order_units"] = (risk_df["On_Order_Units"])

    else:
        risk_df["on_order_units"] = 0
risk_df["on_hand_units"] = numeric_column(risk_df, "on_hand_units")
risk_df["on_order_units"] = numeric_column(risk_df, "on_order_units")


# CATEGORY
if "category" not in risk_df.columns:
    risk_df["category"] = "Unknown"
risk_df["category"] = (risk_df["category"].fillna("Unknown").astype(str))


# RISK LEVEL
if "Risk_Level" not in risk_df.columns:
    if "Risk" in risk_df.columns:
        risk_df["Risk_Level"] = (risk_df["Risk"])

    else:
        inventory_ratio = np.where(risk_df["forecast_units"] > 0, ((risk_df["on_hand_units"] + risk_df["on_order_units"]) / risk_df["forecast_units"]) * 100, 0)
        risk_df["Risk_Level"] = np.select([inventory_ratio < 100, inventory_ratio > 150], ["HIGH", "MEDIUM"], default="LOW")
risk_df["Risk_Level"] = (risk_df["Risk_Level"].fillna("LOW").astype(str))

# SALES AT RISK
if "Sales_at_Risk" not in risk_df.columns:
    price_col = first_existing_column(risk_df, ["selling_price", "list_price", "unit_price", "price"])

    if price_col:
        price = numeric_column(risk_df, price_col)

    else:
        price = pd.Series(0, index=risk_df.index)
    shortage = (risk_df["forecast_units"] - (risk_df["on_hand_units"] + risk_df["on_order_units"])).clip(lower=0)
    risk_df["Sales_at_Risk"] = (shortage * price)
risk_df["Sales_at_Risk"] = numeric_column(risk_df, "Sales_at_Risk")


# CAPITAL LOCKED
if "Capital_Locked" not in risk_df.columns:
    cost_col = first_existing_column(risk_df, ["unit_cost", "Unit_Cost", "cost"])
    if cost_col:
        unit_cost = numeric_column(risk_df, cost_col)
    else:
        unit_cost = pd.Series(0,index=risk_df.index)
    excess = (risk_df["on_hand_units"] - (risk_df["forecast_units"] * 1.20)).clip(lower=0)
    risk_df["Capital_Locked"] = (excess * unit_cost)
risk_df["Capital_Locked"] = numeric_column(risk_df, "Capital_Locked")


# RISK FLAGS
def create_risk_flag(row):
    risk = str(row.get("Risk_Level", "LOW")).upper()
    stockout_score = float(row.get("Stockout_Risk_Score", 0))

    overstock_score = float(row.get("Overstock_Risk_Score", 0))
    if ("HIGH" in risk or stockout_score >= 0.75):
        return "STOCKOUT"
    elif ("MEDIUM" in risk or overstock_score >= 0.75):
        return "OVERSTOCK"
    else:
        return "HEALTHY"
risk_df["Risk_Flag"] = risk_df.apply(create_risk_flag, axis=1)


# PRIORITY SCORE
risk_df["Shortage_Units"] = (risk_df["forecast_units"] - (risk_df["on_hand_units"] + risk_df["on_order_units"])).clip(lower=0)
risk_df["Priority_Score"] = (risk_df["Shortage_Units"] * 2 + risk_df["Sales_at_Risk"] / 1000)


# RECOMMENDED ACTION
def recommended_action(row):
    flag = row["Risk_Flag"]
    if "STOCKOUT" in flag:
        return "REORDER NOW"

    elif "OVERSTOCK" in flag:
        return "MARKDOWN / CLEAR"
    return "MONITOR"

risk_df["Recommended_Action"] = risk_df.apply(recommended_action, axis=1)


# SIDEBAR
st.sidebar.title("PROJECT FORESIGHT")
st.sidebar.markdown("### Demand & Inventory Intelligence")
st.sidebar.markdown("---")
st.sidebar.header("Dashboard Filters")


# Category Filter
categories = sorted(risk_df["category"].dropna().astype(str).unique().tolist())
selected_category = st.sidebar.selectbox("Category", ["All"] + categories)


# SKU Filter
sku_values = sorted(risk_df["sku_id"].dropna().astype(str).unique().tolist())
selected_sku = st.sidebar.selectbox("SKU", ["All"] + sku_values)


# Risk Filter
risk_options = ["All", "STOCKOUT", "OVERSTOCK", "HEALTHY"]
selected_risk = st.sidebar.selectbox("Risk Flag", risk_options)


# APPLY FILTERS
filtered_df = risk_df.copy()
if selected_category != "All":
    filtered_df = filtered_df[filtered_df["category"] == selected_category]

if selected_sku != "All":
    filtered_df = filtered_df[filtered_df["sku_id"] == selected_sku]

if selected_risk != "All":
    filtered_df = filtered_df[filtered_df["Risk_Flag"] == selected_risk]

if filtered_df.empty:
    st.warning("No records found for selected filters.")
    st.stop()


# DASHBOARD HEADER
st.title("PROJECT FORESIGHT")
st.subheader("Demand Forecasting & Inventory Intelligence Dashboard")
st.caption("Weekly SKU-level demand planning, inventory risk and action prioritization")
st.markdown("---")


# KPI CARDS
total_skus = (filtered_df["sku_id"].nunique())
total_forecast = (filtered_df["forecast_units"].sum())
total_inventory = (filtered_df["on_hand_units"].sum())
total_sales_risk = (filtered_df["Sales_at_Risk"].sum())

stockout_count = (filtered_df["Risk_Flag"].eq("STOCKOUT").sum())
overstock_count = (filtered_df["Risk_Flag"].eq("OVERSTOCK").sum())
c1, c2, c3, c4, c5, c6 = st.columns(6)


with c1:
    st.metric("SKUs", f"{total_skus:,}")

with c2:
    st.metric("Forecast Demand", f"{total_forecast:,.0f}")

with c3:
    st.metric("Current Inventory", f"{total_inventory:,.0f}")

with c4:
    st.metric("Sales at Risk", f"{total_sales_risk:,.0f}")

with c5:
    st.metric("Stockout Risks", f"{stockout_count:,}")

with c6:
    st.metric("Overstock Risks", f"{overstock_count:,}")
st.markdown("---")


# 1. FORECAST VS ACTUAL
st.header("1. Forecast vs Actual")
actual_df = pd.DataFrame()


if not processed_df.empty:
    actual_col = first_existing_column(processed_df, ["units_sold", "actual_units", "actual", "sales_units"])

    if (actual_col and "date" in processed_df.columns):
        actual_df = processed_df[["date", "sku_id", actual_col]].copy()
        actual_df["Actual"] = pd.to_numeric(actual_df[actual_col], errors="coerce").fillna(0)
        actual_df["Week"] = (actual_df["date"].dt.to_period("W").dt.start_time)
        actual_df = (actual_df.groupby(["Week", "sku_id"], as_index=False)["Actual"].sum())


if not actual_df.empty:
    if selected_sku != "All":
        actual_chart = actual_df[actual_df["sku_id"] == selected_sku].copy()

    else:
        actual_chart = (actual_df.groupby("Week", as_index=False)["Actual"].sum())
        actual_chart["sku_id"] = "All"


    if selected_sku != "All":
        selected_forecast = filtered_df[filtered_df["sku_id"] == selected_sku]["forecast_units"].sum()

    else:
        selected_forecast = (filtered_df["forecast_units"].sum())
    actual_chart = (actual_chart.sort_values("Week").tail(8))
    actual_chart["Forecast"] = (selected_forecast / max(len(actual_chart), 1))
    chart_long = actual_chart[["Week", "Actual", "Forecast"]].melt(id_vars="Week", var_name="Type", value_name="Units")
    forecast_actual_fig = px.line(chart_long, x="Week", y="Units", color="Type", markers=True, title="Weekly Actual Demand vs Forecast")
    forecast_actual_fig.update_layout(xaxis_title="Week", yaxis_title="Units", height=450)
    st.plotly_chart(forecast_actual_fig, use_container_width=True)


else:
    st.info(
        """
        Actual sales history is not available in processed_data.csv.
        Expected column:
        units_sold
        """
    )

st.markdown("---")


# 2. RISK FLAGS
st.header("2. Risk Flags")
risk_counts = (filtered_df["Risk_Flag"].value_counts().reset_index())
risk_counts.columns = ["Risk Flag", "SKU Count"]


risk_fig = px.pie(risk_counts, names="Risk Flag", values="SKU Count", hole=0.45, title="Inventory Risk Distribution")
st.plotly_chart(risk_fig, use_container_width=True)
risk_table_columns = ["sku_id", "category", "forecast_units", "on_hand_units", "on_order_units", "Shortage_Units", "Risk_Flag", "Sales_at_Risk", "Capital_Locked", "Recommended_Action"]
risk_table_columns = [col for col in risk_table_columns if col in filtered_df.columns]
risk_table = (filtered_df[risk_table_columns].sort_values("Sales_at_Risk", ascending=False).head(20).copy())
risk_table = risk_table.rename(columns={"sku_id": "SKU", "category": "Category", "forecast_units": "Forecast Units", "on_hand_units": "On Hand", "on_order_units": "On Order", "Shortage_Units": "Shortage Units", "Risk_Flag": "Risk", "Sales_at_Risk": "Sales at Risk", "Capital_Locked": "Capital Locked", "Recommended_Action": "Action"})
st.dataframe(risk_table, use_container_width=True, hide_index=True)
st.markdown("---")


# 3. PRIORITIZED REORDER LIST
st.header("3. Prioritized Reorder List")
st.info("SKUs with the highest shortage and business impact are prioritized first.")
reorder_df = filtered_df[filtered_df["Shortage_Units"] > 0].copy()
reorder_df = reorder_df[(reorder_df["Risk_Flag"] == "STOCKOUT") | (reorder_df["Shortage_Units"] > 0)]
reorder_df = reorder_df.sort_values(["Priority_Score", "Sales_at_Risk", "Shortage_Units"], ascending=False)
reorder_df = reorder_df.head(20)


if not reorder_df.empty:
    reorder_display = reorder_df[["sku_id", "category", "forecast_units", "on_hand_units", "on_order_units", "Shortage_Units", "Sales_at_Risk", "Risk_Flag", "Priority_Score", "Recommended_Action"]].copy()
    reorder_display = reorder_display.rename(columns={"sku_id": "SKU", "category": "Category", "forecast_units": "Forecast Demand", "on_hand_units": "On Hand", "on_order_units": "On Order", "Shortage_Units": "Shortage", "Sales_at_Risk": "Sales at Risk", "Risk_Flag": "Risk", "Priority_Score": "Priority", "Recommended_Action": "Action"})
    st.dataframe(reorder_display, use_container_width=True, hide_index=True)
    st.download_button(label="Download Reorder List", data=reorder_display.to_csv(index=False).encode("utf-8"), file_name="prioritized_reorder_list.csv", mime="text/csv", use_container_width=True)


else:
    st.success("No immediate reorder requirement found.")

st.markdown("---")


# 4. MARKDOWN LIST
st.header("4. Markdown / Clearance List")
st.info("SKUs with excess inventory relative to forecast demand are prioritized for markdown/clearance.")
markdown_df = filtered_df.copy()
markdown_df["Excess_Units"] = (markdown_df["on_hand_units"] - (markdown_df["forecast_units"] * 1.20)).clip(lower=0)
markdown_df = markdown_df[markdown_df["Excess_Units"] > 0]
markdown_df["Markdown_Value"] = (markdown_df["Excess_Units"] * (markdown_df["Capital_Locked"] / markdown_df["Excess_Units"].replace(0, np.nan)).fillna(0))
markdown_df = markdown_df.sort_values("Capital_Locked", ascending=False)
markdown_df = markdown_df.head(20)


if not markdown_df.empty:
    markdown_display = markdown_df[["sku_id", "category", "forecast_units", "on_hand_units", "Excess_Units", "Capital_Locked", "Risk_Flag", "Recommended_Action"]].copy()
    markdown_display = markdown_display.rename(columns={"sku_id": "SKU", "category": "Category", "forecast_units": "Forecast Demand", "on_hand_units": "On Hand", "Excess_Units": "Excess Units", "Capital_Locked": "Capital Locked", "Risk_Flag": "Risk", "Recommended_Action": "Action"})
    st.dataframe(markdown_display, use_container_width=True, hide_index=True)
    st.download_button(label="Download Markdown List", data=markdown_display.to_csv(index=False).encode("utf-8"), file_name="markdown_clearance_list.csv", mime="text/csv", use_container_width=True)


else:
    st.success("No significant excess inventory found.")
st.markdown("---")


# 5. FORECAST DEMAND BY CATEGORY
st.header("5. Forecast Demand by Category")
category_forecast = (filtered_df.groupby("category", as_index=False).agg(Forecast_Demand=("forecast_units", "sum"), Current_Inventory=("on_hand_units", "sum")))
category_long = category_forecast.melt(id_vars="category", value_vars=["Forecast_Demand", "Current_Inventory"], var_name="Metric", value_name="Units")
category_fig = px.bar(category_long, x="category", y="Units", color="Metric", barmode="group", title="Forecast Demand vs Current Inventory by Category")
category_fig.update_layout(xaxis_title="Category", yaxis_title="Units", height=450)
st.plotly_chart(category_fig, use_container_width=True)
st.markdown("---")


# 6. TOP SKU FORECAST
st.header("6. Top SKU Forecast Demand")
top_forecast = (filtered_df[["sku_id", "category", "forecast_units"]].sort_values("forecast_units", ascending=False).head(15))
forecast_fig = px.bar(top_forecast, x="sku_id", y="forecast_units", color="category", text_auto=True, title="Top 15 SKU Forecast Demand")
forecast_fig.update_layout(xaxis_title="SKU", yaxis_title="Forecast Units", height=500)
st.plotly_chart(forecast_fig, use_container_width=True)
st.markdown("---")


# 7. INVENTORY POSITION
st.header("7. Inventory Position")
inventory_chart_df = filtered_df[["sku_id", "forecast_units", "on_hand_units", "on_order_units"]].copy()
inventory_chart_df = (inventory_chart_df.sort_values("forecast_units", ascending=False).head(15))
inventory_long = inventory_chart_df.melt(id_vars="sku_id", value_vars=["forecast_units", "on_hand_units", "on_order_units"], var_name="Inventory Metric", value_name="Units")
inventory_long["Inventory Metric"] = (inventory_long["Inventory Metric"].replace({"forecast_units": "Forecast", "on_hand_units": "On Hand", "on_order_units": "On Order"}))
inventory_fig = px.bar(inventory_long, x="sku_id", y="Units", color="Inventory Metric", barmode="group", title="Forecast vs On-Hand vs On-Order Inventory")
inventory_fig.update_layout( xaxis_title="SKU", yaxis_title="Units", height=500)
st.plotly_chart(inventory_fig, use_container_width=True)
st.markdown("---")


# 8. BUSINESS RECOMMENDATIONS
st.header("8. Business Recommendations")
recommendation_col1, recommendation_col2 = st.columns(2)


with recommendation_col1:
    st.subheader("Stockout Actions")
    st.markdown(
        """
        - Prioritize high-shortage SKUs.
        - Review on-hand + on-order inventory.
        - Replenish critical SKUs first.
        - Monitor forecast demand weekly.
        - Protect high-value SKUs from stockouts.
        """
    )


with recommendation_col2:
    st.subheader("Overstock Actions")
    st.markdown(
        """
        - Identify excess inventory.
        - Reduce future procurement.
        - Run markdown/clearance campaigns.
        - Bundle slow-moving products.
        - Improve inventory turnover.
        """
    )
st.markdown("---")


# 9. DASHBOARD SUMMARY
st.header("9. Dashboard Summary")
reorder_count = len(reorder_df)
markdown_count = len(markdown_df)


summary_df = pd.DataFrame(
    {
        "Metric": ["SKUs Analysed", "Forecast Demand", "Current Inventory", "Sales at Risk", "Stockout Risk SKUs", "Overstock Risk SKUs", "Prioritized Reorder SKUs", "Markdown / Clearance SKUs"],
        "Value": [total_skus, f"{total_forecast:,.0f}", f"{total_inventory:,.0f}", f"{total_sales_risk:,.0f}", stockout_count, overstock_count, reorder_count, markdown_count]
    }
)


st.dataframe(summary_df, use_container_width=True, hide_index=True)
st.markdown("---")


# 10. DOWNLOAD COMPLETE DASHBOARD DATA
st.header("10. Download Dashboard Data")
download_columns = ["sku_id", "category", "forecast_units", "on_hand_units", "on_order_units", "Shortage_Units", "Risk_Flag", "Priority_Score", "Sales_at_Risk", "Capital_Locked", "Recommended_Action"]
download_columns = [col for col in download_columns if col in filtered_df.columns]
download_df = filtered_df[download_columns].copy()
download_csv = (download_df.to_csv(index=False).encode("utf-8"))
st.download_button(label="Download Complete Dashboard CSV", data=download_csv, file_name="foresight_dashboard_data.csv", mime="text/csv", use_container_width=True)


# FOOTER
st.markdown("---")
st.caption("Project FORESIGHT | AI-Powered Demand Forecasting & Inventory Intelligence")