import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os


# PAGE CONFIGURATION
st.set_page_config(page_title="Dashboard | Project FORESIGHT", page_icon=" ", layout="wide", initial_sidebar_state="expanded")


# LOAD CSS
def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "style.css")

    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# TITLE
st.title("Project FORESIGHT Dashboard")
st.subheader("Demand Forecasting & Inventory Intelligence")
st.markdown("---")


# DATA PATH
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")


# LOAD DATA
if not os.path.exists(DATA_PATH):
    st.error(f"Processed dataset not found:\n{DATA_PATH}")
    st.stop()
df = pd.read_csv(DATA_PATH)
df.columns = (df.columns.str.strip().str.lower().str.replace(" ", "_"))


# DATA TYPE CLEANING
numeric_columns = ["units_sold", "forecast_units", "actual_units", "on_hand_units", "on_order_units", "selling_price", "unit_cost", "revenue", "inventory_value", "lead_time_days"]

for col in numeric_columns:
    if col in df.columns:
        df[col] = (df[col].astype(str).str.replace(",", "", regex=False).str.replace("₹", "", regex=False).str.strip())
        df[col] = pd.to_numeric(df[col], errors="coerce")


# DATE COLUMN
date_columns = ["date", "week_start", "forecast_date"]
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")


# BASIC VALIDATION
if "sku_id" not in df.columns:
    st.error("Required column 'sku_id' not found in dataset.")
    st.stop()

if "category" not in df.columns:
    df["category"] = "Unknown"


# SIDEBAR
st.sidebar.title("Project FORESIGHT")
st.sidebar.markdown("### Dashboard Filters")
st.sidebar.markdown("---")


# CATEGORY FILTER
categories = sorted(df["category"].dropna().astype(str).unique().tolist())
selected_category = st.sidebar.selectbox("Category", ["All"] + categories)
filtered_df = df.copy()


if selected_category != "All":
    filtered_df = filtered_df[filtered_df["category"].astype(str) == selected_category]


# SKU FILTER
sku_values = sorted(
    filtered_df["sku_id"].dropna().astype(str).unique().tolist())
selected_sku = st.sidebar.selectbox("SKU", ["All"] + sku_values)


if selected_sku != "All":
    filtered_df = filtered_df[filtered_df["sku_id"].astype(str) == selected_sku]


st.sidebar.markdown("---")
st.sidebar.info("Weekly SKU-level Demand & Inventory Intelligence")


# KPI CALCULATIONS
total_skus = (filtered_df["sku_id"].nunique())
if "units_sold" in filtered_df.columns:
    total_demand = (filtered_df["units_sold"].fillna(0).sum())

else:
    total_demand = 0

if "forecast_units" in filtered_df.columns:
    total_forecast = (filtered_df["forecast_units"].fillna(0).sum())

else:
    total_forecast = 0

if "on_hand_units" in filtered_df.columns:
    total_inventory = (filtered_df["on_hand_units"].fillna(0).sum())

else:
    total_inventory = 0


# RISK CALCULATION
if "forecast_units" in filtered_df.columns:
    forecast_for_risk = (filtered_df["forecast_units"].fillna(0))

else:
    forecast_for_risk = pd.Series(0, index=filtered_df.index)


if "on_hand_units" in filtered_df.columns:
    inventory_for_risk = (filtered_df["on_hand_units"].fillna(0))

else:
    inventory_for_risk = pd.Series(0, index=filtered_df.index)


if "on_order_units" in filtered_df.columns:
    on_order_for_risk = (filtered_df["on_order_units"].fillna(0))

else:
    on_order_for_risk = pd.Series(0, index=filtered_df.index)

available_inventory = (inventory_for_risk + on_order_for_risk)


# STOCKOUT / OVERSTOCK FLAGS
filtered_df["stockout_risk"] = (available_inventory < forecast_for_risk)
filtered_df["overstock_risk"] = (available_inventory > forecast_for_risk * 1.5)


def get_risk(row):
    if row["stockout_risk"]:
        return "High Stockout"
    elif row["overstock_risk"]:
        return "Overstock"
    else:
        return "Healthy"
filtered_df["risk_level"] = (filtered_df.apply(get_risk, axis=1))


# RISK COUNTS
high_stockout_count = (filtered_df["risk_level"].eq("High Stockout").sum())
overstock_count = (filtered_df["risk_level"].eq("Overstock").sum())
healthy_count = (filtered_df["risk_level"].eq("Healthy").sum())


# KPI SECTION
st.header("Business Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total SKUs", f"{total_skus:,}")

with col2:
    st.metric("Forecast Demand", f"{total_forecast:,.0f}")

with col3:
    st.metric("Current Inventory", f"{total_inventory:,.0f}")

with col4:
    st.metric("Stockout Risk", f"{high_stockout_count:,}")

st.markdown("---")


# RISK SUMMARY
st.header("Inventory Risk Summary")
r1, r2, r3 = st.columns(3)
with r1:
    st.error(f"High Stockout: {high_stockout_count}")

with r2:
    st.warning(f"Overstock: {overstock_count}")

with r3:
    st.success(f"Healthy: {healthy_count}")

st.markdown("---")


# FORECAST VS ACTUAL
st.header("Forecast vs Actual")
has_actual = "units_sold" in filtered_df.columns
has_forecast = "forecast_units" in filtered_df.columns

if has_actual and has_forecast:
    chart_df = filtered_df.copy()

    if "week_start" in chart_df.columns:
        chart_df = (chart_df.groupby("week_start", as_index=False).agg(Actual=("units_sold", "sum"), Forecast=("forecast_units", "sum")).sort_values("week_start"))
        chart_df = chart_df.dropna(subset=["week_start"])

        if not chart_df.empty:
            chart_df = chart_df.melt(id_vars=["week_start"], value_vars=["Actual", "Forecast"], var_name="Type", value_name="Units")
            fig = px.line(chart_df, x="week_start", y="Units", color="Type", markers=True, title="Weekly Forecast vs Actual Demand")
            fig.update_layout(xaxis_title="Week", yaxis_title="Units")
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No valid weekly data available.")

    else:
        comparison = pd.DataFrame({"Type": ["Actual", "Forecast"], "Units": [filtered_df["units_sold"].fillna(0).sum(), filtered_df["forecast_units"].fillna(0).sum()]})
        fig = px.bar(comparison, x="Type", y="Units", text_auto=True, title="Forecast vs Actual Demand")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Forecast or Actual demand columns are not available.")
st.markdown("---")


# RISK DISTRIBUTION
st.header("Risk Distribution")
risk_count = (filtered_df["risk_level"].value_counts().reset_index())
risk_count.columns = ["Risk Level", "Count"]

if not risk_count.empty:
    risk_fig = px.pie(risk_count, names="Risk Level", values="Count", title="Inventory Risk Distribution", hole=0.4)
    st.plotly_chart(risk_fig, use_container_width=True)

st.markdown("---")


# PRIORITIZED REORDER LIST
st.header("Prioritized Reorder List")
reorder_df = filtered_df[filtered_df["risk_level"] == "High Stockout"].copy()

if not reorder_df.empty:
    reorder_df["shortage_units"] = (forecast_for_risk.loc[reorder_df.index] - available_inventory.loc[reorder_df.index]).clip(lower=0)
    reorder_df["priority"] = (reorder_df["shortage_units"].rank(ascending=False, method="dense"))
    reorder_columns = ["sku_id", "category"]

    if "forecast_units" in reorder_df.columns:
        reorder_columns.append("forecast_units")

    if "on_hand_units" in reorder_df.columns:
        reorder_columns.append("on_hand_units")

    if "on_order_units" in reorder_df.columns:
        reorder_columns.append("on_order_units")

    reorder_columns += ["shortage_units", "priority"]
    reorder_display = (reorder_df[reorder_columns].sort_values("shortage_units", ascending=False).head(20).copy())

    for col in reorder_display.columns:
        if col not in ["sku_id", "category"]:
            reorder_display[col] = pd.to_numeric(reorder_display[col], errors="coerce")

    st.dataframe(
        reorder_display, use_container_width=True, hide_index=True)

else:
    st.success("No high stockout SKUs found.")

st.markdown("---")


# MARKDOWN LIST
st.header("Prioritized Markdown List")
markdown_df = filtered_df[filtered_df["risk_level"] == "Overstock"].copy()

if not markdown_df.empty:
    markdown_df["excess_units"] = (available_inventory.loc[markdown_df.index] - (forecast_for_risk.loc[markdown_df.index] * 1.5)).clip(lower=0)
    markdown_df = (markdown_df.sort_values("excess_units", ascending=False).head(20))
    markdown_columns = ["sku_id", "category"]

    if "forecast_units" in markdown_df.columns:
        markdown_columns.append("forecast_units")

    if "on_hand_units" in markdown_df.columns:
        markdown_columns.append("on_hand_units")

    if "on_order_units" in markdown_df.columns:
        markdown_columns.append("on_order_units")

    markdown_columns.append("excess_units")
    markdown_display = (markdown_df[markdown_columns].copy())

    for col in markdown_display.columns:
        if col not in ["sku_id", "category"]:
            markdown_display[col] = pd.to_numeric(markdown_display[col], errors="coerce")

    st.dataframe(markdown_display, use_container_width=True, hide_index=True)

else:
    st.success("No overstock SKUs found.")

st.markdown("---")


# INVENTORY VS FORECAST
st.header("Inventory vs Forecast")
if ("forecast_units" in filtered_df.columns and "on_hand_units" in filtered_df.columns):
    comparison_df = pd.DataFrame({
        "Metric": ["Forecast Demand", "On-Hand Inventory", "On-Order Inventory"],
        "Units": [filtered_df["forecast_units"].fillna(0).sum(),
            filtered_df["on_hand_units"].fillna(0).sum(),
            filtered_df["on_order_units"].fillna(0).sum()
            if "on_order_units" in filtered_df.columns
            else 0
        ]
    })

    comparison_df["Units"] = pd.to_numeric(comparison_df["Units"], errors="coerce")
    comparison_fig = px.bar(comparison_df, x="Metric", y="Units", text_auto=True, title="Forecast Demand vs Available Inventory")
    comparison_fig.update_layout(xaxis_title="Metric", yaxis_title="Units")
    st.plotly_chart(comparison_fig, use_container_width=True)

st.markdown("---")


# DASHBOARD DATA
st.header("Dashboard Data")
display_df = filtered_df.copy()

for col in display_df.columns:
    if col not in ["sku_id", "category", "risk_level"]:
        if display_df[col].dtype == "object":
            cleaned = (display_df[col].astype(str).str.replace(",", "", regex=False).str.replace("₹", "", regex=False).str.strip())
            converted = pd.to_numeric(cleaned, errors="coerce")
            if converted.notna().mean() >= 0.80:
                display_df[col] = converted

st.dataframe(display_df, use_container_width=True, hide_index=True)
st.markdown("---")


# DOWNLOAD CSV
st.header("Download Dashboard Data")
csv_data = (display_df.to_csv(index=False).encode("utf-8"))
st.download_button(label="Download Dashboard CSV", data=csv_data, file_name="foresight_dashboard.csv", mime="text/csv")
st.markdown("---")


# FOOTER
st.caption("Project FORESIGHT | AI-Powered Demand Forecasting & Inventory Intelligence")