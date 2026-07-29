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
st.set_page_config(page_title="Project FORESIGHT Dashboard", page_icon=" ", layout="wide")


# Sidebar
st.sidebar.title("Project FORESIGHT")
st.sidebar.markdown("---")
st.sidebar.success("Demand Forecasting Dashboard")
st.sidebar.markdown("---")

# Dark Mode Toggle
dark_mode = st.sidebar.toggle("Dark Mode", value=False)
st.sidebar.markdown("---")
st.sidebar.header("Filters")


# Load Dataset
DATA_PATH = r"E:\Zidio Development\Project_FORESIGHT\data\processed/processed_data.csv"
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    st.warning("Processed Dataset not found.")

    df = pd.DataFrame({
        "sku_id":["SKU001", "SKU002", "SKU003","SKU004", "SKU005"],
        "category":["Electronics", "Electronics", "Furniture", "Furniture", "Grocery"],
        "forecast_units":[120, 80, 160, 90, 230],
        "selling_price":[450, 900, 150, 200, 45],
        "unit_cost":[300, 650, 120, 150, 30],
        "on_hand_units":[50, 140, 250, 80,60],
        "risk_level":["High", "Overstock", "Medium", "High", "Overstock"]})


# SKU Search
st.sidebar.subheader("SKU Search")
sku = st.sidebar.text_input("Enter SKU ID", placeholder="Example: SKU001")
filtered_df = df.copy()


# Category Filter
st.sidebar.subheader("Category Filter")

# Get available categories safely
if "category" in df.columns:
    categories = sorted(df["category"].dropna().astype(str).unique().tolist())
    category = st.sidebar.selectbox("Select Category", ["All"] + categories)

else:
    st.sidebar.error("Category column not found in dataset.")
    category = "All"


# Apply Category Filter
filtered_df = df.copy()
if category != "All":
    filtered_df = filtered_df[filtered_df["category"].astype(str) == category]


# Filter Result
st.sidebar.markdown("---")
st.sidebar.write(f"Products Found: {len(filtered_df)}")

# SKU Search
if sku.strip():
    filtered_df = filtered_df[filtered_df["sku_id"].astype(str).str.contains(sku.strip(), case=False, na=False)]

# No Result Handling
if filtered_df.empty:
    st.warning(f"No SKU found matching: {sku}")
filtered_df = df.copy()

if category != "All":
    filtered_df = filtered_df[filtered_df["category"] == category]

if sku != "":
    filtered_df = filtered_df[filtered_df["sku_id"].str.contains(sku, case=False)]


# KPI Calculation
revenue_at_risk = (
    filtered_df.loc[filtered_df["risk_level"]=="High", "forecast_units"] *
    filtered_df.loc[filtered_df["risk_level"]=="High", "selling_price"]
).sum()

capital_locked = (
    filtered_df.loc[filtered_df["risk_level"]=="Overstock", "on_hand_units"] *
    filtered_df.loc[filtered_df["risk_level"]=="Overstock", "unit_cost"]
).sum()

total_stockout = (filtered_df["risk_level"]=="High").sum()
total_overstock = (filtered_df["risk_level"]=="Overstock").sum()
total_products = len(filtered_df)
forecast_units = filtered_df["forecast_units"].sum()


# Dashboard Header
st.title("Business Dashboard")
st.caption("Demand Forecasting & Inventory Intelligence")
st.markdown("---")


# KPI Cards
c1,c2,c3,c4,c5,c6 = st.columns(6)
with c1:
    st.metric("Products", total_products)

with c2:
    st.metric("Forecast Units", int(forecast_units))

with c3:
    st.metric("Revenue at Risk", f"₹ {revenue_at_risk:,.0f}")

with c4:
    st.metric("Capital Locked", f"₹ {capital_locked:,.0f}")

with c5:
    st.metric("Stockout", int(total_stockout))

with c6:
    st.metric("Overstock", int(total_overstock))

st.markdown("---")


# Forecast Trend
st.header("Demand Forecast")

forecast_df = (filtered_df.groupby("sku_id", as_index=False)["forecast_units"].sum())
forecast_fig = px.bar(forecast_df, x="sku_id", y="forecast_units", title="Forecast Units by SKU", text_auto=True)
forecast_fig.update_layout(xaxis_title="SKU", yaxis_title="Forecast Units", height=450)
st.plotly_chart( forecast_fig, use_container_width=True)
st.markdown("---")


# Revenue Graph
st.header("Revenue Analysis")

filtered_df["forecast_revenue"] = (filtered_df["forecast_units"] * filtered_df["selling_price"])
revenue_df = (filtered_df.groupby("category", as_index=False)["forecast_revenue"].sum())
revenue_fig = px.bar(revenue_df, x="category", y="forecast_revenue", title="Forecast Revenue by Category", text_auto=True)
revenue_fig.update_layout(height=450)
st.plotly_chart(revenue_fig, use_container_width=True)
st.markdown("---")


# Inventory Graph
st.header("Inventory Status")

inventory_df = (filtered_df.groupby("category", as_index=False)["on_hand_units"].sum())
inventory_fig = px.pie(inventory_df, names="category", values="on_hand_units", title="Inventory Distribution")
st.plotly_chart(inventory_fig, use_container_width=True)
st.markdown("---")


# Risk Distribution
st.header("Inventory Risk")

risk_df = (filtered_df["risk_level"].value_counts().reset_index())
risk_df.columns = ["Risk", "Count"]
risk_fig = px.bar(risk_df, x="Risk", y="Count", color="Risk", text_auto=True, title="Risk Distribution")
risk_fig.update_layout(height=450)
st.plotly_chart(risk_fig, use_container_width=True)
st.markdown("---")


# ==========================================================
# Stockout Risk Graph
# ==========================================================

st.markdown("---")
st.header("Stockout Risk Analysis")

# Check required columns
required_columns = ["sku_id", "forecast_units", "on_hand_units"]

if all(column in filtered_df.columns for column in required_columns):

    # Create Stockout Risk DataFrame
    stockout_df = filtered_df[["sku_id", "forecast_units", "on_hand_units"]].copy()

    # Calculate shortage
    stockout_df["shortage_units"] = (stockout_df["forecast_units"] - stockout_df["on_hand_units"])

    # Only stockout / shortage products
    stockout_df["stockout_units"] = (stockout_df["shortage_units"].clip(lower=0))

    # Stockout Risk Percentage
    stockout_df["stockout_risk_pct"] = np.where(
        stockout_df["forecast_units"] > 0,
        (stockout_df["stockout_units"] / stockout_df["forecast_units"]) * 100, 0
    )

    # Risk Level
    stockout_df["risk_level"] = np.select(
        [
            stockout_df["stockout_risk_pct"] >= 50,
            stockout_df["stockout_risk_pct"] >= 20,
            stockout_df["stockout_risk_pct"] > 0
        ],
        ["High", "Medium", "Low"], default="No Risk"
    )

    # Sort highest risk first
    stockout_df = stockout_df.sort_values("stockout_risk_pct", ascending=False)

    # Stockout Risk Graph
    stockout_fig = px.bar(
        stockout_df,
        x="sku_id",
        y="stockout_risk_pct",
        color="risk_level",
        text="stockout_risk_pct",
        title="Stockout Risk by SKU"
    )
    stockout_fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")

    stockout_fig.update_layout(
        xaxis_title="SKU",
        yaxis_title="Stockout Risk (%)", 
        yaxis=dict(range=[0, max(100, stockout_df["stockout_risk_pct"].max() + 10)]),  height=500, legend_title="Risk Level"
    )
    st.plotly_chart(stockout_fig, use_container_width=True)


    # Stockout Risk KPI
    high_risk = (stockout_df["risk_level"] == "High").sum()
    medium_risk = (stockout_df["risk_level"] == "Medium").sum()
    low_risk = (stockout_df["risk_level"] == "Low").sum()
    no_risk = (stockout_df["risk_level"] == "No Risk").sum()
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("High Risk", int(high_risk))

    with c2:
        st.metric("Medium Risk", int(medium_risk))

    with c3:
        st.metric("Low Risk", int(low_risk))

    with c4:
        st.metric("No Risk", int(no_risk))


    # Stockout Risk Table
    st.subheader("Stockout Risk Details")
    display_df = stockout_df[
        [
            "sku_id",
            "forecast_units",
            "on_hand_units",
            "stockout_units",
            "stockout_risk_pct",
            "risk_level"
        ]
    ].copy()

    display_df["stockout_risk_pct"] = (display_df["stockout_risk_pct"].round(2))
    st.dataframe(display_df, use_container_width=True, hide_index=True)

else:
    st.warning("Stockout Risk graph cannot be displayed.")
    st.info(
        "Dataset must contain: "
        "sku_id, forecast_units and on_hand_units."
    )

st.markdown("---")


# Overstock Risk Graph
st.markdown("---")
st.header("Overstock Risk Analysis")


# Required columns check
required_columns = ["sku_id", "forecast_units", "on_hand_units"]
if all(column in filtered_df.columns for column in required_columns):

    # Create Overstock Risk DataFrame
    overstock_df = filtered_df[["sku_id", "forecast_units", "on_hand_units"]].copy()

    # Calculate excess inventory
    overstock_df["excess_units"] = (overstock_df["on_hand_units"] - overstock_df["forecast_units"])

    # Only excess stock
    overstock_df["overstock_units"] = (overstock_df["excess_units"].clip(lower=0))

    # Overstock Risk Percentage
    overstock_df["overstock_risk_pct"] = np.where(
        overstock_df["forecast_units"] > 0,
        (overstock_df["overstock_units"] / overstock_df["forecast_units"]) * 100, 0
    )

    # Risk Level
    overstock_df["risk_level"] = np.select(
        [
            overstock_df["overstock_risk_pct"] >= 100,
            overstock_df["overstock_risk_pct"] >= 50,
            overstock_df["overstock_risk_pct"] > 0
        ],
        ["High", "Medium", "Low"], default="No Risk"
    )

    # Sort highest risk first
    overstock_df = overstock_df.sort_values("overstock_risk_pct", ascending=False)

    # Overstock Risk Graph
    overstock_fig = px.bar(
        overstock_df,
        x="sku_id",
        y="overstock_risk_pct",
        color="risk_level",
        text="overstock_risk_pct",
        title="Overstock Risk by SKU"
    )

    overstock_fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")

    overstock_fig.update_layout(
        xaxis_title="SKU",
        yaxis_title="Overstock Risk (%)",
        yaxis=dict(range=[0, max(100, overstock_df["overstock_risk_pct"].max() + 10)]),
        height=500,
        legend_title="Risk Level"
    )
    st.plotly_chart(overstock_fig, use_container_width=True)


    # Overstock Risk KPI Cards
    high_overstock = (overstock_df["risk_level"] == "High").sum()
    medium_overstock = (overstock_df["risk_level"] == "Medium").sum()
    low_overstock = (overstock_df["risk_level"] == "Low").sum()
    no_overstock = (overstock_df["risk_level"] == "No Risk").sum()
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("High Overstock", int(high_overstock))

    with c2:
        st.metric("Medium Overstock", int(medium_overstock))

    with c3:
        st.metric("Low Overstock", int(low_overstock))

    with c4:
        st.metric("No Overstock", int(no_overstock))


    # Overstock Risk Table
    st.subheader("Overstock Risk Details")
    overstock_display = overstock_df[
        [
            "sku_id",
            "forecast_units",
            "on_hand_units",
            "overstock_units",
            "overstock_risk_pct",
            "risk_level"
        ]
    ].copy()

    overstock_display["overstock_risk_pct"] = (overstock_display["overstock_risk_pct"].round(2))
    st.dataframe(overstock_display, use_container_width=True, hide_index=True)

else:
    st.warning("Overstock Risk graph cannot be displayed.")
    st.info(
        "Dataset must contain: "
        "sku_id, forecast_units and on_hand_units."
    )

st.markdown("---")


# Category Summary
st.header("Category Summary")

summary = (
    filtered_df
    .groupby("category")
    .agg(
        Forecast=("forecast_units", "sum"),
        Inventory=("on_hand_units", "sum"),
        Revenue=("forecast_revenue", "sum")
    )
    .reset_index()
)

st.dataframe(summary, use_container_width=True)
st.markdown("---")


# Top Forecast Products
st.header("Top Forecast Products")
top_products = (filtered_df.sort_values("forecast_units", ascending=False).head(10))
st.dataframe(top_products, use_container_width=True)
st.markdown("---")


# Forecast vs Actual Comparison
st.markdown("---")
st.header("Forecast vs Actual")

# Check required columns
if ("units_sold" in filtered_df.columns and "forecast_units" in filtered_df.columns):
    compare_df = filtered_df[["sku_id", "units_sold", "forecast_units"]].copy()

    # Group by SKU
    compare_df = (
        compare_df
        .groupby("sku_id", as_index=False)
        .agg(Actual=("units_sold", "sum"), Forecast=("forecast_units", "sum"))
    )

    # Convert to long format for Plotly
    chart_df = compare_df.melt(
        id_vars="sku_id",
        value_vars=["Actual", "Forecast"],
        var_name="Type",
        value_name="Units"
    )

    # Create Graph
    fig_compare = px.bar(
        chart_df,
        x="sku_id",
        y="Units",
        color="Type",
        barmode="group",
        text_auto=True,
        title="Forecast vs Actual Demand by SKU"
    )

    fig_compare.update_layout(
        xaxis_title="SKU",
        yaxis_title="Units",
        height=500,
        legend_title="Demand Type",
        hovermode="x unified"
    )

    st.plotly_chart(fig_compare, use_container_width=True)

    # Comparison Table
    st.subheader("Forecast vs Actual Summary")
    compare_df["Difference"] = (compare_df["Forecast"] - compare_df["Actual"])
    compare_df["Accuracy (%)"] = np.where(compare_df["Actual"] != 0, (1 - abs(compare_df["Forecast"] - compare_df["Actual"]) / compare_df["Actual"]) * 100,0)
    compare_df["Accuracy (%)"] = (compare_df["Accuracy (%)"].clip(0, 100).round(2))
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

else:
    st.warning("Forecast vs Actual graph cannot be displayed.")
    st.info(
            "Dataset must contain both "
            "'units_sold' and 'forecast_units' columns."
        )

st.markdown("---")


# Download Forecast CSV
st.header("Download Forecast")

csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(label="Download Forecast CSV", data=csv, file_name="forecast_results.csv", mime="text/csv")
st.markdown("---")


# Download Risk Report
st.markdown("---")
st.subheader("Download Risk Report")

# Risk Report Content
risk_report = f"""
----------------------------------------------------------
                PROJECT FORESIGHT
                  RISK REPORT
----------------------------------------------------------
Generated Date:
{pd.Timestamp.now().strftime("%d-%m-%Y %H:%M:%S")}

----------------------------------------------------------
                    KPI SUMMARY
----------------------------------------------------------

Total Products       : {total_products}
Forecast Units       : {forecast_units:,.0f}
Revenue at Risk      : {revenue_at_risk:,.2f}
Capital Locked       : {capital_locked:,.2f}
Total Stockout       : {total_stockout}
Total Overstock      : {total_overstock}

----------------------------------------------------------
                 RISK SUMMARY
----------------------------------------------------------

High Risk / Stockout Products : {total_stockout}
Overstock Products            : {total_overstock}

----------------------------------------------------------
             BUSINESS RECOMMENDATIONS
----------------------------------------------------------

1. Increase inventory for High Risk / Stockout SKUs.
2. Reduce excess inventory for Overstock products.
3. Review demand forecasts before procurement.
4. Monitor inventory levels regularly.
5. Maintain appropriate safety stock.

----------------------------------------------------------
                    PROJECT FORESIGHT
----------------------------------------------------------
AI-Powered Demand Forecasting & Inventory Intelligence System

----------------------------------------------------------
"""

# Download Button
st.download_button(
    label="Download Risk Report",
    data=risk_report,
    file_name="Risk_Report.txt",
    mime="text/plain",
    use_container_width=True
)

st.markdown("---")



# Business Report Download
st.header("Download Business Report")

report = f"""
-----------------------------------------

PROJECT FORESIGHT
BUSINESS REPORT

-----------------------------------------

Total Products : {total_products}
Forecast Units : {forecast_units}
Revenue at Risk : {revenue_at_risk:,.2f}
Capital Locked : {capital_locked:,.2f}
Total Stockout : {total_stockout}
Total Overstock : {total_overstock}

-----------------------------------------

Business Recommendation
    1. Increase inventory for High Risk SKUs.
    2. Reduce inventory for Overstock products.
    3. Review weekly forecast before procurement.
    4. Monitor inventory every week.

-----------------------------------------
"""

st.download_button(label="Download Business Report", data=report, file_name="Business_Report.txt", mime="text/plain")
st.markdown("---")


# Dashboard Summary
st.header("Dashboard Summary")

st.success(f"""
    Total Products : {total_products}
    Forecast Units : {forecast_units}
    Revenue at Risk :  {revenue_at_risk:,.2f}
    Capital Locked : {capital_locked:,.2f}
    High Risk Products : {total_stockout}
    Overstock Products : {total_overstock}
""")

st.markdown("---")


# Footer
st.caption("Project FORESIGHT | AI-Powered Demand Forecasting & Inventory Intelligence")