import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os


# PROJECT PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "processed_data.csv")
RISK_DATA_PATH = os.path.join(DATA_DIR, "risk_analysis", "sku_risk_analysis.csv")
DECISION_GRID_PATH = os.path.join(DATA_DIR, "decisioning_grid", "decisioning_grid.csv")
STYLE_PATH = os.path.join(BASE_DIR, "style.css")


# PAGE CONFIGURATION
st.set_page_config(page_title="Project FORESIGHT Dashboard", page_icon=" ", layout="wide")


# CUSTOM CSS
def load_css():
    if os.path.exists(STYLE_PATH):
        with open(STYLE_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css()


# SIDEBAR
st.sidebar.title("PROJECT FORESIGHT")
st.sidebar.markdown("---")
st.sidebar.success("Demand Forecasting & Inventory Intelligence")
st.sidebar.markdown("---")
dark_mode = st.sidebar.toggle("Dark Mode", value=False)
st.sidebar.markdown("---")
st.sidebar.header("Filters")


# LOAD PROCESSED DATA
if os.path.exists(PROCESSED_DATA_PATH):
    processed_df = pd.read_csv(PROCESSED_DATA_PATH)

else:
    st.error("processed_data.csv not found.")
    st.stop()


# LOAD RISK ANALYSIS
if os.path.exists(RISK_DATA_PATH):
    risk_df = pd.read_csv(RISK_DATA_PATH)

else:
    st.error("sku_risk_analysis.csv not found.")
    st.stop()


# LOAD DECISIONING GRID
if os.path.exists(DECISION_GRID_PATH):
    decision_df = pd.read_csv(DECISION_GRID_PATH)

else:
    st.warning(
        "decisioning_grid.csv not found. "
        "Decisioning Grid will be generated automatically."
    )
    decision_df = risk_df.copy()


# PREPARE PROCESSED DATA
processed_df["date"] = pd.to_datetime(processed_df["date"], errors="coerce")
processed_df = processed_df.sort_values("date")


# GET LATEST SKU INFORMATION
latest_sku_df = (processed_df.sort_values("date").groupby("sku_id", as_index=False).tail(1))


# ADD CATEGORY INFORMATION
if "category" in latest_sku_df.columns:
    category_df = latest_sku_df[["sku_id", "category"]].drop_duplicates(subset=["sku_id"])
    if "category" not in risk_df.columns:
        risk_df = risk_df.merge(category_df, on="sku_id", how="left")

    if "category" not in decision_df.columns:
        decision_df = decision_df.merge(category_df, on="sku_id", how="left")


# FORECAST UNITS
if "forecast_units" not in risk_df.columns:
    if "Average_Daily_Demand" in risk_df.columns:
        risk_df["forecast_units"] = (risk_df["Average_Daily_Demand"] * 7)

    elif "Average_Daily_Demand" in decision_df.columns:
        risk_df["forecast_units"] = (risk_df["Average_Daily_Demand"] * 7)

    else:
        risk_df["forecast_units"] = 0


# SIDEBAR SKU SEARCH
st.sidebar.subheader("SKU Search")
sku_search = st.sidebar.text_input("Enter SKU ID", placeholder="Example: 204")


# CATEGORY FILTER
st.sidebar.subheader("Category Filter")
if "category" in risk_df.columns:
    categories = sorted(risk_df["category"].dropna().astype(str).unique().tolist())
    selected_category = st.sidebar.selectbox("Select Category", ["All"] + categories)

else:
    selected_category = "All"


# APPLY FILTERS
filtered_risk_df = risk_df.copy()
filtered_decision_df = decision_df.copy()
# Category Filter
if selected_category != "All":
    if "category" in filtered_risk_df.columns:
        filtered_risk_df = filtered_risk_df[filtered_risk_df["category"].astype(str) == selected_category]

    if "category" in filtered_decision_df.columns:
        filtered_decision_df = filtered_decision_df[filtered_decision_df["category"].astype(str) == selected_category]


# SKU Search
if sku_search.strip():
    search_text = sku_search.strip()
    filtered_risk_df = filtered_risk_df[filtered_risk_df["sku_id"].astype(str).str.contains(search_text, case=False, na=False)]
    filtered_decision_df = filtered_decision_df[filtered_decision_df["sku_id"].astype(str).str.contains(search_text, case=False, na=False)]


# EMPTY DATA CHECK
if filtered_risk_df.empty:
    st.warning("No SKU found for the selected filters.")
    st.stop()


# KPI CALCULATIONS
total_skus = len(filtered_risk_df)
total_sales_at_risk = (pd.to_numeric(filtered_risk_df["Sales_at_Risk"], errors="coerce").fillna(0).sum())
total_capital_locked = (pd.to_numeric( filtered_risk_df["Capital_Locked"], errors="coerce").fillna(0).sum())
high_risk_count = (filtered_risk_df["Risk_Level"].astype(str).str.upper().eq("HIGH").sum())
medium_risk_count = (filtered_risk_df["Risk_Level"].astype(str).str.upper().eq("MEDIUM").sum())
low_risk_count = (filtered_risk_df["Risk_Level"].astype(str).str.upper().eq("LOW").sum())


# DASHBOARD HEADER
st.title("PROJECT FORESIGHT")
st.caption("Demand Forecasting & Inventory Intelligence Dashboard")
st.markdown("---")


# KPI CARDS
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("SKUs Analysed", f"{total_skus:,}")

with c2:
    st.metric("Sales at Risk", f"{total_sales_at_risk:,.0f}")

with c3:
    st.metric("Capital Locked", f"{total_capital_locked:,.0f}")

with c4:
    st.metric("HIGH Risk", f"{high_risk_count:,}")

with c5:
    st.metric("MEDIUM Risk", f"{medium_risk_count:,}")
st.markdown("---")


# 1. BUSINESS INSIGHTS
st.header("Business Insights")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Sales at Risk", f"{total_sales_at_risk:,.2f}")

with col2:
    st.metric("Total Capital Locked", f"{total_capital_locked:,.2f}")


# RISK DISTRIBUTION
st.subheader("Risk Level Distribution")
risk_distribution = (filtered_risk_df["Risk_Level"].value_counts().reset_index())
risk_distribution.columns = ["Risk Level", "SKU Count"]
risk_fig = px.bar(risk_distribution, x="Risk Level", y="SKU Count", text_auto=True, title="SKU Risk Distribution")
risk_fig.update_layout(height=400)
st.plotly_chart(risk_fig, use_container_width=True)
st.markdown("---")


# 2. RISKY SKUs
st.header("Top 10 Risky SKUs")
top_risky = (filtered_risk_df.sort_values("Sales_at_Risk", ascending=False).head(10))


top_columns = [
    "sku_id",
    "Days_of_Supply",
    "Stockout_Risk_Score",
    "Overstock_Risk_Score",
    "Risk_Level",
    "Primary_Risk",
    "Sales_at_Risk",
    "Capital_Locked",
    "Recommended_Action"
]

top_columns = [col for col in top_columns if col in top_risky.columns]
st.dataframe(top_risky[top_columns], use_container_width=True, hide_index=True)
st.markdown("---")


# 3. FOUR-QUADRANT DECISIONING GRID
st.header("Decisioning Grid")
st.info("Each SKU is positioned according to Stockout Risk" "and Overstock Risk. Bubble size represents Sales at Risk.")


# CHECK REQUIRED COLUMNS
required_grid_columns = ["sku_id", "Stockout_Risk_Score", "Overstock_Risk_Score"]
missing_columns = [col for col in required_grid_columns if col not in filtered_decision_df.columns]


if missing_columns:
    st.error(
        "Decisioning Grid cannot be displayed."
        f"Missing columns: {missing_columns}"
    )

else:
    grid_df = filtered_decision_df.copy()


    # Numeric conversion
    grid_df["Stockout_Risk_Score"] = pd.to_numeric(grid_df["Stockout_Risk_Score"], errors="coerce").fillna(0)
    grid_df["Overstock_Risk_Score"] = pd.to_numeric(grid_df["Overstock_Risk_Score"], errors="coerce").fillna(0)
    if "Sales_at_Risk" in grid_df.columns:
        grid_df["Sales_at_Risk"] = pd.to_numeric(grid_df["Sales_at_Risk"], errors="coerce").fillna(0)

    else:
        grid_df["Sales_at_Risk"] = 1


    # Decisioning Quadrant
    def classify_quadrant(row):
        stockout = row["Stockout_Risk_Score"]
        overstock = row["Overstock_Risk_Score"]

        if stockout >= 0.75 and overstock < 0.75:
            return "Reorder Now"

        elif stockout < 0.75 and overstock >= 0.75:
            return "Markdown / Clear"

        elif stockout >= 0.75 and overstock >= 0.75:
            return "Watch / Volatile"

        else:
            return "Healthy"

    grid_df["Decision_Quadrant"] = grid_df.apply(classify_quadrant, axis=1)


    # Decision Actions
    action_map = {
        "Reorder Now": "Urgently replenish stock",
        "Markdown / Clear": "Run promotions / reduce excess inventory",
        "Watch / Volatile": "Investigate demand volatility",
        "Healthy": "Maintain current inventory"
    }
    grid_df["Decision_Action"] = (grid_df["Decision_Quadrant"].map(action_map))


    # Quadrant Distribution
    quadrant_order = ["Reorder Now", "Markdown / Clear", "Watch / Volatile", "Healthy"]
    quadrant_counts = (grid_df["Decision_Quadrant"].value_counts().reindex(quadrant_order, fill_value=0))


    # Four KPI Cards
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.metric("Reorder Now", int(quadrant_counts["Reorder Now"]))

    with q2:
        st.metric("Markdown / Clear", int(quadrant_counts["Markdown / Clear"]))

    with q3:
        st.metric("Watch / Volatile", int(quadrant_counts["Watch / Volatile"]))

    with q4:
        st.metric("Healthy", int(quadrant_counts["Healthy"]))
    st.markdown("###Stockout vs Overstock Risk")


    # Plotly Decisioning Grid
    grid_fig = px.scatter(grid_df,
        x="Stockout_Risk_Score",
        y="Overstock_Risk_Score",
        size="Sales_at_Risk",
        color="Decision_Quadrant",
        hover_name="sku_id",
        hover_data={"Stockout_Risk_Score": ":.2f", "Overstock_Risk_Score": ":.2f", "Sales_at_Risk": ":,.2f", "Decision_Action": True},
        category_orders={"Decision_Quadrant": quadrant_order},
        title=("SKU Decisioning Grid — " "Stockout vs Overstock Risk")
    )


    # Quadrant Boundary Lines
    grid_fig.add_vline(x=0.75, line_dash="dash")
    grid_fig.add_hline(y=0.75, line_dash="dash")


    # Quadrant Labels
    grid_fig.add_annotation(x=0.375, y=0.875, text="MARKDOWN / CLEAR", showarrow=False)
    grid_fig.add_annotation(x=0.375, y=0.375, text="HEALTHY", showarrow=False)
    grid_fig.add_annotation(x=0.875, y=0.875, text="WATCH / VOLATILE", showarrow=False)
    grid_fig.add_annotation(x=0.875, y=0.375, text="REORDER NOW", showarrow=False)

    grid_fig.update_layout(
        xaxis_title="Stockout Risk Score",
        yaxis_title="Overstock Risk Score",
        xaxis=dict(range=[0, 1.05]),
        yaxis=dict( range=[0, 1.05]),
        height=650, legend_title="Decision Quadrant"
    )

    st.plotly_chart(grid_fig, use_container_width=True)


    # Decisioning Grid Explanation
    st.subheader("Decisioning Rules")
    rule_df = pd.DataFrame({
        "Quadrant": ["Reorder Now", "Markdown / Clear", "Watch / Volatile", "Healthy"],
        "Condition": [ "High Stockout + Low Overstock", "Low Stockout + High Overstock", "High Stockout + High Overstock", "Low Stockout + Low Overstock"],
        "Recommended Action": ["Urgently replenish stock", "Run promotions / reduce inventory", "Investigate demand volatility", "Maintain current inventory"]
    })


    st.dataframe(rule_df,use_container_width=True,hide_index=True)


    # Decisioning Grid Table
    st.subheader("SKU Decisioning Details")

    display_columns = ["sku_id", "Stockout_Risk_Score", "Overstock_Risk_Score", "Decision_Quadrant", "Decision_Action", "Sales_at_Risk", "Capital_Locked"]
    display_columns = [col for col in display_columns if col in grid_df.columns]
    display_grid = grid_df[display_columns].copy()


    if "Stockout_Risk_Score" in display_grid.columns:
        display_grid["Stockout_Risk_Score"] = display_grid["Stockout_Risk_Score"].round(2)


    if "Overstock_Risk_Score" in display_grid.columns:
        display_grid["Overstock_Risk_Score"] = display_grid["Overstock_Risk_Score"].round(2)


    if "Sales_at_Risk" in display_grid.columns:
        display_grid["Sales_at_Risk"] = display_grid["Sales_at_Risk"].round(2)


    if "Capital_Locked" in display_grid.columns:
        display_grid["Capital_Locked"] = display_grid["Capital_Locked"].round(2)
    st.dataframe(display_grid, use_container_width=True, hide_index=True)
st.markdown("---")


# 4. INVENTORY RISK
st.header("Inventory Risk Analysis")
risk_chart_df = (filtered_risk_df[["sku_id", "Stockout_Risk_Score", "Overstock_Risk_Score"]].melt(id_vars="sku_id", var_name="Risk Type", value_name="Risk Score"))
risk_chart_df["Risk Type"] = (risk_chart_df["Risk Type"].replace({"Stockout_Risk_Score": "Stockout Risk", "Overstock_Risk_Score": "Overstock Risk"}))
risk_fig = px.bar(risk_chart_df, x="sku_id", y="Risk Score", color="Risk Type", barmode="group", title="Stockout vs Overstock Risk by SKU")
risk_fig.update_layout(xaxis_title="SKU", yaxis_title="Risk Score", height=500)
st.plotly_chart(risk_fig, use_container_width=True)
st.markdown("---")



# 5. RISK DETAILS
st.header("SKU Risk Details")
risk_columns = ["sku_id", "Days_of_Supply", "Stockout_Risk_Score", "Overstock_Risk_Score", "Risk_Level", "Primary_Risk", "Recommended_Action", "Sales_at_Risk", "Capital_Locked"]

risk_columns = [col for col in risk_columns if col in filtered_risk_df.columns]
risk_display = filtered_risk_df[risk_columns].copy()
st.dataframe(risk_display, use_container_width=True, hide_index=True)
st.markdown("---")


# 6. FORECAST INFORMATION
st.header("Demand Forecast")
if "forecast_units" in filtered_risk_df.columns:
    forecast_df = (filtered_risk_df[["sku_id", "forecast_units"]].sort_values("forecast_units", ascending=False).head(20))
    forecast_fig = px.bar(forecast_df, x="sku_id", y="forecast_units", text_auto=True, title="Forecast Demand by SKU")
    forecast_fig.update_layout(xaxis_title="SKU", yaxis_title="Forecast Units", height=500)
    st.plotly_chart(forecast_fig, use_container_width=True)


else:
    st.info("Forecast data is not available in the risk dataset.")
st.markdown("---")


# 7. BUSINESS RECOMMENDATIONS
st.header("Business Recommendations")
st.markdown("""
    - Prioritize **Reorder Now** SKUs.
    - Monitor **₹ Sales at Risk** for stockout decisions.
    - Monitor **₹ Capital Locked** for excess inventory.
    - Use **Markdown / Clear** for overstocked SKUs.
    - Investigate **Watch / Volatile** SKUs before taking action.
    - Maintain inventory for **Healthy** SKUs.
    - Use the demand forecast for procurement planning.
    - Review forecast uncertainty before major inventory decisions.
    - Retrain the forecasting model periodically with updated sales data.
""")
st.markdown("---")


# 8. DOWNLOAD DECISIONING GRID
st.header("Download Reports")
decision_csv = grid_df.to_csv(index=False).encode("utf-8")
st.download_button(label="Download Decisioning Grid CSV", data=decision_csv, file_name="decisioning_grid_dashboard.csv", mime="text/csv", use_container_width=True)
risk_csv = filtered_risk_df.to_csv(index=False).encode("utf-8")
st.download_button(label="Download Risk Analysis CSV", data=risk_csv, file_name="sku_risk_analysis_dashboard.csv", mime="text/csv", use_container_width=True)


st.markdown("---")


# DASHBOARD SUMMARY
st.header("Dashboard Summary")
st.success(
    f"""
    Total SKUs Analysed : {total_skus}
    Total Sales at Risk : {total_sales_at_risk:,.2f}
    Total Capital Locked : {total_capital_locked:,.2f}
    HIGH Risk SKUs : {high_risk_count}
    MEDIUM Risk SKUs : {medium_risk_count}
    LOW Risk SKUs : {low_risk_count}
    Reorder Now : {int(quadrant_counts["Reorder Now"])}
    Markdown / Clear : {int(quadrant_counts["Markdown / Clear"])}
    Watch / Volatile : {int(quadrant_counts["Watch / Volatile"])}
    Healthy : {int(quadrant_counts["Healthy"])}
    """
)


# FOOTER
st.markdown("---")
st.caption("Project FORESIGHT" "AI-Powered Demand Forecasting & Inventory Intelligence")