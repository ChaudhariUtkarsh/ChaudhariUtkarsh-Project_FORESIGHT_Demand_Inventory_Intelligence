import streamlit as st
import os
import pandas as pd

# PAGE CONFIGURATION
st.set_page_config(page_title="Project FORESIGHT", page_icon=" ", layout="wide", initial_sidebar_state="expanded")


# LOAD CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")

    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()



# SIDEBAR
st.sidebar.title("Project FORESIGHT")
st.sidebar.info("Demand Forecasting & Inventory Intelligence System")

st.sidebar.markdown("---")
st.sidebar.success("Navigation")
st.sidebar.write("Home")
st.sidebar.write("Dashboard")
st.sidebar.write("Forecast")
st.sidebar.write("Risk Scoring")
st.sidebar.write("About")
st.sidebar.markdown("---")
st.sidebar.caption("Version 1.0")


# HEADER
st.title("Project FORESIGHT")
st.subheader("AI-Powered Demand Forecasting & Inventory Intelligence")

st.markdown("---")

st.write(
    """
    Project FORESIGHT helps businesses forecast future demand,
    identify inventory risks, reduce stockouts, minimise overstock,
    and improve supply chain decisions using Machine Learning.
    """
)


# LOAD BUSINESS DATA
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RISK_FILE = os.path.join(BASE_DIR, "data", "risk_analysis", "sku_risk_analysis.csv")
risk_df = pd.read_csv(RISK_FILE)
total_products = risk_df["sku_id"].nunique()
total_sales_at_risk = risk_df["Sales_at_Risk"].sum()
total_capital_locked = risk_df["Capital_Locked"].sum()


# BUSINESS OVERVIEW
st.header("Business Overview")
col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric("Products", f"{total_products:,}")


with col2:
    st.metric("Sales at Risk", f"₹{total_sales_at_risk:,.0f}")


with col3:
    st.metric("Capital Locked", f"{total_capital_locked:,.0f}")


with col4:
    st.metric("Forecast Horizon", "6–8 Weeks")


st.markdown("---")


# FORECAST MODEL OVERVIEW
st.header("Forecast Model Overview")
col1, col2, col3 = st.columns(3)


with col1:
    st.metric("Forecast Level", "SKU + Weekly")


with col2:
    st.metric("Best Model", "LightGBM")


with col3:
    st.metric("Baseline WAPE", "53.60%")


st.markdown("---")



# MODEL PERFORMANCE
st.header("Model Performance")
performance_col1, performance_col2 = st.columns(2)


with performance_col1:
    st.success("LightGBM")
    st.write(
        """
        **CV WAPE:** 40.40%
        **Baseline WAPE:** 53.60%
        **Improvement:** 13.20 percentage points
        """
    )


with performance_col2:
    st.info("XGBoost")
    st.write(
        """
        **CV WAPE:** 40.48%
        **Baseline WAPE:** 53.60%
        **Improvement:** 13.12 percentage points
        """
    )


st.markdown("---")


# PROJECT FEATURES
st.header("Project Features")
c1, c2 = st.columns(2)


with c1:
    st.success("Demand Forecasting")
    st.write(
        """
        - Weekly 6–8 Week Forecast
        - SKU-level Prediction
        - Historical Demand Analysis
        - Forecast Trend Analysis
        """
    )


    st.success("Inventory Intelligence")
    st.write(
        """
        - Stockout Detection
        - Overstock Detection
        - Risk Classification
        - Inventory Monitoring
        """
    )


with c2:
    st.success("Business Analytics")
    st.write(
        """
        - Sales at Risk
        - Capital Locked
        - Business KPIs
        - Executive Dashboard
        """
    )


    st.success("Machine Learning")
    st.write(
        """
        - XGBoost
        - LightGBM
        - Seasonal-Naive Baseline
        - Rolling-Origin CV
        - WAPE Evaluation
        """
    )

st.markdown("---")


# PROJECT WORKFLOW
st.header("Project Workflow")
workflow = [
    "Load Raw Dataset",
    "Data Cleaning",
    "Weekly SKU-Level Aggregation",
    "Feature Engineering",
    "Seasonal-Naive Baseline",
    "Demand Forecasting",
    "Rolling-Origin Model Evaluation",
    "WAPE Comparison",
    "Inventory Risk Scoring",
    "Business Recommendation",
    "Dashboard Visualisation"
]


for i, step in enumerate(workflow, start=1):
    st.write(f"**{i}. {step}**")


st.markdown("---")


# BUSINESS OBJECTIVES
st.header("Business Objectives")
objectives = [
    "Improve Forecast Accuracy",
    "Reduce Stockout Risk",
    "Reduce Overstock Cost",
    "Improve Inventory Planning",
    "Increase Business Profitability",
    "Support Data-Driven Decision Making"
]


for objective in objectives:
    st.write(f"{objective}")


st.markdown("---")


# TECHNOLOGY STACK
st.header("Technology Stack")
tech1, tech2, tech3 = st.columns(3)


with tech1:
    st.info(
        """
        **Data & Programming**
        - Python
        - Pandas
        - NumPy
        """
    )


with tech2:
    st.info(
        """
        **Machine Learning**
        - XGBoost
        - LightGBM
        - Scikit-Learn
        """
    )


with tech3:
    st.info(
        """
        **Dashboard & Visualisation**
        - Streamlit
        - Plotly
        - Matplotlib
        """
    )
st.markdown("---")


# PROJECT DELIVERABLES
st.header("Project Deliverables")
deliverables = [
    "Weekly Demand Forecasting",
    "Seasonal-Naive Baseline",
    "Rolling-Origin Cross Validation",
    "WAPE Model Evaluation",
    "Inventory Risk Scoring",
    "Business Recommendations",
    "Interactive Dashboard",
    "Executive Reports"
]


for item in deliverables:
    st.write(f"{item}")

st.markdown("---")


# PROJECT OVERVIEW
st.header("Project Overview")
st.write(
    """
    Project FORESIGHT is a Demand & Inventory Intelligence
    system designed to provide weekly SKU-level demand forecasts,
    evaluate model performance against a seasonal-naive baseline,
    identify inventory risks, and support business decision-making.
    """
)
st.info("Use the sidebar to navigate through Dashboard, Forecast and Risk Scoring.")


# FOOTER
st.markdown("---")
st.caption("Developed for Project FORESIGHT | " "AI-Powered Demand & Inventory Intelligence")