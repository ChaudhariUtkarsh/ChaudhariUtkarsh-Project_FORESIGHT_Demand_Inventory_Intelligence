import streamlit as st

def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# Page Configuration
st.set_page_config(page_title="About Project", page_icon=" ", layout="wide")


# Header
st.title("About Project FORESIGHT")
st.markdown("""AI-Powered Demand Forecasting & Inventory Intelligence System""")
st.markdown("---")


# Project Overview
st.header("Project Overview")
st.write("""
    Project FORESIGHT is an Artificial Intelligence based Demand Forecasting and Inventory Intelligence platform.
    The system predicts future product demand using Machine Learning algorithms and helps businesses identify Stockout Risk, Overstock Risk, Revenue at Risk and Capital Locked.
    The project enables data-driven inventory planning, improves supply chain efficiency and supports better business decision making.
""")

st.markdown("---")


# Objectives
st.header("Project Objectives")
st.success("""
    1. Improve Forecast Accuracy
    2. Reduce Stockout Risk
    3. Reduce Overstock Cost
    4. Increase Inventory Visibility
    5. Improve Supply Chain Planning
    6. Support Business Decisions
    7. Increase Profitability
""")

st.markdown("---")


# Key Features
st.header("Key Features")
c1, c2 = st.columns(2)

with c1:
    st.info("""
        ### Forecasting
        1. Demand Prediction
        2. Weekly Forecast
        3. Monthly Forecast
        4. SKU-wise Forecast
        5. Forecast Dashboard
    """)

with c2:

    st.info("""
        ### Inventory Intelligence
        1. Risk Scoring
        2. Stockout Detection
        3. Overstock Detection
        4. Revenue at Risk
        5. Capital Locked
    """)

st.markdown("---")


# Technology Stack
st.header("Technology Stack")
tech1, tech2, tech3 = st.columns(3)

with tech1:
    st.write("""
        ### Programming
        - Python
        - Pandas
        - NumPy
    """)

with tech2:
    st.write("""
        ### Machine Learning
        - XGBoost
        - LightGBM
        - Scikit-Learn
    """)

with tech3:
    st.write("""
        ### Dashboard
        - Streamlit
        - Plotly
        - Matplotlib
    """)

st.markdown("---")


# Machine Learning Models
st.header("Machine Learning Models")
models = {
    "Baseline Model":
    "Seasonal Naive Forecast",

    "Model 1":
    "XGBoost Regressor",

    "Model 2":
    "LightGBM Regressor",

    "Evaluation":
    "MAE, RMSE, MAPE, WAPE"
}

st.table(models)
st.markdown("---")


# Project Workflow
st.header("Project Workflow")
workflow = [
    "1. Load Dataset",
    "2. Data Cleaning",
    "3. Feature Engineering",
    "4. Train ML Models",
    "5. Evaluate Models",
    "6. Demand Forecast",
    "7. Risk Scoring",
    "8. Dashboard",
    "9. Business Report"
]

for step in workflow:
    st.write(step)
st.markdown("---")


# Business Benefits
st.header("Business Benefits")
st.write("""
    1. Improve Forecast Accuracy
    2. Reduce Inventory Cost
    3. Reduce Revenue Loss
    4.Reduce Capital Locking
    5. Better Procurement Planning
    6. Better Warehouse Management
    7. Better Business Decisions
""")
st.markdown("---")


# Dashboard Modules
st.header("Dashboard Modules")
modules = [
    "Home",
    "Dashboard",
    "Forecast",
    "Risk Scoring",
    "About"
]

for module in modules:
    st.write(module)

st.markdown("---")


# Project Information
st.header("Project Information")
col1, col2 = st.columns(2)

with col1:
    st.metric("Project Version", "1.0")
    st.metric("Dashboard", "Streamlit")

with col2:
    st.metric("ML Models", "2")
    st.metric("Forecast Horizon", "7-30 Days")

st.markdown("---")


# Developer
st.header("Developer")
st.write("""
    **Project Name**
    Project FORESIGHT

    **Domain**
    Demand Forecasting & Inventory Intelligence

    **Developed Using**
    Python, Streamlit,
    Machine Learning,
    XGBoost,
    LightGBM,
    Plotly,
    Scikit-Learn
""")

st.markdown("---")


# Footer
st.caption("Project FORESIGHT | AI-Powered Demand Forecasting & Inventory Intelligence | Version 1.0")