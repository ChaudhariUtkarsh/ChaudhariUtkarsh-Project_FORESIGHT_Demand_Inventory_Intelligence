import streamlit as st

def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Page Configuration
st.set_page_config(page_title="Project FORESIGHT", page_icon=" ", layout="wide", initial_sidebar_state="expanded")


# Sidebar
st.sidebar.title("Project FORESIGHT")
st.sidebar.info("""Demand Forecasting & Inventory Intelligence System""")
st.sidebar.markdown("---")
st.sidebar.success("Navigation")

st.sidebar.write("Home")
st.sidebar.write("Dashboard")
st.sidebar.write("Forecast")
st.sidebar.write("Risk Scoring")
st.sidebar.write("About")

st.sidebar.markdown("---")
st.sidebar.caption("Version 1.0")


# Header
st.title("Project FORESIGHT")
st.subheader("AI-Powered Demand Forecasting & Inventory Intelligence")
st.markdown("---")

st.write(
    """
    Project FORESIGHT helps businesses forecast future demand, identify inventory risks, reduce stockouts, minimise overstock and improve supply chain decisions using Machine Learning.
    """
)


# KPI Cards
st.header("Business Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Products", "1,250")

with col2:
    st.metric("Revenue at Risk", "2,45,000")

with col3:
    st.metric("Capital Locked", "1,18,000")

with col4:
    st.metric("Forecast Accuracy", "92%")

st.markdown("---")


# Features
st.header("Project Features")
c1, c2 = st.columns(2)

with c1:
    st.success("Demand Forecasting")
    st.write("""
        - Weekly Forecast
        - Monthly Forecast
        - SKU-wise Prediction
        - Historical Analysis
    """)

    st.success("Inventory Intelligence")
    st.write("""
        - Stockout Detection
        - Overstock Detection
        - Risk Classification
        - Inventory Monitoring
    """)

with c2:
    st.success("Business Analytics")
    st.write("""
        - Revenue at Risk
        - Capital Locked
        - Business KPIs
        - Executive Dashboard
    """)

    st.success("Machine Learning")
    st.write("""
        - XGBoost
        - LightGBM
        - Baseline Comparison
        - WAPE Evaluation
    """)

st.markdown("---")


# Workflow
st.header("Project Workflow")
workflow = [
    "Load Raw Dataset",
    "Data Cleaning",
    "Feature Engineering",
    "Demand Forecasting",
    "Model Evaluation",
    "Inventory Risk Scoring",
    "Business Recommendation",
    "Dashboard Visualisation"
]

for i, step in enumerate(workflow, start=1):
    st.write(f"**{i}. {step}**")
st.markdown("---")


# Objectives
st.header("Business Objectives")
st.write("""
    Improve Forecast Accuracy
    Reduce Stockout Risk
    Reduce Overstock Cost
    Improve Inventory Planning
    Increase Business Profitability
    Support Data-Driven Decision Making
""")

st.markdown("---")


# Technology Stack
st.header("Technology Stack")
tech1, tech2, tech3 = st.columns(3)
with tech1:
    st.info("""
        - Python
        - Pandas
        - NumPy
    """)

with tech2:
    st.info("""
        - XGBoost
        - LightGBM
        - Scikit-Learn
    """)

with tech3:
    st.info("""
        - Streamlit
        - Plotly
        - Matplotlib
    """)

st.markdown("---")


# Deliverables
st.header("Project Deliverables")
st.write("""
    Demand Forecasting
    Inventory Risk Scoring
    Business Recommendation
    Dashboard
    Reports
    Model Evaluation
    Forecast Comparison
""")

st.markdown("---")


# Footer
st.caption("Developed for Project FORESIGHT | AI-Powered Demand & Inventory Intelligence")