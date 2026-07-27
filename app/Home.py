import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Project FORESIGHT",
    page_icon="",
    layout="wide"
)

# Title
st.title("Project FORESIGHT")
st.subheader("Demand & Inventory Intelligence")
st.markdown("---")
st.write(
    """
Welcome to **Project FORESIGHT**.

This dashboard helps businesses to:
- Forecast Future Product Demand
- Identify Stockout Risk
- Identify Overstock Risk
- Improve Inventory Planning
- Support Business Decision Making
"""
)

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.info("Forecast upcoming product demand.")
with col2:
    st.success("Reduce inventory cost using AI.")

st.markdown("---")
st.header("Project Workflow")
st.write(
"""
1. Load Dataset
2. Data Cleaning
3. Feature Engineering
4. Demand Forecasting
5. Risk Scoring
6. Dashboard Visualization
"""
)

st.markdown("---")
st.caption("Developed using Python 1) Streamlit 2) Scikit-Learn") 