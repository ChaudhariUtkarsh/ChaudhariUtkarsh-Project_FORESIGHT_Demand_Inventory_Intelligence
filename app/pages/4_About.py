import streamlit as st
import os


st.set_page_config(page_title="About | Project FORESIGHT", page_icon=" ", layout="wide", initial_sidebar_state="expanded")


def load_css():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    css_path = os.path.join(base_dir, "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css()


st.title("About Project FORESIGHT")
st.markdown("""**Project FORESIGHT** is an AI-powered Demand Forecasting and Inventory Intelligence system designed to help businesses make better inventory and replenishment decisions.""")
st.markdown("---")


st.header("Project Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Project", "FORESIGHT")

with col2:
    st.metric("Primary Task", "Demand Forecasting")

with col3:
    st.metric("Risk Engine", "Inventory Intelligence")

st.markdown(
    """
    ### Key Capabilities
        - SKU-level demand forecasting
        - Inventory risk identification
        - Stockout risk detection
        - Overstock risk detection
        - Reorder prioritization
        - Markdown / clearance recommendations
        - Revenue-at-risk estimation
        - Capital-locked estimation
    """
)
st.markdown("---")


st.header("Model Evaluation")
st.markdown("""The forecasting models are evaluated using multiple error metrics. However, **WAPE (Weighted Absolute Percentage Error)** is the primary evaluation metric for Project FORESIGHT.""")


st.subheader("Primary Evaluation Metric")
metric_col1, metric_col2 = st.columns([1, 2])

with metric_col1:
    st.metric("Primary Metric", "WAPE")

with metric_col2:
    st.info("""**WAPE (Weighted Absolute Percentage Error)** is used as the primary metric because it evaluates forecast error relative to the total actual demand and provides a business-oriented measure of forecasting accuracy.""")
st.markdown("---")


st.subheader("Baseline vs Production Model")
baseline_wape = 12.17
lightgbm_wape = 8.16
improvement = ((baseline_wape - lightgbm_wape) / baseline_wape) * 100


comparison_col1, comparison_col2 = st.columns(2)
with comparison_col1:
    st.markdown("###Baseline Model")
    st.write("**Model:** Seasonal Naive")
    st.metric("Baseline WAPE", f"{baseline_wape:.2f}%")
with comparison_col2:
    st.markdown("###Production Model")
    st.write("**Model:** LightGBM")
    st.metric("LightGBM WAPE", f"{lightgbm_wape:.2f}%")
st.markdown("---")


st.subheader("Model Improvement")
st.metric("WAPE Improvement", f"{improvement:.2f}%")

st.success(
    f"""
    The production **LightGBM** model improves WAPE from **{baseline_wape:.2f}%** for the Seasonal Naive baseline to **{lightgbm_wape:.2f}%**.
    This represents a **{improvement:.2f}% improvement** over the baseline.
    """
)
st.markdown("---")

st.subheader("Forecasting Model Comparison")
comparison_data = {"Model": ["Seasonal Naive", "LightGBM"], "Role": ["Baseline", "Production"], "WAPE": [f"{baseline_wape:.2f}%", f"{lightgbm_wape:.2f}%"]}
st.table(comparison_data)
st.markdown("---")


st.subheader("Additional Evaluation Metrics")
st.markdown(
    """
    Project FORESIGHT may also track the following supporting metrics during model evaluation:
        - **MAE** — Mean Absolute Error
        - **RMSE** — Root Mean Squared Error
        - **MAPE** — Mean Absolute Percentage Error
        - **WAPE** — Weighted Absolute Percentage Error
    **WAPE remains the primary model-selection and reporting metric for the production forecasting workflow.**
    """
)
st.markdown("---")


st.header("Production Model")
prod_col1, prod_col2, prod_col3 = st.columns(3)
with prod_col1:
    st.metric("Production Model", "LightGBM")
with prod_col2:
    st.metric("Baseline", "Seasonal Naive")
with prod_col3:
    st.metric("Primary Metric", "WAPE")


st.markdown(
    """
    ### Production Model Selection
    The production forecasting model is selected by comparing machine-learning performance against a Seasonal Naive baseline.

    The current evaluation shows:
    **Seasonal Naive WAPE:** 12.17%
    **LightGBM WAPE:** 8.16%

    Therefore, LightGBM provides the better forecasting performance and is used as the production model.
    """
)
st.markdown("---")


st.header("Business Impact")
impact_col1, impact_col2 = st.columns(2)

with impact_col1:
    st.markdown(
        """
        ### Demand Planning
            - Improve demand forecasting
            - Identify future demand patterns
            - Support SKU-level planning
            - Reduce forecast uncertainty
        """
    )

with impact_col2:
    st.markdown(
        """
        ### Inventory Intelligence
            - Detect stockout risk
            - Detect overstock risk
            - Prioritize replenishment
            - Identify excess inventory
            - Estimate financial impact
        """
    )

st.markdown("---")


st.header("Technology Stack")
tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)

with tech_col1:
    st.markdown(
        """
        **Data**
            - Python
            - Pandas
            - NumPy
        """
    )

with tech_col2:
    st.markdown(
        """
        **Machine Learning**
            - LightGBM
            - Seasonal Naive
            - Scikit-learn
        """
    )

with tech_col3:
    st.markdown(
        """
        **Dashboard**
            - Streamlit
            - Plotly
        """
    )

with tech_col4:
    st.markdown(
        """
        **Deployment**
            - FastAPI
            - Render
        """
    )

st.markdown("---")


st.header("Inventory Risk Intelligence")
st.markdown(
    """
    The risk engine uses demand forecasts and inventory position to identify potential inventory problems.
    ### Main Risk Categories
    **High Stockout Risk**
    
    Inventory may not be sufficient to satisfy expected demand.
    **Medium Risk**
    
    Inventory position requires monitoring and may need intervention depending on demand and supply conditions.
    **Low Risk**
    
    Inventory position is considered relatively healthy.
    **Overstock**
    
    Inventory exceeds expected demand and may result in excess capital being locked in stock.
    """
)
st.markdown("---")


st.header("Project Workflow")
st.markdown(
    """
    **Historical Data**
             ↓
    **Data Processing & Feature Engineering**
             ↓
    **Seasonal Naive Baseline**
             ↓
    **LightGBM Forecasting Model**
             ↓
    **WAPE Evaluation**
             ↓
    **Demand Forecast**
             ↓
    **Inventory Risk Scoring**
             ↓
    **Reorder / Markdown Recommendations**
             ↓
    **Business Dashboard**
    """
)
st.markdown("---")


st.header("Model Evaluation Summary")
summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

with summary_col1:
    st.metric("Primary Metric", "WAPE")

with summary_col2:
    st.metric("Baseline", "Seasonal Naive")

with summary_col3:
    st.metric("Production Model", "LightGBM")

with summary_col4:
    st.metric("Improvement", f"{improvement:.2f}%")

st.success("""**Final Model Decision:** LightGBM is the production forecasting model because it achieves a lower WAPE than the Seasonal Naive baseline.""")


st.markdown("---")
st.caption("Project FORESIGHT | AI-Powered Demand Forecasting & Inventory Intelligence")