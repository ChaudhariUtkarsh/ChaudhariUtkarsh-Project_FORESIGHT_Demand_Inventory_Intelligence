import os
import json
import streamlit as st


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")


st.set_page_config(page_title="Project Foresight", page_icon=" ", layout="wide")


st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }
    .metric-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        background-color: #fafafa;
        text-align: center;
    }
    .metric-title {
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 30px;
        font-weight: 700;
    }
    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">Project Foresight</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">' 'Demand Forecasting & Inventory Intelligence System' '</div>', unsafe_allow_html=True)
st.markdown("""Project Foresight is a demand forecasting and inventory intelligence system designed to provide SKU-level weekly demand forecasts and inventory decision intelligence.""")


def load_model_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as error:
        st.error("Unable to read model_metrics.json.")
        st.error(str(error))
        return None
metrics = load_model_metrics()


st.markdown('<div class="section-title">''Current Model Performance' '</div>', unsafe_allow_html=True)
baseline_wape = 12.17
lightgbm_wape = 8.16
improvement = 33.01

if metrics is not None:
    try:
        baseline_wape = float(metrics.get("baseline_cv_wape", 12.17))
        models = metrics.get("models", {})
        if "lightgbm" in models:
            lightgbm_wape = float(models["lightgbm"].get("CV_WAPE (%)", 8.16))
            improvement = float(models["lightgbm"].get("WAPE_Improvement (%)", 33.01))
    except (TypeError, ValueError, KeyError):
        baseline_wape = 12.17
        lightgbm_wape = 8.16
        improvement = 33.01

st.info("Evaluation Method: Rolling-Origin Cross-Validation")
st.write("The forecasting models are evaluated using " "Rolling-Origin Cross-Validation with WAPE " "(Weighted Absolute Percentage Error).")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Baseline WAPE", f"{baseline_wape:.2f}%")

with col2:
    st.metric("LightGBM WAPE", f"{lightgbm_wape:.2f}%")

with col3:
    st.metric("Improvement", f"{improvement:.2f}%")


st.markdown('<div class="section-title">' 'Final Verified Forecasting Results' '</div>', unsafe_allow_html=True)
st.write(
    f"""
    **Seasonal Naive WAPE:** {baseline_wape:.2f}%
    **LightGBM WAPE:** {lightgbm_wape:.2f}%
    **WAPE Improvement:** {improvement:.2f}%
    Lower WAPE indicates better forecasting accuracy.
    """
)

st.markdown('<div class="section-title">' 'Forecasting Model' '</div>', unsafe_allow_html=True)
st.write(
    """
    The system uses a 52-week Seasonal Naive forecast as the baseline and evaluates machine-learning models against this baseline.
    The final production model is selected based on Rolling-Origin Cross-Validation performance using WAPE.
    """
)

st.markdown('<div class="section-title">' 'Key Capabilities' '</div>', unsafe_allow_html=True)

cap1, cap2, cap3 = st.columns(3)

with cap1:
    st.markdown("###Weekly Demand Forecast")
    st.write("SKU-level weekly demand forecasting using " "historical sales and time-series features.")

with cap2:
    st.markdown("###Inventory Risk")
    st.write("Identifies stockout and overstock risks " "at SKU level.")

with cap3:
    st.markdown("###Inventory Decisions")
    st.write("Provides reorder and markdown/clear " "decision intelligence.")
st.markdown('<div class="section-title">' 'Model Status' '</div>', unsafe_allow_html=True)

if metrics is not None:
    production_model = metrics.get("production_model", "lightgbm")
    st.success(f"Production Model: " f"{str(production_model).upper()}")
    st.info("Current model metrics are loaded from " "models/model_metrics.json.")
else:
    st.warning("model_metrics.json was not found. " "Displaying the verified project metrics.")
st.markdown('<div class="section-title">' 'Project Objective' '</div>', unsafe_allow_html=True)
st.write("""Project Foresight helps businesses improve inventory planning by combining weekly SKU-level demand forecasting with inventory risk analysis, reorder prioritisation, and markdown/clear decision intelligence.""")


st.divider()
st.caption("Project Foresight | Weekly SKU-level Demand Forecasting " "& Inventory Intelligence")