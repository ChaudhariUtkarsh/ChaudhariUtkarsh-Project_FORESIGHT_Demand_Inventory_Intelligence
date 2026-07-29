import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# Page Configuration
st.set_page_config(page_title="Demand Forecast", page_icon=" ", layout="wide")


# Title
st.title("Demand Forecast")
st.markdown("Predict future product demand using the trained Machine Learning model.")
st.markdown("---")


# Sidebar
st.sidebar.header("Forecast Settings")
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", min_value=1, max_value=30, value=7)
st.sidebar.markdown("---")


# Model Paths
MODEL_PATH = "models/best_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"


# Load Model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("best_model.pkl not found.")
        st.stop()
    return joblib.load(MODEL_PATH)


# Load Encoder
@st.cache_resource
def load_encoder():
    if os.path.exists(ENCODER_PATH):
        return joblib.load(ENCODER_PATH)
    return None

model = load_model()
encoder = load_encoder()


# Prediction Form
st.header("Prediction Input")
with st.form("forecast_form"):
    sku_id = st.text_input("SKU ID", value="SKU001")
    year = st.number_input("Year", min_value=2024, max_value=2035, value=2026)
    month = st.slider("Month", 1, 12, 7)
    week = st.slider("Week", 1, 53, 30)
    day = st.slider( "Day", 1, 31, 15)
    day_of_week = st.selectbox("Day of Week", [ 0, 1, 2, 3, 4, 5, 6])
    quarter = st.selectbox("Quarter", [1, 2, 3, 4])
    is_weekend = st.selectbox("Weekend", [0, 1])
    lag_1 = st.number_input("Lag 1 Sales", value=100)
    lag_7 = st.number_input("Lag 7 Sales", value=95)
    lag_14 = st.number_input( "Lag 14 Sales", value=90)
    rolling_mean_7 = st.number_input("Rolling Mean (7)", value=100.0)
    rolling_std_7 = st.number_input( "Rolling Std (7)", value=8.5)
    rolling_mean_30 = st.number_input( "Rolling Mean (30)", value=98.0)
    price_difference = st.number_input("Price Difference", value=10.0)
    discount_percentage = st.number_input("Discount %", value=5.0)
    inventory_gap = st.number_input("Inventory Gap", value=20)
    total_inventory = st.number_input("Total Inventory", value=500)
    on_hand_units = st.number_input("On Hand Units", value=300)
    on_order_units = st.number_input("On Order Units", value=100)
    reorder_point = st.number_input("Reorder Point", value=80)
    submitted = st.form_submit_button("Predict Demand")


# Prediction Logic
if submitted:
    try:
        # Encode SKU
        if encoder is not None:
            try:
                sku_id_enc = encoder.transform([sku_id])[0]
            except:
                sku_id_enc = 0
        else:
            sku_id_enc = 0

        # Create Input DataFrame
        input_df = pd.DataFrame({
            "year":[year],
            "month":[month],
            "week":[week],
            "day":[day],
            "day_of_week":[day_of_week],
            "quarter":[quarter],
            "is_weekend":[is_weekend],

            "lag_1":[lag_1],
            "lag_7":[lag_7],
            "lag_14":[lag_14],

            "rolling_mean_7":[rolling_mean_7],
            "rolling_std_7":[rolling_std_7],
            "rolling_mean_30":[rolling_mean_30],

            "price_difference":[price_difference],
            "discount_percentage":[discount_percentage],

            "inventory_gap":[inventory_gap],
            "total_inventory":[total_inventory],

            "on_hand_units":[on_hand_units],
            "on_order_units":[on_order_units],
            "reorder_point":[reorder_point],

            "sku_id_enc":[sku_id_enc]
        })

        # Prediction
        prediction = model.predict(input_df)[0]

        if prediction < 0:
            prediction = 0

        prediction = round(float(prediction),2)
        st.markdown("---")
        st.success("Prediction Completed Successfully")


        # KPI Cards
        c1,c2,c3 = st.columns(3)

        with c1:
            st.metric("Forecast Units", prediction)

        with c2:
            expected_revenue = prediction * 100 st.metric("Expected Revenue", f"{expected_revenue:,.2f}")

        with c3:
            if prediction > reorder_point:
                risk = "High"
            elif prediction > reorder_point*0.70:
                risk = "Medium"
            else:
                risk = "Low"
            st.metric("Risk Level", risk)
        st.markdown("---")

 
        # Forecast Summary
        summary = pd.DataFrame({"SKU":[sku_id], "Forecast Units":[prediction], "Forecast Days":[forecast_days], "Risk":[risk]})
        st.subheader("Forecast Summary")
        st.dataframe(summary, use_container_width=True)


        # Forecast Chart
        chart = pd.DataFrame({"Day":list(range(1,forecast_days+1)), "Forecast":[prediction]*forecast_days})
        fig = px.line(chart, x="Day", y="Forecast", markers=True, title="Forecast Trend")
        st.plotly_chart(fig, use_container_width=True)


        # Forecast Bar Chart
        fig2 = px.bar(summary, x="SKU", y="Forecast Units", color="Risk", text_auto=True, title="Forecast by SKU")
        st.plotly_chart(fig2, use_container_width=True)

        
        # Forecast Gauge
        gauge = px.bar(x=["Forecast"], y=[prediction], text_auto=True, title="Forecast Indicator")
        st.plotly_chart(gauge, use_container_width=True)

    except Exception as e:
        st.error(e)



# Download Forecast CSV
st.markdown("---")
st.header("Download Forecast")
forecast_download = summary.copy()
csv = forecast_download.to_csv(index=False).encode("utf-8")
st.download_button(label="Download Forecast CSV", data=csv, file_name="forecast_prediction.csv", mime="text/csv")


# Forecast Report
report = f"""
-------------------------------------------

PROJECT FORESIGHT
Demand Forecast Report

-------------------------------------------

SKU ID : {sku_id}
Forecast Units : {prediction}
Forecast Horizon : {forecast_days} Days
Expected Revenue : {expected_revenue:,.2f}
Risk Level : {risk}

-------------------------------------------

Recommendation
"""

if risk == "High":
    report += """
        1. Increase Inventory
        2. Place Reorder Immediately
        3. Increase Safety Stock
    """

elif risk == "Medium":
    report += """
        1. Monitor Inventory
        2. Review Weekly Forecast
        3. Keep Current Reorder Policy
    """

else:
    report += """
        1. Inventory is Healthy
        2. No Immediate Action Required
    """

report += """
-------------------------------------------
Generated by Project FORESIGHT
-------------------------------------------
"""

st.download_button(label="Download Forecast Report", data=report, file_name="Forecast_Report.txt", mime="text/plain")


# Business Recommendation
st.markdown("---")
st.header("Business Recommendation")

if risk == "High":
    st.error("""
        High Demand Expected Recommended Action
        1. Increase Procurement
        2. Increase Safety Stock
        3. Inform Supply Chain Team
        4. Monitor Daily Sales
    """)

elif risk == "Medium":
    st.warning("""
        Moderate Demand Expected Recommended Action
        1. Weekly Monitoring
        2. Continue Normal Procurement
        3. Review Inventory Weekly
    """)

else:
    st.success("""
        Demand is Stable Recommended Action
        1. Maintain Current Inventory
        2. No Immediate Procurement Required
    """)


# Forecast History
st.markdown("---")
st.header("Forecast History")
history = pd.DataFrame({"SKU":[sku_id], "Forecast":[prediction], "Revenue":[expected_revenue], "Risk":[risk]})
st.dataframe(history, use_container_width=True)


# Business Insights
st.markdown("---")
st.header("Business Insights")
col1,col2,col3 = st.columns(3)

with col1:
    st.metric("Forecast", prediction)

with col2:
    st.metric("Revenue", f"{expected_revenue:,.2f}")

with col3:
    st.metric("Risk", risk)


# Footer
st.markdown("---")
st.caption("Project FORESIGHT | AI-Powered Demand Forecasting & Inventory Intelligence")