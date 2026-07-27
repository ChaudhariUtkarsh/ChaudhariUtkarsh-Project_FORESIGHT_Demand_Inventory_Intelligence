import streamlit as st

st.title("Inventory Risk")
st.write("Risk Scoring Module")
st.selectbox(
    "Select Risk Type",
    [
        "Stockout",
        "Overstock"
    ]
)
st.info("Risk Prediction will be available after model integration.")