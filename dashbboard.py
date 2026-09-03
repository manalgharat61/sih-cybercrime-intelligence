import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Cybercrime Predictive Intelligence", layout="wide")

# Load model, encoder, and data directly
@st.cache_resource
def load_assets():
    model = joblib.load('hotspot_model.pkl')
    encoder = joblib.load('fraud_encoder.pkl')
    atm_df = pd.read_csv('atm_locations.csv')
    return model, encoder, atm_df

try:
    model, encoder, atm_df = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")

st.markdown("🚨 **Cybercrime Predictive Intelligence**")
st.markdown("Forecast Likely Cash Withdrawal Locations in Advance")

st.sidebar.header("Log New Cyber Complaint")
amount_lost = st.sidebar.number_input("Amount Lost (INR)", min_value=1000, value=25000, step=1000)

fraud_types = list(encoder.classes_) if hasattr(encoder, 'classes_') else ["UPI Phishing", "Credit Card Fraud", "Identity Theft"]
fraud_type = st.sidebar.selectbox("Fraud Type", fraud_types)

if st.sidebar.button("Generate Hotspot Prediction"):
    try:
        encoded_fraud = encoder.transform([fraud_type])[0]
    except:
        encoded_fraud = 0
        
    # Predict probabilities or target zone
    input_data = pd.DataFrame([[amount_lost, encoded_fraud]], columns=['amount_lost', 'fraud_type'])
    prediction = model.predict(input_data)[0]
    
    st.success(f"⚠️ Predicted High-Risk Cashout Zone: **{prediction}**")
    
    # Render map
    st.subheader("Predicted Hotspot Location Map")
    m = folium.Map(location=[19.0760, 72.8777], zoom_start=12)
    folium.Marker(
        [19.0760, 72.8777],
        popup=f"High Risk Zone: {prediction}\nLoss: INR {amount_lost}",
        icon=folium.Icon(color="red", icon="warning")
    ).add_to(m)
    st_folium(m, width=700, height=450)
else:
    st.info("Configure complaint details in the sidebar and click **Generate Hotspot Prediction**.")
