import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import datetime

st.set_page_config(page_title="Cybercrime Predictive Intelligence", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load('hotspot_model.pkl')
    fraud_encoder = joblib.load('fraud_encoder.pkl')
    district_encoder = joblib.load('district_encoder.pkl')
    payment_encoder = joblib.load('payment_encoder.pkl')
    atm_df = pd.read_csv('atm_locations.csv')
    return model, fraud_encoder, district_encoder, payment_encoder, atm_df

try:
    model, fraud_encoder, district_encoder, payment_encoder, atm_df = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")

st.markdown("🚨 **Cybercrime Predictive Intelligence**")
st.markdown("Forecast Likely Cash Withdrawal Locations in Advance")

st.sidebar.header("Log New Cyber Complaint")

# User Inputs
amount_lost = st.sidebar.number_input("Amount Lost (INR)", min_value=1000, value=25000, step=1000)

fraud_types = list(fraud_encoder.classes_) if hasattr(fraud_encoder, 'classes_') else []
fraud_type = st.sidebar.selectbox("Fraud Type", fraud_types)

districts = list(district_encoder.classes_) if hasattr(district_encoder, 'classes_') else []
victim_district = st.sidebar.selectbox("Victim District", districts)

payment_channels = list(payment_encoder.classes_) if hasattr(payment_encoder, 'classes_') else []
payment_channel = st.sidebar.selectbox("Payment Channel", payment_channels)

if 'prediction' not in st.session_state:
    st.session_state.prediction = None
    st.session_state.loss_val = None

if st.sidebar.button("Generate Hotspot Prediction"):
    
    def safe_encode(encoder, value):
        try:
            return encoder.transform([value])[0]
        except:
            return 0
            
    encoded_fraud = safe_encode(fraud_encoder, fraud_type)
    encoded_district = safe_encode(district_encoder, victim_district)
    encoded_payment = safe_encode(payment_encoder, payment_channel)
    
    # Generate Time Features dynamically
    now = datetime.datetime.now()
    hour_of_day = now.hour
    day_of_week = now.weekday() # 0 = Monday
    is_weekend = 1 if day_of_week >= 5 else 0

    # Build input dataframe matching your training script exactly
    input_data = pd.DataFrame([[
        amount_lost, 
        encoded_fraud, 
        encoded_district, 
        encoded_payment, 
        hour_of_day, 
        day_of_week, 
        is_weekend
    ]], columns=[
        'amount_lost', 
        'fraud_type_encoded', 
        'victim_district_encoded', 
        'payment_channel_encoded', 
        'hour_of_day', 
        'day_of_week', 
        'is_weekend'
    ])
    
    st.session_state.prediction = model.predict(input_data)[0]
    st.session_state.loss_val = amount_lost

if st.session_state.prediction:
    pred_zone = st.session_state.prediction
    st.success(f"⚠️ Predicted High-Risk Cashout Zone: **{pred_zone}**")
    
    st.subheader(f"Predicted Hotspot Locations for {pred_zone}")
    
    zone_col = next((col for col in atm_df.columns if 'zone' in col.lower() or 'region' in col.lower() or 'area' in col.lower()), None)
    
    if zone_col and pred_zone in atm_df[zone_col].values:
        filtered_atms = atm_df[atm_df[zone_col] == pred_zone]
    else:
        filtered_atms = atm_df.head(5) 

    lat_col = next((c for c in atm_df.columns if 'lat' in c.lower()), 'latitude')
    lon_col = next((c for c in atm_df.columns if 'lon' in c.lower() or 'lng' in c.lower()), 'longitude')
    
    if not filtered_atms.empty and lat_col in filtered_atms.columns and lon_col in filtered_atms.columns:
        center_lat = filtered_atms[lat_col].mean()
        center_lon = filtered_atms[lon_col].mean()
    else:
        center_lat, center_lon = 19.0760, 72.8777

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    for _, row in filtered_atms.iterrows():
        lat = row.get(lat_col, 19.0760)
        lon = row.get(lon_col, 72.8777)
        name = row.get('atm_name', row.get('name', 'High Risk ATM'))
        
        folium.Marker(
            [lat, lon],
            popup=f"<b>{name}</b><br>Zone: {pred_zone}<br>Risk Loss: INR {st.session_state.loss_val}",
            icon=folium.Icon(color="red", icon="warning")
        ).add_to(m)
        
    st_folium(m, width=700, height=450, key="multi_hotspot_map")
else:
    st.info("Configure complaint details in the sidebar and click **Generate Hotspot Prediction**.")
