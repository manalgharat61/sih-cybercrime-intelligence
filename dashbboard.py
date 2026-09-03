import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Cybercrime Predictive Intelligence", layout="wide")

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

if 'prediction' not in st.session_state:
    st.session_state.prediction = None
    st.session_state.loss_val = None

if st.sidebar.button("Generate Hotspot Prediction"):
    try:
        encoded_fraud = encoder.transform([fraud_type])[0]
    except:
        encoded_fraud = 0
        
    input_data = pd.DataFrame([[amount_lost, encoded_fraud]], columns=['amount_lost', 'fraud_type'])
    
    if hasattr(model, "feature_names_in_"):
        for col in model.feature_names_in_:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[model.feature_names_in_]

    st.session_state.prediction = model.predict(input_data)[0]
    st.session_state.loss_val = amount_lost

if st.session_state.prediction:
    pred_zone = st.session_state.prediction
    st.success(f"⚠️ Predicted High-Risk Cashout Zone: **{pred_zone}**")
    
    st.subheader(f"Predicted Hotspot Locations for {pred_zone}")
    
    # Filter ATM dataset by predicted zone if a zone column exists, otherwise show sample cluster
    zone_col = next((col for col in atm_df.columns if 'zone' in col.lower() or 'region' in col.lower() or 'area' in col.lower()), None)
    
    if zone_col and pred_zone in atm_df[zone_col].values:
        filtered_atms = atm_df[atm_df[zone_col] == pred_zone]
    else:
        # Fallback: slice top matching rows if specific zone string doesn't match column values directly
        filtered_atms = atm_df.head(5) 

    # Determine center map coordinates dynamically
    lat_col = next((c for c in atm_df.columns if 'lat' in c.lower()), 'latitude')
    lon_col = next((c for c in atm_df.columns if 'lon' in c.lower() or 'lng' in c.lower()), 'longitude')
    
    if not filtered_atms.empty and lat_col in filtered_atms.columns and lon_col in filtered_atms.columns:
        center_lat = filtered_atms[lat_col].mean()
        center_lon = filtered_atms[lon_col].mean()
    else:
        center_lat, center_lon = 19.0760, 72.8777

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    # Add multiple markers for each location in the filtered dataset
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
