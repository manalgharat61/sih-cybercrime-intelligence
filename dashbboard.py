import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components  # Required to embed external websites

st.set_page_config(page_title="Cybercrime Predictive Intelligence", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load('hotspot_model.pkl')
    fraud_encoder = joblib.load('fraud_encoder.pkl')
    atm_df = pd.read_csv('atm_locations.csv')
    atm_df.columns = atm_df.columns.str.strip()
    return model, fraud_encoder, atm_df

try:
    model, fraud_encoder, atm_df = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")

# --- COMBINE PROJECTS USING TABS ---
tab1, tab2 = st.tabs(["🗺️ Predictive Hotspot Map", "🤖 AI Cyber Assistant"])

# ==========================================
# TAB 1: YOUR PREDICTIVE MAP
# ==========================================
with tab1:
    st.markdown("🚨 **Cybercrime Predictive Intelligence**")
    st.markdown("Forecast Likely Cash Withdrawal Locations in Advance")

    st.sidebar.header("Log New Cyber Complaint")
    amount_lost = st.sidebar.number_input("Amount Lost (INR)", min_value=1000, value=25000, step=1000)

    fraud_types = list(fraud_encoder.classes_) if hasattr(fraud_encoder, 'classes_') else []
    fraud_type = st.sidebar.selectbox("Fraud Type", fraud_types)

    hour_of_day = st.sidebar.slider("Hour of Incident (0-23)", min_value=0, max_value=23, value=14)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_name = st.sidebar.selectbox("Day of Week", days)
    day_of_week = days.index(day_name)
    is_weekend = 1 if day_of_week >= 5 else 0

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

        input_data = pd.DataFrame([[amount_lost, encoded_fraud, hour_of_day, day_of_week, is_weekend]], 
                                  columns=['amount_lost', 'fraud_type_encoded', 'hour_of_day', 'day_of_week', 'is_weekend'])
        
        st.session_state.prediction = model.predict(input_data)[0]
        st.session_state.loss_val = amount_lost

    if st.session_state.prediction:
        pred_zone = st.session_state.prediction
        st.success(f"⚠️ Predicted High-Risk Cashout Zone: **{pred_zone}**")
        
        target_col = None
        for col in ['city_zone', 'zone', 'region', 'area']:
            if col in atm_df.columns:
                target_col = col
                break
                
        if target_col and pred_zone in atm_df[target_col].values:
            filtered_atms = atm_df[atm_df[target_col] == pred_zone]
        else:
            filtered_atms = atm_df.head(5) 

        lon_col = next((c for c in atm_df.columns if 'lon' in c.lower() or 'lng' in c.lower()), 'longitude')
        lat_col = next((c for c in atm_df.columns if 'lat' in c.lower()), 'latitude')
        
        if lon_col in filtered_atms.columns:
            filtered_atms = filtered_atms[filtered_atms[lon_col] > 72.81]

        st.subheader(f"Predicted Hotspot Locations for {pred_zone}")
        name_col = next((c for c in filtered_atms.columns if 'bank' in c.lower() or 'name' in c.lower()), None)
        if name_col and not filtered_atms.empty:
            specific_targets = filtered_atms[name_col].dropna().unique()
            if len(specific_targets) > 0:
                st.warning(f"📍 **Specific Targeted Locations:** {', '.join(specific_targets[:8])}")
        
        if not filtered_atms.empty and lat_col in filtered_atms.columns and lon_col in filtered_atms.columns:
            center_lat = filtered_atms[lat_col].mean()
            center_lon = filtered_atms[lon_col].mean()
        else:
            center_lat, center_lon = 19.0760, 72.8777

        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        for _, row in filtered_atms.iterrows():
            lat = row.get(lat_col, 19.0760)
            lon = row.get(lon_col, 72.8777)
            name = row.get('atm_name', row.get('bank_name', 'High Risk ATM'))
            
            folium.Marker(
                [lat, lon],
                popup=f"<b>{name}</b><br>Zone: {pred_zone}<br>Risk Loss: INR {st.session_state.loss_val}",
                icon=folium.Icon(color="red", icon="warning")
            ).add_to(m)
            
        st_folium(m, width=700, height=450, key="multi_hotspot_map")
    else:
        st.info("Configure complaint details in the sidebar and click **Generate Hotspot Prediction**.")
        # ==========================================
# TAB 2: HIS AI STUDIO APP
# ==========================================
with tab2:
    st.markdown("🤖 **AI Cybercrime Assistant**")
    st.info("To ensure maximum security and processing speed, our Generative AI Assistant runs in a dedicated Google environment.")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Creates a large, primary-colored button to open his app
    st.link_button(
        "Launch AI Cyber Assistant ↗️", 
        "https://ai.studio/apps/8b92e452-9168-4581-8606-c19e28717c68", 
        type="primary", 
        use_container_width=True
    )

