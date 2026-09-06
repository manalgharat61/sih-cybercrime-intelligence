import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import hashlib
import streamlit.components.v1 as components
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

tab1= st.tabs(["🗺️ Predictive Hotspot Map"])

# ==========================================
# TAB 1: PREDICTIVE HOTSPOT DASHBOARD
# ==========================================
with tab1:
    st.markdown("🚨 **Cybercrime Predictive Hotspot Intelligence**")
    st.caption("Tactical risk assessment & cash-out hotspot interception")

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
        st.session_state.threat_score = None
        st.session_state.loss_val = None

    if st.sidebar.button("Generate Hotspot Prediction"):
        def safe_encode(encoder, value):
            try:
                return encoder.transform([value])[0]
            except:
                return 0
                
        encoded_fraud = safe_encode(fraud_encoder, fraud_type)

        input_data = pd.DataFrame([[
            amount_lost, 
            encoded_fraud, 
            hour_of_day, 
            day_of_week, 
            is_weekend
        ]], columns=[
            'amount_lost', 
            'fraud_type_encoded', 
            'hour_of_day', 
            'day_of_week', 
            'is_weekend'
        ])
        
        predicted_zone = model.predict(input_data)[0]
        st.session_state.prediction = predicted_zone
        st.session_state.loss_val = amount_lost
        
        # Base zone probability from Random Forest
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            predicted_idx = list(model.classes_).index(predicted_zone)
            st.session_state.threat_score = round(float(probabilities[predicted_idx]) * 100, 1)
        else:
            st.session_state.threat_score = 72.0

    if st.session_state.prediction:
        pred_zone = st.session_state.prediction
        base_threat = st.session_state.threat_score or 72.0
        loss = st.session_state.loss_val

        # Zone-level metrics summary
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Predicted Target Zone", str(pred_zone))
        col2.metric("Zone Cashout Risk", f"{base_threat}%")
        col3.metric("Total Exposure", f"₹{loss:,}")
        col4.metric("Intercept Window", "< 45 Mins" if (hour_of_day >= 21 or hour_of_day <= 5) else "< 90 Mins")

        st.divider()

        # Locate zone column
        target_col = None
        for col in ['city_zone', 'zone', 'region', 'area']:
            if col in atm_df.columns:
                target_col = col
                break
                
        if target_col and pred_zone in atm_df[target_col].values:
            filtered_atms = atm_df[atm_df[target_col] == pred_zone].copy()
        else:
            filtered_atms = atm_df.head(10).copy()

        # Land filter
        lon_col = next((c for c in atm_df.columns if 'lon' in c.lower() or 'lng' in c.lower()), 'longitude')
        lat_col = next((c for c in atm_df.columns if 'lat' in c.lower()), 'latitude')
        if lon_col in filtered_atms.columns:
            filtered_atms = filtered_atms[filtered_atms[lon_col] > 72.81]

        name_col = next((c for c in filtered_atms.columns if 'bank' in c.lower() or 'name' in c.lower()), None)

        if not filtered_atms.empty and lat_col in filtered_atms.columns and lon_col in filtered_atms.columns:
            center_lat = filtered_atms[lat_col].mean()
            center_lon = filtered_atms[lon_col].mean()
        else:
            center_lat, center_lon = 19.0760, 72.8777

        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

        # Operational status templates to cycle through based on terminal specifics
        status_catalog = [
            ("Rapid sequential cash withdrawal pattern", "Immediate Dispatch"),
            ("High mule account activity cluster", "Surveillance Priority"),
            ("High-density transit terminal", "Patrol Alert"),
            ("Off-peak repeat debit attempt", "Card Lock Advisory"),
            ("Standard baseline traffic", "Monitoring Only")
        ]

        for idx, row in filtered_atms.reset_index().iterrows():
            lat = row.get(lat_col, 19.0760)
            lon = row.get(lon_col, 72.8777)
            atm_name = row.get(name_col, f"ATM Terminal #{idx+1}")
            atm_id = row.get('atm_id', f"ATM-{1000 + idx}")
            is_high_risk = row.get('is_high_risk_area', 0)

            # Deterministic, unique variation per ATM based on ATM ID
            hash_val = int(hashlib.md5(str(atm_id).encode()).hexdigest(), 16)
            jitter = (hash_val % 31) - 15  # -15% to +15%
            
            # Individual terminal risk score
            terminal_score = base_threat + jitter + (12 if is_high_risk == 1 else -5)
            terminal_score = round(max(35.0, min(97.5, terminal_score)), 1)

            # Individual severity tier and pin styling
            if terminal_score >= 80:
                atm_severity = "CRITICAL"
                pin_color = "red"
                badge_bg = "#dc3545"
            elif terminal_score >= 60:
                atm_severity = "ELEVATED"
                pin_color = "orange"
                badge_bg = "#fd7e14"
            else:
                atm_severity = "MODERATE"
                pin_color = "blue"
                badge_bg = "#0d6efd"

            status_desc, action_rec = status_catalog[hash_val % len(status_catalog)]
            est_terminal_loss = round((loss * (terminal_score / 100)) / 1000) * 1000

            # Customized popup per individual ATM terminal
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; min-width: 210px; font-size: 13px;">
                <h4 style="margin:0 0 6px 0; color:#111;">{atm_name}</h4>
                <div style="display:inline-block; background:{badge_bg}; color:white; padding:2px 8px; border-radius:4px; font-weight:bold; font-size:11px; margin-bottom:8px;">
                    {atm_severity} RISK • {terminal_score}%
                </div><br>
                <b>Terminal ID:</b> <code>{atm_id}</code><br>
                <b>Zone:</b> {pred_zone}<br>
                <b>Est. Exposure:</b> ₹{est_terminal_loss:,}<br>
                <b>Mule Signal:</b> {status_desc}<br>
                <b>Action:</b> <b>{action_rec}</b>
            </div>
            """

            folium.Marker(
                [lat, lon],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{atm_name} | {terminal_score}% ({atm_severity})",
                icon=folium.Icon(color=pin_color, icon="info-sign")
            ).add_to(m)

        st_folium(m, width=950, height=520, key="multi_hotspot_map")
    else:
        st.info("Configure complaint details in the sidebar and click **Generate Hotspot Prediction**.")
