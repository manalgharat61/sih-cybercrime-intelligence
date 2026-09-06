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
    fraud_encoder = joblib.load('fraud_encoder.pkl')
    atm_df = pd.read_csv('atm_locations.csv')
    atm_df.columns = atm_df.columns.str.strip()
    return model, fraud_encoder, atm_df

try:
    model, fraud_encoder, atm_df = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")

tab1, tab2 = st.tabs(["🗺️ Predictive Hotspot Map", "🤖 AI Cyber Assistant"])

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
        
        # Predicted Zone
        predicted_zone = model.predict(input_data)[0]
        st.session_state.prediction = predicted_zone
        st.session_state.loss_val = amount_lost
        
        # Calculate real Threat Probability from Random Forest class probabilities
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            predicted_idx = list(model.classes_).index(predicted_zone)
            st.session_state.threat_score = round(float(probabilities[predicted_idx]) * 100, 1)
        else:
            st.session_state.threat_score = 78.5

    if st.session_state.prediction:
        pred_zone = st.session_state.prediction
        threat_pct = st.session_state.threat_score or 75.0
        loss = st.session_state.loss_val

        # Classify Threat Level
        if threat_pct >= 70 or loss >= 100000:
            threat_level = "CRITICAL"
            badge_color = "red"
        elif threat_pct >= 40 or loss >= 30000:
            threat_level = "ELEVATED"
            badge_color = "orange"
        else:
            threat_level = "MODERATE"
            badge_color = "blue"

        # Threat Metrics Panel
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Predicted Target Zone", str(pred_zone))
        col2.metric("Hotspot Probability", f"{threat_pct}%")
        col3.metric("Threat Severity", threat_level)
        col4.metric("Est. Intercept Window", "< 45 Mins" if hour_of_day >= 20 or hour_of_day <= 6 else "< 90 Mins")

        st.divider()

        # Locate zone column
        target_col = None
        for col in ['city_zone', 'zone', 'region', 'area']:
            if col in atm_df.columns:
                target_col = col
                break
                
        if target_col and pred_zone in atm_df[target_col].values:
            filtered_atms = atm_df[atm_df[target_col] == pred_zone]
        else:
            filtered_atms = atm_df.head(5)

        # Coordinate filtering
        lon_col = next((c for c in atm_df.columns if 'lon' in c.lower() or 'lng' in c.lower()), 'longitude')
        lat_col = next((c for c in atm_df.columns if 'lat' in c.lower()), 'latitude')
        
        if lon_col in filtered_atms.columns:
            filtered_atms = filtered_atms[filtered_atms[lon_col] > 72.81]

        # Targeted Banks summary
        name_col = next((c for c in filtered_atms.columns if 'bank' in c.lower() or 'name' in c.lower()), None)
        if name_col and not filtered_atms.empty:
            targets = filtered_atms[name_col].dropna().unique()
            if len(targets) > 0:
                st.info(f"🏦 **High-Risk ATMs In Zone:** {', '.join(targets[:6])}")

        # Map generation
        if not filtered_atms.empty and lat_col in filtered_atms.columns and lon_col in filtered_atms.columns:
            center_lat = filtered_atms[lat_col].mean()
            center_lon = filtered_atms[lon_col].mean()
        else:
            center_lat, center_lon = 19.0760, 72.8777

        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

        for _, row in filtered_atms.iterrows():
            lat = row.get(lat_col, 19.0760)
            lon = row.get(lon_col, 72.8777)
            atm_name = row.get(name_col, 'High-Risk Terminal')
            atm_id = row.get('atm_id', 'N/A')
            is_high_risk = row.get('is_high_risk_area', 1)

            # Enhanced popup card with operational metrics
            popup_html = f"""
            <div style="font-family: Arial; min-width: 170px;">
                <h4 style="margin:0 0 5px 0; color:#d9534f;">{atm_name}</h4>
                <b>ATM ID:</b> {atm_id}<br>
                <b>Zone:</b> {pred_zone}<br>
                <b>Threat Probability:</b> {threat_pct}%<br>
                <b>Severity:</b> <span style="color:{badge_color}; font-weight:bold;">{threat_level}</span><br>
                <b>Loss Exposure:</b> INR {loss:,}<br>
                <b>Flagged Area:</b> {'Yes' if is_high_risk == 1 else 'Standard'}
            </div>
            """

            folium.Marker(
                [lat, lon],
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"{atm_name} ({threat_level} Risk)",
                icon=folium.Icon(color="red" if is_high_risk == 1 else "orange", icon="exclamation-sign")
            ).add_to(m)

        st_folium(m, width=900, height=500, key="multi_hotspot_map")
    else:
        st.info("Configure complaint details in the sidebar and click **Generate Hotspot Prediction**.")

# ==========================================
# TAB 2: AI ASSISTANT PORTAL
# ==========================================
with tab2:
    st.markdown("🤖 **AI Cybercrime Assistant**")
    st.info("The Generative AI Assistant runs in a dedicated sandboxed environment.")
    st.link_button(
        "Launch AI Cyber Assistant ↗️", 
        "https://ai.studio/apps/8b92e452-9168-4581-8606-c19e28717c68", 
        type="primary", 
        use_container_width=True
    )
    
