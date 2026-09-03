import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import pandas as pd

# Load ATM Data for mapping base locations
try:
    atms = pd.read_csv("atm_locations.csv")
except FileNotFoundError:
    st.error("Missing atm_locations.csv file. Run generate_data.py first.")
    st.stop()

st.title("🚨 Cybercrime Predictive Intelligence")
st.markdown("Forecast Likely Cash Withdrawal Locations in Advance")

st.sidebar.header("Log New Cyber Complaint")

# Form inputs for the dashboard user
amount = st.sidebar.number_input("Amount Lost (INR)", min_value=1000, value=25000, step=1000)
fraud_type = st.sidebar.selectbox("Fraud Type", [
    'UPI Phishing', 
    'Investment Scam', 
    'Credit Card Fraud', 
    'Part-time Job Scam', 
    'Loan App Extortion'
])

# 1. Initialize session state memory
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None

# 2. Fetch data on button click and save it to memory
if st.sidebar.button("Generate Hotspot Prediction"):
    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict_hotspot",
            json={"amount_lost": amount, "fraud_type": fraud_type}
        )
        response.raise_for_status()
        
        # Save the API response to Streamlit's memory
        st.session_state.prediction_data = response.json()
        
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to API. Ensure your FastAPI server is running on port 8000.")

# 3. Render the map permanently if there is data in memory
if st.session_state.prediction_data:
    data = st.session_state.prediction_data
    target_zone = data["prediction"]["high_risk_zone"]
    confidence = data["prediction"]["confidence_score"]
    
    st.success(data["actionable_intel"])
    
    # Filter ATMs located in the predicted high-risk zone
    risk_atms = atms[atms['city_zone'] == target_zone]
    
    # Render the map centered on our base coordinates
    m = folium.Map(location=[19.0760, 72.8777], zoom_start=11)
    
    # Plot the predicted high-probability withdrawal nodes
    for _, row in risk_atms.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.7,
            popup=f"{row['bank_name']} - {row['atm_id']} (Risk: {confidence})"
        ).add_to(m)
        
    # Display the interactive Folium map inside Streamlit
    st_data = st_folium(m, width=700, height=500)