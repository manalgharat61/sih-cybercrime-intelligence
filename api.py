from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

# Initialize the FastAPI app
app = FastAPI(title="Cybercrime Predictive Hotspot API")

# Load the trained model and label encoder into memory at startup
try:
    model = joblib.load("hotspot_model.pkl")
    encoder = joblib.load("fraud_encoder.pkl")
except Exception as e:
    raise RuntimeError("Failed to load model files. Ensure hotspot_model.pkl and fraud_encoder.pkl are in the same directory.")

# Define the expected JSON structure of incoming complaints using Pydantic
class ComplaintInput(BaseModel):
    amount_lost: float
    fraud_type: str

@app.post("/predict_hotspot")
def predict_hotspot(complaint: ComplaintInput):
    try:
        # 1. Preprocess the incoming text data into numerical features
        fraud_encoded = encoder.transform([complaint.fraud_type])[0]
        
        # 2. Format as a Pandas DataFrame to match the model's training schema
        features = pd.DataFrame(
            [[complaint.amount_lost, fraud_encoded]], 
            columns=['amount_lost', 'fraud_type_encoded']
        )
        
        # 3. Generate the prediction and confidence score
        predicted_zone = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = max(probabilities) * 100
        
        # 4. Return actionable intelligence
        return {
            "status": "success",
            "prediction": {
                "high_risk_zone": predicted_zone,
                "confidence_score": f"{confidence:.2f}%"
            },
            "actionable_intel": f"Dispatch patrol to ATMs in {predicted_zone}. {confidence:.0f}% probability of mule cashout."
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid fraud type. Must match trained categories.")

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)