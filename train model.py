import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

print("Loading datasets...")
atms = pd.read_csv("atm_locations.csv")
complaints = pd.read_csv("complaints.csv")
txs = pd.read_csv("mule_transactions.csv")

# ==========================================
# 1. FEATURE ENGINEERING & DATA MERGING
# ==========================================
# Link the complaint details to the ATM zone where the cash was ultimately withdrawn
print("Merging data and engineering features...")
df_merged = txs.merge(atms, left_on='withdrawal_atm_id', right_on='atm_id')
df_final = df_merged.merge(complaints, on='complaint_id')

# Convert text-based 'fraud_type' into numerical values the ML model can understand
le_fraud = LabelEncoder()
df_final['fraud_type_encoded'] = le_fraud.fit_transform(df_final['fraud_type'])

# Define Inputs (Features) and Output (Target)
X = df_final[['amount_lost', 'fraud_type_encoded']] # What we know when a complaint is filed
y = df_final['city_zone']                           # What we want to predict (Withdrawal Location)

# ==========================================
# 2. MODEL TRAINING
# ==========================================
print("Training Random Forest AI Model...")
# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test) * 100
print(f"Model Accuracy on Test Data: {accuracy:.2f}%")

# ==========================================
# 3. EXPORT MODEL FOR BACKEND API
# ==========================================
# Save the trained model and the encoder so our future FastAPI backend can use them
joblib.dump(model, 'hotspot_model.pkl')
joblib.dump(le_fraud, 'fraud_encoder.pkl')
print("Model saved successfully -> hotspot_model.pkl")