import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

print("Loading datasets...")
atms = pd.read_csv("atm_locations.csv")
complaints = pd.read_csv("complaints.csv")
txs = pd.read_csv("mule_transactions.csv")

# ==========================================
# 1. FEATURE ENGINEERING & DATA MERGING
# ==========================================
print("Merging data and engineering features...")

# Link each complaint to the ATM zone where the cash was ultimately withdrawn
df = txs.merge(atms, left_on='withdrawal_atm_id', right_on='atm_id')
df = df.merge(complaints, on='complaint_id')

# --- Time-based features (known the instant a complaint is filed) ---
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour_of_day'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek       # 0 = Monday
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# --- Categorical encodings ---
le_fraud = LabelEncoder()
le_district = LabelEncoder()
le_payment = LabelEncoder()

df['fraud_type_encoded'] = le_fraud.fit_transform(df['fraud_type'])
df['victim_district_encoded'] = le_district.fit_transform(df['victim_district'])
df['payment_channel_encoded'] = le_payment.fit_transform(df['payment_channel'])

# --- Final feature set & target ---
FEATURE_COLUMNS = [
    'amount_lost',
    'fraud_type_encoded',
    'victim_district_encoded',
    'payment_channel_encoded',
    'hour_of_day',
    'day_of_week',
    'is_weekend',
]
TARGET_COLUMN = 'city_zone'   # withdrawal zone we are trying to predict

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

# ==========================================
# 2. MODEL TRAINING
# ==========================================
print("Training Random Forest AI Model...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
baseline_accuracy = y.value_counts(normalize=True).max() * 100

print(f"Model Accuracy on Test Data: {accuracy:.2f}%")
print(f"Majority-class baseline:      {baseline_accuracy:.2f}%")
print("\nClassification report:\n", classification_report(y_test, y_pred))

# ==========================================
# 3. EVALUATION ARTIFACTS (for dashboard/report)
# ==========================================
print("Saving evaluation artifacts...")

# Confusion matrix as a plain dict-of-dicts (zone -> predicted zone -> count)
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_dict = {
    true_label: {pred_label: int(cm[i][j]) for j, pred_label in enumerate(labels)}
    for i, true_label in enumerate(labels)
}

# Feature importances
importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_.round(4).tolist()))
importances = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))

metrics = {
    "accuracy_pct": round(accuracy, 2),
    "baseline_accuracy_pct": round(baseline_accuracy, 2),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "labels": labels,
    "confusion_matrix": cm_dict,
    "feature_importances": importances,
}
with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ==========================================
# 4. EXPORT MODEL, ENCODERS & FEATURE LIST FOR API/DASHBOARD
# ==========================================
joblib.dump(model, 'hotspot_model.pkl')
joblib.dump(le_fraud, 'fraud_encoder.pkl')
joblib.dump(le_district, 'district_encoder.pkl')
joblib.dump(le_payment, 'payment_encoder.pkl')
joblib.dump(FEATURE_COLUMNS, 'feature_columns.pkl')

print("Model, encoders, and metrics saved successfully:")
print("  -> hotspot_model.pkl")
print("  -> fraud_encoder.pkl")
print("  -> district_encoder.pkl")
print("  -> payment_encoder.pkl")
print("  -> feature_columns.pkl")
print("  -> model_metrics.json")
