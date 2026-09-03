import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()
random.seed(42)
np.random.seed(42)

# ==========================================
# 1. GENERATE ATM / POS LOCATIONS
# ==========================================
def generate_atms(n_atms=150, base_lat=19.0760, base_lon=72.8777):
    banks = ['State Bank of India', 'HDFC Bank', 'ICICI Bank', 'Axis Bank', 'Bank of Baroda']
    zones = ['Zone_North', 'Zone_South', 'Zone_East', 'Zone_West', 'Zone_Central']
    
    atms = []
    for i in range(1, n_atms + 1):
        lat = base_lat + np.random.normal(0, 0.08)
        lon = base_lon + np.random.normal(0, 0.08)
        
        atms.append({
            "atm_id": f"ATM_{i:04d}",
            "bank_name": random.choice(banks),
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "city_zone": random.choice(zones),
            "is_high_risk_area": random.choice([0, 1]) if random.random() < 0.25 else 0
        })
    df_atms = pd.DataFrame(atms)
    df_atms.to_csv("atm_locations.csv", index=False)
    print(f"Generated {len(df_atms)} ATM locations -> atm_locations.csv")
    return df_atms

# ==========================================
# 2. GENERATE COMPLAINTS & MULE TRANSACTIONS
# ==========================================
def generate_complaints_and_mule_logs(df_atms, n_complaints=1000):
    fraud_types = ['UPI Phishing', 'Investment Scam', 'Credit Card Fraud', 'Part-time Job Scam', 'Loan App Extortion']
    atm_ids = df_atms['atm_id'].tolist()
    high_risk_atms = df_atms[df_atms['is_high_risk_area'] == 1]['atm_id'].tolist()
    
    start_date = datetime.now() - timedelta(days=60)
    
    complaints = []
    mule_txs = []
    
    for i in range(1, n_complaints + 1):
        complaint_time = start_date + timedelta(
            days=random.randint(0, 59),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        amount = random.choice([15000, 25000, 50000, 100000, 200000, 500000])
        mule_acc = f"MULE_ACC_{random.randint(1000, 9999)}"
        complaint_id = f"CMP_{i:05d}"
        fraud_type = random.choice(fraud_types)
        
        complaints.append({
            "complaint_id": complaint_id,
            "timestamp": complaint_time.strftime("%Y-%m-%d %H:%M:%S"),
            "fraud_type": fraud_type,
            "amount_lost": amount,
            "mule_account_id": mule_acc
        })
        
        # Simulating cashout delay (20 to 180 mins)
        time_to_cashout = random.randint(20, 180)
        cashout_time = complaint_time + timedelta(minutes=time_to_cashout)
        
        # 60% probability of using known high-risk cluster ATMs
        if high_risk_atms and random.random() < 0.60:
            target_atm = random.choice(high_risk_atms)
        else:
            target_atm = random.choice(atm_ids)
            
        mule_txs.append({
            "tx_id": f"TX_{i:06d}",
            "complaint_id": complaint_id,
            "mule_account_id": mule_acc,
            "tx_timestamp": cashout_time.strftime("%Y-%m-%d %H:%M:%S"),
            "withdrawal_atm_id": target_atm,
            "amount_withdrawn": min(amount, random.choice([20000, 40000, 50000])),
            "time_to_cashout_mins": time_to_cashout
        })
        
    df_complaints = pd.DataFrame(complaints)
    df_mule_txs = pd.DataFrame(mule_txs)
    
    df_complaints.to_csv("complaints.csv", index=False)
    df_mule_txs.to_csv("mule_transactions.csv", index=False)
    
    print(f"Generated {len(df_complaints)} complaints -> complaints.csv")
    print(f"Generated {len(df_mule_txs)} mule transactions -> mule_transactions.csv")

if __name__ == "__main__":
    atms_df = generate_atms(n_atms=150)
    generate_complaints_and_mule_logs(atms_df, n_complaints=1000)