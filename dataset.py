import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()
random.seed(42)
np.random.seed(42)

ZONES = ['Zone_North', 'Zone_South', 'Zone_East', 'Zone_West', 'Zone_Central']

# ==========================================
# 1. GENERATE ATM / POS LOCATIONS
# ==========================================
def generate_atms(n_atms=150, base_lat=19.0760, base_lon=72.8777):
    banks = ['State Bank of India', 'HDFC Bank', 'ICICI Bank', 'Axis Bank', 'Bank of Baroda']

    atms = []
    for i in range(1, n_atms + 1):
        lat = base_lat + np.random.normal(0, 0.08)
        lon = base_lon + np.random.normal(0, 0.08)

        atms.append({
            "atm_id": f"ATM_{i:04d}",
            "bank_name": random.choice(banks),
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "city_zone": random.choice(ZONES),
            "is_high_risk_area": random.choice([0, 1]) if random.random() < 0.25 else 0
        })
    df_atms = pd.DataFrame(atms)
    df_atms.to_csv("atm_locations.csv", index=False)
    print(f"Generated {len(df_atms)} ATM locations -> atm_locations.csv")
    return df_atms


# ==========================================
# 2. DEFINE REALISTIC CORRELATION RULES
# ==========================================
# Each fraud type has a "specialist" zone where that scam ring's mule
# network tends to cash out. This mirrors real intel patterns where
# specific mule rings operate out of specific localities.
FRAUD_ZONE_BIAS = {
    'UPI Phishing':          {'Zone_West': 0.40, 'Zone_Central': 0.20, 'Zone_North': 0.15, 'Zone_South': 0.15, 'Zone_East': 0.10},
    'Investment Scam':       {'Zone_Central': 0.40, 'Zone_North': 0.20, 'Zone_West': 0.15, 'Zone_South': 0.15, 'Zone_East': 0.10},
    'Credit Card Fraud':     {'Zone_South': 0.35, 'Zone_East': 0.25, 'Zone_North': 0.15, 'Zone_West': 0.15, 'Zone_Central': 0.10},
    'Part-time Job Scam':    {'Zone_East': 0.40, 'Zone_South': 0.20, 'Zone_North': 0.15, 'Zone_Central': 0.15, 'Zone_West': 0.10},
    'Loan App Extortion':    {'Zone_North': 0.40, 'Zone_East': 0.20, 'Zone_Central': 0.15, 'Zone_West': 0.15, 'Zone_South': 0.10},
}

DISTRICTS = ZONES  # victim's district uses the same zone labels for direct comparability


def pick_withdrawal_zone(fraud_type, amount, victim_district):
    """
    Realistic zone selection logic:
    - Base probability of cashing out NEAR the victim's district (logistics/speed).
    - Larger amounts reduce 'local' probability (mules spread bigger hauls further).
    - Remaining probability follows the fraud ring's typical operating zone.
    """
    if amount <= 25000:
        p_local = 0.65
    elif amount <= 100000:
        p_local = 0.45
    else:
        p_local = 0.25

    if random.random() < p_local:
        return victim_district

    bias = FRAUD_ZONE_BIAS[fraud_type]
    zones = list(bias.keys())
    weights = list(bias.values())
    return random.choices(zones, weights=weights, k=1)[0]


# ==========================================
# 3. GENERATE COMPLAINTS & MULE TRANSACTIONS
# ==========================================
def generate_complaints_and_mule_logs(df_atms, n_complaints=1000):
    fraud_types = list(FRAUD_ZONE_BIAS.keys())
    payment_channels = ['UPI', 'Net Banking', 'Debit Card', 'Credit Card', 'Wallet']

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
        victim_district = random.choice(DISTRICTS)
        payment_channel = random.choice(payment_channels)

        complaints.append({
            "complaint_id": complaint_id,
            "timestamp": complaint_time.strftime("%Y-%m-%d %H:%M:%S"),
            "fraud_type": fraud_type,
            "amount_lost": amount,
            "victim_district": victim_district,
            "payment_channel": payment_channel,
            "mule_account_id": mule_acc
        })

        # Determine the actual withdrawal zone using the correlation rules
        target_zone = pick_withdrawal_zone(fraud_type, amount, victim_district)

        # Pick a real ATM located in that zone (fallback to any ATM if zone empty)
        zone_atms = df_atms[df_atms['city_zone'] == target_zone]['atm_id'].tolist()
        target_atm = random.choice(zone_atms) if zone_atms else random.choice(df_atms['atm_id'].tolist())

        # Simulating cashout delay (20 to 180 mins)
        time_to_cashout = random.randint(20, 180)
        cashout_time = complaint_time + timedelta(minutes=time_to_cashout)

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
