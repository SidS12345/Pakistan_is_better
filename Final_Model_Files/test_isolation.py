#!/usr/bin/env python3
import os
import joblib
import pandas as pd
import numpy as np

# adjust this if your class is in another file
from SuperCreateIsolation import IsolationForestModel

# 1) Load raw transactions
RAW_CSV = "fraudTrain_with_dates.csv"
if not os.path.exists(RAW_CSV):
    raise FileNotFoundError(f"Can't find {RAW_CSV} in cwd={os.getcwd()}")
df = pd.read_csv(RAW_CSV)

# 2) Load & patch your RandomForest fraud detector
RF_CKPT = (
    "/Users/rauf/Desktop/Sid's ML/Final_Model_Files/"
    "Trained_SK_Models/RandomForestModel_20250503_021350.joblib"
)
rf_ckpt = joblib.load(RF_CKPT)
rf_model = rf_ckpt['model']
# alias sklearn's predict_proba → predict_prob
rf_model.predict_prob = rf_model.predict_proba
# re-attach its scaler if it was saved
rf_model.scalar = rf_ckpt.get('scalar', None)

# 3) Load the IsolationForest + scaler you trained
ISO_CKPT = "./models/IsoForest_20250504_184928.joblib"
iso_ckpt = joblib.load(ISO_CKPT)
iso = IsolationForestModel(
    fraud_model=rf_model,
    contamination=iso_ckpt['contamination']
)
iso.iso_model = iso_ckpt['iso_model']
iso.scaler    = iso_ckpt['scaler']
iso.is_fitted = True

# 4) Pick a random (cc_num, date) from your data
combos = df[['cc_num','date']].drop_duplicates()
sample = combos.sample(1).iloc[0]
cc   = sample['cc_num']
date = sample['date']
print(f"→ Testing merchant {cc} on date {date}")

# 5) Compute & print its fraud probability
try:
    prob = iso.predict_for_merchant_day(raw_df=df, cc_num=cc, date_str=date)
    print(f"   Fraud probability = {prob:.4f}")
except Exception as e:
    print("Error while scoring:", e)
