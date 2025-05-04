import time
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os

class IsolationForestModel:
    def __init__(self, fraud_model, contamination=0.01):
        self.fraud_model   = fraud_model
        self.iso_model     = IsolationForest(contamination=contamination, n_jobs=-1)
        self.scaler        = StandardScaler()
        self.contamination = contamination
        self.is_fitted     = False

    def build_daily_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate raw transactions into one row per (cc_num, day) with your
        9 summary features, using your pretrained fraud_model for per-txn scores.
        """
        # 1) Copy & clean up
        df = df.copy()
        df = df.drop(columns=['Unnamed: 0'], errors='ignore')

        # 2) Parse date and hour
        df['day']  = pd.to_datetime(df['date'],   format='%d%m%Y').dt.date
        df['hour'] = pd.to_datetime(df['unix_time'], unit='s').dt.hour

        daily = defaultdict(lambda: defaultdict(dict))

        # 3) Group by merchant & day
        for (cc, day), grp in df.groupby(['cc_num', 'day']):
            # 3a) Keep exactly the features your RF was trained on:
            tx_df = grp.drop(columns=[
                'cc_num', 'date', 'day'
            ], errors='ignore')

            # 3b) Scale them if your RF has a scalar attached
            if hasattr(self.fraud_model, 'scalar') and self.fraud_model.scalar is not None:
                X_tx = self.fraud_model.scalar.transform(tx_df.values)
            else:
                X_tx = tx_df.values

            # 3c) Get per-transaction fraud probabilities (shape n_txn × 2)
            probs = self.fraud_model.predict_prob(X_tx)[:, 1]

            # 4) Compute your five fraud stats
            cnt   = len(probs)
            mean_f, max_f = probs.mean(), probs.max()
            std_f, med_f = probs.std(), np.median(probs)

            # 5) Compute your four spend/time stats
            amts    = grp['amt'].values
            odd_cnt = ((grp['hour'] >= 23) | (grp['hour'] <= 6)).sum()
            mean_sp, max_sp, std_sp = amts.mean(), amts.max(), amts.std()

            # 6) Store them
            daily[cc][day] = {
                'num_transactions':      cnt,
                'mean_fraud':            mean_f,
                'max_fraud':             max_f,
                'std_fraud':             std_f,
                'median_fraud':          med_f,
                'mean_spend':            mean_sp,
                'max_spend':             max_sp,
                'std_spend':             std_sp,
                'odd_hour_transactions': odd_cnt
            }

        # 7) Flatten into a DataFrame
        rows = []
        for cc, days in daily.items():
            for day, stats in days.items():
                row = {'cc_num': cc, 'day': day}
                row.update(stats)
                rows.append(row)

        return pd.DataFrame(rows)

    def fit(self, summary_df: pd.DataFrame):
        # drop grouping columns, scale & fit
        X  = summary_df.drop(columns=['cc_num', 'day'], errors='ignore')
        Xs = self.scaler.fit_transform(X)
        self.iso_model.fit(Xs)
        self.is_fitted = True

    def save(self, path: str = "."):
        if not self.is_fitted:
            raise RuntimeError("Fit before saving.")

        # 1) Ensure the directory exists
        os.makedirs(path, exist_ok=True)

        # 2) Build the filename inside that directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(path, f"IsoForest_{timestamp}.joblib")

        # 3) Dump
        joblib.dump({
            'scaler':        self.scaler,
            'iso_model':     self.iso_model,
            'contamination': self.contamination
        }, fname)

        print(f"Saved IsolationForestModel to {fname}")

    def predict_prob(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each cc_num/day in df, compute an anomaly‐based fraud_prob ∈ [0,1].
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")

        summary = self.build_daily_summary(df)
        X       = summary.drop(columns=['cc_num','day'], errors='ignore')
        Xs      = self.scaler.transform(X)
        scores  = self.iso_model.decision_function(Xs)

        # sigmoid on negative score: higher score ⇒ more anomalous ⇒ higher fraud_prob
        probs   = 1.0 / (1.0 + np.exp(scores))
        summary['fraud_prob'] = probs

        return summary[['cc_num','day','fraud_prob']]


def train_isolation_forest(
    csv_path: str,
    fraud_model,
    contamination: float = 0.01,
    save_dir: str = None
) -> IsolationForestModel:
    """
    1) Read your CSV (with cc_num,amt,...,date columns),
    2) Build daily summary,
    3) Fit IsolationForest,
    4) Optionally save to save_dir.
    """
    # load
    df = pd.read_csv(csv_path)

    # instantiate & fit
    iso = IsolationForestModel(fraud_model, contamination=contamination)
    summary = iso.build_daily_summary(df)
    iso.fit(summary)
    print(f"Fitted on {len(summary)} merchant-day records.")

    # save if requested
    if save_dir:
        iso.save(save_dir)

    return iso


ckpt = joblib.load(
    "/Users/rauf/Desktop/Sid's ML/Final_Model_Files/"
    "Trained_SK_Models/RandomForestModel_20250503_021350.joblib"
)
rf = ckpt['model']

# alias sklearn's predict_proba as predict_prob
rf.predict_prob = rf.predict_proba

fraud_model = rf
# if your saved dict included a scaler, re-attach it:
fraud_model.scalar = ckpt.get('scalar', None)

# 2) Train your IsolationForest on the CSV with the new date column
iso = train_isolation_forest(
    csv_path    = "fraudTrain_with_dates.csv",
    fraud_model = fraud_model,
    contamination=0.02,
    save_dir    = "./models"
)

# 3) Score (merchant,day) fraud‐probabilities
df        = pd.read_csv("fraudTrain_with_dates.csv")
results   = iso.predict_prob(df)
print(results.head())