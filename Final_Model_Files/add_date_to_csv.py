import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1) Load your CSV
df = pd.read_csv('fraudTrain.csv')

# 2) Define your date range
start_date = datetime(2025, 6, 1)
end_date   = datetime(2025, 6, 15)
n_days     = (end_date - start_date).days + 1

# 3) Generate a random date for each row in ddmmyyyy format
df['date'] = [
    (start_date + timedelta(days=int(np.random.randint(0, n_days)))).strftime('%d%m%Y')
    for _ in range(len(df))
]

# 4) Save the new CSV
output_path = 'fraudTrain_with_dates.csv'
df.to_csv(output_path, index=False)

# 5) Quick check
print(df.head())
print(f"\nSaved new file to: {output_path}")
