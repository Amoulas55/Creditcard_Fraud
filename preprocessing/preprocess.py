import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Load original dataset
df = pd.read_csv("/home/u762545/opekepe/creditcard.csv")

# Drop Time
df = df.drop(columns=['Time'])

# Separate features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Normalize all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Combine with target into final DataFrame
df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
df_scaled['Class'] = y.values

# Save to CSV
output_path = "/home/u762545/opekepe/preprocessed/creditcard_normalized.csv"
df_scaled.to_csv(output_path, index=False)

print(f"✅ Saved full preprocessed dataset to: {output_path}")
