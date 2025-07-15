import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Paths
data_path = "/home/u762545/opekepe/creditcard.csv"
save_dir = "/home/u762545/opekepe/preprocessed"
os.makedirs(save_dir, exist_ok=True)

# Load data
df = pd.read_csv(data_path)

# Separate features and target
X = df.drop(columns=["Class"]).values
y = df["Class"].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for CNN-LSTM: [samples, timesteps=1, features]
X_cnnlstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Save to .npy files
np.save(os.path.join(save_dir, "X_full_cnnlstm.npy"), X_cnnlstm)
np.save(os.path.join(save_dir, "y_full.npy"), y)

print("✅ Saved X_full_cnnlstm.npy and y_full.npy to", save_dir)
