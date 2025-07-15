import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Paths
base_dir = "/home/u762545/opekepe"
input_csv = os.path.join(base_dir, "preprocessed", "creditcard_normalized.csv")
output_dir = os.path.join(base_dir, "preprocessed")
os.makedirs(output_dir, exist_ok=True)

# Load normalized data
df = pd.read_csv(input_csv)
X = df.drop(columns=["Class"])
y = df["Class"]

# Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Apply SMOTE on training data only
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Save .npy arrays
np.save(os.path.join(output_dir, "X_train.npy"), X_train_res)
np.save(os.path.join(output_dir, "y_train.npy"), y_train_res)
np.save(os.path.join(output_dir, "X_test.npy"), X_test)
np.save(os.path.join(output_dir, "y_test.npy"), y_test)

# Optional: Save resampled training set as CSV
df_train_resampled = pd.DataFrame(X_train_res, columns=X.columns)
df_train_resampled["Class"] = y_train_res.values
df_train_resampled.to_csv(os.path.join(output_dir, "train_smote.csv"), index=False)

# Save test set as CSV
df_test = pd.DataFrame(X_test, columns=X.columns)
df_test["Class"] = y_test.values
df_test.to_csv(os.path.join(output_dir, "test_untouched.csv"), index=False)

# Reporting
print("✅ Split + SMOTE complete.")
print(f"🔸 Resampled train shape: {X_train_res.shape}")
print(f"🔹 Test shape (untouched): {X_test.shape}")
print(f"📁 Files saved in: {output_dir}")
