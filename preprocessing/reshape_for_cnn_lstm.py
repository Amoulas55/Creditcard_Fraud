import numpy as np
import os

# Set paths
base_dir = "/home/u762545/opekepe/preprocessed"
X_train = np.load(os.path.join(base_dir, "X_train.npy"))
X_test = np.load(os.path.join(base_dir, "X_test.npy"))

# Reshape for CNN-LSTM: (samples, time_steps=1, features=29)
X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Save reshaped versions
np.save(os.path.join(base_dir, "X_train_cnnlstm.npy"), X_train_reshaped)
np.save(os.path.join(base_dir, "X_test_cnnlstm.npy"), X_test_reshaped)

print(f"✅ Reshaped for CNN-LSTM:")
print(f"🔸 X_train_cnnlstm shape: {X_train_reshaped.shape}")
print(f"🔹 X_test_cnnlstm shape: {X_test_reshaped.shape}")
