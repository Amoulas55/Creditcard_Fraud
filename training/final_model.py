import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score, classification_report, precision_recall_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# 📁 Paths
data_dir = "/home/u762545/opekepe/preprocessed"
X = np.load(os.path.join(data_dir, "X_full_cnnlstm.npy"))
y = np.load(os.path.join(data_dir, "y_full.npy"))

# 🧪 Resplit into train (64%), val (16%), test (20%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42)

print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

# 🖥️ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🖥️ Using device:", device)

# 🧠 Model
class CNNLSTM(nn.Module):
    def __init__(self, conv_filters, lstm_units_1, lstm_units_2, dense_units, dropout_1, dropout_2):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv_filters, kernel_size=1)
        self.pool = nn.MaxPool1d(kernel_size=1)
        self.lstm1 = nn.LSTM(input_size=conv_filters, hidden_size=lstm_units_1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_1)
        self.lstm2 = nn.LSTM(input_size=lstm_units_1, hidden_size=lstm_units_2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_2)
        self.fc1 = nn.Linear(lstm_units_2, dense_units)
        self.fc2 = nn.Linear(dense_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x[:, -1, :])
        x = self.fc1(x)
        x = self.fc2(x)
        return self.sigmoid(x)

# ✅ Best hyperparameters from Optuna
params = {
    "conv_filters": 64,
    "lstm_units_1": 128,
    "lstm_units_2": 64,
    "dense_units": 64,
    "dropout_1": 0.3,
    "dropout_2": 0.3,
    "lr": 0.0005,
    "batch_size": 64
}

# 🔧 Data preparation
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)),
    batch_size=params["batch_size"],
    shuffle=True
)

val_X_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
val_y_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
test_X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
test_y_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# 🚀 Model setup
model = CNNLSTM(
    params["conv_filters"],
    params["lstm_units_1"],
    params["lstm_units_2"],
    params["dense_units"],
    params["dropout_1"],
    params["dropout_2"]
).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=params["lr"])

print("🔧 Training final model...")
train_losses, val_losses = [], []

for epoch in range(10):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        val_preds = model(val_X_tensor)
        val_loss = criterion(val_preds, val_y_tensor).item()

    train_losses.append(total_loss / len(train_loader))
    val_losses.append(val_loss)

print("✅ Training complete.")

# 🧪 Best threshold on val set
model.eval()
with torch.no_grad():
    val_probs = model(val_X_tensor).cpu().numpy().ravel()
    test_probs = model(test_X_tensor).cpu().numpy().ravel()

best_thresh, best_f1 = 0.5, 0
for t in np.arange(0.1, 0.95, 0.01):
    f1 = f1_score(y_val, (val_probs > t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

# 🎯 Final evaluation on test set
test_preds = (test_probs > best_thresh).astype(int)
f1 = f1_score(y_test, test_preds)
roc = roc_auc_score(y_test, test_probs)
prec, rec, _ = precision_recall_curve(y_test, test_probs)
prc_auc = auc(rec, prec)
cm = confusion_matrix(y_test, test_preds)
report = classification_report(y_test, test_preds, output_dict=True)

print("\n🏁 Final Test Set Results:")
print(f"Best Threshold (from val): {best_thresh:.2f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc:.4f}")
print(f"PRC AUC: {prc_auc:.4f}\n")

# 💾 Save CSVs for later plotting
output_dir = "/home/u762545/opekepe"
pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses}).to_csv(os.path.join(output_dir, "final_losses.csv"), index=False)
pd.DataFrame({"precision": prec, "recall": rec}).to_csv(os.path.join(output_dir, "final_prc.csv"), index=False)
pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"]).to_csv(os.path.join(output_dir, "final_confusion_matrix.csv"))
pd.DataFrame(report).transpose().to_csv(os.path.join(output_dir, "final_classification_report.csv"))
np.save(os.path.join(output_dir, "final_test_probs.npy"), test_probs)
np.save(os.path.join(output_dir, "final_test_preds.npy"), test_preds)

print("📦 Final model and test report saved.")
