import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

# Define your CNN–LSTM model class structure (must match how the model was trained)
class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cnn_out_channels, kernel_size, lstm_layers, num_classes):
        super(CNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=cnn_out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(input_size=cnn_out_channels, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, features, seq_len) for CNN
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, features) for LSTM
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Load the test data
X_test = np.load("X_test_cnnlstm.npy")
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Load the trained model
input_size = X_test.shape[2]
sequence_length = X_test.shape[1]
model = CNNLSTM(input_size=input_size, hidden_size=64, cnn_out_channels=32, kernel_size=3, lstm_layers=1, num_classes=1)
model.load_state_dict(torch.load("final_cnnlstm_model_pytorch.pth", map_location=torch.device('cpu')))
model.eval()

# Define a wrapper to use with SHAP
def model_predict(x_numpy):
    x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(x_tensor)
        return torch.sigmoid(outputs).numpy()

# Use SHAP DeepExplainer
background = X_test[:100]  # smaller sample for SHAP baseline
explainer = shap.DeepExplainer(model, torch.tensor(background, dtype=torch.float32))
shap_values = explainer.shap_values(torch.tensor(X_test[:50], dtype=torch.float32))  # sample for speed

# Plot summary
shap.summary_plot(shap_values[0], X_test[:50], feature_names=[f'feat_{i}' for i in range(X_test.shape[2])])
