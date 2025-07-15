import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, roc_auc_score, average_precision_score
)
import seaborn as sns

# === Example placeholders ===
# Replace these with your actual predictions and ground truths
y_true = np.load("y_true.npy")
y_probs = np.load("y_probs.npy")  # predicted probabilities
threshold = 0.48
y_pred = (y_probs >= threshold).astype(int)

# === Output Directory ===
output_dir = "final_outputs"
os.makedirs(output_dir, exist_ok=True)

# === Metrics ===
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_probs)
prc_auc = average_precision_score(y_true, y_probs)
conf_matrix = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred)

# === Save metrics to .txt ===
with open(os.path.join(output_dir, "final_metrics.txt"), "w") as f:
    f.write(f"Threshold: {threshold:.2f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n")
    f.write(f"PRC AUC: {prc_auc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

# === ROC Curve ===
fpr, tpr, _ = roc_curve(y_true, y_probs)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(output_dir, "roc_curve.png"))
plt.close()

# === PRC Curve ===
precision, recall, _ = precision_recall_curve(y_true, y_probs)
plt.figure()
plt.plot(recall, precision, label=f"AUC = {prc_auc:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig(os.path.join(output_dir, "prc_curve.png"))
plt.close()

# === Confusion Matrix ===
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

# === Threshold vs F1 Score Curve ===
thresholds = np.linspace(0, 1, 100)
f1_scores = [f1_score(y_true, (y_probs >= t).astype(int)) for t in thresholds]
plt.figure()
plt.plot(thresholds, f1_scores)
plt.axvline(threshold, color='red', linestyle='--', label=f'Best Threshold = {threshold}')
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("Threshold vs. F1 Score")
plt.legend()
plt.savefig(os.path.join(output_dir, "threshold_vs_f1.png"))
plt.close()
