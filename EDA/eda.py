import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths
base_dir = "/home/u762545/opekepe"
eda_dir = os.path.join(base_dir, "EDA")
os.makedirs(eda_dir, exist_ok=True)

# Load dataset
file_path = os.path.join(base_dir, "creditcard.csv")
df = pd.read_csv(file_path)

# Prepare EDA summary
eda_report = []

eda_report.append(f"📊 Shape of dataset: {df.shape}\n\n")
eda_report.append("🧾 Data types:\n")
eda_report.append(f"{df.dtypes}\n\n")

eda_report.append("🔎 Missing values:\n")
eda_report.append(f"{df.isnull().sum()}\n\n")

# Class distribution
class_counts = df['Class'].value_counts()
fraud_ratio = class_counts[1] / class_counts.sum()
eda_report.append("⚖️ Class distribution:\n")
eda_report.append(f"{class_counts}\n")
eda_report.append(f"Fraud Ratio: {fraud_ratio:.6f}\n\n")

# Descriptive stats
eda_report.append("📈 Descriptive Statistics:\n")
eda_report.append(f"{df.describe()}\n\n")

# Save summary to txt
with open(os.path.join(eda_dir, "eda_summary.txt"), "w") as f:
    f.writelines(eda_report)

# Class distribution plot
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Class', palette='Set2')
plt.title("Class Distribution")
plt.savefig(os.path.join(eda_dir, "class_distribution.png"))
plt.close()

# Amount distribution using matplotlib
plt.figure(figsize=(8,5))
plt.hist(df['Amount'], bins=100, color='skyblue', edgecolor='black')
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.savefig(os.path.join(eda_dir, "amount_distribution.png"))
plt.close()

# Correlation matrix
plt.figure(figsize=(20, 15))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix")
plt.savefig(os.path.join(eda_dir, "correlation_matrix.png"))
plt.close()
