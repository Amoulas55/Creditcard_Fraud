# Creditcard\_Fraud

This project implements a deep learning pipeline for **credit card fraud detection** using a **hybrid CNN-LSTM model**. It includes comprehensive preprocessing, model training with hyperparameter tuning via Optuna, evaluation with multiple metrics and plots, and model explainability using SHAP. All experiments are based on the **Kaggle Credit Card Fraud Detection Dataset**.

---

## 📁 Project Structure

```
Creditcard_Fraud/
│
├── EDA/
│   ├── amount_distribution.png
│   ├── class_distribution.png
│   ├── correlation_matrix.png
│   ├── eda.py
│   └── eda_summary.txt
│
├── evaluation/
│   ├── confusion_matrix.png
│   ├── final_metrics.txt
│   ├── prc_curve.png
│   ├── roc_curve.png
│   ├── save_results.py
│   └── threshold_vs_f1.png
│
├── explainability/
│   ├── shap_analysis.py
│   └── shap_summary.png
│
├── preprocessing/
│   ├── preprocess.py
│   ├── reshape_for_cnn_lstm.py
│   └── split_and_smote.py
│
├── training/
│   ├── final_model.py
│   ├── generate_cnnlstm_full.py
│   └── train_cnn_lstm_optuna.py
│
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* GPU recommended (for training with PyTorch)

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install requirements
pip install -r requirements.txt
```

### Dataset

Download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in a private `data/` directory (excluded from GitHub).

---

## 🧹 Pipeline Overview

### 1. Preprocessing

* File: `preprocessing/preprocess.py`
* Normalizes and scales features, drops irrelevant columns

### 2. SMOTE Balancing & Splitting

* File: `preprocessing/split_and_smote.py`
* Applies SMOTE to training data and splits into train/val/test

### 3. Reshaping for CNN-LSTM

* File: `preprocessing/reshape_for_cnn_lstm.py`
* Reshapes data into sequences for hybrid model

### 4. Training with Optuna

* File: `training/train_cnn_lstm_optuna.py`
* Optimizes hyperparameters

### 5. Final Model Training

* File: `training/final_model.py`
* Trains with best parameters on full training data

---

## 📊 Results

Metrics based on best CNN–LSTM model:

```
Accuracy      : 1.00
F1 Score      : 0.8394
ROC AUC       : 0.9651
PRC AUC       : 0.8093

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.85      0.83      0.84        98

    accuracy                           1.00     56962
   macro avg       0.93      0.91      0.92     56962
weighted avg       1.00      1.00      1.00     56962
```

---

## 📈 Visualizations

Located in `EDA/`, `evaluation/`, and `explainability/`:

* Confusion matrix
* ROC curve, PRC curve
* SHAP feature importance
* Threshold vs F1
* Distribution plots
* Correlation heatmap

---

## 🧠 Explainability

* File: `explainability/shap_analysis.py`
* Uses SHAP to show feature impact on CNN-LSTM predictions

---

## 📄 License

This project is open-source under the MIT License.

---

## 👨‍💻 Author

**Angelos Moulas**
MSc in Data Science & Society
Tilburg University

Feel free to ⭐ the repository if you find it helpful!

---

## 📬 Contact

For questions or feedback, open an issue or contact via [GitHub](https://github.com/Amoulas55).
