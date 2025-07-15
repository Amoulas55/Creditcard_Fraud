# Creditcard\_Fraud

This repository contains a deep learning-based approach to detect fraudulent transactions in credit card datasets using a CNN-LSTM hybrid architecture. The model is trained and evaluated on the popular Kaggle dataset for credit card fraud detection.

## Project Structure

```
Creditcard_Fraud/
├── EDA/
│   ├── amount_distribution.png
│   ├── class_distribution.png
│   ├── correlation_matrix.png
│   ├── eda.py
│   └── eda_summary.txt
├── evaluation/
│   ├── confusion_matrix.png
│   ├── final_metrics.txt
│   ├── prc_curve.png
│   ├── roc_curve.png
│   ├── save_results.py
│   └── threshold_vs_f1.png
├── explainability/
│   ├── shap_analysis.py
│   └── shap_summary.png
├── preprocessing/
│   ├── preprocess.py
│   ├── reshape_for_cnn_lstm.py
│   └── split_and_smote.py
├── training/
│   ├── final_model.py
│   ├── generate_cnnlstm_full.py
│   └── train_cnn_lstm_optuna.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Description

This project implements a pipeline for:

* **Exploratory Data Analysis (EDA)**: Understanding class imbalance, transaction amount distribution, and feature correlations.
* **Preprocessing**: Standardizing data, applying SMOTE for class balancing, and reshaping for CNN-LSTM input.
* **Model Training**: A hybrid CNN-LSTM model is trained using PyTorch. Hyperparameters are optimized with Optuna.
* **Evaluation**: Includes confusion matrix, ROC, PRC, F1-threshold optimization, and final performance metrics.
* **Explainability**: SHAP values are computed to interpret the model's predictions.

## Getting Started

### Requirements

Install all necessary packages using:

```bash
pip install -r requirements.txt
```

### Running the Code

1. **Run preprocessing:**

   ```bash
   python preprocessing/preprocess.py
   python preprocessing/split_and_smote.py
   python preprocessing/reshape_for_cnn_lstm.py
   ```

2. **Train model with Optuna tuning:**

   ```bash
   python training/train_cnn_lstm_optuna.py
   ```

3. **Train final model:**

   ```bash
   python training/generate_cnnlstm_full.py
   python training/final_model.py
   ```

4. **Run evaluation and visualization:**

   ```bash
   python evaluation/save_results.py
   ```

5. **Run SHAP analysis (optional):**

   ```bash
   python explainability/shap_analysis.py
   ```

## Results

* **ROC AUC**: 0.97
* **PRC AUC**: 0.81
* **Best F1 Threshold**: 0.48
* **Confusion Matrix**: Excellent fraud detection with very few false negatives

## License

MIT License

## Author

Angelos Moulas
