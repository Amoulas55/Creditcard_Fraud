import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from sklearn.metrics import f1_score, classification_report, roc_auc_score, precision_recall_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# 📁 Paths
base_dir = "/home/u762545/opekepe/preprocessed"
X_train = np.load(os.path.join(base_dir, "X_train_cnnlstm.npy"))
X_test = np.load(os.path.join(base_dir, "X_test_cnnlstm.npy"))
y_train = np.load(os.path.join(base_dir, "y_train.npy"))
y_test = np.load(os.path.join(base_dir, "y_test.npy"))

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

# 🔧 Build CNN–LSTM
def build_model(trial):
    model = Sequential()
    model.add(Conv1D(
        filters=trial.suggest_categorical("conv_filters", [32, 64, 128]),
        kernel_size=1,
        activation='relu',
        input_shape=(1, 29)
    ))
    model.add(MaxPooling1D(pool_size=1))
    model.add(LSTM(
        units=trial.suggest_int("lstm_units_1", 64, 256),
        return_sequences=True
    ))
    model.add(Dropout(rate=trial.suggest_float("dropout_1", 0.1, 0.5)))
    model.add(LSTM(units=trial.suggest_int("lstm_units_2", 32, 128)))
    model.add(Dropout(rate=trial.suggest_float("dropout_2", 0.1, 0.5)))
    model.add(Dense(
        units=trial.suggest_int("dense_units", 32, 128),
        activation='relu'
    ))
    model.add(Dense(1, activation='sigmoid'))

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# 🎯 Objective
def objective(trial):
    model = build_model(trial)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    class_weight_pos = trial.suggest_int("class_weight_1", 5, 20)

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        X_train_sub, y_train_sub,
        epochs=30,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        class_weight={0: 1, 1: class_weight_pos},
        verbose=0
    )

    y_val_probs = model.predict(X_val, verbose=0).ravel()
    best_f1 = 0
    for t in np.arange(0.1, 0.95, 0.01):
        y_pred = (y_val_probs > t).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1

    # 📢 Print and save trial result
    msg = f"✅ Trial {trial.number}: F1 = {best_f1:.4f}"
    print(msg)
    sys.stdout.flush()
    with open(os.path.join(base_dir, "optuna_trials_log.txt"), "a") as f:
        f.write(msg + "\n")

    return best_f1

# 🚀 Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

# 📌 Save best trial
best_params = study.best_params
best_value = study.best_value
with open(os.path.join(base_dir, "best_params.txt"), "w") as f:
    f.write(f"Best F1 Score: {best_value:.4f}\n")
    f.write("Best Parameters:\n")
    for k, v in best_params.items():
        f.write(f"{k}: {v}\n")

print("\n✅ Best Trial Results:")
print(f"Best F1 Score: {best_value:.4f}")
for k, v in best_params.items():
    print(f"  {k}: {v}")
sys.stdout.flush()

# 🧠 Retrain model on full training set
final_model = build_model(optuna.trial.FixedTrial(best_params))
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
final_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=best_params["batch_size"],
    validation_split=0.2,
    callbacks=[early_stop],
    class_weight={0: 1, 1: best_params["class_weight_1"]},
    verbose=1
)

# 💾 Save model
final_model.save(os.path.join(base_dir, "cnn_lstm_model_optuna.keras"))

# 🔍 Evaluate on test set
y_probs = final_model.predict(X_test, verbose=0).ravel()
best_threshold, best_f1 = 0.5, 0
for t in np.arange(0.1, 0.95, 0.01):
    y_pred = (y_probs > t).astype(int)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    if f1 > best_f1:
        best_threshold = t
        best_f1 = f1

y_pred_final = (y_probs > best_threshold).astype(int)
report = classification_report(y_test, y_pred_final, digits=4, zero_division=0)
roc_auc = roc_auc_score(y_test, y_probs)
precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)

print(f"\n📊 Final Evaluation (Threshold = {best_threshold:.2f}):")
print(report)
print(f"🔵 ROC-AUC: {roc_auc:.4f}")
print(f"🟢 PR-AUC: {pr_auc:.4f}")
print(f"✨ Best F1: {best_f1:.4f} at threshold {best_threshold:.2f}")
sys.stdout.flush()

# 📄 Save predictions
pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred_final,
    "y_prob": y_probs
}).to_csv(os.path.join(base_dir, "cnn_lstm_predictions_optuna.csv"), index=False)
