# ----------------------------------------------------
# Essential Libraries and Dependencies
# ----------------------------------------------------
import pandas as pd
import numpy as np
import warnings
import tensorflow as tf
import json # Import json for saving metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ----------------------------------------------------
# Data Loading and Preprocessing
# ----------------------------------------------------

# NOTE: This script assumes 'data.csv' is in the same directory.
try:
    df = pd.read_csv("data.csv")
except FileNotFoundError:
    print("Error: 'data.csv' not found. Please ensure the data file is in the script directory.")
    exit()

# Remove garbage columns
df = df.drop(columns=['Unnamed: 32'], errors='ignore')
df = df.drop(columns=['id'], errors='ignore')

# Encode diagnosis: Malignant (M) -> 1, Benign (B) -> 1, since this is a classification task
# and our model uses 0/1 labels.
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

print("Final dataset ready!")
print(f"Shape: {df.shape} → (569, 31) with 30 features + diagnosis")

# Separate features (X) and label (y)
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Train/test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y       # Keeps M/B ratio the same in both splits
)

print("\nSplit shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\nScaled versions ready!")

# ----------------------------------------------------
# Evaluation and Logging Function
# ----------------------------------------------------
# This function is modified to return the calculated metrics as a dictionary
# instead of appending to a global list.

def calculate_metrics(name, y_true, y_pred, y_prob, epochs):
    """Calculates and returns key classification metrics as a structured dictionary."""
    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    # Confusion matrix: tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Derived rates
    FPR = fp / (fp + tn) if (fp + tn) > 0 else 0 # False Positive Rate
    FNR = fn / (fn + tp) if (fn + tp) > 0 else 0 # False Negative Rate
    TPR = tp / (tp + fn) if (tp + fn) > 0 else 0 # True Positive Rate (Sensitivity/Recall)
    TNR = tn / (tn + fp) if (tn + fp) > 0 else 0 # True Negative Rate (Specificity)

    data_points = len(y_true)

    # Print results to console
    print(f"\n{name:18} → Accuracy: {acc:.4f} | AUC: {auc:.4f}")
    print(f"  FPR: {FPR:.4f} | FNR: {FNR:.4f} | TPR: {TPR:.4f} | TNR: {TNR:.4f}")

    # Return structured metrics matching the format expected by model_metrics.json
    return {
        "model_name": name,
        "data_points": data_points,
        "epochs": epochs,
        "metrics": {
            "Accuracy": round(acc, 4),
            "AUC (ROC Area)": round(auc, 4),
            "FPR (False Positive Rate)": round(FPR, 4),
            "FNR (False Negative Rate)": round(FNR, 4),
            "TPR (True Positive Rate / Sensitivity)": round(TPR, 4),
            "TNR (True Negative Rate / Specificity)": round(TNR, 4),
            "Test Set Size": data_points
        }
    }


# ----------------------------------------------------
# Keras Linear Classifier (User's Exact Model)
# ----------------------------------------------------
EPOCHS = 3000
MODEL_NAME = "Keras Linear Classifier (WDBC Dataset)"
MODEL_FILENAME = "models/train_linear_regression_model.h5"
SCALER_FILENAME = "models/scaler.pkl"
METRICS_FILENAME = "models/linear_regression_model_metrics.json"

print(f"\n--- Training {MODEL_NAME} ---")

# 1. Model Definition: Single Dense layer with linear activation
model_lr = Sequential([
    Dense(1, input_shape=(30,), activation='linear')
])

# 2. Compilation: Using SGD and MSE loss (as requested)
model_lr.compile(
    optimizer=SGD(learning_rate=1e-3), 
    loss='mse'
)

# 3. Training
history = model_lr.fit(
    X_train_scaled, 
    y_train, 
    epochs=EPOCHS, 
    batch_size=128, 
    verbose=0
)

# 4. Evaluation
prob_lr = model_lr.predict(X_test_scaled, verbose=0).flatten()
pred_lr = (prob_lr > 0.5).astype(int)

# Calculate and store all metrics
model_results = calculate_metrics(MODEL_NAME, y_test, pred_lr, prob_lr, epochs=EPOCHS)


# ----------------------------------------------------
# Saving Assets
# ----------------------------------------------------

# A. Save Keras Model
try:
    model_lr.save(MODEL_FILENAME)
    print(f"\n--- SUCCESS: Model saved to {MODEL_FILENAME} ---")
except Exception as e:
    print(f"\n--- ERROR: Could not save Keras model ---")
    print(f"Detail: {e}")

# B. Save Scaler
import pickle
try:
    with open(SCALER_FILENAME, 'wb') as file:
        pickle.dump(scaler, file)
    print(f"--- SUCCESS: Scaler saved to {SCALER_FILENAME} ---")
except Exception as e:
    print(f"--- ERROR: Could not save Scaler ---")
    print(f"Detail: {e}")

# C. Save Metrics to JSON (NEW)
try:
    # Ensure the model_results structure matches what app.py expects
    with open(METRICS_FILENAME, 'w') as f:
        json.dump(model_results, f, indent=2)
    print(f"--- SUCCESS: Metrics saved to {METRICS_FILENAME} ---")
except Exception as e:
    print(f"--- ERROR: Could not save Metrics to JSON ---")
    print(f"Detail: {e}")