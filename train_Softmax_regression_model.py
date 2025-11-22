# ----------------------------------------------------
# Script: train_softmax_model.py
# Purpose: Trains a Softmax Regression Classifier for WDBC dataset.
# Saves the model, scaler, and performance metrics to the 'models/' directory.
# ----------------------------------------------------
import pandas as pd
import numpy as np
import warnings
import tensorflow as tf
import json 
import pickle # For saving the scaler
import os # <--- ADDED: Required for os.makedirs

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ----------------------------------------------------
# Data Loading and Preprocessing
# ----------------------------------------------------

# NOTE: This script assumes 'data.csv' is accessible.
try:
    df = pd.read_csv("data.csv")
except FileNotFoundError:
    print("Error: 'data.csv' not found. Please ensure the data file is available.")
    exit()

# Data Cleaning
df = df.drop(columns=['Unnamed: 32', 'id'], errors='ignore')

# Encode diagnosis: Malignant (M) -> 1, Benign (B) -> 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

print("Final dataset ready!")
print(f"Shape: {df.shape} → ({df.shape[0]}, {df.shape[1]}) with 30 features + diagnosis")

# Separate features (X) and label (y)
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Train/test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y 
)

print("\nSplit shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# Scale features (StandardScaler is shared across all models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\nScaled versions ready!")

# ----------------------------------------------------
# Evaluation and Logging Function
# ----------------------------------------------------

def calculate_metrics(name, y_true, y_pred, y_prob, epochs):
    """Calculates and returns key classification metrics as a structured dictionary."""
    
    # y_true must be the original 0/1 labels for sklearn metrics
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) # y_prob must be the probability of class 1

    # Confusion matrix: tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Derived rates
    FPR = fp / (fp + tn) if (fp + tn) > 0 else 0 
    FNR = fn / (fn + tp) if (fn + tp) > 0 else 0 
    TPR = tp / (tp + fn) if (tp + fn) > 0 else 0 
    TNR = tn / (tn + fp) if (tn + fp) > 0 else 0 

    data_points = len(y_true)

    # Print results to console
    print(f"\n{name:30} → Accuracy: {acc:.4f} | AUC: {auc:.4f}")
    print(f"  FPR: {FPR:.4f} | FNR: {FNR:.4f} | TPR: {TPR:.4f} | TNR: {TNR:.4f}")

    # Return structured metrics 
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
# Keras Softmax Regression Classifier
# ----------------------------------------------------
# User-specified hyperparameters
EPOCHS = 3000
BATCH_SIZE = 128
LEARNING_RATE = 1e-3 

MODEL_NAME = "Keras Softmax Regression Classifier"
MODEL_FILENAME = "models/train_softmax_regression_model.h5"
SCALER_FILENAME = "models/scaler.pkl"
METRICS_FILENAME = "models/softmax_regression_model_metrics.json"

print(f"\n--- Training {MODEL_NAME} ---")

# Convert labels to categorical (one-hot encoding) for softmax output layer (2 units)
y_train_categorical = to_categorical(y_train)

# 1. Model Definition: Single Dense layer with 2 units (M/B) and Softmax activation
model_soft = Sequential([
    Dense(2, input_shape=(30,), activation='softmax')
])

# 2. Compilation: Using SGD and categorical_crossentropy loss
model_soft.compile(
    optimizer=SGD(learning_rate=LEARNING_RATE), 
    loss='categorical_crossentropy'
)

# 3. Training
history = model_soft.fit(
    X_train_scaled, 
    y_train_categorical, # Use one-hot encoded labels here
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    verbose=0
)

# 4. Evaluation
# Model predicts probabilities for both classes: [P(B), P(M)]
prob_full = model_soft.predict(X_test_scaled, verbose=0) 
prob_soft = prob_full[:, 1] # Extract the probability of class 1 (Malignant)
pred_soft = (prob_soft > 0.5).astype(int)

# Calculate and store all metrics (using original y_test and prob_soft for AUC)
model_results = calculate_metrics(MODEL_NAME, y_test, pred_soft, prob_soft, epochs=EPOCHS)


# ----------------------------------------------------
# Saving Assets
# ----------------------------------------------------

# A. Save Keras Model
try:
    # Ensure the 'models' directory exists
    os.makedirs('models', exist_ok=True)
    model_soft.save(MODEL_FILENAME)
    print(f"\n--- SUCCESS: Softmax Model saved to {MODEL_FILENAME} ---")
except Exception as e:
    print(f"\n--- ERROR: Could not save Softmax Keras model ---")
    print(f"Detail: {e}")

# B. Save Scaler (Overwrites the old one, but ensures it's up-to-date)
try:
    with open(SCALER_FILENAME, 'wb') as file:
        pickle.dump(scaler, file)
    print(f"--- SUCCESS: Scaler saved to {SCALER_FILENAME} ---")
except Exception as e:
    print(f"--- ERROR: Could not save Scaler ---")
    print(f"Detail: {e}")

# C. Save Metrics to JSON
try:
    with open(METRICS_FILENAME, 'w') as f:
        json.dump(model_results, f, indent=2)
    print(f"--- SUCCESS: Metrics saved to {METRICS_FILENAME} ---")
except Exception as e:
    print(f"--- ERROR: Could not save Metrics to JSON ---")
    print(f"Detail: {e}")