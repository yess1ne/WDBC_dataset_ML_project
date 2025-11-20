# ----------------------------------------------------
# Essential Libraries and Dependencies
# ----------------------------------------------------
import pandas as pd
import numpy as np
import warnings
import tensorflow as tf
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

# Encode diagnosis: Malignant (M) -> 1, Benign (B) -> 0
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
# We keep this function as it relies only on numpy and sklearn metrics
results = []

def log_result(name, y_true, y_pred, y_prob, epochs=None):
    """Calculates and logs key classification metrics."""
    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    # Confusion matrix: tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Derived rates
    FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
    FNR = fn / (fn + tp) if (fn + tp) > 0 else 0
    TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
    TNR = tn / (tn + fp) if (tn + fp) > 0 else 0

    data_points = len(y_true)

    results.append([
        name, acc, auc, data_points, epochs, FPR, FNR, TPR, TNR
    ])

    print(f"\n{name:18} → Accuracy: {acc:.4f} | AUC: {auc:.4f}")
    print(f"  FPR: {FPR:.4f} | FNR: {FNR:.4f} | TPR: {TPR:.4f} | TNR: {TNR:.4f}")

# ----------------------------------------------------
# Keras Linear Classifier (User's Exact Model)
# ----------------------------------------------------
print("\n--- Training Linear Classifier (Keras) ---")

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
# Hyperparameters: epochs=3000, batch_size=128
history = model_lr.fit(
    X_train_scaled, 
    y_train, 
    epochs=3000, 
    batch_size=128, 
    verbose=0
)

# 4. Evaluation
# Prediction: The raw output score (prob_lr is continuous, not strictly probability)
prob_lr = model_lr.predict(X_test_scaled, verbose=0).flatten()

# Classification: Apply the 0.5 threshold to get binary prediction (0 or 1)
pred_lr = (prob_lr > 0.5).astype(int)

# Log the results
log_result("Keras LinReg Sim", y_test, pred_lr, prob_lr, epochs=3000)

# ----------------------------------------------------
# Saving the Model
# ----------------------------------------------------
# Keras models MUST be saved using the model.save() method.
# We will save it as 'train_linear_regression_model.h5' for reliability.
# Note: The extension is changed from .pkl to .h5 because Keras models do not reliably pickle.
# The user's intent to use it later requires the proper Keras save format.

model_filename = "models/train_linear_regression_model.h5"

try:
    model_lr.save(model_filename)
    print(f"\n--- SUCCESS: Model saved to {model_filename} ---")
    print(f"You can load this model in app.py using: tf.keras.models.load_model('{model_filename}')")
except Exception as e:
    print(f"\n--- ERROR: Could not save Keras model ---")
    print(f"Detail: {e}")

# ----------------------------------------------------
# Saving the Scaler
# ----------------------------------------------------
# It is CRITICAL to save the scaler object as well, so incoming data in app.py can be scaled correctly.
import pickle

scaler_filename = "models/scaler.pkl"

try:
    with open(scaler_filename, 'wb') as file:
        pickle.dump(scaler, file)
    print(f"--- SUCCESS: Scaler saved to {scaler_filename} ---")
    print(f"You can load this scaler in app.py using: pickle.load(open('{scaler_filename}', 'rb'))")
except Exception as e:
    print(f"--- ERROR: Could not save Scaler ---")
    print(f"Detail: {e}")