# ----------------------------------------------------
# Essential Libraries and Dependencies
# ----------------------------------------------------
import pandas as pd
import numpy as np
import warnings
import json 
import joblib # Using joblib for saving scikit-learn models/scalers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.svm import SVC # Support Vector Classifier

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

# ----------------------------------------------------
# SVM SCALER MANDATE (CRITICAL)
# ----------------------------------------------------
# SVM is a distance-based algorithm. Without scaling, features with larger
# magnitudes (like 'area_worst' in the thousands) would numerically dominate 
# the distance calculations, causing the smaller, but equally important, features
# (like 'smoothness_mean' in the hundredths) to be effectively ignored.
# Standard Scaling (mean=0, std=1) ensures all 30 features contribute equally.
# The exact same fitted scaler (mean/std derived from X_train) must be saved
# and used for all future predictions on new data.

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\nScaled versions ready! (Crucial for SVM)")

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
    # labels=[0, 1] ensures that 0 is the negative class (Benign) and 1 is the positive (Malignant)
    # The .ravel() output order is (tn, fp, fn, tp)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Derived rates
    FPR = fp / (fp + tn) if (fp + tn) > 0 else 0 # False Positive Rate
    FNR = fn / (fn + tp) if (fn + tp) > 0 else 0 # False Negative Rate
    TPR = tp / (tp + fn) if (tp + fn) > 0 else 0 # True Positive Rate (Sensitivity/Recall)
    TNR = tn / (tn + fp) if (tn + fp) > 0 else 0 # True Negative Rate (Specificity)

    data_points = len(y_true)

    # Print results to console
    print(f"\n{name:18} → Accuracy: {acc:.4f} | AUC: {auc:.4f}")
    print(f" FPR: {FPR:.4f} | FNR: {FNR:.4f} | TPR: {TPR:.4f} | TNR: {TNR:.4f}")

    # Return structured metrics matching the format expected by model_metrics.json
    return {
        "model_name": name,
        "data_points": data_points,
        # Note: SVM training time is very fast and iterations are not true 'epochs'
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
# Support Vector Machine (SVM) Classifier
# ----------------------------------------------------
# Hyperparameters from the Configuration Table/Prompt:
C_VALUE = 5.0
KERNEL_TYPE = 'rbf'
MAX_ITER = 3000 # Corresponds to 'Epochs'
MODEL_NAME = "Support Vector Machine (C=5.0, RBF)"
MODEL_FILENAME = "models/train_svm_model.joblib"
SCALER_FILENAME = "models/scaler_svm.joblib"
METRICS_FILENAME = "models/svm_model_metrics.json"

print(f"\n--- Training {MODEL_NAME} ---")

# 1. Model Definition
# probability=True is required to output prediction probabilities (for AUC score)
# L2 Norm (Regularization) is implicitly used via the C parameter in SVC
model_svm = SVC(
    C=C_VALUE,
    kernel=KERNEL_TYPE,
    max_iter=MAX_ITER,
    probability=True,
    random_state=42
)

# 2. Training
# Note: Batch size 128 is a concept for iterative optimizers (like Keras/SGD). 
# SVC uses the libsvm solver which handles optimization internally, so batch_size 
# and learning_rate are not directly passed here.
model_svm.fit(X_train_scaled, y_train)

# 3. Evaluation
# Get predicted class (0 or 1)
pred_svm = model_svm.predict(X_test_scaled)
# Get probability of class 1 (Malignant) for AUC calculation
prob_svm = model_svm.predict_proba(X_test_scaled)[:, 1] 

# Calculate and store all metrics
model_results = calculate_metrics(MODEL_NAME, y_test, pred_svm, prob_svm, epochs=MAX_ITER)


# ----------------------------------------------------
# Saving Assets
# ----------------------------------------------------

# A. Save Scikit-learn Model using joblib (Standard practice)
try:
    joblib.dump(model_svm, MODEL_FILENAME)
    print(f"\n--- SUCCESS: Model saved to {MODEL_FILENAME} ---")
except Exception as e:
    print(f"\n--- ERROR: Could not save SVM model ---")
    print(f"Detail: {e}")

# B. Save Scaler using joblib
try:
    joblib.dump(scaler, SCALER_FILENAME)
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