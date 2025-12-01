# --------------------------------------------------------------------------------
# GRU-SVM Hybrid Model Training Script for WDBC
# 
# This script uses a GRU (RNN) for feature extraction and an SVM for final classification.
# The GRU is trained first to generate high-level feature vectors, which are then
# used to train the final, highly accurate SVM classifier.
# --------------------------------------------------------------------------------
import os
import joblib
import json
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_recall_fscore_support

# TensorFlow/Keras imports for the Feature Extractor
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Configuration & Hyperparameters (based on configuration table) ---
HYPER_PARAMETERS = {
    # GRU (Feature Extractor) Params
    "GRU_units": 100,             # RNN units
    "Dense_layer_size": 30,       # Size of the final feature vector for SVM
    "GRU_epochs": 100,            # Max epochs for the preliminary GRU training
    "learning_rate": 0.001,
    "batch_size": 128,
    
    # SVM (Classifier) Params
    "SVM_C": 5,
    "SVM_kernel": "rbf",
    "SVM_max_iter": 3000,         # Max iterations for the SVM solver
    
    "random_state": 42
}

# --- File Paths ---
DATA_PATH = 'data.csv' 
MODELS_DIR = 'models'
MODEL_FILENAME = os.path.join(MODELS_DIR, 'train_gru_svm_model.joblib')       # The final SVM classifier
SCALER_FILENAME = os.path.join(MODELS_DIR, 'scaler_gru_svm.joblib')           # The feature scaler
METRICS_FILENAME = os.path.join(MODELS_DIR, 'gru_svm_model_metrics.json')     # Performance metrics
FEATURE_EXTRACTOR_FILENAME = os.path.join(MODELS_DIR, 'gru_feature_extractor.h5') # Keras model for prediction

# --- Feature Names (Crucial for consistent order) ---
FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", 
    "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", 
    "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", 
    "perimeter_se", "area_se", "smoothness_se", "compactness_se", 
    "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se", 
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", 
    "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", 
    "symmetry_worst", "fractal_dimension_worst"
]

# ----------------------------------------------------
# Data Loading and Preprocessing
# ----------------------------------------------------

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: '{DATA_PATH}' not found. Please ensure the data file is in the script directory.")
    exit()

# Remove garbage columns
df = df.drop(columns=['Unnamed: 32'], errors='ignore')
df = df.drop(columns=['id'], errors='ignore')

# Rename columns to match the list above for safety and clarity
df.columns = ['diagnosis'] + FEATURE_NAMES

# Encode diagnosis: Malignant (M) -> 1, Benign (B) -> 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

print("Final dataset ready!")
print(f"Shape: {df.shape}")

# Separate features (X) and label (y)
X = df[FEATURE_NAMES]
y = df['diagnosis']

# Train/test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=HYPER_PARAMETERS['random_state'],
    stratify=y 
)

print("\nSplit shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\nScaled versions ready! (Crucial for Hybrid Model)")

# ----------------------------------------------------
# GRU RESHAPING MANDATE (CRITICAL)
# ----------------------------------------------------
# GRU layers require 3D input: [samples, timesteps, features_per_timestep]
# We treat the 30 features as 30 timesteps, each with 1 feature.
# [1, 30] -> [1, 30, 1]

X_train_reshaped = X_train_scaled[:, :, np.newaxis]
X_test_reshaped  = X_test_scaled[:, :, np.newaxis]

print(f"GRU Input Shape: {X_train_reshaped.shape}")

# ----------------------------------------------------
# Evaluation and Logging Function
# ----------------------------------------------------

def calculate_metrics(name, y_true, y_pred, y_prob, epochs):
    """Calculates and returns key classification metrics as a structured dictionary."""
    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)

    # Confusion matrix: tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Derived rates
    FPR = fp / (fp + tn) if (fp + tn) > 0 else 0 # False Positive Rate
    FNR = fn / (fn + tp) if (fn + tp) > 0 else 0 # False Negative Rate
    TPR = tp / (tp + fn) if (tp + fn) > 0 else 0 # True Positive Rate (Sensitivity/Recall)
    TNR = tn / (tn + fp) if (tn + fp) > 0 else 0 # True Negative Rate (Specificity)

    data_points = len(y_true)

    # Print results to console
    print(f"\n{name:18} → Accuracy: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")
    print(f" FPR: {FPR:.4f} | FNR: {FNR:.4f} | TPR: {TPR:.4f} | TNR: {TNR:.4f}")

    # Return structured metrics matching the format expected by model_metrics.json
    return {
        "model_name": name,
        "data_points": data_points,
        "epochs": epochs, 
        "metrics": {
            "Accuracy": round(acc, 4),
            "AUC (ROC Area)": round(auc, 4),
            "Precision (PPV)": round(prec, 4),
            "F1 Score": round(f1, 4),
            "FPR (False Positive Rate)": round(FPR, 4),
            "FNR (False Negative Rate)": round(FNR, 4),
            "TPR (True Positive Rate / Sensitivity)": round(TPR, 4),
            "TNR (True Negative Rate / Specificity)": round(TNR, 4),
            "Test Set Size": data_points
        }
    }

# ----------------------------------------------------
# GRU Feature Extractor Definition and Training
# ----------------------------------------------------
print("\n--- Training GRU Feature Extractor (Stage 1) ---")
#[Image of deep learning feature extraction for classification]

def build_gru_feature_extractor():
    """Builds a Keras model to extract features using a GRU layer."""
    # Input shape: (30 timesteps, 1 feature per timestep)
    input_layer = Input(shape=(30, 1))
    
    # GRU layer: returns the output for the last time step (the final feature vector)
    gru_out = GRU(
        HYPER_PARAMETERS['GRU_units'], 
        return_sequences=False, 
        dropout=0.2 # Standard dropout
    )(input_layer)
    
    # Feature vector layer
    feature_vector = Dense(HYPER_PARAMETERS['Dense_layer_size'], activation='relu', name='feature_output')(gru_out)
    
    # Final, temporary classification head (used only for training the GRU weights)
    temp_output = Dense(1, activation='sigmoid', name='temp_classifier')(feature_vector)
    
    # Full Keras model for training
    temp_model = Model(inputs=input_layer, outputs=temp_output)
    
    # Keras model for extraction (only the GRU and Feature Vector part)
    # We slice the model to only output the feature vector layer
    extractor_model = Model(inputs=input_layer, outputs=feature_vector)
    
    return temp_model, extractor_model

# 1. Build models
temp_training_model, gru_extractor = build_gru_feature_extractor()

# 2. Compile and train the full model (to learn optimal GRU weights)
temp_training_model.compile(
    optimizer=Adam(learning_rate=HYPER_PARAMETERS['learning_rate']),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Use early stopping to prevent overfitting the GRU
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)

temp_training_model.fit(
    X_train_reshaped, y_train,
    epochs=HYPER_PARAMETERS['GRU_epochs'],
    batch_size=HYPER_PARAMETERS['batch_size'],
    validation_split=0.1, # Using 10% of train set for internal validation
    callbacks=[es],
    verbose=0
)

print("GRU weights are optimized for feature extraction.")

# 3. Generate Features for SVM
# Use the stripped-down extractor model to generate the new, high-level features
X_train_features = gru_extractor.predict(X_train_reshaped, verbose=0)
X_test_features = gru_extractor.predict(X_test_reshaped, verbose=0)

print(f"Features extracted. New feature dimension for SVM: {X_train_features.shape}")

# ----------------------------------------------------
# Support Vector Machine (SVM) Classification (Stage 2)
# ----------------------------------------------------
MODEL_NAME = "GRU-SVM Hybrid Classifier"

print(f"\n--- Training {MODEL_NAME} ---")

# 1. Model Definition
# Train the final SVM on the extracted features (X_train_features)
model_svm = SVC(
    C=HYPER_PARAMETERS['SVM_C'],
    kernel=HYPER_PARAMETERS['SVM_kernel'],
    max_iter=HYPER_PARAMETERS['SVM_max_iter'],
    probability=True,
    random_state=HYPER_PARAMETERS['random_state']
)

# 2. Training
model_svm.fit(X_train_features, y_train)

# 3. Evaluation
# Get predicted class (0 or 1)
pred_svm = model_svm.predict(X_test_features)
# Get probability of class 1 (Malignant) for AUC calculation
prob_svm = model_svm.predict_proba(X_test_features)[:, 1] 

# Calculate and store all metrics
model_results = calculate_metrics(MODEL_NAME, y_test, pred_svm, prob_svm, epochs=HYPER_PARAMETERS['GRU_epochs'])


# ----------------------------------------------------
# Saving Assets
# ----------------------------------------------------
os.makedirs(MODELS_DIR, exist_ok=True)
print("\n--- Saving Assets ---")

# A. Save Final SVM Model (the Classifier component)
try:
    joblib.dump(model_svm, MODEL_FILENAME)
    print(f"--- SUCCESS: SVM Classifier saved to {MODEL_FILENAME} ---")
except Exception as e:
    print(f"--- ERROR: Could not save SVM Classifier ---")
    print(f"Detail: {e}")

# B. Save Initial Feature Scaler
try:
    joblib.dump(scaler, SCALER_FILENAME)
    print(f"--- SUCCESS: Scaler saved to {SCALER_FILENAME} ---")
except Exception as e:
    print(f"--- ERROR: Could not save Scaler ---")
    print(f"Detail: {e}")

# C. Save GRU Feature Extractor (the essential Keras component for prediction)
try:
    gru_extractor.save(FEATURE_EXTRACTOR_FILENAME)
    print(f"--- SUCCESS: GRU Extractor saved to {FEATURE_EXTRACTOR_FILENAME} ---")
except Exception as e:
    print(f"--- ERROR: Could not save GRU Extractor ---")
    print(f"Detail: {e}")

# D. Save Metrics to JSON
try:
    metrics_data = {
        "model_name": MODEL_NAME,
        "metrics": model_results['metrics'],
        "hyperparameters": HYPER_PARAMETERS
    }
    with open(METRICS_FILENAME, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"--- SUCCESS: Metrics saved to {METRICS_FILENAME} ---")
except Exception as e:
    print(f"--- ERROR: Could not save Metrics to JSON ---")
    print(f"Detail: {e}")

print("\n--- GRU-SVM Training Script Finished ---")