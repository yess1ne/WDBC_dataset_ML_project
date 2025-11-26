# ----------------------------------------------------
# MLP Training Script for WDBC Dataset (Non-Linear Classifier)
# Optimized with Custom Hyperparameters (Heavy Training)
# ----------------------------------------------------
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam # Import Adam optimizer

# --- Configuration ---
DATA_FILE = "data.csv"
MODEL_FILENAME = "models/train_MLP_model.h5"
METRICS_FILENAME = "models/MLP_model_metrics.json"
SCALER_FILENAME = "models/scaler.pkl" # Reuse the existing scaler
os.makedirs("models", exist_ok=True)

# --- CUSTOM HYPERPARAMETERS ---
BATCH_SIZE = 128 
HIDDEN_LAYERS = [500, 500, 500] # Cell Size: 3 layers, 500 neurons each
EPOCHS = 3000
LEARNING_RATE = 0.01 # 1e-2

ACTIVATION = 'relu'
OUTPUT_ACTIVATION = 'sigmoid' # Binary classification

# --- Custom Metrics Calculation ---
def calculate_all_metrics(y_true, y_pred_proba):
    """Calculates all key classification metrics from true labels and probabilities."""
    
    # Convert probabilities to binary predictions using 0.5 threshold
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    
    # Calculate core metrics
    acc = accuracy_score(y_true, y_pred_binary)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # Calculate Confusion Matrix: TN, FP, FN, TP
    # Handle case where confusion matrix might be 1x1 if only one class is predicted
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    except ValueError:
        # Fallback for perfect classification or edge cases
        cm = confusion_matrix(y_true, y_pred_binary)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        elif cm.shape == (1, 1):
             # If y_true is all 0s and y_pred is all 0s, tn = cm[0,0], else tp = cm[0,0]
             if np.sum(y_true) == 0:
                 tn, fp, fn, tp = cm[0,0], 0, 0, 0
             else:
                 tn, fp, fn, tp = 0, 0, 0, cm[0,0]
        else:
             tn, fp, fn, tp = 0, 0, 0, 0
    
    # Calculate Rates
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Sensitivity)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate (Specificity)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate

    return {
        "Accuracy": f"{acc:.4f}",
        "AUC (ROC Area)": f"{auc:.4f}",
        "TPR (True Positive Rate / Sensitivity)": f"{tpr:.4f}",
        "TNR (True Negative Rate / Specificity)": f"{tnr:.4f}",
        "FPR (False Positive Rate)": f"{fpr:.4f}",
        "FNR (False Negative Rate)": f"{fnr:.4f}",
    }

# --- Data Loading and Preparation ---
print("--- 1. Data Loading and Preparation ---")
try:
    df = pd.read_csv(DATA_FILE)
    df.drop('id', axis=1, inplace=True)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    # FIX: Drop the extraneous unnamed column common in WDBC dataset
    df = df.drop(columns=['Unnamed: 32'], errors='ignore')
    
    X = df.drop('diagnosis', axis=1).values
    y = df['diagnosis'].values
    
    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for deployment if it doesn't already exist
    if not os.path.exists(SCALER_FILENAME):
        with open(SCALER_FILENAME, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"SUCCESS: StandardScaler saved to {SCALER_FILENAME}.")
    else:
        print(f"StandardScaler already exists at {SCALER_FILENAME}. Skipping save.")

except Exception as e:
    print(f"ERROR during data loading/scaling: {e}")
    exit()

# --- Model Definition ---
print("\n--- 2. Building MLP Model ---")
try:
    model = Sequential(name="MLP_Classifier")
    
    # Input Layer (implicitly defined by the first hidden layer's input_shape)
    
    # Hidden Layers
    model.add(Dense(units=HIDDEN_LAYERS[0], activation=ACTIVATION, input_shape=(X_train_scaled.shape[1],)))
    for units in HIDDEN_LAYERS[1:]:
        model.add(Dense(units=units, activation=ACTIVATION))
        
    # Output Layer (1 neuron with sigmoid for binary classification)
    model.add(Dense(units=1, activation=OUTPUT_ACTIVATION))

    # Define custom optimizer with the specified learning rate
    custom_optimizer = Adam(learning_rate=LEARNING_RATE)

    # Compile the model
    model.compile(optimizer=custom_optimizer, 
                  loss='binary_crossentropy', # Appropriate loss for binary probability output
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    model.summary()

except Exception as e:
    print(f"ERROR defining/compiling model: {e}")
    exit()
    
# --- Model Training ---
print(f"\n--- 3. Training MLP Model for {EPOCHS} Epochs (Heavy Load) ---")
try:
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test_scaled, y_test),
        verbose=1 # Increased verbosity to monitor progress during heavy training
    )
    print("SUCCESS: Model training complete.")
except Exception as e:
    print(f"ERROR during model training: {e}")
    exit()

# --- Evaluation and Saving ---
print("\n--- 4. Evaluating and Saving Assets ---")
try:
    # 4A. Evaluate on Test Set
    y_test_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
    metrics = calculate_all_metrics(y_test, y_test_pred_proba)
    
    # 4B. Save Metrics to JSON
    metrics_data = {
        "model_name": "MLP Classifier (Deep NN)",
        "metrics": metrics
    }
    with open(METRICS_FILENAME, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print(f"SUCCESS: Metrics saved to {METRICS_FILENAME}.")
    
    print("\n--- Test Set Performance ---")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # 4C. Save Model
    model.save(MODEL_FILENAME)
    print(f"SUCCESS: Model saved to {MODEL_FILENAME}.")

except Exception as e:
    print(f"ERROR during evaluation/saving: {e}")

print("\n--- MLP Training Complete ---")