# ----------------------------------------------------
# Flask Application: WDBC Prediction Dashboard - Scalable Version
# ----------------------------------------------------
import os
import pickle
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf

# Suppress TensorFlow warnings/logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Initialization ---
app = Flask(__name__)

# --- Global Configuration Variables ---
# SCALER remains global as it is shared across all models.
SCALER = None 

# NEW: Configuration dictionary to hold all models and their associated assets.
# This makes it easy to add new models (e.g., 'logistic_regression', 'svm', etc.) later.
MODELS_CONFIG = {
    "linear_regression": {
        "name": "Linear Regression Classifier", # Descriptive name for display
        "path": "models/train_linear_regression_model.h5",
        "metrics": "models/linear_regression_model_metrics.json",
        "instance": None # Placeholder for the loaded Keras model instance
    }
    # Future models will be added here:
    # "logistic_regression": { ... }
}

# Feature names (used to ensure input order matches training order)
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

# Path to the scaler file (remains the same as it is shared)
SCALER_FILENAME = "models/scaler.pkl"


# --- Asset Loading Function (Modified for Scalability) ---
def load_assets():
    """Loads the shared StandardScaler and all models defined in MODELS_CONFIG."""
    global SCALER
    
    # 1. Load Scaler (Remains the same)
    if not os.path.exists(SCALER_FILENAME):
        print(f"ERROR: Scaler file not found at {SCALER_FILENAME}. Cannot scale inputs.")
        return False
    try:
        with open(SCALER_FILENAME, 'rb') as f:
            SCALER = pickle.load(f)
        print("SUCCESS: StandardScaler loaded.")
    except Exception as e:
        print(f"ERROR loading scaler: {e}")
        return False

    # 2. Load Models and Metrics (Iterates through MODELS_CONFIG)
    all_loaded_successfully = True
    
    for model_key, config in MODELS_CONFIG.items():
        print(f"\nAttempting to load assets for model: {config['name']}")
        
        # --- A. Load Model Instance (.h5) ---
        if not os.path.exists(config['path']):
            print(f"ERROR: Model file not found at {config['path']}. Using MOCK data.")
            MODELS_CONFIG[model_key]['instance'] = "MOCK"
            all_loaded_successfully = False
        else:
            try:
                # Load Keras model without compilation to avoid environment issues
                model_instance = tf.keras.models.load_model(config['path'], compile=False) 
                MODELS_CONFIG[model_key]['instance'] = model_instance
                print(f"SUCCESS: Model instance loaded from {config['path']}.")
            except Exception as e:
                print(f"ERROR loading model instance: {e}")
                MODELS_CONFIG[model_key]['instance'] = "MOCK" 
                all_loaded_successfully = False

        # --- B. Load Model Metrics (.json) ---
        if not os.path.exists(config['metrics']):
            print(f"WARNING: Metrics file not found at {config['metrics']}. Metrics will be unavailable.")
            MODELS_CONFIG[model_key]['metrics_data'] = {"model_name": config['name'], "metrics": {}}
        else:
            try:
                with open(config['metrics'], 'r') as f:
                    metrics_data = json.load(f)
                # Store the loaded data in a new key
                MODELS_CONFIG[model_key]['metrics_data'] = metrics_data
                print("SUCCESS: Model Metrics loaded.")
            except Exception as e:
                print(f"ERROR loading metrics: {e}")
                MODELS_CONFIG[model_key]['metrics_data'] = {"model_name": config['name'], "metrics": {}}
                
    return True # We return True even if some models failed, as long as SCALER loaded.

# --- Routes ---

@app.route('/')
def home():
    """Renders the main dashboard page."""
    # Pass all model config to the home page for a potential dashboard summary
    return render_template('index.html', models_config=MODELS_CONFIG)


@app.route('/predict')
def show_prediction_form():
    """Renders the 30-feature input form with the randomizer."""
    return render_template('predict.html')


@app.route('/predict_result', methods=['POST'])
def handle_prediction():
    """Handles the form submission, performs prediction, and shows results."""
    
    # NOTE: For now, we hardcode to use the 'linear_regression' model key
    # In a scalable version, the user form would submit which model_key to use.
    CURRENT_MODEL_KEY = "linear_regression" 
    config = MODELS_CONFIG.get(CURRENT_MODEL_KEY)
    
    if not SCALER or not config or not config['instance']:
        return "Application assets not loaded properly or model configuration missing. Check console output.", 500

    try:
        # 1. Extract and Validate Input Data
        input_data = {}
        for feature in FEATURE_NAMES:
            value = request.form.get(feature, type=float)
            if value is None:
                 return f"Missing required feature: {feature}", 400
            input_data[feature] = value
        
        # 2. Convert to DataFrame and Scale
        data_df = pd.DataFrame([input_data])
        input_scaled = SCALER.transform(data_df[FEATURE_NAMES])
        
        # 3. Prediction Logic
        model_instance = config['instance']
        model_used_name = config['name'] # Get name from config
        model_metrics = config['metrics_data'].get("metrics", {}) # Get metrics from config

        if model_instance == "MOCK":
            # Mock prediction logic (same as before)
            prediction_score = np.random.uniform(0.1, 0.9)
            prediction_class = 1 if prediction_score > 0.5 else 0
            if data_df['radius_mean'].iloc[0] > 15:
                prediction_score = np.random.uniform(0.7, 0.9)
                prediction_class = 1 
            else:
                prediction_score = np.random.uniform(0.1, 0.3)
                prediction_class = 0 
            model_used_name += " (MOCK - File Missing)"
        
        else:
            # Actual Model Prediction
            prediction_raw = model_instance.predict(input_scaled, verbose=0).flatten()[0]
            prediction_score = float(prediction_raw)
            prediction_class = int(prediction_score > 0.5)

        # 4. Determine Result Label
        result_label = "Malignant (M)" if prediction_class == 1 else "Benign (B)"
        
        # 5. Render Result Page
        return render_template('prediction_result.html', 
                               prediction_class=prediction_class,
                               result_label=result_label,
                               prediction_score=f"{prediction_score:.4f}",
                               model_name=model_used_name,
                               model_metrics=model_metrics) # Pass metrics from the config

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return f"An internal error occurred: {e}", 500


# --- Main Application Execution ---
# Load assets before starting the Flask server
if __name__ == '__main__':
    print("Attempting to load ML assets...")
    if load_assets():
        print("Starting Flask server...")
        app.run(debug=True)
    else:
        # Note: App runs even if a model fails, as long as SCALER loads.
        # This message will only show if SCALER fails.
        print("FATAL ERROR: Could not load required assets. Application will not start.")