# ----------------------------------------------------
# Flask Application: WDBC Prediction Dashboard
# ----------------------------------------------------
import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf

# Suppress TensorFlow warnings/logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Initialization ---
# Create Flask app instance. It will automatically look for templates in a 'templates' folder.
app = Flask(__name__)

# --- Global Model and Scaler Variables ---
# These will be loaded once when the application starts.
MODEL = None
SCALER = None

MODEL_FILENAME = "models/train_linear_regression_model.h5"
SCALER_FILENAME = "models/scaler.pkl"

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


# --- Model Loading Function ---
def load_assets():
    """Loads the trained Keras model and the StandardScaler object."""
    global MODEL, SCALER
    
    # 1. Load Scaler
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

    # 2. Load Model (Keras H5 file)
    if not os.path.exists(MODEL_FILENAME):
        print(f"ERROR: Model file not found at {MODEL_FILENAME}. Prediction will use mock data.")
        # If the model is missing, we still run the app but use a placeholder prediction
        MODEL = "MOCK"
        return True # Continue app execution
    
    try:
        MODEL = tf.keras.models.load_model(MODEL_FILENAME)
        print(f"SUCCESS: Keras Model loaded from {MODEL_FILENAME}.")
    except Exception as e:
        print(f"ERROR loading Keras model: {e}")
        MODEL = "MOCK" # Fallback to mock if load fails
        return True

    return True

# --- Routes ---

@app.route('/')
def home():
    """Renders the main dashboard page."""
    return render_template('index.html')


@app.route('/predict')
def show_prediction_form():
    """Renders the 30-feature input form with the randomizer."""
    return render_template('predict.html')


@app.route('/predict_result', methods=['POST'])
def handle_prediction():
    """Handles the form submission, performs prediction, and shows results."""
    
    if not SCALER or not MODEL:
        return "Application assets not loaded properly. Check console output.", 500

    try:
        # 1. Extract and Validate Input Data
        input_data = {}
        for feature in FEATURE_NAMES:
            # Get data from the POST request form
            value = request.form.get(feature, type=float)
            if value is None:
                 return f"Missing required feature: {feature}", 400
            input_data[feature] = value
        
        # 2. Convert to DataFrame and Scale
        # Ensure the order of features is correct for the scaler/model
        data_df = pd.DataFrame([input_data])
        input_scaled = SCALER.transform(data_df)
        
        # 3. Prediction Logic
        if MODEL == "MOCK":
            # Mock prediction if the model file is missing
            prediction_score = np.random.uniform(0.1, 0.9)
            prediction_class = 1 if prediction_score > 0.5 else 0
            
            # For demonstration, let's make the mock prediction match the radius mean
            if data_df['radius_mean'].iloc[0] > 15:
                prediction_score = np.random.uniform(0.7, 0.9)
                prediction_class = 1 # Malignant likely
            else:
                prediction_score = np.random.uniform(0.1, 0.3)
                prediction_class = 0 # Benign likely

            model_used = "MOCK (Model File Missing)"
        
        else:
            # Actual Keras Model Prediction
            # Keras expect a 2D array (1 sample, 30 features)
            prediction_raw = MODEL.predict(input_scaled, verbose=0)[0][0]
            prediction_score = float(prediction_raw)
            prediction_class = int(prediction_score > 0.5)
            model_used = "Keras Linear Classifier"

        # 4. Determine Result Label
        result_label = "Malignant (M)" if prediction_class == 1 else "Benign (B)"
        
        # 5. Render Result Page (Placeholder, you'll create this later)
        # For now, we redirect back to the form with the result in the URL (not best practice, but simplest for demo)
        
        # You will replace this with render_template('result.html', ...)
        return render_template('prediction_result.html', 
                               prediction_class=prediction_class,
                               result_label=result_label,
                               prediction_score=f"{prediction_score:.4f}",
                               model_name=model_used)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return f"An internal error occurred: {e}", 500


# --- Main Application Execution ---
# Load assets before starting the Flask server
if __name__ == '__main__':
    print("Attempting to load ML assets...")
    if load_assets():
        # Only run the app if assets were loaded or mock was set up
        print("Starting Flask server...")
        app.run(debug=True)
    else:
        print("FATAL ERROR: Could not load required assets. Application will not start.")

# NOTE: You will also need a 'prediction_result.html' template later to display the results!