# ----------------------------------------------------
# Flask Application: WDBC Prediction Dashboard - Scalable Version
# ----------------------------------------------------
import os
import pickle
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, Response
import tensorflow as tf

# Suppress TensorFlow warnings/logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Initialization ---
app = Flask(__name__)

# --- Global Configuration Variables ---
# SCALER remains global as it is shared across all models.
SCALER = None 

# Configuration dictionary holds Linear, Softmax, and the new MLP models.
MODELS_CONFIG = {
    "linear_regression": {
        "name": "Linear Regression Classifier", # Descriptive name for display
        "path": "models/train_linear_regression_model.h5",
        "metrics": "models/linear_regression_model_metrics.json",
        "instance": None # Placeholder for the loaded Keras model instance
    },
    "softmax_regression": {
        "name": "Softmax Regression Classifier",
        "path": "models/train_softmax_regression_model.h5",
        "metrics": "models/softmax_regression_model_metrics.json",
        "instance": None
    },
    # NEW MODEL INTEGRATED
    "mlp_classifier": {
        "name": "MLP Classifier (Deep NN)",
        "path": "models/train_MLP_model.h5",
        "metrics": "models/MLP_model_metrics.json",
        "instance": None
    }
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


# --- Asset Loading Function ---
def load_assets():
    """Loads the shared StandardScaler and all models defined in MODELS_CONFIG."""
    global SCALER
    
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
                MODELS_CONFIG[model_key]['metrics_data'] = metrics_data
                print("SUCCESS: Model Metrics loaded.")
            except Exception as e:
                print(f"ERROR loading metrics: {e}")
                MODELS_CONFIG[model_key]['metrics_data'] = {"model_name": config['name'], "metrics": {}}
                
    return True 

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
    """
    Handles the form submission, performs prediction using ALL available models,
    and shows the comparative results dashboard.
    """
    
    if not SCALER:
        return "Application assets not loaded properly. Check console output.", 500

    try:
        # 1. Extract and Validate Input Data
        input_data = {}
        for feature in FEATURE_NAMES:
            value = request.form.get(feature, type=float)
            if value is None:
                 return f"Missing required feature: {feature}", 400
            input_data[feature] = value
        
        # Convert to DataFrame for scaling
        data_df = pd.DataFrame([input_data])
        # Scale the single input point
        input_scaled = SCALER.transform(data_df[FEATURE_NAMES])
        
        # List to store results from all models
        all_model_results = []

        # 2. Iterate through ALL Models and Predict
        for model_key, config in MODELS_CONFIG.items():
            
            model_instance = config['instance']
            model_used_name = config['name']
            model_metrics = config['metrics_data'].get("metrics", {})
            prediction_score = None
            prediction_class = None
            
            if model_instance == "MOCK":
                # Mock prediction logic
                if data_df['radius_mean'].iloc[0] > 15:
                    prediction_score = np.random.uniform(0.7, 0.9)
                else:
                    prediction_score = np.random.uniform(0.1, 0.3)
                prediction_class = int(prediction_score > 0.5)
                model_used_name += " (MOCK)"

            else:
                # Actual Model Prediction
                prediction_raw = model_instance.predict(input_scaled, verbose=0).flatten()
                
                # Determine score based on model output shape
                if model_instance.output_shape == (None, 2):
                     # Softmax output: [P(B), P(M)]. Use P(M) [index 1]
                     prediction_score = float(prediction_raw[1]) 
                else:
                     # Linear/MLP output: [Score]
                     prediction_score = float(prediction_raw[0])
                     
                prediction_class = int(prediction_score > 0.5)

            # Store the prediction result for this specific model
            all_model_results.append({
                'model_key': model_key,
                'model_name': model_used_name,
                'prediction_score': f"{prediction_score:.4f}",
                'prediction_class': prediction_class,
                'result_label': "Malignant (M)" if prediction_class == 1 else "Benign (B)",
                'metrics': model_metrics
            })
        
        # 3. Prepare data for the download link (simplified format)
        # We need to serialize the complex data (all_model_results, input_data) for the URL
        download_data = {
            'inputs': input_data,
            'results': [
                {
                    'model_name': res['model_name'].replace(' (MOCK)', ''), # Clean name
                    'score': res['prediction_score'],
                    'label': res['result_label'],
                    'metrics': res['metrics']
                } for res in all_model_results
            ]
        }
        download_data_json = json.dumps(download_data)

        # 4. Render Comparative Dashboard
        return render_template('prediction_result.html', 
                               all_model_results=all_model_results,
                               input_features=data_df.iloc[0].to_dict(),
                               download_data_json=download_data_json) # Pass serialized data

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return f"An internal error occurred: {e}", 500


@app.route('/download_report', methods=['GET'])
def download_report():
    """
    Generates a downloadable CSV report based on the prediction results
    passed as a JSON string in the URL parameter 'data'.
    """
    
    data_json = request.args.get('data')
    if not data_json:
        return "Error: No data provided for download.", 400
        
    try:
        download_data = json.loads(data_json)
        
        # --- 1. Compile CSV Content ---
        csv_output = []
        
        # A. Input Features Section
        csv_output.append("--- Input Patient Features ---")
        csv_output.append("Feature,Value")
        for feature_name, value in download_data['inputs'].items():
            csv_output.append(f"{feature_name},{value}")
            
        # B. Prediction Results Section
        csv_output.append("\n--- Model Prediction Results ---")
        csv_output.append("Model Name,Predicted Diagnosis,Confidence Score (P(M))")
        for res in download_data['results']:
            csv_output.append(f"{res['model_name']},{res['label']},{res['score']}")

        # C. Performance Metrics Section (From Test Set)
        csv_output.append("\n--- Model Test Set Performance Metrics ---")
        
        # Use metrics from the first model result to define all columns (assumes keys are consistent)
        if download_data['results'] and download_data['results'][0]['metrics']:
            metric_keys = list(download_data['results'][0]['metrics'].keys())
            csv_output.append("Model Name," + ",".join(metric_keys))
            
            for res in download_data['results']:
                metric_values = [str(res['metrics'].get(k, 'N/A')) for k in metric_keys]
                csv_output.append(f"{res['model_name']}," + ",".join(metric_values))

        # Join lines into a single CSV string
        csv_content = "\n".join(csv_output)
        
        # --- 2. Create Flask Response ---
        # Note: We use text/csv content type to force a download
        response = Response(csv_content, mimetype='text/csv')
        response.headers["Content-Disposition"] = "attachment; filename=model_prediction_report.csv"
        return response

    except json.JSONDecodeError:
        return "Error: Invalid JSON data.", 400
    except Exception as e:
        print(f"Error during report generation: {e}")
        return f"An internal error occurred during report generation: {e}", 500


# --- Main Application Execution ---
# Load assets before starting the Flask server
if __name__ == '__main__':
    print("Attempting to load ML assets...")
    if load_assets():
        print("Starting Flask server...")
        # Use a higher port if needed, but 5000 is standard for development
        app.run(debug=True, port=5000)
    else:
        print("FATAL ERROR: Could not load required assets. Application will not start.")