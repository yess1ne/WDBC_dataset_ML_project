# ----------------------------------------------------
# Flask Application: WDBC Prediction Dashboard - Scalable Version
# ----------------------------------------------------
import os
import pickle
import json
import joblib # REQUIRED: For loading scikit-learn models/scalers
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, Response
import tensorflow as tf

# Suppress TensorFlow warnings/logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Initialization ---
app = Flask(__name__)

# --- Global Configuration Variables ---
# NOTE: We now support per-model scalers, so global SCALER is less critical 
# but kept for models that might share one (like Linear/Softmax/MLP).
SCALER = None 

# --- MODELS CONFIGURATION ---
# This dictionary defines all available models, their paths, types, and display names.
MODELS_CONFIG = {
    "linear_regression": {
        "name": "Linear Regression Classifier",
        "path": "models/train_linear_regression_model.h5",
        "metrics": "models/linear_regression_model_metrics.json",
        "instance": None,
        "type": "keras",
        "scaler_path": "models/scaler.pkl" # Uses the standard scaler
    },
    "softmax_regression": {
        "name": "Softmax Regression Classifier",
        "path": "models/train_softmax_regression_model.h5",
        "metrics": "models/softmax_regression_model_metrics.json",
        "instance": None,
        "type": "keras",
        "scaler_path": "models/scaler.pkl"
    },
    "mlp_classifier": {
        "name": "MLP Classifier (Deep NN)",
        "path": "models/train_MLP_model.h5",
        "metrics": "models/MLP_model_metrics.json",
        "instance": None,
        "type": "keras",
        "scaler_path": "models/scaler.pkl"
    },
    # NEW: Support Vector Machine (SVM) Configuration
    "svm_classifier": {
        "name": "Support Vector Machine (RBF)",
        "path": "models/train_svm_model.joblib", 
        "metrics": "models/svm_model_metrics.json",
        "instance": None,
        "type": "sklearn",
        "scaler_path": "models/scaler_svm.joblib" # Uses its own dedicated scaler
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

# --- Asset Loading Function ---
def load_assets():
    """Loads models, metrics, and scalers defined in MODELS_CONFIG."""
    print("\n--- Loading Assets ---")
    
    for model_key, config in MODELS_CONFIG.items():
        print(f"Loading {config['name']}...")
        
        # 1. Load Metrics (.json)
        if os.path.exists(config['metrics']):
            try:
                with open(config['metrics'], 'r') as f:
                    config['metrics_data'] = json.load(f)
                print(f"  - Metrics loaded.")
            except Exception as e:
                print(f"  - ERROR loading metrics: {e}")
                config['metrics_data'] = {"model_name": config['name'], "metrics": {}}
        else:
            print(f"  - WARNING: Metrics file not found at {config['metrics']}")
            config['metrics_data'] = {"model_name": config['name'], "metrics": {}}

        # 2. Load Scaler (.pkl or .joblib)
        # We store the loaded scaler instance directly in the config for this model
        if os.path.exists(config['scaler_path']):
            try:
                if config['scaler_path'].endswith('.joblib'):
                    config['scaler_instance'] = joblib.load(config['scaler_path'])
                else:
                    with open(config['scaler_path'], 'rb') as f:
                        config['scaler_instance'] = pickle.load(f)
                print(f"  - Scaler loaded.")
            except Exception as e:
                print(f"  - ERROR loading scaler: {e}")
                config['scaler_instance'] = None
        else:
            print(f"  - ERROR: Scaler file not found at {config['scaler_path']}")
            config['scaler_instance'] = None

        # 3. Load Model Instance (.h5 or .joblib)
        if os.path.exists(config['path']):
            try:
                if config['type'] == 'keras':
                    config['instance'] = tf.keras.models.load_model(config['path'], compile=False)
                elif config['type'] == 'sklearn':
                    config['instance'] = joblib.load(config['path'])
                print(f"  - Model instance loaded.")
            except Exception as e:
                print(f"  - ERROR loading model: {e}")
                config['instance'] = "MOCK"
        else:
            print(f"  - ERROR: Model file not found at {config['path']}")
            config['instance'] = "MOCK"
            
    return True 

# --- Routes ---

@app.route('/')
def home():
    """Renders the main dashboard page."""
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
    
    try:
        # 1. Extract and Validate Input Data
        input_data = {}
        for feature in FEATURE_NAMES:
            value = request.form.get(feature, type=float)
            if value is None:
                return f"Missing required feature: {feature}", 400
            input_data[feature] = value
        
        # Convert to DataFrame (needed for scalers that expect DataFrame structure or feature names)
        data_df = pd.DataFrame([input_data])
        
        # List to store results from all models
        all_model_results = []

        # 2. Iterate through ALL Models and Predict
        for model_key, config in MODELS_CONFIG.items():
            
            model_instance = config['instance']
            scaler_instance = config.get('scaler_instance')
            model_used_name = config['name']
            model_metrics = config['metrics_data'].get("metrics", {})
            prediction_score = None
            prediction_class = None
            
            # --- Pre-Check: Asset Availability ---
            if not scaler_instance or not model_instance:
                model_used_name += " (Error: Assets Missing)"
                prediction_score = 0.5
                prediction_class = 0 # Default Safe
                # Logic to handle missing assets gracefully in loop

            elif model_instance == "MOCK":
                # Mock prediction logic
                if data_df['radius_mean'].iloc[0] > 15:
                    prediction_score = np.random.uniform(0.7, 0.9)
                else:
                    prediction_score = np.random.uniform(0.1, 0.3)
                prediction_class = int(prediction_score > 0.5)
                model_used_name += " (MOCK)"

            else:
                # --- A. Scale Input ---
                # IMPORTANT: Use the specific scaler loaded for this model
                try:
                    # Transform expects 2D array. Keras models might be picky about DataFrames vs Arrays,
                    # but StandardScaler handles both usually. Converting to values ensures consistency.
                    input_scaled = scaler_instance.transform(data_df[FEATURE_NAMES].values)
                except Exception as e:
                     print(f"Scaling error for {model_key}: {e}")
                     continue # Skip this model if scaling fails

                # --- B. Predict ---
                if config['type'] == 'keras':
                    # Keras Prediction
                    prediction_raw = model_instance.predict(input_scaled, verbose=0).flatten()
                    if model_instance.output_shape == (None, 2):
                         prediction_score = float(prediction_raw[1]) # Softmax P(M)
                    else:
                         prediction_score = float(prediction_raw[0]) # Linear/MLP P(M)
                
                elif config['type'] == 'sklearn':
                    # Scikit-learn Prediction (SVM)
                    if hasattr(model_instance, "predict_proba"):
                        # Returns [[prob_0, prob_1]] -> We want prob_1 (Malignant)
                        prediction_score = float(model_instance.predict_proba(input_scaled)[0][1])
                    else:
                        # Fallback if predict_proba is not available (e.g., specific SVM configs)
                        prediction_class = int(model_instance.predict(input_scaled)[0])
                        prediction_score = 1.0 if prediction_class == 1 else 0.0

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
        
        # 3. Prepare data for the download link
        download_data = {
            'inputs': input_data,
            'results': [
                {
                    'model_name': res['model_name'].replace(' (MOCK)', ''),
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
                               download_data_json=download_data_json)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        import traceback
        traceback.print_exc() # Print full stack trace for debugging
        return f"An internal error occurred: {e}", 500


@app.route('/download_report', methods=['GET'])
def download_report():
    """Generates a downloadable CSV report."""
    data_json = request.args.get('data')
    if not data_json:
        return "Error: No data provided.", 400
    
    try:
        download_data = json.loads(data_json)
        csv_output = []
        
        # A. Input Features
        csv_output.append("--- Input Patient Features ---")
        csv_output.append("Feature,Value")
        for feature_name, value in download_data['inputs'].items():
            csv_output.append(f"{feature_name},{value}")
            
        # B. Results
        csv_output.append("\n--- Model Prediction Results ---")
        csv_output.append("Model Name,Predicted Diagnosis,Confidence Score (P(M))")
        for res in download_data['results']:
            csv_output.append(f"{res['model_name']},{res['label']},{res['score']}")

        # C. Metrics
        csv_output.append("\n--- Model Test Set Performance Metrics ---")
        if download_data['results'] and download_data['results'][0]['metrics']:
            metric_keys = list(download_data['results'][0]['metrics'].keys())
            csv_output.append("Model Name," + ",".join(metric_keys))
            for res in download_data['results']:
                metric_values = [str(res['metrics'].get(k, 'N/A')) for k in metric_keys]
                csv_output.append(f"{res['model_name']}," + ",".join(metric_values))

        csv_content = "\n".join(csv_output)
        response = Response(csv_content, mimetype='text/csv')
        response.headers["Content-Disposition"] = "attachment; filename=model_prediction_report.csv"
        return response

    except Exception as e:
        return f"Error creating report: {e}", 500

# --- Main Application Execution ---
if __name__ == '__main__':
    print("Starting Flask Application...")
    if load_assets():
        app.run(debug=True, port=5000, host='0.0.0.0')
    else:
        print("WARNING: Some assets failed to load. Application starting in potential degraded state.")
        app.run(debug=True, port=5000, host='0.0.0.0')