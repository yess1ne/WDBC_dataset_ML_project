Breast Cancer Prediction Dashboard (WDBC Dataset)

This repository hosts a comparative machine learning application built with Flask and Keras/TensorFlow for predicting breast mass malignancy (Benign/Malignant). The predictions are based on 30 geometric features extracted from digitized images of fine needle aspirates (FNAs), sourced from the widely-used Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

The application's primary goal is to provide a platform for comparative analysis of different machine learning paradigms on a critical medical classification task.

1. Core Functionality

The application provides two main views:

Prediction Input: A form to input 30 diagnostic features for a patient sample.

Comparative Results Dashboard: Displays the prediction (Benign or Malignant) and the confidence score for all five integrated models, along with their individual performance metrics on the test set.

2. Machine Learning Models

The application compares the output of five distinct pre-trained classification models to give a comprehensive prediction profile:

Model Key

Model Name

Type

Key Features

linear_regression

Linear Regression Classifier

Keras/Deep Learning

Simple single-layer network acting as logistic regression.

softmax_regression

Softmax Regression Classifier

Keras/Deep Learning

A basic multi-class classification layer, providing probability distributions.

mlp_classifier

MLP Classifier (Deep NN)

Keras/Deep Learning

A standard Multi-Layer Perceptron architecture utilizing deep learning.

svm_classifier

Support Vector Machine (RBF)

scikit-learn

A classical non-linear classifier, highly effective for pattern recognition.

gru_svm_classifier

GRU-SVM Classifier (Recurrent NN Hybrid)

Keras/Deep Learning

A complex hybrid that uses a Gated Recurrent Unit (GRU) for sequence/feature processing followed by a classification layer.

3. Local Setup and Startup

This section details the necessary steps to get the Flask application running locally.

Step 3.1: Repository Setup

Clone the repository and navigate into the main application directory:

git clone <your-repository-url>
cd WDBC_webapp


Step 3.2: Environment Activation

Activate your Python virtual environment (venv) where all project dependencies have been installed:

On Windows:

.\venv\Scripts\activate


On macOS/Linux:

source venv/bin/activate


Step 3.3: Running the Server

Start the Flask application using Python:

python app.py


Look for the server output in the terminal, confirming the models have loaded and the application is running:

Starting Flask Application...
--- Loading ML Assets ---
...
* Running on http://127.0.0.1:5000 (Press CTRL+C to quit)


4. Usage

Access the Application: Open your web browser and go to http://127.0.0.1:5000.

Input Data: Fill in the 30 features on the prediction input page. A randomizer is available for rapid testing.

Analyze Results: Review the Comparative Results Dashboard to see the consensus prediction and evaluate individual model performance metrics (Accuracy, Precision, Recall, F1-Score) from the test set.

Reporting: Use the Download Report feature to save the input data, predictions, and metrics as a CSV file for documentation.
