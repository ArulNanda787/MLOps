# Experiment 4: ETL, Feature Store and MLflow

## Aim
To implement an ETL data pipeline, organise reusable machine-learning
features, train multiple classification models, and track experiments
using MLflow.

## Dataset
The Iris dataset from scikit-learn is used. It contains 150 observations,
four original numeric measurements, and three target classes.

## 1. Data Collection and Cleaning
- Extracted the Iris dataset and saved it as a raw CSV.
- Inspected missing values, duplicate rows, and target labels.
- Removed exact duplicate rows and rows without target labels.
- Replaced infinite feature values with missing values.
- Imputed missing numeric feature values using column medians.
- Saved the cleaned dataset and a JSON data-quality report.

The original Iris dataset may contain no missing values or duplicate rows.
The cleaning report records the actual results.

## 2. Feature Engineering and Feature Store
- Created the engineered feature petal_ratio.
- Saved the complete feature dataset in CSV format.
- Saved separate training and testing feature datasets.
- Stored feature names, target information, class names, and split
  settings in feature_metadata.json.
- Used an 80:20 stratified train/test split with random_state=42.

The CSV-based feature store is a simple feature repository for this
laboratory experiment.

## 3. ETL Pipeline
Extract: Load the Iris dataset and save the raw CSV.
Transform: Validate and clean the data, engineer features, and prepare
the training and testing datasets.
Load: Save the cleaned dataset and reusable features to their respective
directories.

## 4. MLflow Experiment Tracking
Experiment name: ETL_FeatureStore_Experiment

Models:
1. Logistic Regression
2. Random Forest

Logged information includes model parameters, accuracy, macro precision,
macro recall, macro F1 score, model artifacts, classification reports,
and confusion matrices.

## 5. Evaluation
The models use the same stratified training and testing split.
Accuracy is the primary metric for selecting the best-performing model.
Macro precision, macro recall, and macro F1 score provide additional
evidence about classification performance across the three classes.

See reports/model_comparison.csv and reports/experiment_summary.json
for the measured results.

## 6. Results Interpretation
Compare the actual test metrics of both models. Identify the model with
the highest test accuracy, discuss its other evaluation metrics, and
mention limitations such as the small dataset and the need for testing
on additional data.

## 7. How to Run
Install the dependencies:

    pip install -r requirements.txt

Run the experiment:

    python experiment.py

Launch the MLflow UI:

    mlflow ui --backend-store-uri "$(pwd)/mlruns" --host 127.0.0.1 --port 5000

Open http://127.0.0.1:5000 in a browser.

## Output Directories
- data/raw: Original dataset
- data/processed: Cleaned dataset
- feature_store: Reusable features and metadata
- artifacts: Saved models and evaluation files
- reports: Data-quality report and model comparison
- mlruns: Local MLflow tracking data
