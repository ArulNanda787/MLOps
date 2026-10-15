from pathlib import Path
import json
import warnings

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FEATURE_DIR = BASE_DIR / "feature_store"
ARTIFACT_DIR = BASE_DIR / "artifacts"
REPORT_DIR = BASE_DIR / "reports"
MLFLOW_DIR = BASE_DIR / "mlruns"

for folder in [
    RAW_DIR, PROCESSED_DIR, FEATURE_DIR,
    ARTIFACT_DIR, REPORT_DIR, MLFLOW_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. EXTRACT DATA
# --------------------------------------------------
print("\n========== 1. DATA EXTRACTION ==========")

iris = load_iris(as_frame=True)
raw_df = iris.frame.copy()

raw_path = RAW_DIR / "iris_raw.csv"
raw_df.to_csv(raw_path, index=False)

print("Dataset: Iris")
print("Raw dataset shape:", raw_df.shape)
print("Raw data saved to:", raw_path.relative_to(BASE_DIR))
print(raw_df.head())

# --------------------------------------------------
# 2. VALIDATE AND CLEAN DATA
# --------------------------------------------------
print("\n========== 2. DATA VALIDATION ==========")

missing_before = int(raw_df.isnull().sum().sum())
duplicates_before = int(raw_df.duplicated().sum())
rows_before = len(raw_df)

df = raw_df.copy()

# Remove exact duplicate rows.
df = df.drop_duplicates().copy()

# Rows without a target cannot be used for supervised learning.
df = df.dropna(subset=["target"]).copy()

# Replace infinite values with missing values.
feature_columns = [
    column for column in df.columns if column != "target"
]
df[feature_columns] = df[feature_columns].replace(
    [float("inf"), float("-inf")], float("nan")
)

# Fill missing numeric feature values with their column medians.
for column in feature_columns:
    df[column] = df[column].fillna(df[column].median())

# Keep target labels as integers.
df["target"] = df["target"].astype(int)

# Check that all target labels are valid Iris classes.
valid_targets = set(iris.target.unique())
df = df[df["target"].isin(valid_targets)].copy()

missing_after = int(df.isnull().sum().sum())
duplicates_after = int(df.duplicated().sum())

cleaning_report = {
    "dataset": "Iris",
    "rows_before": rows_before,
    "rows_after": len(df),
    "columns": list(df.columns),
    "missing_values_before": missing_before,
    "missing_values_after": missing_after,
    "duplicate_rows_before": duplicates_before,
    "duplicate_rows_after": duplicates_after,
    "cleaning_steps": [
        "Checked missing values and duplicate rows",
        "Removed exact duplicate rows",
        "Removed rows with missing target labels",
        "Replaced infinite feature values with missing values",
        "Imputed missing numeric features using column medians",
        "Validated target labels"
    ]
}

clean_path = PROCESSED_DIR / "iris_clean.csv"
df.to_csv(clean_path, index=False)

with open(REPORT_DIR / "data_quality_report.json", "w") as file:
    json.dump(cleaning_report, file, indent=4)

print("Rows before cleaning:", rows_before)
print("Rows after cleaning:", len(df))
print("Missing values before:", missing_before)
print("Missing values after:", missing_after)
print("Duplicate rows before:", duplicates_before)
print("Duplicate rows after:", duplicates_after)
print("Clean data saved to:", clean_path.relative_to(BASE_DIR))

# --------------------------------------------------
# 3. TRANSFORM AND ENGINEER FEATURES
# --------------------------------------------------
print("\n========== 3. FEATURE ENGINEERING ==========")

X = df.drop(columns=["target"]).copy()
y = df["target"].copy()

# Create an additional useful feature.
X["petal_ratio"] = (
    X["petal length (cm)"] /
    X["sepal length (cm)"].replace(0, float("nan"))
)

# Handle any missing or infinite engineered feature values.
X = X.replace([float("inf"), float("-inf")], float("nan"))
X["petal_ratio"] = X["petal_ratio"].fillna(
    X["petal_ratio"].median()
)

features = X.copy()
features["target"] = y

# Use readable class names in the feature metadata.
class_names = {
    int(index): name for index, name in enumerate(iris.target_names)
}

features_path = FEATURE_DIR / "iris_features.csv"
features.to_csv(features_path, index=False)

# Stratified split preserves class proportions.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

train_features = X_train.copy()
train_features["target"] = y_train

test_features = X_test.copy()
test_features["target"] = y_test

train_features.to_csv(
    FEATURE_DIR / "train_features.csv", index=False
)
test_features.to_csv(
    FEATURE_DIR / "test_features.csv", index=False
)

feature_metadata = {
    "feature_store_name": "Iris Feature Store",
    "dataset": "Iris",
    "feature_columns": list(X.columns),
    "target_column": "target",
    "target_classes": class_names,
    "total_rows": len(features),
    "training_rows": len(train_features),
    "testing_rows": len(test_features),
    "split_ratio": "80:20",
    "random_state": 42,
    "feature_description": {
        "petal_ratio":
            "Petal length divided by sepal length"
    }
}

with open(FEATURE_DIR / "feature_metadata.json", "w") as file:
    json.dump(feature_metadata, file, indent=4)

print("Engineered feature columns:", list(X.columns))
print("Feature store saved to:", features_path.relative_to(BASE_DIR))
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Feature metadata saved successfully.")

# --------------------------------------------------
# 4. CREATE MLFLOW EXPERIMENT
# --------------------------------------------------
print("\n========== 4. MLFLOW EXPERIMENTS ==========")

mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
mlflow.set_experiment("ETL_FeatureStore_Experiment")

# Two models with different algorithms and configurations.
models = {
    "LogisticRegression": (
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=300))
        ]),
        {
            "model": "LogisticRegression",
            "max_iter": 300,
            "scaling": "StandardScaler"
        }
    ),
    "RandomForest": (
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            ))
        ]),
        {
            "model": "RandomForest",
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42
        }
    )
}

results = []

# --------------------------------------------------
# 5. TRAIN, EVALUATE AND TRACK
# --------------------------------------------------
for run_name, (model, params) in models.items():
    print(f"\nTraining model: {run_name}")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=iris.target_names,
        zero_division=0
    )

    matrix = confusion_matrix(y_test, predictions).tolist()

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("dataset", "Iris")
        mlflow.set_tag("stage", "ETL and Feature Store Experiment")

        mlflow.log_params(params)
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("random_state_split", 42)
        mlflow.log_param("feature_count", X.shape[1])
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("testing_rows", len(X_test))

        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.log_metric("precision_macro", float(precision))
        mlflow.log_metric("recall_macro", float(recall))
        mlflow.log_metric("f1_macro", float(f1))

        # Save evaluation evidence as artifacts.
        report_path = ARTIFACT_DIR / f"{run_name}_report.txt"
        report_path.write_text(
            f"Model: {run_name}\n"
            f"Accuracy: {accuracy:.4f}\n"
            f"Macro precision: {precision:.4f}\n"
            f"Macro recall: {recall:.4f}\n"
            f"Macro F1: {f1:.4f}\n\n"
            f"Classification report:\n{report}\n"
            f"Confusion matrix:\n{matrix}\n"
        )

        matrix_path = ARTIFACT_DIR / f"{run_name}_confusion_matrix.json"
        matrix_path.write_text(json.dumps(matrix, indent=4))

        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(matrix_path))
        mlflow.log_artifact(str(REPORT_DIR / "data_quality_report.json"))
        mlflow.log_artifact(str(FEATURE_DIR / "feature_metadata.json"))

        # Log the trained model.
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Save a local copy for convenient reuse.
        joblib.dump(model, ARTIFACT_DIR / f"{run_name}.joblib")

        results.append({
            "model": run_name,
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1
        })

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(report)

# --------------------------------------------------
# 6. COMPARE MODELS AND SAVE RESULTS
# --------------------------------------------------
print("\n========== 5. MODEL COMPARISON ==========")

results_df = pd.DataFrame(results).sort_values(
    by="accuracy", ascending=False
)

results_path = REPORT_DIR / "model_comparison.csv"
results_df.to_csv(results_path, index=False)

best_model = results_df.iloc[0]

summary = {
    "experiment": "ETL_FeatureStore_Experiment",
    "selection_metric": "accuracy",
    "best_model": str(best_model["model"]),
    "best_accuracy": float(best_model["accuracy"]),
    "model_results": results
}

with open(REPORT_DIR / "experiment_summary.json", "w") as file:
    json.dump(summary, file, indent=4)

print(results_df.to_string(index=False))
print("\nBest model:", best_model["model"])
print("Best accuracy:", f"{best_model['accuracy']:.4f}")
print("Comparison saved to:", results_path.relative_to(BASE_DIR))
print("\nExperiment completed successfully.")
print("Open MLflow UI to compare both experiment runs.")
