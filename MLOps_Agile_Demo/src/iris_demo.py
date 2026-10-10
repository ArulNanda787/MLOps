from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

print("Dataset loaded successfully")
print("Number of samples:", len(X))
print("Number of features:", X.shape[1])
print("Classes:", iris.target_names)

# Inspect the data
print("\nMissing values:", 0)
print("Dataset shape:", X.shape)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Train the Decision Tree classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Generate predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))

# Save the trained model
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/iris_model.joblib")
print("Model saved to models/iris_model.joblib")
