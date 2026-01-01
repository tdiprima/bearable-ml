"""
This script runs a small machine learning experiment and keeps track of it using MLflow.

What it does:
- Trains a RandomForest model on the Iris dataset
- Records the settings used (parameters)
- Saves performance results (metrics)
- Stores the trained model for later reference

Think of MLflow here as a lab notebook for experiments.
"""

import mlflow
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from my_timer import timer

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)  # Add split for proper evaluation

with timer("MLFlow experiment"):
    # Initialize a new MLFlow run
    with mlflow.start_run():
        n_estimators = 100

        # Log the parameters
        mlflow.log_param("n_estimators", n_estimators)

        # Train the model
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        # Log the metric
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        # Create predictions for signature inference
        predictions = model.predict(X_test)

        # Infer the model signature
        signature = infer_signature(X_test, predictions)

        # Create an input example (first test sample)
        input_example = X_test[:1]

        # Log the model artifact with signature and input example
        mlflow.sklearn.log_model(
            model,
            name="random_forest_model",
            signature=signature,
            input_example=input_example,
        )

        print(f"Logged accuracy: {accuracy}")
