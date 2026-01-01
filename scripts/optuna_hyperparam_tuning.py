"""
This script uses Optuna to automatically search for good settings
(hyperparameters) for an SVM classifier on the Iris dataset.

What it does:
- Defines a goal for Optuna (the objective function)
- Tries different SVM settings over many runs
- Compares results and finds the best-performing configuration

In short: it experiments so you don't have to guess.
"""

import optuna
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from my_timer import timer


def objective(trial):
    X, y = load_iris(return_X_y=True)

    # Suggest hyperparameters
    C = trial.suggest_float("C", 1e-4, 1e-2, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])

    classifier = SVC(C=C, kernel=kernel, gamma="auto")
    return cross_val_score(classifier, X, y, n_jobs=-1, cv=3).mean()


with timer("Optuna hyperparameter optimization"):
    # Create and optimize study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print(f"Best trial value: {study.best_value}")
    print(f"Best params: {study.best_params}")
