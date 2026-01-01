"""
This script trains and compares multiple prediction models using AutoGluon.

AutoGluon automatically:
- Tries several different models on tabular data
- Tunes them to improve performance
- Combines models when that helps accuracy

After training, it prints a leaderboard showing which models performed best.

Note:
This version is configured to be macOS-friendly and avoids LightGBM
to prevent known OpenMP-related crashes.
"""

import os

# Prevent OpenMP issues (set before imports)
os.environ["OMP_NUM_THREADS"] = "1"

from autogluon.tabular import TabularDataset, TabularPredictor
from my_timer import timer

path = "../data/autogluon_house_prices.csv"

# Models that work on this macOS system
# (LightGBM crashes due to OpenMP issues, XGBoost/CatBoost not installed)
WORKING_MODELS = {
    "RF": {},   # Random Forest - generally best performer
    "XT": {},   # Extra Trees
    "KNN": {},  # K-Nearest Neighbors
    "LR": {},   # Logistic Regression
}

with timer("AutoGluon training"):
    # Load data
    print("Loading data...")
    train_data = TabularDataset(path)
    print(f"Loaded {len(train_data)} rows, {len(train_data.columns)} columns\n")

    # Train the predictor with working models only
    predictor = TabularPredictor(
        label="price",
        verbosity=2,
    ).fit(
        train_data,
        hyperparameters=WORKING_MODELS,
        time_limit=120,
        presets="medium",
    )

    # Print leaderboard
    print("\n" + "=" * 60)
    print("LEADERBOARD")
    print("=" * 60)
    print(predictor.leaderboard(train_data))

    # Summary
    print(f"\nBest model: {predictor.model_best}")
    print(f"Validation accuracy: {predictor.info()['best_model_score_val']:.4f}")
