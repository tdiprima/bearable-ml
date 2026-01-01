"""
This script uses PyCaret to quickly train and compare multiple
classification models on a dataset.

What it does:
- Prepares the data and modeling pipeline
- Tests several classification algorithms
- Picks the best-performing model
- Tunes it to squeeze out better performance

In short: fast model testing with minimal effort.
"""

import pandas as pd
from pycaret.classification import *

from my_timer import timer

with timer("PyCaret model comparison"):
    # Load data
    data = pd.read_csv("../data/pycaret_churn_data.csv")

    # Setup pipeline
    setup(data, target="churned", session_id=123, verbose=False)

    # Compare models
    best_model = compare_models()

    # Tune the best model
    tuned_best = tune_model(best_model)

    # Print results
    print(tuned_best)
