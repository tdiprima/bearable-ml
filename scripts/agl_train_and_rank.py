"""
This script lets AutoGluon try a bunch of different prediction models on your
table-shaped data, figure out which ones work best, and combine them if that helps.

At the end, it shows a ranked list so you can see which model performed the best
and by how much.
"""
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split

from my_timer import timer

path = "../data/autogluon_house_prices.csv"

with timer("AutoGluon training"):
    # Load and split
    data = TabularDataset(path)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Train on training set
    predictor = TabularPredictor(label="price").fit(train_data)

    # Evaluate on held-out test set
    print(predictor.leaderboard(test_data))
