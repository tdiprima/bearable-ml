"""
This script uses NeuralProphet to forecast time-based data.

What it does:
- Learns patterns from past time series data
- Fits a forecasting model
- Predicts future values based on those patterns

In short: it looks at the past and makes an educated guess about the future.
"""

import pandas as pd
from neuralprophet import NeuralProphet

from my_timer import timer

with timer("NeuralProphet forecasting"):
    # Load data (must have 'ds' for date and 'y' for value)
    data = pd.read_csv("../data/neuralprophet_website_traffic.csv")

    # Initialize and fit model
    m = NeuralProphet()
    metrics = m.fit(data, freq="D")

    # Make future dataframe and predict
    future = m.make_future_dataframe(data, periods=30)
    forecast = m.predict(future)

    # Print tail of forecast
    print(forecast[["ds", "yhat1"]].tail())
