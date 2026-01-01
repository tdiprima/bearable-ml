# Data

## 1. PyCaret: Customer Churn Classification
Create a classification dataset for PyCaret - customer churn prediction with realistic patterns

**File:** `pycaret_churn_data.csv` (200 rows)

**Target column:** `churned`

This simulates telecom customer churn with realistic patterns:

- **Short tenure + high monthly cost + many support tickets → likely to churn**
- **Long tenure + two-year contract + premium support → likely to stay**
- Mix of numeric features (tenure, charges, usage) and categorical features (contract type, payment method)
- Some edge cases that aren't perfectly predictable

Expected accuracy: 85-95% (challenging but learnable)

---

## 2. AutoGluon: House Price Regression
Create a regression dataset for AutoGluon - house price prediction with multiple features and realistic relationships

**File:** `autogluon_house_prices.csv` (200 rows)

**Target column:** `price`

Multiple correlated features affect price:

- Square footage, bedrooms, bathrooms (primary drivers)
- Neighborhood quality, school rating, crime index
- Pool, garage, lot size
- Distance to downtown, age of home

The price formula has realistic interactions (e.g., a pool matters more in nice neighborhoods). Expected R² around 0.95+.

---

## 3. NeuralProphet: Website Traffic Time Series

Create a time series dataset for NeuralProphet - daily website traffic with weekly seasonality, yearly seasonality, trend, and some holidays

**File:** `neuralprophet_website_traffic.csv` (730 rows = 2 years)

**Columns:** `ds` (date), `y` (daily visitors)

Built-in patterns for NeuralProphet to discover:

- **Weekly seasonality:** weekdays ~2x higher than weekends
- **Yearly trend:** steady growth (~40 visitors/month)
- **Holiday effects:** spikes on Valentine's Day, Easter, Mother's Day, July 4th, Halloween, Thanksgiving/Black Friday, Christmas/New Year's

This gives NeuralProphet enough data to detect both weekly and yearly patterns.

<br>
