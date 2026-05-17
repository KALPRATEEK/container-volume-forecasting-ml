# Methodology

The project follows a supervised time-series machine learning workflow.

## 1. Data

The public version uses synthetic monthly logistics data. The generated data includes:

- date
- region
- import/export direction
- container volume
- industrial production index
- truck toll index
- business climate index

## 2. Feature Engineering

The model uses:

- lag features
- rolling means
- rolling standard deviations
- month encoding
- sine/cosine seasonality encoding
- lagged external indicators

## 3. Model Comparison

The project compares:

- Seasonal Naive
- Linear Regression
- Elastic Net
- XGBoost internal-only
- XGBoost with external indicators

## 4. Evaluation

The evaluation includes:

- MAE
- RMSE
- MAPE
- directional accuracy
- actual-vs-predicted plots
