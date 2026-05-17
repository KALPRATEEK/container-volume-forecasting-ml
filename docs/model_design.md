# Model Design

## Internal-Only Model

The internal-only model uses:

- previous monthly volume
- 3-month, 6-month, and 12-month lags
- rolling means
- rolling standard deviation
- calendar month
- sine/cosine seasonality encoding

## External-Indicator Model

The external-indicator model adds lagged macroeconomic/logistics indicators:

- industrial production index
- truck toll index
- business climate index

The research-style goal is to test whether these external indicators improve forecasting accuracy compared with volume-only models.
