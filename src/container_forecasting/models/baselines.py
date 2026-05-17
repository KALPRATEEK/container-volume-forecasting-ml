import pandas as pd


def seasonal_naive_forecast(test_df: pd.DataFrame) -> pd.Series:
    """Use previous year's same month as a simple seasonal benchmark."""

    if "volume_lag_12" not in test_df.columns:
        raise ValueError("volume_lag_12 is required for the seasonal naive benchmark.")

    return test_df["volume_lag_12"]
