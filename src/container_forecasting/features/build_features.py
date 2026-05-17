import numpy as np
import pandas as pd

from container_forecasting.config import DATE_COLUMN, GROUP_COLUMNS


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create forecasting features from monthly volume and external indicators."""

    required_columns = {
        "date",
        "month",
        "region",
        "direction",
        "container_volume",
        "industrial_production_index",
        "truck_toll_index",
        "business_climate_index",
    }

    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    data = df.copy()
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    data = data.sort_values(GROUP_COLUMNS + [DATE_COLUMN])

    for lag in [1, 2, 3, 6, 12]:
        data[f"volume_lag_{lag}"] = data.groupby(GROUP_COLUMNS)["container_volume"].shift(lag)

    data["rolling_mean_3"] = data.groupby(GROUP_COLUMNS)["container_volume"].transform(
        lambda series: series.shift(1).rolling(window=3).mean()
    )
    data["rolling_std_3"] = data.groupby(GROUP_COLUMNS)["container_volume"].transform(
        lambda series: series.shift(1).rolling(window=3).std()
    )
    data["rolling_mean_6"] = data.groupby(GROUP_COLUMNS)["container_volume"].transform(
        lambda series: series.shift(1).rolling(window=6).mean()
    )

    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)

    external_columns = [
        "industrial_production_index",
        "truck_toll_index",
        "business_climate_index",
    ]

    for column in external_columns:
        for lag in [1, 2]:
            data[f"{column}_lag_{lag}"] = data.groupby(GROUP_COLUMNS)[column].shift(lag)

    return data.dropna().reset_index(drop=True)


def get_internal_feature_columns() -> list[str]:
    """Features that use only historical volume and calendar signals."""

    return [
        "month",
        "month_sin",
        "month_cos",
        "volume_lag_1",
        "volume_lag_2",
        "volume_lag_3",
        "volume_lag_6",
        "volume_lag_12",
        "rolling_mean_3",
        "rolling_std_3",
        "rolling_mean_6",
    ]


def get_external_feature_columns() -> list[str]:
    """Features that add macroeconomic/logistics indicators."""

    return get_internal_feature_columns() + [
        "industrial_production_index_lag_1",
        "industrial_production_index_lag_2",
        "truck_toll_index_lag_1",
        "truck_toll_index_lag_2",
        "business_climate_index_lag_1",
        "business_climate_index_lag_2",
    ]
