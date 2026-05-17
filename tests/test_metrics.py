import numpy as np
import pandas as pd

from container_forecasting.evaluation.metrics import (
    directional_accuracy,
    mean_absolute_percentage_error,
    regression_metrics,
)


def test_mean_absolute_percentage_error():
    y_true = pd.Series([100, 200, 300])
    y_pred = np.array([110, 190, 330])

    result = mean_absolute_percentage_error(y_true, y_pred)

    assert result > 0


def test_directional_accuracy():
    y_true = pd.Series([100, 120, 110, 140])
    y_pred = np.array([90, 130, 100, 150])

    result = directional_accuracy(y_true, y_pred)

    assert 0 <= result <= 100


def test_regression_metrics_returns_expected_keys():
    y_true = pd.Series([100, 200, 300])
    y_pred = np.array([110, 190, 330])

    result = regression_metrics(y_true, y_pred)

    assert {"MAE", "RMSE", "MAPE", "Directional_Accuracy"}.issubset(result.keys())
