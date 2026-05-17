import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mean_absolute_percentage_error(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Calculate MAPE while avoiding division by zero."""

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    non_zero_mask = y_true_array != 0
    if not non_zero_mask.any():
        raise ValueError("MAPE cannot be calculated because all true values are zero.")

    return float(
        np.mean(
            np.abs(
                (y_true_array[non_zero_mask] - y_pred_array[non_zero_mask])
                / y_true_array[non_zero_mask]
            )
        )
        * 100
    )


def directional_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Measure whether predicted month-to-month direction matches actual direction."""

    actual_direction = np.sign(np.diff(np.asarray(y_true)))
    predicted_direction = np.sign(np.diff(np.asarray(y_pred)))

    if len(actual_direction) == 0:
        return 0.0

    return float(np.mean(actual_direction == predicted_direction) * 100)


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Return common forecasting metrics."""

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "RMSE": round(float(rmse), 2),
        "MAPE": round(mean_absolute_percentage_error(y_true, y_pred), 2),
        "Directional_Accuracy": round(directional_accuracy(y_true, y_pred), 2),
    }
