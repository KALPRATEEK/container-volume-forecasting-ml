import pandas as pd
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from container_forecasting.config import FIGURES_DIR, REPORTS_DIR, SYNTHETIC_DATA_DIR, TARGET_COLUMN
from container_forecasting.data.make_synthetic_data import generate_synthetic_container_data
from container_forecasting.evaluation.metrics import regression_metrics
from container_forecasting.features.build_features import (
    add_time_series_features,
    get_external_feature_columns,
    get_internal_feature_columns,
)
from container_forecasting.models.baselines import seasonal_naive_forecast
from container_forecasting.visualization.plot_results import plot_actual_vs_predicted


def load_or_create_data() -> pd.DataFrame:
    """Load synthetic data, or create it when missing."""

    path = SYNTHETIC_DATA_DIR / "synthetic_container_volume.csv"

    if not path.exists():
        SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
        data = generate_synthetic_container_data()
        data.to_csv(path, index=False)

    return pd.read_csv(path)


def train_and_evaluate() -> pd.DataFrame:
    """Train baseline and ML models, then write reports."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = load_or_create_data()
    data = add_time_series_features(raw_data)

    data["date"] = pd.to_datetime(data["date"])

    train_df = data[data["date"] < "2025-01-01"].copy()
    test_df = data[data["date"] >= "2025-01-01"].copy()

    target = TARGET_COLUMN

    internal_features = get_internal_feature_columns()
    external_features = get_external_feature_columns()

    model_specs = [
        {
            "name": "Seasonal Naive",
            "type": "baseline",
            "features": internal_features,
            "model": None,
        },
        {
            "name": "Linear Regression Internal",
            "type": "sklearn",
            "features": internal_features,
            "model": LinearRegression(),
        },
        {
            "name": "Elastic Net External",
            "type": "sklearn",
            "features": external_features,
            "model": make_pipeline(
                StandardScaler(),
                ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42, max_iter=50_000),
            ),
        },
        {
            "name": "XGBoost Internal Only",
            "type": "sklearn",
            "features": internal_features,
            "model": XGBRegressor(
                n_estimators=80,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                objective="reg:squarederror",
                tree_method="hist",
                n_jobs=1,
            ),
        },
        {
            "name": "XGBoost With External Indicators",
            "type": "sklearn",
            "features": external_features,
            "model": XGBRegressor(
                n_estimators=80,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                objective="reg:squarederror",
                tree_method="hist",
                n_jobs=1,
            ),
        },
    ]

    results = []

    for spec in model_specs:
        model_name = spec["name"]

        if spec["type"] == "baseline":
            predictions = seasonal_naive_forecast(test_df).to_numpy()
        else:
            model = spec["model"]
            feature_columns = spec["features"]

            model.fit(train_df[feature_columns], train_df[target])
            predictions = model.predict(test_df[feature_columns])

        metrics = regression_metrics(test_df[target], predictions)
        metrics["model"] = model_name
        results.append(metrics)

        prediction_output = test_df[["date", "region", "direction", target]].copy()
        prediction_output["prediction"] = predictions.round(0).astype(int)
        prediction_output["model"] = model_name

        safe_model_name = model_name.lower().replace(" ", "_")
        prediction_path = REPORTS_DIR / f"predictions_{safe_model_name}.csv"
        prediction_output.to_csv(prediction_path, index=False)

        sample_plot_data = prediction_output[
            (prediction_output["region"] == "REGION_A")
            & (prediction_output["direction"] == "EXPORT")
        ]

        if not sample_plot_data.empty:
            plot_actual_vs_predicted(
                sample_plot_data,
                FIGURES_DIR / f"actual_vs_predicted_{safe_model_name}.png",
                title=f"{model_name}: REGION_A EXPORT",
            )

    results_df = pd.DataFrame(results)
    results_df = results_df[["model", "MAE", "RMSE", "MAPE", "Directional_Accuracy"]]
    results_df.to_csv(REPORTS_DIR / "model_results.csv", index=False)

    print(results_df)
    return results_df


def main() -> None:
    train_and_evaluate()


if __name__ == "__main__":
    main()
