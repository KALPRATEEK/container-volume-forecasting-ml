from container_forecasting.data.make_synthetic_data import generate_synthetic_container_data
from container_forecasting.features.build_features import (
    add_time_series_features,
    get_external_feature_columns,
    get_internal_feature_columns,
)


def test_add_time_series_features_creates_expected_features():
    raw = generate_synthetic_container_data()
    featured = add_time_series_features(raw)

    for column in set(get_internal_feature_columns() + get_external_feature_columns()):
        assert column in featured.columns

    assert featured.isna().sum().sum() == 0
    assert len(featured) < len(raw)
