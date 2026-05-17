from container_forecasting.data.make_synthetic_data import generate_synthetic_container_data


def test_generate_synthetic_container_data_has_expected_columns():
    data = generate_synthetic_container_data()

    expected_columns = {
        "date",
        "year",
        "month",
        "region",
        "direction",
        "container_volume",
        "industrial_production_index",
        "truck_toll_index",
        "business_climate_index",
    }

    assert expected_columns.issubset(data.columns)
    assert len(data) > 0
    assert data["container_volume"].min() > 0
