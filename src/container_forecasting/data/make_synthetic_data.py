import numpy as np
import pandas as pd

from container_forecasting.config import SYNTHETIC_DATA_DIR


def generate_synthetic_container_data(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-01",
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate public-safe monthly container-volume data.

    The generated data mimics realistic logistics behavior:
    - yearly seasonality
    - regional differences
    - import/export direction differences
    - gradual trend
    - external macroeconomic/logistics indicator effects
    - random operational noise

    No confidential company data is used.
    """

    rng = np.random.default_rng(random_state)
    dates = pd.date_range(start=start_date, end=end_date, freq="MS")
    months = dates.month.to_numpy()
    n_periods = len(dates)

    regions = ["REGION_A", "REGION_B", "REGION_C", "REGION_D"]
    directions = ["IMPORT", "EXPORT"]

    rows: list[dict] = []

    for region in regions:
        for direction in directions:
            base_volume = rng.integers(4_000, 9_000)
            direction_factor = 1.10 if direction == "EXPORT" else 0.95

            trend = np.linspace(0, rng.integers(500, 1_800), n_periods)
            yearly_seasonality = 800 * np.sin(2 * np.pi * months / 12)
            half_year_seasonality = 300 * np.cos(2 * np.pi * months / 6)

            industrial_production_index = (
                100 + np.linspace(0, 5, n_periods) + rng.normal(0, 2, n_periods)
            )
            truck_toll_index = (
                100 + 4 * np.sin(2 * np.pi * months / 12) + rng.normal(0, 1.5, n_periods)
            )
            business_climate_index = (
                95 + np.linspace(0, 4, n_periods) + rng.normal(0, 2.5, n_periods)
            )

            macro_effect = (
                24 * (industrial_production_index - industrial_production_index.mean())
                + 18 * (truck_toll_index - truck_toll_index.mean())
                + 12 * (business_climate_index - business_climate_index.mean())
            )

            noise = rng.normal(0, 450, n_periods)

            volume = (
                base_volume
                + trend
                + yearly_seasonality
                + half_year_seasonality
                + macro_effect
                + noise
            ) * direction_factor

            volume = np.maximum(volume, 500).round().astype(int)

            for index, date in enumerate(dates):
                rows.append(
                    {
                        "date": date,
                        "year": date.year,
                        "month": date.month,
                        "region": region,
                        "direction": direction,
                        "container_volume": int(volume[index]),
                        "industrial_production_index": round(
                            float(industrial_production_index[index]), 2
                        ),
                        "truck_toll_index": round(float(truck_toll_index[index]), 2),
                        "business_climate_index": round(float(business_climate_index[index]), 2),
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SYNTHETIC_DATA_DIR / "synthetic_container_volume.csv"

    data = generate_synthetic_container_data()
    data.to_csv(output_path, index=False)

    print(f"Saved synthetic dataset to {output_path}")
    print(f"Rows: {len(data)}")


if __name__ == "__main__":
    main()
