from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_actual_vs_predicted(
    prediction_df: pd.DataFrame,
    output_path: Path,
    title: str = "Actual vs Predicted Container Volume",
) -> None:
    """Save an actual-vs-predicted line chart."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = prediction_df.copy()
    data["date"] = pd.to_datetime(data["date"])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data["date"], data["container_volume"], marker="o", label="Actual")
    ax.plot(data["date"], data["prediction"], marker="o", label="Predicted")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Container Volume")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
