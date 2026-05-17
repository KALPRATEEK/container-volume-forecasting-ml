from pathlib import Path
from container_forecasting.models.train import train_and_evaluate
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Container Volume Forecasting", layout="wide")

st.title("Container Volume Forecasting Dashboard")

st.markdown("""
    This dashboard demonstrates a public-safe machine learning workflow for monthly logistics
    volume forecasting. The data is synthetic and used only for portfolio demonstration.
    """)

results_path = Path("reports/model_results.csv")
if not results_path.exists():
    with st.spinner("Generating synthetic demo results..."):
        train_and_evaluate()
if results_path.exists():
    results = pd.read_csv(results_path)
    st.subheader("Model Comparison")
    st.dataframe(results, use_container_width=True)
else:
    st.warning("Model results not found. Run `python -m container_forecasting.models.train` first.")

prediction_files = sorted(Path("reports").glob("predictions_*.csv"))

if prediction_files:
    selected_file = st.selectbox(
        "Select prediction file",
        prediction_files,
        format_func=lambda path: path.name,
    )

    data = pd.read_csv(selected_file)
    data["date"] = pd.to_datetime(data["date"])

    col1, col2 = st.columns(2)

    with col1:
        region = st.selectbox("Region", sorted(data["region"].unique()))

    with col2:
        direction = st.selectbox("Direction", sorted(data["direction"].unique()))

    filtered = data[(data["region"] == region) & (data["direction"] == direction)]

    st.subheader("Actual vs Predicted Volume")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(filtered["date"], filtered["container_volume"], marker="o", label="Actual")
    ax.plot(filtered["date"], filtered["prediction"], marker="o", label="Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Container Volume")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    st.subheader("Prediction Table")
    st.dataframe(filtered, use_container_width=True)

else:
    st.info("No prediction files found yet. Run the training pipeline first.")
