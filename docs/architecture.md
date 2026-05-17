# Architecture

```text
Data Generation
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Evaluation
      ↓
Reports
      ↓
Dashboard
```

## Components

| Component | Role |
|---|---|
| `src/container_forecasting/data` | Synthetic data generation |
| `src/container_forecasting/features` | Time-series feature engineering |
| `src/container_forecasting/models` | Model training and comparison |
| `src/container_forecasting/evaluation` | Forecasting metrics |
| `src/container_forecasting/visualization` | Plot creation |
| `dashboard` | Streamlit app |
| `.github/workflows` | CI/CD automation |
