# Container Volume Forecasting with Machine Learning

![CI](https://github.com/your-username/container-volume-forecasting-ml/actions/workflows/ci.yml/badge.svg)
![Docs](https://github.com/your-username/container-volume-forecasting-ml/actions/workflows/docs.yml/badge.svg)

A public, anonymized machine learning project for forecasting monthly container-volume movements using time-series feature engineering, external macroeconomic/logistics indicators, model comparison, and an interactive dashboard.

> This repository is a public-safe demonstration version. It uses synthetic data only and does not contain confidential company data.

---

## Project Objective

The project demonstrates how different forecasting model classes can be designed, compared, and evaluated for German import/export container-volume forecasting.

The main research-style question is:

> How can different forecasting model classes be designed, compared, and evaluated for German import/export container-volume forecasting, and to what extent do external macroeconomic and logistics indicators improve their forecasting performance compared with volume-only models?

---

## What This Project Shows

- End-to-end data science project structure
- Synthetic logistics dataset generation
- Time-series feature engineering
- Lag and rolling-window features
- External macro/logistics indicator integration
- Baseline forecasting models
- XGBoost model comparison
- Model evaluation using MAE, RMSE, MAPE, and directional accuracy
- Streamlit dashboard
- Automated tests with pytest
- Linting with Ruff and formatting with Black
- CI pipeline with GitHub Actions
- Documentation deployment workflow with MkDocs
- Docker-ready project setup

---

## Architecture

```text
Synthetic/Public-Safe Data
        ↓
Data Validation
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Evaluation
        ↓
Reports + Dashboard
        ↓
CI/CD + Documentation
```

---

## Repository Structure

```text
container-volume-forecasting-ml/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   └── docs.yml
│   └── dependabot.yml
├── dashboard/
│   └── streamlit_app.py
├── data/
│   ├── synthetic/
│   └── external/
├── docs/
├── reports/
│   └── figures/
├── src/
│   └── container_forecasting/
├── tests/
├── Dockerfile
├── Makefile
├── mkdocs.yml
├── pyproject.toml
└── README.md
```

---

## Quick Start

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install the project

```bash
pip install -e ".[dev,docs]"
```

### 3. Generate synthetic data

```bash
python -m container_forecasting.data.make_synthetic_data
```

### 4. Train and compare models

```bash
python -m container_forecasting.models.train
```

### 5. Open the dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

---

## CI/CD

The project includes:

| Workflow | Purpose |
|---|---|
| `ci.yml` | Install, lint, test, run synthetic training smoke test |
| `docs.yml` | Build and deploy documentation to GitHub Pages |
| `dependabot.yml` | Weekly dependency and GitHub Actions update checks |

---

## Confidentiality

This project does not include:

- Real company data
- Customer information
- Internal trade lanes
- Internal system exports
- Proprietary business rules
- Confidential documentation

All data is synthetic and generated only to demonstrate the technical workflow.
