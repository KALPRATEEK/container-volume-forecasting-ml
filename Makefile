.PHONY: install lint test data train dashboard docs

install:
	pip install -e ".[dev,docs]"

lint:
	ruff check src tests dashboard
	black --check src tests dashboard

test:
	pytest -q

data:
	python -m container_forecasting.data.make_synthetic_data

train:
	python -m container_forecasting.models.train

dashboard:
	streamlit run dashboard/streamlit_app.py

docs:
	mkdocs serve
