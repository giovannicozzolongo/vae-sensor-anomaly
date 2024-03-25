.PHONY: train eval test lint format download clean

train:
	python -m src.models.train

eval:
	python -m src.evaluation.anomaly
	python -m src.evaluation.rul

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

download:
	python -m src.data.download

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
