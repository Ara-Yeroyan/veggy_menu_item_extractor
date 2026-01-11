.PHONY: install dev run-api run-mcp docker-up docker-down test lint clean

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt pytest pytest-cov ruff

run-api:
	PYTHONPATH=. uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-mcp:
	PYTHONPATH=. uvicorn src.mcp.main:app --host 0.0.0.0 --port 8001 --reload

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

docker-up-d:
	docker-compose up -d --build

# Run tests locally (requires: pip install -r requirements.txt pytest)
test:
	PYTHONPATH=. python3 -m pytest tests/ -v

# Run tests with coverage
test-cov:
	PYTHONPATH=. python3 -m pytest tests/ --cov=src --cov-report=html

# Run tests inside Docker container (no local deps needed)
docker-test:
	docker-compose exec api python -m pytest tests/ -v

lint:
	ruff check src/ tests/

generate-menus:
	PYTHONPATH=. python scripts/generate_test_menus.py

# Evaluate system against ground truth
evaluate:
	PYTHONPATH=. python scripts/evaluate_system.py

# Evaluate with JSON output
evaluate-json:
	PYTHONPATH=. python scripts/evaluate_system.py --json

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
