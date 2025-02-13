.PHONY: install test clean lint format

install:
	@pip install -r requirements-dev.txt

test:
	@python -m unittest discover -s tests/api -p "test_*.py"

stress-test:
	@echo "🏋️ Ejecutando prueba de estrés con Locust..."
	@locust -f tests/stress/api_stress.py --host=https://api-inference-deploy-873360345531.us-central1.run.app --headless -u 50 -r 10 -t 30s

clean:
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name '__pycache__' -delete

lint:
	@flake8 api_inference/ tests/

format:
	@black api_inference/ tests/
