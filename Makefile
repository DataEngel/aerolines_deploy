.PHONY: install test clean lint format

install:
	@pip install -r requirements-dev.txt

test:
	@python -m unittest discover -s tests/api -p "test_*.py"

clean:
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name '__pycache__' -delete

lint:
	@flake8 api_inference/ tests/

format:
	@black api_inference/ tests/
