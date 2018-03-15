test:
	pipenv run pytest

init:
	pipenv install --dev

lint:
	flake8
	pylint janalysis tests/*.py
