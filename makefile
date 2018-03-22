test:
	pipenv run pytest

init:
	pipenv install --dev

lint:
	pipenv run flake8
	pipenv run pylint janalysis tests/*.py
