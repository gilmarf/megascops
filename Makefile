run:
	poetry run python megascops/main.py

black:
	black -l 86 $$(find * -name '*.py')