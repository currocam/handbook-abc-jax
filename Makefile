.PHONY: format check export-all all

format:
	uvx ruff format

check:
	uvx marimo check

export-all:
	for file in notebooks/*.py; do \
		uv run marimo export ipynb "$$file" -o "$${file%.py}.ipynb"; \
	done

all: format check export-all
