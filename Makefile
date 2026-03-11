.PHONY: format check export all

format:
	uvx ruff format

check:
	grep -rl "marimo\.App" notebooks/*.py | xargs uvx marimo check

export:
	grep -rl "marimo\.App" notebooks/*.py | xargs -P 4 -I{} sh -c '\
		ipynb="$${1%.py}.ipynb"; \
		run=0; \
		[ "$$1" -nt "$$ipynb" ] && run=1; \
		if [ ! -f "$$ipynb" ]; then \
			uv run marimo export ipynb "$$1" -o "$$ipynb"; \
			run=1; \
		fi; \
		if [ "$$run" = 1 ]; then \
			uv run jupyter nbconvert --to notebook --execute --inplace "$$ipynb"; \
		fi' _ {}

all: format check export
