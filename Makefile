PYTHON = python3.14

all: run

run:
	$(PYTHON) -m src

debug:
	$(PYTHON) -m src

clean:
	rm -rf .mypy_cache .DS_Store
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

.PHONY: all run debug clean