install:
	uv sync

all: run

run:
	uv run python -m src

debug:
	uv run python -m src

clean:
	rm -rf .mypy_cache .DS_Store
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

.PHONY: install all run debug clean