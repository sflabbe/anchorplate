.PHONY: sync test lint format lock clean smoke

sync:
	uv sync --dev

lock:
	uv lock

test:
	MPLBACKEND=Agg uv run pytest -q

lint:
	uv run ruff check src tests examples --select=E,F,W,B --ignore=E501

format:
	uv run ruff format src tests examples

smoke:
	MPLBACKEND=Agg uv run anchorplate-run-case examples/toml/simple_case.toml --dry-run

clean:
	rm -rf .venv .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info outputs
