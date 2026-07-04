# cricstate — thin wrappers over uv. `make help` lists targets.
.DEFAULT_GOAL := help
.PHONY: help install check lint type test test-corpus figures leaderboard reproduce

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Sync the locked environment (Python 3.12)
	uv sync

check: lint type test ## Run the full CI check set locally

lint: ## Ruff lint + format check
	uv run ruff check . && uv run ruff format --check .

type: ## mypy --strict
	uv run mypy

test: ## Unit + property tests (the CI set)
	uv run pytest -m "not corpus"

test-corpus: ## Corpus tests (needs the built data/v1 corpus)
	uv run pytest -m corpus

figures: ## Regenerate paper figures + tables from results/summary.json
	uv run python scripts/generate_figures.py

leaderboard: ## Regenerate the leaderboard from the artifact cache
	uv run evalkit run-all

reproduce: ## Full pipeline from the pinned snapshot (see docs/REPRODUCE.md)
	uv run python -m cricstate.download
	uv run python -m cricstate.build
	uv run evalkit run-all --cold
