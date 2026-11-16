# Autonomous Shopping Agent

Turns a natural-language shopping list into the cheapest basket across nearby Israeli supermarkets, split across a small, reasonable number of stores.

## Development

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Documentation is in `docs/`: the system architecture, the work breakdown, and the data sources and chain adapters.
