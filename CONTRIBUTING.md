# Contributing

Thanks for your interest! Issues and pull requests are welcome.

## Setup

```bash
uv sync
uv run pre-commit install   # optional but recommended
```

## Before opening a PR

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -v
```

## Guidelines

- Keep the decoder architecture strictly modular and faithful to modern Llama-3 building blocks.
- Add or update unit tests for any component or behavior changes.
- Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
- Type-hint public functions and add clear docstrings with tensor shapes.

