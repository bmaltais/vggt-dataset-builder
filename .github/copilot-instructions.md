# Copilot Instructions

> Full project context is in `CLAUDE.md` at the repo root.
> Key points for IDE-based workflows:

- Always use `uv` (not `pip`): `uv run python`, `uv pip install`
- Format with `black` before commits: `uv run black .`
- Run tests with `uv run pytest`; markers are `unit` and `integration`
