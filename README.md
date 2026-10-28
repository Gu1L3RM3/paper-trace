# PaperTrace AI

Multi-agent research assistant: ask a question, get a synthesis that cites only the academic papers
it actually retrieved from OpenAlex.

> Work in progress. See [`PLAN.md`](PLAN.md) for the staged roadmap and [`CONTEXT.md`](CONTEXT.md)
> for the full project context.

## Quick start (development)

```bash
uv sync
cp .env.example .env   # fill LLM_API_KEY
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
uv run pytest
```

## Retrieval CLI

```bash
uv run python -m papertrace.cli "O jejum intermitente melhora a memória?"
```

Plans 2–4 English queries with the configured LLM, searches OpenAlex in parallel and prints the
deduplicated papers. Switch provider by editing `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL` (and
`LLM_BASE_URL` for OpenAI-compatible endpoints) in the env file; no code changes.

Real-API tests are opt-in: `uv run pytest -m live`.
