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
