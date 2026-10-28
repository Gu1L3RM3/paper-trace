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

## Bring your own LLM

Any model API works as long as an adapter implements the two-method `LLMClient` port
(`call_tool`, `complete`; see `src/papertrace/llm/base.py`). Two ways to plug one in, no edits to
agents or existing adapters:

- Point the env file at it: `LLM_PROVIDER=my_package.my_module:build` where `build(settings, model,
  semaphore)` returns your client.
- Or register it in code: `register_llm_provider("name", build)` from `papertrace.llm.factory`.

`openai_compatible` already covers OpenAI, OpenRouter, Groq, Ollama and vLLM via `LLM_BASE_URL`;
`LLM_API_KEY` is optional for keyless local servers.
