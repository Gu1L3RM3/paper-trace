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
deduplicated papers. Point it at any server that speaks the chat/completions protocol with function calling
(Ollama, vLLM, LM Studio, OpenRouter, Groq, a gateway, a hosted API) by setting `LLM_BASE_URL`
and `LLM_MODEL` in the env file. `LLM_API_KEY` is optional; no code changes, no vendor SDK.

Real-API tests are opt-in: `uv run pytest -m live`.

## API

```bash
uv run uvicorn papertrace.api.app:app --reload
curl -N -X POST localhost:8000/api/research -H 'content-type: application/json' \
  -d '{"text":"Does sleep improve memory?"}'
```

Streams SSE events in order: `plan`, `papers`, `synthesis`, `verification`, `done`. Failures arrive as one
`error` event (`{"message": ...}`) instead of an HTTP 500. `synthesis.unmatched_citations` lists
any `(Author, Year)` the answer cites that no retrieved paper backs.

## Bring your own LLM

Any model API works as long as an adapter implements the two-method `LLMClient` port
(`call_tool`, `complete`; see `src/papertrace/llm/base.py`). Two ways to plug one in, no edits to
agents or existing adapters:

- Point the env file at it: `LLM_PROVIDER=my_package.my_module:build` where `build(settings, model,
  semaphore)` returns your client.
- Or register it in code: `register_llm_provider("name", build)` from `papertrace.llm.factory`.

The built-in `chat_completions` provider needs only `LLM_BASE_URL` and `LLM_MODEL`; `LLM_API_KEY`
is optional for keyless local servers.

## Abstract evidence verification

The API keeps the draft internal, verifies its factual claims against retrieved abstracts, and
revises at most once before verifying the final text again. Only the final synthesis and its
corresponding verification report are streamed. Verification failures emit `error`, without a
final synthesis or `done`. Persistent scientific issues are delivered explicitly in the report.
`done` means processing completed, not that scientific correctness is guaranteed.

`verification` includes `claims`, `findings`, `status`, `reason` and `revision_count` (0 or 1).
Each claim includes a literal synthesis span, an assessment (`supported`, `partially_supported`,
`contradicted`, `unsupported`), reason, evidence and counterevidence. Evidence contains a one-based
`paper_index` into the streamed paper list and an exact abstract `quote` (whitespace may differ).
Findings describe relevant omissions, contradictions or ambiguous author/year citations.
Status is computed in code: `no_issues_found`, `issues_found`, or `not_verifiable`. Empty reports
cannot establish support. References and quoted spans are validated deterministically; semantic
support and extraction completeness still depend on the LLM. Titles alone are never evidence.

Without usable abstracts, synthesis and verification skip their LLM calls and return an explicit
insufficient-evidence response. Partial search failures remain visible in `papers.failed_queries`.
The retrieval-only CLI is unchanged. No new sources or full-text retrieval are introduced.

Set optional `VERIFICATION_MODEL` to override the verifier; otherwise it uses the effective
`SYNTHESIS_MODEL` (or `LLM_MODEL`). All agents share `LLM_MAX_CONCURRENCY`. A normal run uses three
logical LLM calls; one revision raises this to five, excluding transport retries. Clients must
consume the new `verification` event and wait for `done` before treating the run as complete.

Live verification tests are opt-in (`uv run pytest -m live`). Inspect the emitted report against
its abstracts manually; fake-based tests establish flow and validation, not scientific accuracy.
