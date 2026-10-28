# PaperTrace AI

Multi-agent research assistant: ask a question, get a synthesis that cites only the academic papers
it actually retrieved from OpenAlex.

See [`PLAN.md`](PLAN.md) for the staged roadmap and [`CONTEXT.md`](CONTEXT.md) for the full
project context.

## Run it

```bash
uv sync
cp .env.example .env   # set LLM_API_KEY and LLM_MODEL
uv run papertrace      # http://127.0.0.1:8000
```

Open the page, ask a question, and watch the plan, papers, draft synthesis and verification appear
as the server streams them. Citations link to the reference list, and each reference links to its
DOI. A revised synthesis replaces the draft. Missing configuration stops the command at startup
with the names of the variables to set.

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

Plans 2–4 English queries with the configured OpenAI model, searches OpenAlex in parallel and
prints the deduplicated papers. Set `LLM_API_KEY` and `LLM_MODEL` in the env file (`LLM_BASE_URL`
only for an OpenAI-compatible gateway).

Real-API tests are opt-in: `uv run pytest -m live`.

## API

```bash
uv run uvicorn papertrace.api.app:app --reload
curl -N -X POST localhost:8000/api/research -H 'content-type: application/json' \
  -d '{"text":"Does sleep improve memory?"}'
```

Streams SSE events in order: `plan`, `papers`, `synthesis`, `verification`, `done`. Failures arrive as one
`error` event (`{"message": ...}`) for failures inside the pipeline. Invalid requests return HTTP 422;
configuration errors at startup prevent the app from starting. `synthesis.unmatched_citations` lists
any `(Author, Year)` the answer cites that no retrieved paper backs.

## Architecture and control flow

FastAPI passes the question to a deterministic `Pipeline`, which coordinates all four agents.
The Orchestrator plans searches; it does not supervise other agents. Each agent returns its typed
result to the pipeline, never directly to another agent.

| Agent | Input | Output to pipeline | External dependency |
|---|---|---|---|
| Orchestrator | Question | 2–4 English queries | LLM forced tool call |
| Research | Search plan | Deduplicated papers and failed queries | OpenAlex |
| Synthesis | Question and papers | Internal draft with citation check | LLM completion |
| Verification | Question, draft and papers | Claim/evidence report | LLM tool call and domain validation |

The pipeline requests at most one revision and full re-verification after streaming the draft;
an accepted revision arrives as a second `synthesis` (`revision: true`) and `verification`. Stages within a request are sequential; only Research's OpenAlex
queries run in parallel. Separate API requests can overlap under the shared LLM semaphore.
Research uses no LLM. The retrieval CLI runs only planning and research.

Partial search failures can continue when papers were retrieved, with `failed_queries` exposed.
All-query failure is terminal. No papers plus partial failure is terminal too; a successful empty
search instead returns insufficient evidence. Missing usable abstracts skip both synthesis and
verification LLM calls. A synthesis or first-verification technical failure emits `error` without `done`; a failed
revision keeps the draft. Invalid verifier reports become `not_verifiable`. Remaining scientific
concerns after one revision stay visible in the report.

The built-in LLM adapter retries transport failures and HTTP 429/500/502/503/504 up to three
attempts, with 1s and 2s backoff. Invalid credentials, invalid tool output and fabricated report
provenance do not trigger these retries. OpenAlex has no explicit retry loop.

The vanilla JS frontend (`src/papertrace/web/static/`) consumes the same SSE stream.
See [architecture, contracts and failure matrix](CONTEXT.md#4-architecture) for details.

## LLM provider

OpenAI is the only provider, through the official `openai` SDK (`src/papertrace/llm/openai_client.py`).
Agents depend on the two-method `LLMClient` port (`call_tool`, `complete`), not on the SDK.

## Abstract evidence verification

The API streams the draft, verifies its factual claims against retrieved abstracts, and revises
at most once, verifying the revised text again. A failed first verification emits `error` without
`done`; a failed revision keeps the draft. Persistent scientific issues stay visible in the report.
`done` means processing completed, not that scientific correctness is guaranteed.
The current `insufficient_evidence` flag indicates missing usable abstracts; it does not summarize
semantic verification findings. Inspect the report even when that flag is false.

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
