# PaperTrace AI — Project Context

This file is the single source of truth for *what* PaperTrace AI is and *why* it is built the way it is.
`PLAN.md` says *when* each part gets built. `CLAUDE.md` / `AGENTS.md` say *how* to work in the repo.

## 1. Product

PaperTrace AI is a scientific research assistant. A user asks a question in natural language
(Portuguese or English). The system:

1. plans the search,
2. queries a real academic source,
3. writes a grounded synthesis that cites **only what was actually retrieved**.

The system must never invent papers, authors, DOIs or findings that the retrieved sources do not support.
When the evidence is thin, it says so instead of filling the gap.

The whole application (UI, prompts, logs, docs) is in English. User questions may be in PT or EN.

## 2. Developer and purpose

- Author: Guilherme Abreu Cavazzani — Electrical Engineering student (UFSJ), Data Analysis intern in the
  CI&T NextGen AI program, Claude Certified Architect – Foundations (CCAF).
- GitHub: https://github.com/Gu1L3RM3
- Independent public portfolio project aimed at AI Engineering roles.
- **No employer code or data may be used.**
- Local project only: no deploy, no production infrastructure.

## 3. v1 scope

Three agents, one data source, no persistence, no cross-database verification.
Anything beyond this is v2 and needs explicit approval before being added.

| Agent | Input | Output | Notes |
|---|---|---|---|
| **Orchestrator** | `ResearchQuestion` | `SearchPlan` (2–4 `SearchQuery`) | Forced tool call (structured output). Queries are always in English and each covers a different angle. |
| **Research Agent** | `SearchPlan` | `list[Paper]` | Runs all queries against OpenAlex concurrently with `asyncio.gather` over real I/O. Deduplicates by DOI (normalized title when DOI is missing). |
| **Synthesis Agent** | `list[Paper]` + question | `Synthesis` (Markdown) | Sees only the structured paper list, never free web text. Cites as (Author, Year). Must state when evidence is insufficient. |

Contracts between agents are Pydantic models, never loose dicts.

### Domain models (target shape)

```python
class ResearchQuestion(BaseModel): text: str
class SearchQuery(BaseModel):     query: str; angle: str
class SearchPlan(BaseModel):      queries: list[SearchQuery]  # 2..4
class Paper(BaseModel):           title: str; authors: list[str]; year: int | None
                                  doi: str | None; abstract: str | None; url: str
class Citation(BaseModel):        author: str; year: int | None; doi: str | None
class Synthesis(BaseModel):       markdown: str; citations: list[Citation]
                                  insufficient_evidence: bool; unmatched_citations: list[str]
```

### Grounding guarantee in v1

There is no separate verification agent in v1. Grounding relies on:
1. the Synthesis Agent prompt (cite only listed papers, admit gaps), and
2. a deterministic post-check in code: every (Author, Year) cited must match a paper in the input list.
   Mismatches are surfaced in `unmatched_citations`, never silently dropped.

## 4. Architecture

```
Browser (vanilla JS) ──POST /api/research──► FastAPI ──SSE events──► Browser
                                              │
                                          Pipeline
                     ┌────────────────────────┼─────────────────────────┐
              Orchestrator              Research Agent             Synthesis Agent
                     │                        │                         │
                LLMClient (port)        PaperSource (port)         LLMClient (port)
                     │                        │                         │
       AnthropicClient | OpenAICompatible   OpenAlexSource      AnthropicClient | OpenAICompatible
```

Source layout:

```
src/papertrace/
  domain/      Pydantic models, pure functions (dedup, citation check). No I/O.
  llm/         LLMClient port, provider adapters, factory.
  sources/     PaperSource port, OpenAlex adapter.
  agents/      Orchestrator, Research, Synthesis. Depend on ports only.
  pipeline.py  Runs the 3 stages, emits typed StageEvents.
  api/         FastAPI app, routes, SSE, dependency wiring (composition root).
  web/static/  index.html, styles.css, app.js.
tests/
  unit/        domain, agents with fakes.
  contract/    same test suite run against every adapter of a port.
  integration/ API + SSE with fake agents; live tests opt-in (@pytest.mark.live).
```

### Streaming events (SSE)

`plan` → `papers` → `synthesis` → `done`; `error` may replace any step. Each event payload is a
serialized Pydantic model, so the frontend renders typed data, not scraped text.

## 5. LLM provider independence

The project must run on any LLM that supports tool calling, and switching provider must be a `.env` change.

- `LLMClient` is a small async protocol:
  - `call_tool(system, messages, tool: ToolSpec[T]) -> T` — forced tool call, output validated into a Pydantic model.
  - `complete(system, messages) -> str` — free text.
- Adapters (the **only** modules allowed to import a vendor SDK):
  - `AnthropicClient` — Anthropic SDK, `tool_choice={"type": "tool", "name": ...}`.
  - `OpenAICompatibleClient` — OpenAI SDK with `base_url`; covers OpenAI, OpenRouter, Groq, Ollama, vLLM,
    and any other OpenAI-API endpoint with function calling.
- `build_llm_client(settings)` picks the adapter from `LLM_PROVIDER`.
- Each agent can use a different model (`ORCHESTRATOR_MODEL`, `SYNTHESIS_MODEL` override `LLM_MODEL`).
- All adapters pass the same contract test suite (Liskov substitution, proven by tests).
- Calls are async and stateless, so agents and queries run in parallel; `LLM_MAX_CONCURRENCY`
  (an `asyncio.Semaphore`) protects provider rate limits.

### Environment variables (generic names on purpose)

| Variable | Example | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` / `openai_compatible` | Selects the adapter |
| `LLM_API_KEY` | — | Provider key |
| `LLM_MODEL` | `claude-opus-5-5` | Default model for all agents |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Only for OpenAI-compatible endpoints |
| `ORCHESTRATOR_MODEL`, `SYNTHESIS_MODEL` | — | Optional per-agent overrides |
| `LLM_MAX_CONCURRENCY` | `4` | Parallel LLM calls cap |
| `OPENALEX_MAILTO` | `you@example.com` | OpenAlex polite pool |

## 6. Engineering principles

### SOLID (explicit, enforced in review)

- **S** — one class, one reason to change: each agent does one job; the pipeline only coordinates;
  adapters only translate between vendor APIs and domain models.
- **O** — new LLM providers or paper sources are new classes implementing a port. Agents and the
  pipeline are not edited to add them.
- **L** — every implementation of a port passes the same contract tests; fakes used in tests are
  real substitutes, not mocks of internals.
- **I** — ports are small: `PaperSource.search(query, limit)`, `LLMClient.call_tool/complete`.
- **D** — agents receive ports through the constructor. Concrete classes are wired only in
  `api/dependencies.py` (the composition root).

### TDD

Red → green → refactor for every behavior. Tests never touch the network in CI: HTTP is mocked with
`respx`, LLMs are replaced by fakes implementing `LLMClient`. Live tests exist but are opt-in.
Fixtures are recorded from real OpenAlex responses; no invented papers even in tests.

### Git

Atomic commits, Conventional Commits, short subject (≤72 chars). Work happens on a branch per stage,
merged through a PR with green CI. `main` is protected.

## 7. Stack

Python 3.12, uv, FastAPI, Uvicorn, httpx (async), Pydantic v2, pydantic-settings, sse-starlette,
Anthropic SDK, OpenAI SDK. Tests: pytest, pytest-asyncio, pytest-cov, respx. Quality: ruff, mypy, pre-commit.
Frontend: vanilla HTML/CSS/JS served by FastAPI, no framework, no build step.

## 8. Frontend direction

A distinctive visual identity drawn from the domain: academic research, citations, traceability.
Think library catalog cards, footnotes, citation trails, the margin notes of a paper. Avoid generic SaaS
looks: no identical rounded cards, no decorative gradients, no cream+terracotta "AI" palette.
The UI shows the three stages live (plan, papers, synthesis) as SSE events arrive — not a spinner.
Every citation in the synthesis links to the paper it came from. The `frontend-design` skill is used
before any CSS is written.

## 9. Constraints (v1)

- OpenAlex is the only data source.
- No separate citation-verification agent.
- No persistence, no auth, no deploy.
- Do not expand scope without asking.

## 10. Known v1 limitations

- OpenAlex abstracts are partial or missing for some papers; synthesis works from abstracts only.
- Citation post-check matches (Author, Year) pairs, not the content of claims.
- No caching: each question costs fresh LLM and OpenAlex calls.

## 11. v2 roadmap

- Additional sources: arXiv, Crossref; cross-source deduplication and verification.
- Dedicated citation-verification agent checking claims against abstracts/full text.
- Persistence of research sessions and exports (BibTeX, Markdown).

## 12. Glossary

- **Port** — a protocol the core depends on (`LLMClient`, `PaperSource`).
- **Adapter** — a concrete implementation of a port for one vendor.
- **Composition root** — the one place where adapters are instantiated and injected.
- **Grounded** — every claim and citation traces back to a retrieved paper.
