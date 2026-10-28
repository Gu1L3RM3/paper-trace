# PaperTrace AI — Plan

Stages are defined by **outcomes**, not by tasks. A stage is done only when every "Done when" item is
verifiably true. Context and rationale live in `CONTEXT.md`.

Every stage follows the same rules:

- **TDD**: write the failing test, make it pass with the minimum code, refactor. Commit at green, test and code together.
- **SOLID**: each stage lists which principle shapes its design (see `CONTEXT.md` §6).
- **Atomic commits**: Conventional Commits, short subject, one logical change per commit.
- **Flow**: branch `stage-N-<name>` → PR → green CI → rebase merge into `main` (linear history, atomic commits kept).

---

## Stage 0 — Foundation

**Outcome:** an empty but runnable app whose quality gates run locally and in CI, and `main` cannot
receive code without a green PR.

Work:
- Own git repo, `uv` project (Python 3.12), runtime and dev dependencies installed.
- Source/test layout from `CONTEXT.md` §4.
- TDD smoke: test for `GET /health` → `{"status": "ok"}`, then the endpoint.
- `pre-commit`: ruff (lint + format), mypy, whitespace/EOF fixers, private key detection,
  `commit-msg` hook enforcing Conventional Commits with subject ≤72 chars.
- Claude Code hook: format + lint the file after every Edit/Write.
- CI (GitHub Actions): ruff, mypy, pytest with coverage ≥85%.
- GitHub repo, branch protection on `main` (PR required, `ci` check required, linear history,
  no force-push, no deletion).
- `CLAUDE.md`, `AGENTS.md` (≤200 lines each), `.env.example`, README stub.

**Done when:**
- [x] `uv run pytest` and `uv run pre-commit run --all-files` pass.
- [x] CI is green on GitHub.
- [x] A direct push to `main` is rejected.
- [x] `wc -l CLAUDE.md AGENTS.md` shows ≤200 each.

---

## Stage 1 — Evidence retrieval

**Outcome:** a natural-language question (PT or EN) produces a real, deduplicated list of OpenAlex
papers, and switching LLM provider is a `.env` change.

TDD order:
1. Domain models and validation (`SearchPlan` rejects <2 or >4 queries; `Paper` normalizes DOI).
2. Pure functions: OpenAlex `abstract_inverted_index` → text; dedup by DOI, fallback normalized title.
3. `LLMClient` port + shared contract tests → `AnthropicClient` → `OpenAICompatibleClient` → factory.
4. `PaperSource` port → `OpenAlexSource` (httpx async, `respx` with recorded real responses).
5. `OrchestratorAgent` with a fake `LLMClient`: question → `SearchPlan` through a forced tool call.
6. `ResearchAgent` with a fake `PaperSource`: concurrency proven by a timing test (N queries of
   delay d finish in ≈d, not N·d); dedup applied; one failing query does not kill the others.
7. CLI `python -m papertrace.cli "<question>"` (thin wiring, no logic).

SOLID focus: **D** (agents get ports by constructor), **L** (contract tests per port),
**O** (adding a provider touches only `llm/` adapter + factory).

**Done when:**
- [ ] CLI prints real papers with `LLM_PROVIDER=anthropic` and with `LLM_PROVIDER=openai_compatible`,
      changing only `.env`.
- [x] Both LLM adapters pass the same contract suite.
- [x] Concurrency test proves queries run in parallel.
- [ ] Opt-in live test (`pytest -m live`) passes locally; CI stays offline.

---

## Stage 2 — Grounded synthesis over a streaming API

**Outcome:** the full 3-agent pipeline streams its stages over SSE, and any citation not backed by a
retrieved paper is detected and reported.

TDD order:
1. Citation extraction + check (pure): (Author, Year) pairs vs. input papers → `unmatched_citations`.
2. `SynthesisAgent` with fake `LLMClient`: receives only `list[Paper]`; empty list →
   `insufficient_evidence=True` without calling the LLM.
3. Prompt rules: cite only listed papers, admit gaps — asserted in tests on the built prompt.
4. `Pipeline`: runs the 3 agents, yields typed `StageEvent`s (`plan`, `papers`, `synthesis`, `error`, `done`).
5. `POST /api/research` → SSE; integration test with fake agents checks event order and payload schema.
6. Concurrency cap (`LLM_MAX_CONCURRENCY`) shared by all agents.

SOLID focus: **S** (pipeline coordinates, agents decide, check is a pure function),
**I** (API depends on `Pipeline`, not on agents).

**Done when:**
- [ ] `curl -N -X POST localhost:8000/api/research -d '{"text":"..."}'` shows the 3 stages in order.
- [ ] A synthesis citing an unlisted paper is flagged in tests and in the real run's payload.
- [ ] Errors (OpenAlex down, LLM failure) arrive as an `error` event, not a 500.

---

## Stage 3 — Portfolio-ready product

**Outcome:** someone clones the repo, follows the README, runs one command, and watches the traced
answer build up live in a distinctive UI.

Work:
1. Visual direction with the `frontend-design` skill (academic/traceability identity, see `CONTEXT.md` §8).
2. `web/static/` vanilla HTML/CSS/JS served by FastAPI; streaming via `fetch` + `ReadableStream`
   reader; plan, papers and synthesis render as each event arrives; citations link to DOI/URL.
3. Error and empty states: no results, provider error, missing API key (checked at startup).
4. `papertrace` console script starts the server.
5. Full README: architecture diagram, technical decisions, how to run, example run, v1 limitations,
   v2 roadmap (arXiv, Crossref, citation-verification agent).
6. Tests: static route served, settings validation, end-to-end SSE with fakes.

SOLID focus: **S** in the frontend too (render functions per event type, one stream reader).

**Done when:**
- [ ] `uv run papertrace` + browser shows the 3 stages live on a real question.
- [ ] README steps work from a fresh clone.
- [ ] CI green, coverage ≥85%.
- [ ] Manual check at desktop and phone width; keyboard-usable form.

---

## Out of scope (v1)

Extra data sources, a verification agent, persistence, auth, deploy. See `CONTEXT.md` §9–11.
