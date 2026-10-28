# CLAUDE.md — PaperTrace AI

Guidance for Claude Code in this repository. Product context is in `CONTEXT.md`; the staged plan
with outcomes and "Done when" checklists is in `PLAN.md`. Read the current stage before coding.

## What this is

Research assistant: question (PT/EN) → Orchestrator plans 2–4 English queries (forced tool call)
→ Research Agent searches OpenAlex in parallel and dedups by DOI → Synthesis Agent writes Markdown
citing only retrieved papers as (Author, Year) → Verification Agent assesses abstract support.
The pipeline streams the draft, coordinates one revision maximum and re-verifies it.
The vanilla JS UI is planned. Never invent papers, authors, DOIs or findings.

## Commands

```bash
uv sync
uv run pytest                      # offline test suite
uv run pytest -m live              # opt-in, real APIs, needs .env
uv run pytest --cov                # ≥85% required
uv run ruff check --fix . && uv run ruff format .
uv run mypy
uv run pre-commit run --all-files
uv run python tools/prompt_eval/run.py --out report.html   # prompt/tool/schema lint + 12 live cases
uv run uvicorn papertrace.api.app:app --reload
```

## Layout

- `src/papertrace/domain/` — Pydantic models, pure functions. No I/O.
- `src/papertrace/llm/` — `LLMClient` port, generic `HttpLLMClient`, factory.
- `src/papertrace/sources/` — `PaperSource` port, `OpenAlexSource`.
- `src/papertrace/agents/` — Orchestrator, Research, Synthesis, Verification (ports only).
- `src/papertrace/pipeline.py` — runs the stages, yields `StageEvent`s.
- `src/papertrace/api/` — FastAPI, SSE, `dependencies.py` = composition root.
- `src/papertrace/web/static/` — planned frontend.
- `tests/{unit,contract,integration}/`, fakes in `tests/fakes.py`, recorded fixtures in `tests/fixtures/`.

## SOLID — apply explicitly

- **S**: one agent = one job; pipeline only coordinates; adapters only translate.
- **O**: new provider/source = new class + one wiring line. Do not touch agents to add one.
- **L**: every port implementation passes `tests/contract/`. Callers never special-case an adapter.
- **I**: keep ports minimal (`call_tool`, `complete`, `search`).
- **D**: inject ports through `__init__`; instantiate concretes only in `api/dependencies.py` / CLI.

When proposing or reviewing code, name the principle a design choice serves if it is not obvious.

## LLM provider independence

- No vendor SDK is a dependency: the built-in `HttpLLMClient` speaks chat/completions over
  `httpx`. Other APIs plug in as a new `LLMClient` class under `llm/` or an external
  `LLM_PROVIDER=package.module:function`.
- Switching endpoint/model is a `.env` change: `LLM_BASE_URL`, `LLM_MODEL`, optional
  `LLM_API_KEY`, `LLM_PROVIDER`, optional `ORCHESTRATOR_MODEL` / `SYNTHESIS_MODEL`, `LLM_MAX_CONCURRENCY`.
- Structured output always via forced tool calling into a Pydantic model.
- Parallel work: `asyncio.gather` over real async I/O, bounded by the shared semaphore.
- Read settings through `pydantic-settings`, never `os.environ`.

## TDD loop (always)

1. Write one failing test. Run it; confirm it fails for the expected reason.
2. Minimum code to pass.
3. Refactor with tests green.
4. Commit at green: the test and the code it drives go in one atomic commit.

- No network in the default suite: `respx` for HTTP, fake `LLMClient` for models.
- `@pytest.mark.live` for real API tests; skipped unless `-m live`.
- Fixtures are recorded real OpenAlex data. Never fabricate papers, even in tests.
- Prove concurrency with timing tests, not by reading code.

## Grounding (do not weaken)

- Synthesis receives the question and structured papers, never raw web text. Revision also receives
  the draft and verification report; all supplied content is untrusted data.
- Agents return to the pipeline; they never call each other.
- Stream the draft first; an accepted revision follows as a second `synthesis` (`revision: true`)
  and `verification`. Synthesis or first-verification failures emit `error` without `done`; a failed
  revision keeps the draft. Invalid verifier reports degrade to `not_verifiable`.
- Prompt: cite only listed papers, say when evidence is insufficient.
- Deterministic citation check runs on every synthesis; mismatches go to `unmatched_citations`.
- Empty paper list → `insufficient_evidence=True`, no LLM call.

## Scope (v1)

OpenAlex and abstract-only verification only. No full-text verification, persistence, auth, deploy,
or frontend framework. `VERIFICATION_MODEL` defaults to the effective synthesis model.
Ask before expanding scope; propose extras as v2 in `CONTEXT.md` §11.

## Commits and branches

- Atomic: one logical change, tests pass at every commit.
- Conventional Commits, subject ≤72 chars, imperative, concise: `feat: dedup papers by DOI`.
- Short body only when the why is not obvious.
- `main` is protected: work on `stage-N-<name>` or `feat/<topic>`, PR, green CI, rebase-merge.
- Never `--no-verify`, never force-push `main`, never commit `.env` or `.project-status`.
- End commit messages with the Co-Authored-By trailer configured for the session.

## Agent hooks (`.claude/settings.json`)

- **PreToolUse** `.claude/hooks/guard.py`: blocks `--no-verify` / `commit -n`, force pushes,
  pushes to `main` (or any push while on `main`), and any Bash/Read/Edit/Write/Grep touching
  `.env` or `.env.*` (`.env.example` is allowed). It matches text anywhere in a Bash command, so
  write docs mentioning those words with Edit/Write, not shell heredocs. If blocked, change
  approach or ask the user; never work around it.
- **PostToolUse**: `ruff format` + `ruff check --fix` on every edited `.py` file.
- **Stop** `.claude/hooks/stop-checks.sh`: with uncommitted Python changes, the turn cannot end
  while `pytest` or `mypy` fail. Guard behavior is tested in `tests/hooks/`.
- Codex runs the same scripts via `.codex/hooks.json`; keep both configs in sync when a hook
  changes, and keep `guard.py` handling both payload shapes (Codex edits arrive as `apply_patch`).

## Style

- Python 3.12, full type hints, `mypy --strict` clean, ruff (line length 100).
- Small functions, clear names, no dead or speculative code. Comments explain why.
- Adapters raise domain errors (`SourceError`, `LLMError`); the pipeline emits `error` events.

## Frontend

Use the `frontend-design` skill before writing CSS. Academic/traceability identity (`CONTEXT.md` §8):
no generic SaaS cards, no decorative gradients, no cream+terracotta palette. Render each SSE event
as it arrives; link every citation to its DOI/URL. Semantic, keyboard-usable, phone-width friendly.

## Project tracking

Status is tracked in ProjectGui via `.project-status` (git-excluded). When a stage or task finishes,
update its status there and run `pgui sync`.

## Done means

Tests first and green, coverage ≥85%, pre-commit clean, docs updated if commands/env/behavior
changed, and the matching `PLAN.md` "Done when" box checked only when verifiably true.
