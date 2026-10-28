# AGENTS.md — PaperTrace AI

Instructions for any coding agent working in this repository (Codex, Claude Code, Cursor, others).
Read `CONTEXT.md` (what and why) and `PLAN.md` (current stage and its "Done when") before changing code.

## Project in one paragraph

PaperTrace AI answers research questions (PT or EN) with a grounded synthesis of real academic papers.
Three agents: Orchestrator (question → 2–4 English search queries via forced tool call),
Research Agent (parallel OpenAlex search + DOI dedup), Synthesis Agent (Markdown with (Author, Year)
citations from the retrieved list only). FastAPI streams each stage over SSE to a vanilla JS frontend.
The app is in English. It must never invent papers, authors, DOIs or findings.

## Commands

```bash
uv sync                                  # install everything
uv run pytest                            # tests (offline)
uv run pytest -m live                    # opt-in live tests (needs .env)
uv run pytest --cov                      # coverage, must stay ≥85%
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pre-commit run --all-files        # all hooks
uv run uvicorn papertrace.api.app:app --reload
```

## Layout

```
src/papertrace/
  domain/      Pydantic models + pure functions. No I/O, no SDK imports.
  llm/         LLMClient port, provider adapters, factory.
  sources/     PaperSource port, OpenAlex adapter.
  agents/      Orchestrator, Research, Synthesis. Depend on ports only.
  pipeline.py  Runs the 3 stages, yields typed StageEvents.
  api/         FastAPI app, routes, SSE, dependencies.py (composition root).
  web/static/  index.html, styles.css, app.js (no framework, no build).
tests/
  unit/  contract/  integration/
```

## Architecture rules (SOLID, enforced)

- **Single responsibility.** One agent, one job. The pipeline coordinates only. Adapters only
  translate vendor formats to domain models. Pure logic (dedup, citation check, abstract rebuild)
  lives in `domain/` as functions.
- **Open/closed.** A new LLM provider or paper source is a new class implementing a port plus one
  factory/wiring line. Never edit agents or the pipeline to add one.
- **Liskov.** Every implementation of a port passes the shared contract tests in `tests/contract/`.
  If an adapter needs a special case in a caller, the port is wrong; fix the port.
- **Interface segregation.** Ports stay small:
  `LLMClient.call_tool(system, messages, tool) -> T`, `LLMClient.complete(system, messages) -> str`,
  `PaperSource.search(query, limit) -> list[Paper]`.
- **Dependency inversion.** Agents receive ports via `__init__`. Concrete classes are instantiated
  only in `api/dependencies.py` (and the CLI entry point). No module-level clients.

Additional rules:
- Vendor SDKs (`anthropic`, `openai`) are imported only inside their adapter module in `llm/`.
- Agent contracts are Pydantic models. No `dict[str, Any]` crossing module boundaries.
- Concurrency uses `asyncio.gather` over real async I/O; LLM calls go through the shared
  `LLM_MAX_CONCURRENCY` semaphore.
- Settings come from `pydantic-settings` with the generic names below; never read `os.environ` directly.

## Environment

Generic, provider-agnostic variables (see `.env.example`):
`LLM_PROVIDER` (`anthropic` | `openai_compatible`), `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL`,
`ORCHESTRATOR_MODEL`, `SYNTHESIS_MODEL`, `LLM_MAX_CONCURRENCY`, `OPENALEX_MAILTO`.
Never commit `.env`. Never print or log keys.

## TDD workflow (mandatory)

1. **Red** — write one failing test describing the behavior. Run it, see it fail for the right reason.
2. **Green** — write the minimum code that passes it.
3. **Refactor** — clean up with tests green.
4. Commit (test and implementation may be separate commits: `test: ...` then `feat: ...`).

Testing rules:
- Tests never hit the network by default. HTTP is mocked with `respx`; LLMs are replaced by fakes
  implementing `LLMClient` (in `tests/fakes.py`).
- Live tests are marked `@pytest.mark.live` and are skipped unless `-m live` is passed.
- Fixtures come from recorded **real** OpenAlex responses (`tests/fixtures/`). Do not invent papers,
  authors or DOIs, not even in tests.
- Async tests use `pytest-asyncio` (auto mode).
- Prove parallelism with timing tests (N tasks of delay d complete in ≈d).
- Coverage must stay ≥85% (`fail_under` in `pyproject.toml`).

## Grounding rules (product-critical)

- The Synthesis Agent receives only `list[Paper]`; never raw web text.
- Its prompt requires: cite only listed papers as (Author, Year); say explicitly when evidence is
  insufficient; do not generalize beyond the abstracts.
- The deterministic citation check in `domain/` must run on every synthesis; mismatches are returned
  in `unmatched_citations`, never hidden.
- Empty paper list → `insufficient_evidence=True` without calling the LLM.

## Scope guard (v1)

Do not add, without explicit approval: data sources besides OpenAlex, a verification agent,
persistence, auth, deploy configuration, frontend frameworks or build steps. Propose them as v2.

## Git and commits

- **Atomic commits**: one logical change per commit; the tree builds and tests pass at every commit.
- **Conventional Commits**, short subject (≤72 chars, imperative, no trailing period):
  `feat:`, `fix:`, `test:`, `refactor:`, `docs:`, `chore:`, `ci:`, `build:`, `perf:`.
  Examples: `test: add DOI dedup cases`, `feat: dedup papers by DOI`.
- Body only when the "why" is not obvious; keep it to a few lines.
- `main` is protected: branch `stage-N-<name>` or `feat/<topic>`, open a PR, wait for green CI,
  rebase-merge. Never force-push `main`.
- The `commit-msg` hook rejects messages that do not follow the format. Do not bypass hooks
  (`--no-verify`).
- Never commit `.env`, keys, `.project-status`, or generated files.

## Code style

- Python 3.12, full type hints, `mypy --strict` clean.
- ruff for lint and format (line length 100).
- Small functions, intention-revealing names, no dead code, no speculative abstractions.
- Comments explain *why*, not *what*.
- Errors: raise domain exceptions from adapters (`SourceError`, `LLMError`); the pipeline turns them
  into `error` events. Never swallow exceptions silently.

## Frontend

Vanilla HTML/CSS/JS in `web/static/`. Distinctive academic/traceability identity (see `CONTEXT.md` §8).
No generic SaaS look, no decorative gradients. Stream rendering per SSE event type; citations link to
DOI/URL. Accessible: semantic HTML, keyboard-usable, visible focus, works at phone width.

## Definition of done (any change)

- Tests written first and passing; coverage ≥85%.
- `uv run pre-commit run --all-files` clean.
- Docs updated when behavior, commands or env vars change (`README.md`, `CONTEXT.md`, `PLAN.md`).
- The "Done when" checklist in `PLAN.md` updated if a stage criterion was met.
