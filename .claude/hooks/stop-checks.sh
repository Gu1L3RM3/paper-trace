#!/usr/bin/env bash
# Stop hook: the agent cannot end its turn while tests or type checks fail (exit 2 = keep working).
input=$(cat)
# Already continuing because of this hook: let the turn end to avoid an endless loop.
[ "$(jq -r '.stop_hook_active // false' <<<"$input")" = "true" ] && exit 0
# Codex has no CLAUDE_PROJECT_DIR; fall back to the git root.
cd "${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel)}" || exit 0
# No uncommitted Python/dependency changes: nothing new to check this turn.
[ -z "$(git status --porcelain -- '*.py' pyproject.toml uv.lock)" ] && exit 0

if ! out=$(uv run --quiet pytest -q -x 2>&1 && uv run --quiet mypy 2>&1); then
  echo "Stop blocked: pytest/mypy failing. Fix before finishing:" >&2
  tail -n 40 <<<"$out" >&2
  exit 2
fi
