#!/usr/bin/env python3
"""PreToolUse guard for AI agents: blocks hook bypass, force/main pushes and secret .env access.

Exit code 2 blocks the tool call and shows stderr to the agent.
shortcut: regex-based, so obfuscated commands (variables, eval) can slip through; CI and branch
protection remain the real enforcement.
"""

import json
import re
import subprocess
import sys

# .env and .env.<anything> except .env.example; not .envrc or foo.env.
ENV_FILE = re.compile(r"(?<![\w.-])\.env(?:\.(?!example\b)[\w-]+)?(?![\w.-])")
PATCH_FILE = re.compile(r"^\*\*\* (?:(?:Add|Update|Delete) File|Move to): (.+)$", re.M)
GIT_PUSH = re.compile(r"\bgit\b[^;&|]*\bpush\b([^;&|]*)")
BLOCKED_BASH = [
    (re.compile(r"--no-verify\b"), "Do not bypass git hooks (--no-verify)."),
    (re.compile(r"\bgit\b[^;&|]*\bcommit\b[^;&|]*\s-n(\s|$)"), "Do not bypass git hooks (-n)."),
    (ENV_FILE, "Secret .env files are off-limits to agents; ask the user."),
]
FORCE_PUSH = re.compile(r"\s(--force(-with-lease)?|-f)\b|\s\+\S")


def current_branch(cwd: str) -> str:
    result = subprocess.run(
        ["git", "-C", cwd, "branch", "--show-current"], capture_output=True, text=True
    )
    return result.stdout.strip()


def check_bash(command: str, cwd: str) -> str | None:
    for pattern, reason in BLOCKED_BASH:
        if pattern.search(command):
            return reason
    for push in GIT_PUSH.finditer(command):
        args = push.group(1)
        if FORCE_PUSH.search(args):
            return "Force push is not allowed."
        if re.search(r"\bmain\b", args) or current_branch(cwd) == "main":
            return "main is protected: push a branch and open a PR."
    return None


def edited_paths(tool_name: str, tool_input: dict[str, str]) -> list[str]:
    if tool_name == "apply_patch":  # Codex: paths only exist inside the patch headers
        return PATCH_FILE.findall(tool_input.get("command", ""))
    return [tool_input.get("file_path") or tool_input.get("path") or ""]


def check_paths(paths: list[str]) -> str | None:
    if any(ENV_FILE.search(path) for path in paths):
        return "Secret .env files are off-limits to agents; ask the user."
    return None


def main() -> int:
    payload = json.load(sys.stdin)
    tool_input = payload.get("tool_input", {})
    if payload.get("tool_name") == "Bash":
        reason = check_bash(tool_input.get("command", ""), payload.get("cwd", "."))
    else:
        reason = check_paths(edited_paths(payload.get("tool_name", ""), tool_input))
    if reason:
        print(f"Blocked by agent guard (.claude/hooks/guard.py): {reason}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
