import json
import subprocess
import sys
from pathlib import Path

import pytest

GUARD = Path(__file__).parents[2] / ".claude" / "hooks" / "guard.py"


def run_guard(tool: str, tool_input: dict[str, str], cwd: Path) -> int:
    payload = {"tool_name": tool, "tool_input": tool_input, "cwd": str(cwd)}
    result = subprocess.run(
        [sys.executable, str(GUARD)], input=json.dumps(payload), capture_output=True, text=True
    )
    return result.returncode


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", "-b", "feature", str(tmp_path)], check=True)
    return tmp_path


@pytest.mark.parametrize(
    "command",
    [
        "git commit --no-verify -m 'feat: x'",
        "git commit -n -m 'feat: x'",
        "git push --force origin feature",
        "git push -f origin feature",
        "git push --force-with-lease",
        "git push origin +feature",
        "git push origin main",
        "cat .env",
        "source ./.env && env",
        "grep KEY .env.local",
    ],
)
def test_blocks_risky_bash(command: str, repo: Path) -> None:
    assert run_guard("Bash", {"command": command}, repo) == 2


@pytest.mark.parametrize(
    "command",
    [
        "git commit -m 'feat: add x'",
        "git push -u origin feature",
        "cat .env.example",
        "uv run pytest",
        "ls .envrc",
    ],
)
def test_allows_safe_bash(command: str, repo: Path) -> None:
    assert run_guard("Bash", {"command": command}, repo) == 0


def test_blocks_any_push_while_on_main(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)

    assert run_guard("Bash", {"command": "git push"}, tmp_path) == 2


@pytest.mark.parametrize("tool", ["Read", "Edit", "Write"])
def test_blocks_env_file_tools(tool: str, repo: Path) -> None:
    assert run_guard(tool, {"file_path": str(repo / ".env")}, repo) == 2


def test_blocks_grep_on_env_path(repo: Path) -> None:
    assert run_guard("Grep", {"pattern": "KEY", "path": ".env"}, repo) == 2


def test_allows_env_example_read(repo: Path) -> None:
    assert run_guard("Read", {"file_path": str(repo / ".env.example")}, repo) == 0


def codex_patch(header: str) -> dict[str, str]:
    return {"command": f"*** Begin Patch\n{header}\n+X=1\n*** End Patch\n"}


@pytest.mark.parametrize(
    "header",
    ["*** Update File: .env", "*** Add File: config/.env.local", "*** Delete File: .env"],
)
def test_blocks_codex_patch_on_env(header: str, repo: Path) -> None:
    assert run_guard("apply_patch", codex_patch(header), repo) == 2


@pytest.mark.parametrize("header", ["*** Update File: .env.example", "*** Add File: src/app.py"])
def test_allows_codex_patch_elsewhere(header: str, repo: Path) -> None:
    assert run_guard("apply_patch", codex_patch(header), repo) == 0
