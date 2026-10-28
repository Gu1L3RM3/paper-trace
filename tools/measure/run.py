"""Measure tokens and latency per stage on real questions (live APIs, needs .env).

    uv run python tools/measure/run.py --out docs/metrics.json

Usage is read from each chat completion response by wrapping the SDK call, so `src/` is untouched.
"""

import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
from openai.resources.chat.completions import AsyncCompletions

from papertrace.api.dependencies import build_pipeline
from papertrace.domain.models import ResearchQuestion
from papertrace.settings import Settings

QUESTIONS = [
    "Does intermittent fasting improve cardiovascular health in adults?",
    "Qual a evidência de que a atividade física reduz sintomas de depressão?",
    "How effective are large language models at detecting software vulnerabilities?",
]

_usage: list[dict[str, Any]] = []


def _patch_sdk() -> None:
    original = AsyncCompletions.create

    async def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        response = await original(self, *args, **kwargs)
        usage = response.usage
        _usage.append(
            {
                "model": kwargs.get("model"),
                "tool": bool(kwargs.get("tools")),
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
            }
        )
        return response

    AsyncCompletions.create = create  # type: ignore[method-assign,assignment]


async def _measure(settings: Settings, http: httpx.AsyncClient, question: str) -> dict[str, Any]:
    pipeline = build_pipeline(settings, http)
    _usage.clear()
    started = last = time.perf_counter()
    stages: dict[str, float] = {}
    papers = 0
    revised = False
    status = ""
    async for event in pipeline.run(ResearchQuestion(text=question)):
        now = time.perf_counter()
        key = event.name
        if key in stages:  # a second synthesis/verification is the revision
            key += "_revision"
            revised = True
        stages[key] = round(now - last, 2)
        last = now
        payload = event.payload
        if event.name == "papers" and payload is not None:
            papers = len(getattr(payload, "papers", []))
        if event.name == "verification" and payload is not None:
            status = str(getattr(payload, "status", ""))
        if event.name == "error":
            status = f"error: {getattr(payload, 'message', '')}"
    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for call in _usage:
        totals[call["model"]]["calls"] += 1
        totals[call["model"]]["prompt_tokens"] += call["prompt_tokens"]
        totals[call["model"]]["completion_tokens"] += call["completion_tokens"]
    return {
        "question": question,
        "total_seconds": round(time.perf_counter() - started, 2),
        "stage_seconds": stages,
        "papers": papers,
        "revised": revised,
        "verification_status": status,
        "llm_calls": _usage.copy(),
        "by_model": {model: dict(counts) for model, counts in totals.items()},
    }


async def main(out: str) -> None:
    _patch_sdk()
    settings = Settings()  # type: ignore[call-arg]
    runs = []
    async with httpx.AsyncClient(timeout=30.0) as http:
        for question in QUESTIONS:
            runs.append(await _measure(settings, http, question))
            print(runs[-1]["question"], runs[-1]["total_seconds"], "s")
    models = {
        "orchestrator": settings.model_for("orchestrator"),
        "synthesis": settings.model_for("synthesis"),
        "verification": settings.model_for("verification"),
    }
    report = json.dumps({"models": models, "runs": runs}, indent=2, ensure_ascii=False)
    await asyncio.to_thread(Path(out).write_text, report, "utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/metrics.json")
    asyncio.run(main(parser.parse_args().out))
