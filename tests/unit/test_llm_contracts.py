import re

import pytest
from pydantic import BaseModel

from papertrace.agents import orchestrator, synthesis, verification
from papertrace.domain.verification import VerificationReport
from papertrace.llm.base import ToolSpec
from papertrace.llm.openai_client import tool_parameters

TOOLS = [orchestrator._TOOL, verification._TOOL]
PROMPTS = [orchestrator._SYSTEM, synthesis._SYSTEM, synthesis._REVISE_SYSTEM, verification._SYSTEM]


@pytest.mark.parametrize("tool", TOOLS, ids=lambda t: t.name)
def test_tool_description_says_what_when_and_what_not(
    tool: ToolSpec[BaseModel],
) -> None:
    sentences = re.split(r"(?<=\.)\s+", tool.description)
    assert len(sentences) >= 3
    assert "Call it" in tool.description and "not" in tool.description


@pytest.mark.parametrize("tool", TOOLS, ids=lambda t: t.name)
def test_every_sent_property_is_described(tool: ToolSpec[BaseModel]) -> None:
    def walk(node: object) -> list[str]:
        if not isinstance(node, dict):
            return []
        missing = [n for n, s in node.get("properties", {}).items() if "description" not in s]
        return missing + [m for v in node.values() for m in walk(v)]

    assert walk(tool_parameters(tool.output)) == []


@pytest.mark.parametrize("prompt", PROMPTS)
def test_prompts_have_no_shouting_or_mid_sentence_breaks(prompt: str) -> None:
    assert not re.search(r"\b(CRITICAL|MUST|ALWAYS|IMPORTANT)\b", prompt)
    assert not re.search(r"[a-z,]\n[a-z]", prompt)


def test_report_schema_stays_valid_for_the_verifier_contract() -> None:
    assert VerificationReport.model_json_schema()["required"] == ["claims", "findings"]


@pytest.mark.parametrize(
    ("prompt", "examples"),
    [
        (orchestrator._SYSTEM, 3),
        (synthesis._SYSTEM, 3),
        (synthesis._REVISE_SYSTEM, 4),
        (verification._SYSTEM, 4),
    ],
    ids=["orchestrator", "synthesis", "revise", "verification"],
)
def test_prompts_are_xml_sections_with_few_shot_examples(prompt: str, examples: int) -> None:
    assert prompt.count("<example>") >= examples
    assert "<instructions>" in prompt or "<method>" in prompt or "<rules>" in prompt
    for tag in set(re.findall(r"<([a-z_]+)>", prompt)):
        assert prompt.count(f"<{tag}>") == prompt.count(f"</{tag}>"), tag
