from papertrace.agents.synthesis import SynthesisAgent
from papertrace.domain.models import Paper, ResearchQuestion
from tests.fakes import FakeLLMClient

QUESTION = ResearchQuestion(text="Does sleep help memory?")


def _paper(title: str, authors: list[str], year: int | None, abstract: str | None) -> Paper:
    return Paper(
        title=title, authors=authors, year=year, doi="10.1/x", abstract=abstract, url="https://x"
    )


PAPERS = [
    _paper("Sleep and memory", ["Ana Silva", "Bo Chen"], 2020, "Sleep consolidates memory."),
    _paper("Naps", ["Eve Stone"], None, None),
]


def _llm(text: str) -> FakeLLMClient:
    return FakeLLMClient(QUESTION, text=text)


async def test_no_papers_means_insufficient_evidence_without_llm_call() -> None:
    llm = _llm("should not be used")

    result = await SynthesisAgent(llm).synthesize(QUESTION, [])

    assert result.insufficient_evidence is True
    assert result.unmatched_citations == []
    assert llm.calls == []


async def test_prompt_contains_only_the_retrieved_papers_and_the_grounding_rules() -> None:
    llm = _llm("Sleep helps (Silva et al., 2020).")

    await SynthesisAgent(llm).synthesize(QUESTION, PAPERS)

    system, messages, _ = llm.calls[0]
    prompt = messages[0].content
    assert "only" in system.lower() and "listed papers" in system
    assert "insufficient" in system.lower()
    assert QUESTION.text in prompt
    for paper in PAPERS:
        assert paper.title in prompt
    assert "Sleep consolidates memory." in prompt
    assert "Silva" in prompt and "Stone" in prompt


async def test_unlisted_citations_are_reported() -> None:
    llm = _llm("Sleep helps (Silva et al., 2020) and more (Ghost, 2019).")

    result = await SynthesisAgent(llm).synthesize(QUESTION, PAPERS)

    assert result.markdown.startswith("Sleep helps")
    assert result.unmatched_citations == ["Ghost, 2019"]
    assert result.insufficient_evidence is False


async def test_papers_without_any_abstract_count_as_insufficient_evidence() -> None:
    llm = _llm("Cannot tell.")

    result = await SynthesisAgent(llm).synthesize(QUESTION, [PAPERS[1]])

    assert result.insufficient_evidence is True
    assert len(llm.calls) == 1
