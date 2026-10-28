from papertrace.domain.citations import find_unmatched_citations
from papertrace.domain.models import Paper, ResearchQuestion, SynthesisResult
from papertrace.llm.base import LLMClient, Message

_SYSTEM = (
    "You write a short literature-based answer in Markdown, in the language of the question. "
    "Use only the listed papers: every claim must cite a listed paper as (Surname, Year), or "
    "(Surname et al., Year) when it has several authors, using the surname of its first author. "
    "Never cite or mention anything that is not in the list, and never invent findings. If the "
    "listed papers give insufficient evidence for part of the question, say so explicitly."
)


def _render(index: int, paper: Paper) -> str:
    authors = ", ".join(paper.authors) or "Unknown"
    return (
        f"[{index}] {paper.title}\n"
        f"Authors: {authors}\n"
        f"Year: {paper.year or 'n.d.'}\n"
        f"Abstract: {paper.abstract or 'not available'}"
    )


class SynthesisAgent:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def synthesize(self, question: ResearchQuestion, papers: list[Paper]) -> SynthesisResult:
        if not papers:
            return SynthesisResult(
                markdown="No papers were found, so there is insufficient evidence to answer.",
                unmatched_citations=[],
                insufficient_evidence=True,
            )
        listing = "\n\n".join(_render(i, p) for i, p in enumerate(papers, 1))
        prompt = f"Question: {question.text}\n\nListed papers:\n\n{listing}"
        markdown = await self._llm.complete(
            system=_SYSTEM, messages=[Message(role="user", content=prompt)]
        )
        return SynthesisResult(
            markdown=markdown,
            unmatched_citations=find_unmatched_citations(markdown, papers),
            insufficient_evidence=False,
        )
