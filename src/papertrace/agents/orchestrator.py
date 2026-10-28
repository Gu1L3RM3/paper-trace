from papertrace.domain.models import ResearchQuestion, SearchPlan
from papertrace.llm.base import LLMClient, Message, ToolSpec

_SYSTEM = (
    "You plan literature searches for a research assistant. The user's question may be in any "
    "language. Submit 2 to 4 search queries, always in English, each covering a different angle "
    "of the question, phrased as keywords suited to a scholarly search engine."
)
_TOOL = ToolSpec(
    name="submit_search_plan",
    description="Submit the English search queries for the question.",
    output=SearchPlan,
)


class OrchestratorAgent:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def plan(self, question: ResearchQuestion) -> SearchPlan:
        return await self._llm.call_tool(
            system=_SYSTEM,
            messages=[Message(role="user", content=question.text)],
            tool=_TOOL,
        )
