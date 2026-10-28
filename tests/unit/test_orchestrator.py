from papertrace.agents.orchestrator import OrchestratorAgent
from papertrace.domain.models import ResearchQuestion, SearchPlan, SearchQuery
from tests.fakes import FakeLLMClient

PLAN = SearchPlan(
    queries=[
        SearchQuery(query="intermittent fasting cognition", angle="direct effect"),
        SearchQuery(query="time-restricted eating memory", angle="mechanism"),
    ]
)


async def test_plan_comes_from_forced_tool_call() -> None:
    llm = FakeLLMClient(PLAN)
    plan = await OrchestratorAgent(llm).plan(ResearchQuestion(text="Jejum melhora a memória?"))

    assert plan == PLAN
    system, messages, tool_name = llm.calls[0]
    assert tool_name == "submit_search_plan"
    assert messages[0].content == "Jejum melhora a memória?"
    assert "English" in system
