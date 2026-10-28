from papertrace.domain.models import SearchPlan, SearchQuery
from tests.fakes import FakeLLMClient
from tools.prompt_eval import run


def test_static_lint_scores_every_component_between_0_and_10() -> None:
    components = run.static_components()
    assert len(components) == 8
    assert all(0 <= c.score <= 10 for c in components)


async def test_live_case_passes_and_report_renders() -> None:
    plan = SearchPlan(
        queries=[
            SearchQuery(query="intermittent fasting cognition", angle="a"),
            SearchQuery(query="fasting memory", angle="b"),
        ]
    )
    papers = run.load_papers()
    results = [await run._run_case(c, FakeLLMClient(plan), "fake-model", papers) for c in run.CASES]
    assert next(r for r in results if r.case.id == "orch-en").outcome.passed
    assert "fake-model" in run.render(run.static_components(), results, "")
