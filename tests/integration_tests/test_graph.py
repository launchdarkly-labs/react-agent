import pytest

from react_agent import graph

pytestmark = pytest.mark.anyio


async def test_react_agent_simple_passthrough() -> None:
    res = await graph.ainvoke(
        {"messages": [("user", "Who is the founder of LangChain?")]},  # type: ignore
    )

    assert "harrison" in str(res["messages"][-1].content).lower()
