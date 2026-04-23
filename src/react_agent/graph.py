"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support. Model, system prompt, and
attached tools are sourced from a LaunchDarkly AI Config; tracking events use
a single tracker created once per agent run so every metric event shares the
same ``runId`` (the unit LaunchDarkly groups billing and analytics by — see
``launchdarkly-server-sdk-ai`` 0.18.0).
"""

import time
from datetime import UTC, datetime
from typing import Any, Dict, Literal, cast

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from ldai.tracker import TokenUsage
from ldai_langchain import create_langchain_model, get_ai_usage_from_response
from ldai_langchain.langchain_helper import build_structured_tools

from react_agent.context import Context
from react_agent.ld_client import get_agent_config
from react_agent.state import InputState, State
from react_agent.tools import TOOL_FACTORIES


async def call_model(
    state: State, runtime: Runtime[Context]
) -> Dict[str, Any]:
    """Call the LLM powering our "agent".

    On the first tick of a run, resolve the AI Config and create the single
    tracker reused by every subsequent tick. Per-step events (``track_tool_calls``)
    fire here; at-most-once metrics (duration, tokens, success/error) are
    deferred to ``finalize`` so they emit exactly once per run.
    """
    update: Dict[str, Any] = {}
    ai_config = state.ai_config
    if ai_config is None:
        ai_config = get_agent_config(system_time=datetime.now(tz=UTC).isoformat())
        update["ai_config"] = ai_config
        if not ai_config.enabled:
            update["messages"] = [
                AIMessage(content="This assistant is currently disabled. Please try again later.")
            ]
            return update
        update["tracker"] = ai_config.create_tracker()
        update["start_perf_ns"] = time.perf_counter_ns()
        # Materialize the tool factories once per run so each tool closes
        # over this turn's max_search_results (etc.) and never re-evaluates
        # the AI Config from inside its own body.
        built = {name: factory(ai_config) for name, factory in TOOL_FACTORIES.items()}
        update["tools"] = build_structured_tools(ai_config, built)

    tracker = state.tracker or update["tracker"]
    tools = state.tools or update["tools"]

    # create_langchain_model forwards every variation parameter (temperature,
    # max_tokens, ...) to the provider SDK; init_chat_model alone would drop them.
    model = create_langchain_model(ai_config).bind_tools(tools)
    system_message = ai_config.instructions or ""

    try:
        response = cast(
            AIMessage,
            await model.ainvoke(
                [{"role": "system", "content": system_message}, *state.messages]
            ),
        )
    except Exception:
        # Defer the track_error to finalize so duration + error fire on the
        # same tracker (and therefore the same runId) as the rest of the run.
        update["errored"] = True
        return update

    if response.tool_calls:
        tracker.track_tool_calls([call["name"] for call in response.tool_calls])

    usage = get_ai_usage_from_response(response)
    if usage is not None:
        update["input_tokens"] = usage.input
        update["output_tokens"] = usage.output
        update["total_tokens"] = usage.total

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        update["messages"] = [
            AIMessage(
                id=response.id,
                content="Sorry, I could not find an answer to your question in the specified number of steps.",
            )
        ]
    else:
        update["messages"] = [response]

    return update


async def finalize(state: State) -> Dict[str, Any]:
    """Emit run-level metrics exactly once."""
    if state.tracker is None:
        return {}
    state.tracker.track_duration((time.perf_counter_ns() - state.start_perf_ns) // 1_000_000)
    if state.errored:
        state.tracker.track_error()
        return {}
    if state.total_tokens or state.input_tokens or state.output_tokens:
        state.tracker.track_tokens(
            TokenUsage(total=state.total_tokens, input=state.input_tokens, output=state.output_tokens)
        )
    state.tracker.track_success()
    return {}


# Define a new graph

builder = StateGraph(State, input_schema=InputState, context_schema=Context)

async def run_tools(state: State) -> Dict[str, Any]:
    """Dispatch tool calls using the tools materialized for this run.

    ToolNode dispatches by tool name and only sees what's in its constructor
    list, so we build it fresh per step from state.tools (populated by
    call_model on the first tick) instead of pre-binding a static set.
    """
    return await ToolNode(list(state.tools)).ainvoke({"messages": list(state.messages)})


# Define the nodes we will cycle between
builder.add_node(call_model)
builder.add_node("tools", run_tools)
builder.add_node(finalize)

# Set the entrypoint as `call_model`
# This means that this node is the first one called
builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["finalize", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("finalize" or "tools").
    """
    if state.errored or (state.ai_config and not state.ai_config.enabled):
        return "finalize"
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish (via finalize so metrics emit)
    if not last_message.tool_calls:
        return "finalize"
    # Otherwise we execute the requested actions
    return "tools"


# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "call_model")
builder.add_edge("finalize", "__end__")

# Compile the builder into an executable graph
graph = builder.compile(name="ReAct Agent")
