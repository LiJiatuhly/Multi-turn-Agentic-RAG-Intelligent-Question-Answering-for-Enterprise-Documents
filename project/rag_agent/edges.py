from typing import Literal
from langgraph.types import Send
from .graph_state import State, AgentState
from config import MAX_ITERATIONS, MAX_TOOL_CALLS
from core.execution_logger import log_route


def route_after_rewrite(state: State) -> Literal["request_clarification", "agent"]:
    """问题改写完之后，决定走哪条路。"""
    if not state.get("questionIsClear", False):
        decision = "request_clarification"
    else:
        decision = [
            Send("agent", {"question": query, "question_index": idx, "messages": []})
            for idx, query in enumerate(state["rewrittenQuestions"])
        ]

    log_route("after_rewrite", decision, state)
    return decision


def route_after_orchestrator_call(state: AgentState) -> Literal["tools", "fallback_response", "collect_answer"]:
    """orchestrator 每次调用之后，决定下一步去哪。"""
    iteration = state.get("iteration_count", 0)
    tool_count = state.get("tool_call_count", 0)
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    if not tool_calls:
        decision = "collect_answer"
    elif iteration >= MAX_ITERATIONS or tool_count > MAX_TOOL_CALLS:
        decision = "fallback_response"
    else:
        decision = "tools"

    log_route("after_orchestrator_call", decision, state)
    return decision

