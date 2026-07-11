# 图的条件边(路由)：决定每一步该往哪走。
# 条件边 = 不是固定连线，而是根据当前状态动态返回"下一个节点名"。

from typing import Literal
from langgraph.types import Send        # Send 是"发射并行任务"的核心对象
from .graph_state import State, AgentState
from config import MAX_ITERATIONS, MAX_TOOL_CALLS
from core.execution_logger import log_route


# ============================================================
# 🅰️ route_after_rewrite：Send 并行的发源地，返回值有两种完全不同的类型
# ============================================================
def route_after_rewrite(state: State) -> Literal["request_clarification", "agent"]:
    """问题改写完之后，决定走哪条路。

    这个函数有一个非常特殊的地方：它的返回值有两种完全不同的类型：
      ① 字符串   "request_clarification"   → 去问用户补充信息
      ② Send 列表 [Send(...), Send(...)]    → 并行启动多个 Agent

    LangGraph 的条件边支持这两种返回：
      - 返回字符串 → 去那个节点
      - 返回 Send 列表 → 同时启动列表里的每一个任务（并行）

    ── Send 是什么 ────────────────────────────────────────────────────
    Send("agent", {...}) 的意思是：
      "向 'agent' 节点发送一个任务，初始状态是 {...}"
    多个 Send 同时返回 = 多个 agent 同时跑，互不干扰，各自有独立的 AgentState。
    ──────────────────────────────────────────────────────────────────
    """
    # 情况1：问题不清晰 → 去要求用户澄清
    if not state.get("questionIsClear", False):
        decision = "request_clarification"

    # 情况2：问题清晰 → 对每个改写后的子问题，各发射一个 Agent
    else:
        decision = [
            # enumerate 同时拿到编号(idx)和问题文本(query)
            # idx 用于最终汇总时按顺序排列答案
            Send("agent", {"question": query, "question_index": idx, "messages": []})
            for idx, query in enumerate(state["rewrittenQuestions"])
        ]
        # 例如 rewrittenQuestions = ["报销上限是多少", "申请流程是什么"]
        # 就会同时启动两个 Agent，一个查报销，一个查流程
        # 两个 Agent 各自独立运行，最后在 aggregate_answers 汇总

    log_route("after_rewrite", decision, state)
    return decision


# ============================================================
# 🅰️ route_after_orchestrator_call：Agent 的全部决策逻辑就是这 3 个 if
# ============================================================
def route_after_orchestrator_call(state: AgentState) -> Literal["tools", "fallback_response", "collect_answer"]:
    """orchestrator（编排模型）每次调用之后，决定下一步去哪。

    这就是 Agent "while 循环"的路由逻辑。每次模型回复后，只有三条路：
      tools            → 执行工具（继续搜）
      fallback_response → 预算耗尽，用现有材料强行作答
      collect_answer   → 模型自己说"我答完了"，收尾
    """
    # 从状态里取当前已用的迭代次数和工具调用次数
    # 注意：这两个计数器已经把"本次"算进去了（在 orchestrator 节点里 +1）
    iteration = state.get("iteration_count", 0)
    tool_count = state.get("tool_call_count", 0)

    # 取最后一条消息（就是模型刚刚的回复）
    last_message = state["messages"][-1]
    # getattr 安全取属性，tool_calls 不存在时返回 None，再 or [] 保证拿到列表
    tool_calls = getattr(last_message, "tool_calls", None) or []

    # ── 判断1：模型没有调用工具 → 说明它认为答完了，去收尾 ──────────
    if not tool_calls:
        decision = "collect_answer"
        log_route("after_orchestrator_call", decision, state)
        return decision

    # ── 判断2：预算耗尽 → 不再执行新工具，用已有材料兜底作答 ─────────
    # 为什么是 tool_count > MAX_TOOL_CALLS 而不是 >=？
    # 因为计数器在本次回复时已经 +N（N = 本次调用的工具数），
    # 所以 > MAX_TOOL_CALLS 意味着"执行这次工具调用会超预算"，要阻断。
    if iteration >= MAX_ITERATIONS or tool_count > MAX_TOOL_CALLS:
        decision = "fallback_response"
        log_route("after_orchestrator_call", decision, state)
        return decision

    # ── 判断3：有工具调用且预算充足 → 去执行工具，继续搜 ────────────
    decision = "tools"
    log_route("after_orchestrator_call", decision, state)
    return decision
