# 图的条件边(路由)：决定每一步该往哪走。
# 条件边 = 不是固定连线，而是根据当前状态动态返回"下一个节点名"。
#
# ┌─ 这个文件在整个项目里扮演什么角色？ ──────────────────────────────
# │ 只有两个函数，但它们是全图仅有的两个"岔路口"，图的走向全在这里拐弯：
# │   route_after_rewrite            主图岔路：清晰→并行发 Agent / 不清晰→去澄清
# │   route_after_orchestrator_call  子图岔路：继续搜 / 兜底 / 收尾（Agent 循环的方向盘）
# │
# │ 路由函数的共同套路：只【读】状态、不【写】状态，读完返回一个"去哪"的指示。
# │ 返回的形式有两类：
# │   · 返回【字符串】"节点名"    → 图去那个节点(见 graph.py 里 add_conditional_edges 的对照表)
# │   · 返回【Send 列表】         → 同时启动列表里的每个任务(并行)，只有 route_after_rewrite 用
# └──────────────────────────────────────────────────────────────────
#
# ★ 两个路由函数各读哪些状态字段、这些字段是谁写进来的（数据流转）★
#   route_after_rewrite 读：
#       State.questionIsClear     ← 由 nodes.rewrite_query 写(判断清晰度的结果)
#       State.rewrittenQuestions  ← 由 nodes.rewrite_query 写(改写后的 1~3 个子问题)
#   route_after_orchestrator_call 读：
#       AgentState.iteration_count   ← 由 nodes.orchestrator 每轮 +1(operator.add 累加)
#       AgentState.tool_call_count   ← 由 nodes.orchestrator 每轮 +本轮工具数
#       末条消息的 .tool_calls       ← 由 nodes.orchestrator 写入的模型回复(AIMessage)带的
#   想追这些字段的来龙去脉，去 graph_state.py 看它们的【写】【读】，再去 nodes.py 看写入点。

# ── 陌生语法讲解：from typing import Literal ─────────────────────────
# Literal["a","b"] 是"字面量类型"，用在函数返回值标注上，意思是"这个函数只会返回
# 'a' 或 'b' 这几个固定字符串之一"。它纯粹是给人和工具看的【说明】，运行时不强制、
# 不影响逻辑——写不写、写得全不全，都不改变程序行为。看时把它当"返回值取值范围提示"即可。
from typing import Literal
# ── 陌生对象讲解：from langgraph.types import Send（并行的核心）─────────
# Send 是 LangGraph 里"发射一个并行任务"的对象(下面 route_after_rewrite 详解)。
from langgraph.types import Send        # Send 是"发射并行任务"的核心对象
from .graph_state import State, AgentState
# 两个预算上限常量(在 config.py)：MAX_ITERATIONS=最多循环几圈，MAX_TOOL_CALLS=最多调几次工具。
# 下面 route_after_orchestrator_call 用它们判断"预算是否耗尽"。
from config import MAX_ITERATIONS, MAX_TOOL_CALLS
from core.execution_logger import log_route   # 只打路由日志，不影响逻辑，可忽略


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

    ── 读什么、写什么、去哪（数据流转）────────────────────────────────
    读：State.questionIsClear、State.rewrittenQuestions（都由 nodes.rewrite_query 写）。
    写：无（路由函数不改状态）。
    去向：不清晰 → "request_clarification"(graph.py 在它之前设了 interrupt，会暂停等用户)；
          清晰 → 对 rewrittenQuestions 里每个子问题发一个 Send("agent", {...})，
                 把 question / question_index 传进各自独立的 AgentState(见 graph_state.py)。
                 这些子图全部跑完后，靠 graph.py 里 ["agent"] 的 fan-in 汇合到 aggregate_answers。
    ──────────────────────────────────────────────────────────────────
    """
    # 情况1：问题不清晰 → 去要求用户澄清
    # state.get("questionIsClear", False)：取该字段，没有就默认 False(按不清晰处理，更保守)。
    if not state.get("questionIsClear", False):
        decision = "request_clarification"

    # 情况2：问题清晰 → 对每个改写后的子问题，各发射一个 Agent
    else:
        decision = [
            # enumerate 同时拿到编号(idx)和问题文本(query)
            # idx 用于最终汇总时按顺序排列答案(会随 Send 传进子图，最后 collect_answer 放进 index)
            # "messages": [] → 每个子图从一张空白草稿纸开始，和主图 messages 是两个独立列表
            Send("agent", {"question": query, "question_index": idx, "messages": []})
            for idx, query in enumerate(state["rewrittenQuestions"])
        ]
        # 例如 rewrittenQuestions = ["报销上限是多少", "申请流程是什么"]
        # 就会同时启动两个 Agent，一个查报销，一个查流程
        # 两个 Agent 各自独立运行，最后在 aggregate_answers 汇总

    log_route("after_rewrite", decision, state)   # 打日志，可忽略
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

    ── 读什么、写什么（数据流转）──────────────────────────────────────
    读：iteration_count、tool_call_count（都由 nodes.orchestrator 累加写入）、
        以及 messages 末条(orchestrator 刚写入的模型回复)上的 .tool_calls。
    写：无。返回的字符串会被 graph.py 的对照表映射到真实节点名。
    """
    # 从状态里取当前已用的迭代次数和工具调用次数
    # 注意：这两个计数器已经把"本次"算进去了（在 orchestrator 节点里 +1）
    iteration = state.get("iteration_count", 0)
    tool_count = state.get("tool_call_count", 0)

    # 取最后一条消息（就是模型刚刚的回复）
    last_message = state["messages"][-1]
    # ── 陌生语法讲解：getattr(对象, "属性名", 默认值) ──────────────────
    # getattr(x, "tool_calls", None) = "安全地取 x.tool_calls；万一 x 没有这个属性，
    # 不报错，返回默认值 None"。后面再 `or []`：None 视为假 → 兜成空列表 []，
    # 保证 tool_calls 一定是个可迭代的列表，下面 `if not tool_calls` 才不会出错。
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
