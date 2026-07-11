# 组装两张图：内层 Agent 子图(检索循环) + 外层主图(改写/并行/汇总)。
#
# 🅲 整体扫一遍，知道"谁连着谁"就行，不需要逐行背。
# 核心结构：
#   主图：summarize_history → rewrite_query →（条件边）→ [agent×N 并行] → aggregate_answers
#   子图：orchestrator →（条件边）→ tools → should_compress_context →（条件边）→ orchestrator
#                                                                              ↘ compress_context ↗

from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver   # 内存 checkpointer，保存会话状态（重启后消失）
from langgraph.prebuilt import ToolNode                 # 内置节点：自动执行工具调用列表
from functools import partial                           # partial(f, x=1) → 预先绑定参数，生成新函数

from .graph_state import State, AgentState
from core.execution_logger import logged_node           # 装饰器：自动打印节点输入输出（调试用）
from .nodes import (
    aggregate_answers, collect_answer, compress_context,
    fallback_response, orchestrator, request_clarification,
    rewrite_query, should_compress_context, summarize_history,
)
from .edges import route_after_orchestrator_call, route_after_rewrite


def create_agent_graph(llm, tools_list):
    """🅲 组装并编译完整的 Agent 图，返回可调用的图对象。

    输入：
        llm        : 已初始化的聊天模型（ChatOpenAI）
        tools_list : 工具列表（search_child_chunks, retrieve_parent_chunks）
    输出：编译好的 LangGraph 图（支持流式调用和 checkpointing）
    """
    # ── 内置方法讲解：llm.bind_tools(工具列表)（第一次见）──────────────
    # bind_tools 是模型对象的方法，作用 = 把工具"挂"到模型上，返回一个【新模型】。
    #   输入：工具列表   输出：一个"知道有哪些工具可调"的新模型对象
    # 只有绑过工具的模型，回复时才可能在 AIMessage.tool_calls 里说"我要调 XX 工具"。
    # 没绑的原始 llm 只会说话、不会调工具。两个都保留：不需要工具的节点用 llm，
    # 只有 orchestrator 需要调工具，用 llm_with_tools。
    llm_with_tools = llm.bind_tools(tools_list)
    # ── 内置对象讲解：ToolNode(工具列表)（第一次见）───────────────────
    # ToolNode 是 LangGraph 预制好的一个"节点对象"。加括号传入工具列表就造出来。
    # 它的本事：当上一步模型说"我要调 search_child_chunks(query='…')"，
    # 它就【真的去执行】那个工具函数，把结果包成 ToolMessage 返回。执行逻辑内置，不用自己写。
    tool_node = ToolNode(tools_list)
    # InMemorySaver：把每次图运行的状态存在内存里，支持多轮对话（进程重启后消失）
    checkpointer = InMemorySaver()

    # ── 先建内层：Agent 子图（单个子问题的检索循环）──────────────────
    print("正在编译 Agent 图...")
    # ── 内置对象讲解：StateGraph(状态类型)（第一次见）─────────────────
    # StateGraph 是"造图的模具"。加括号传入状态类型(这里 AgentState)，
    # 造出一个【空的拼装台】agent_builder。传 AgentState 是告诉它：
    # "在我这张图里流动的数据，长 AgentState 那个样子"。之后往台子上加节点、连边。
    agent_builder = StateGraph(AgentState)

    # add_node：注册节点。logged_node 是调试装饰器，加不加都不影响功能。
    # partial(orchestrator, llm_with_tools=llm_with_tools) = 预绑定参数，
    # 因为 LangGraph 节点函数只能接收 state，多余参数要提前绑好。
    agent_builder.add_node("orchestrator",          logged_node("agent.orchestrator",          partial(orchestrator,          llm_with_tools=llm_with_tools)))
    agent_builder.add_node("tools",                 tool_node)
    agent_builder.add_node("compress_context",      logged_node("agent.compress_context",      partial(compress_context,      llm=llm)))
    agent_builder.add_node("fallback_response",     logged_node("agent.fallback_response",     partial(fallback_response,     llm=llm)))
    agent_builder.add_node("should_compress_context", logged_node("agent.should_compress_context", should_compress_context))
    agent_builder.add_node("collect_answer",        logged_node("agent.collect_answer",        collect_answer))

    # 固定边：A → B，总是走这条
    agent_builder.add_edge(START, "orchestrator")
    # 条件边：orchestrator 结束后，由 route_after_orchestrator_call 决定去哪。
    # 第3个参数那个字典 = "判断函数返回值 → 去哪个节点" 的对照表：
    #   路由函数返回 "tools" → 去名叫 "tools" 的节点，返回 "collect_answer" → 去 collect_answer。
    #   (键是"路由函数的返回字符串"，值是"节点名"，这里刚好同名，但概念是两回事。)
    agent_builder.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator_call,
        {"tools": "tools", "fallback_response": "fallback_response", "collect_answer": "collect_answer"}
    )
    agent_builder.add_edge("tools",            "should_compress_context")
    agent_builder.add_edge("compress_context", "orchestrator")   # 压缩完回去继续搜
    agent_builder.add_edge("fallback_response","collect_answer")
    agent_builder.add_edge("collect_answer",   END)

    # ── 内置方法讲解：.compile()（第一次见）──────────────────────────
    # compile() 把拼好的图【冻结】成一个可运行的对象。冻结后才能 .invoke()/.stream() 跑它。
    # ⭐ 记住 agent_subgraph 这个变量：它下面会作为"一个节点"塞进主图。
    agent_subgraph = agent_builder.compile()

    # ── 再建外层：主图（多轮对话 + 并行分发）────────────────────────
    graph_builder = StateGraph(State)

    graph_builder.add_node("summarize_history",   logged_node("main.summarize_history",   partial(summarize_history,   llm=llm)))
    graph_builder.add_node("rewrite_query",       logged_node("main.rewrite_query",       partial(rewrite_query,       llm=llm)))
    graph_builder.add_node("request_clarification", logged_node("main.request_clarification", request_clarification))
    # ⭐ 最关键的一行：把上面编译好的【整张子图】当成主图的一个普通节点，取名 "agent"。
    # 主图不关心它内部有循环/工具/压缩，只当它是"喂进子问题、吐出答案"的黑盒方框。
    graph_builder.add_node("agent",               agent_subgraph)
    graph_builder.add_node("aggregate_answers",   logged_node("main.aggregate_answers",   partial(aggregate_answers,   llm=llm)))

    graph_builder.add_edge(START,                  "summarize_history")
    graph_builder.add_edge("summarize_history",    "rewrite_query")
    # 条件边：rewrite_query 后，route_after_rewrite 决定去 request_clarification 或并行启动 agent。
    # 这里没写第3个字典参数，因为该路由函数直接返回真实节点名或 Send 列表，不需要对照表。
    graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)
    graph_builder.add_edge("request_clarification","rewrite_query")  # 澄清后回来重新改写
    # ⭐ 注意第一个参数是【列表】["agent"] 而不是字符串 "agent"。
    # 列表形式 = fan-in(汇合)：等 "agent" 的【所有并行实例】都跑完，才走到 aggregate_answers。
    graph_builder.add_edge(["agent"],              "aggregate_answers")
    graph_builder.add_edge("aggregate_answers",    END)

    # ── 内置方法讲解：.compile(参数)（第一次见）──────────────────────
    # 冻结主图。这次传两个参数：
    #   checkpointer=checkpointer          → 装上存档器，每步存状态 → 多轮对话能接上
    #   interrupt_before=["request_clarification"]
    #       → 一个"节点名列表"，意思是"每次快到 request_clarification 节点【之前】先暂停"，
    #         把控制权交回给用户等他补充信息（这就是澄清机制/Human-in-the-loop）。
    agent_graph = graph_builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["request_clarification"]
    )

    print("✓ Agent 图编译成功。")
    return agent_graph
