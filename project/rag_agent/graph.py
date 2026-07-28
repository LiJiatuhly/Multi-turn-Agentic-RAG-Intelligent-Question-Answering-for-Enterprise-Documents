# 组装两张图：内层 Agent 子图(检索循环) + 外层主图(改写/并行/汇总)。
#
# ┌─ 这个文件在整个项目里扮演什么角色？ ──────────────────────────────
# │ 它是"总装车间"：把 nodes.py 的节点函数、edges.py 的路由函数、graph_state.py 的
# │ 状态模板、tools.py 的工具，全部拼成两张能跑的图。本文件几乎不含业务逻辑，
# │ 只做三件事：①把函数注册成节点 ②在节点之间连边(固定边/条件边) ③compile 冻结成图。
# │
# │ 🅲 整体扫一遍，知道"谁连着谁"就行，不需要逐行背。核心结构：
# │   主图(State)     ：summarize_history → rewrite_query →(条件边)→ [agent×N 并行] → aggregate_answers
# │   子图(AgentState)：orchestrator →(条件边)→ tools → should_compress_context →(条件边)→ orchestrator
# │                                                                          ↘ compress_context ↗
# │
# │ ⚠️ 例外点名：整个文件是 🅲，但里面【有四行是 🅰️】，看着平平无奇、实则是全项目最容易迷路的地方：
# │   🅰️ ① llm.bind_tools(…) + ToolNode(…)  ── "模型说要调" 和 "框架真的调" 是两码事（下面详解）
# │   🅰️ ② add_node("agent", agent_subgraph) ── 把【整张子图】当成一个普通节点塞进主图
# │   🅰️ ③ add_edge(["agent"], …)            ── 第一个参数是【列表】= fan-in 汇合，不是普通边
# │   🅰️ ④ should_compress_context 【没有出边】 ── 它的去向写在 Command 里，这里查不到
# └──────────────────────────────────────────────────────────────────
#
# ★ 状态怎么在这两张图之间流(装配层看数据流转)★
#   · 两张图用【两套状态模板】：主图 StateGraph(State)、子图 StateGraph(AgentState)。
#     "在这张图里流动的数据长什么样"就由这里传的类决定(见 graph_state.py)。
#   · 子图是被主图当【一个普通节点】(取名 "agent")塞进去的——主图不关心它内部的
#     循环/工具/压缩，只当它"喂进子问题、吐出答案"。子问题靠 edges.py 的 Send 传进去，
#     答案靠 nodes.collect_answer 写进 agent_answers 冒泡回主图。
#   · 具体每条边上"读写了哪些状态字段"，看 数据流转对照表.md + graph_state.py 的【写】【读】。

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
        tools_list : 工具列表（search_child_chunks, retrieve_parent_chunks，来自 tools.py 的 create_tools）
    输出：编译好的 LangGraph 图（支持流式调用和 checkpointing）
    """
    # 🔢 传进来的 tools_list 真实长这样（来自 tools.py 的 create_tools()，两个工具对象）：
    #   [
    #       StructuredTool(
    #           name="search_child_chunks",              # tool("search_child_chunks")(…) 显式起的名
    #           description="在文档中搜索与用户问题相关的片段（子块）。这是检索的第一步…",
    #                                                    # ⭐ 这段就是 tools.py 里函数的 docstring！
    #                                                    #   @tool 自动把它抽出来当"给模型看的使用手册"
    #           args_schema=<pydantic模型: {query: str, limit: int}>,  # 从函数签名自动生成
    #           func=<ToolFactory._search_child_chunks 方法>,          # 真正干活的那个函数
    #       ),
    #       StructuredTool(name="retrieve_parent_chunks", description="根据 parent_id 取完整父块…", …),
    #   ]

    # 🅰️ ── 内置方法讲解：llm.bind_tools(工具列表)（第一次见）──────────────
    # bind_tools 是模型对象的方法，作用 = 把工具"挂"到模型上，返回一个【新模型】。
    #   输入：工具列表   输出：一个"知道有哪些工具可调"的新模型对象
    # 只有绑过工具的模型，回复时才可能在 AIMessage.tool_calls 里说"我要调 XX 工具"。
    # 没绑的原始 llm 只会说话、不会调工具。两个都保留：不需要工具的节点用 llm，
    # 只有 orchestrator 需要调工具，用 llm_with_tools。
    # （呼应：edges.route_after_orchestrator_call 正是靠检查回复里有没有 tool_calls 来分叉。）
    #
    # 🔢 bind_tools 到底"绑"了什么？—— 变形链（Python 函数 → 模型能读的 JSON）：
    #   ① tools.py 里的普通方法        def _search_child_chunks(self, query: str, limit: int = 5) -> str:
    #              │  @tool 装饰：读签名 → 生成 args_schema；读 docstring → 生成 description
    #              ▼
    #   ② StructuredTool 对象          （就是上面 tools_list 里那个）
    #              │  bind_tools 翻译成 OpenAI 的函数调用格式
    #              ▼
    #   ③ 塞进每次 HTTP 请求体的 "tools" 字段，模型真正看到的是这串 JSON：
    #      "tools": [{
    #          "type": "function",
    #          "function": {
    #              "name": "search_child_chunks",
    #              "description": "在文档中搜索与用户问题相关的片段（子块）。…",   ← docstring 变成了它
    #              "parameters": {"type": "object",
    #                             "properties": {"query": {"type": "string"},
    #                                            "limit": {"type": "integer"}},
    #                             "required": ["query"]}
    #          }
    #      }, { …retrieve_parent_chunks… }]
    #   ⭐ 所以 tools.py 里那句"docstring 是写给模型看的，不是给人看的注释"，指的就是 ③ 这一步。
    llm_with_tools = llm.bind_tools(tools_list)

    # 🅰️ ── 内置对象讲解：ToolNode(工具列表)（第一次见）───────────────────
    # ToolNode 是 LangGraph 预制好的一个"节点对象"。加括号传入工具列表就造出来。
    # 它的本事：当上一步模型说"我要调 search_child_chunks(query='…')"，
    # 它就【真的去执行】那个工具函数，把结果包成 ToolMessage 返回。执行逻辑内置，不用自己写。
    # ⭐ 这就是 tools.py 里两个函数返回的字符串"进入 messages"的地方——包成 ToolMessage 后，
    #    nodes._retrieval_contexts 再按固定格式把它解析出来(见 tools.py 顶部"隐形合同")。
    #
    # 🔢 ⭐【一问一答】bind_tools 和 ToolNode 是配对的两半，必须放一起看：
    #
    #   bind_tools 让模型能【说】"我要调"          ToolNode 【真的去】调
    #   ─────────────────────────────           ─────────────────────────────
    #   ① llm_with_tools 回复（orchestrator 写进 messages）：
    #      AIMessage(
    #          content="",                                    # 只调工具不说话 → 空字符串
    #          tool_calls=[{
    #              "name": "search_child_chunks",             # 点名要哪个工具
    #              "args": {"query": "上海住宿费报销上限"},     # 参数（受上面 ③ 的 JSON Schema 约束）
    #              "id": "call_abc123"                        # ⭐ 这次调用的唯一编号
    #          }]
    #      )
    #                    │  图走到 "tools" 节点 → ToolNode 接手
    #                    │  它按 name 找到对应的 StructuredTool，把 args 拆开传进去真的执行
    #                    ▼
    #   ② ToolNode 执行完，把返回的字符串包成 ToolMessage 塞回 messages：
    #      ToolMessage(
    #          content="Parent ID: 公司规章制度_p0\nFile Name: 规章制度.pdf\nContent: 上海出差住宿费上限500元/天",
    #                                                         # ⭐ 就是 tools.py 拼的那个固定格式
    #          tool_call_id="call_abc123",                    # ⭐ 和上面的 id 对上，模型才知道这是哪次调用的结果
    #          name="search_child_chunks"
    #      )
    #
    #   💡 一句话记住：模型【只会说话】，它永远碰不到向量库。
    #      "我要搜" 是 bind_tools 给的能力；"真的搜" 是 ToolNode 干的活。两件事，两个对象。
    tool_node = ToolNode(tools_list)

    # InMemorySaver：把每次图运行的状态存在内存里，支持多轮对话（进程重启后消失）
    #
    # 🔢 它存的是什么？—— 按 thread_id 存一份完整状态快照：
    #   {
    #       "user_123": {                                  # ← thread_id（调用时在 config 里传，见文件末尾）
    #           "messages": [HumanMessage("上海出差能报多少"), AIMessage("住宿每天500元…")],
    #           "conversation_summary": "用户在问差旅报销规则…",
    #           "agent_answers": [],                       # 每轮开头被 summarize_history 清空
    #           "pendingQuery": "",  "pendingClarifications": [],
    #       },
    #       "user_456": { … },                             # 另一个用户的会话，互不干扰
    #   }
    # 🔢 空态：进程一重启 → {} → 所有会话历史全丢（这就是"内存版"的代价）
    #    要跨重启保存，得换成 SqliteSaver / PostgresSaver。
    checkpointer = InMemorySaver()

    # ── 先建内层：Agent 子图（单个子问题的检索循环）──────────────────
    print("正在编译 Agent 图...")
    # ── 内置对象讲解：StateGraph(状态类型)（第一次见）─────────────────
    # StateGraph 是"造图的模具"。加括号传入状态类型(这里 AgentState)，
    # 造出一个【空的拼装台】agent_builder。传 AgentState 是告诉它：
    # "在我这张图里流动的数据，长 AgentState 那个样子"。之后往台子上加节点、连边。
    #
    # 🔢 此刻 agent_builder 真实长这样（一个空拼装台）：
    #   StateGraph(nodes={}, edges=set(), state_schema=AgentState)
    #   ⭐ 注意 state_schema 里存的是【类本身】，不是实例——运行时真正流动的是普通 dict
    #      （见 graph_state.py：State/AgentState 只是"模板"，跑起来是 dict）
    agent_builder = StateGraph(AgentState)

    # add_node：注册节点。logged_node 是调试装饰器，加不加都不影响功能。
    # partial(orchestrator, llm_with_tools=llm_with_tools) = 预绑定参数，
    # 因为 LangGraph 节点函数只能接收 state，多余参数(llm/llm_with_tools)要提前绑好。
    # （所以 nodes.py 里 orchestrator(state, llm_with_tools) 的第二个参数在这里被喂上。）
    #
    # 🔢 一个节点被注册进图，中间经过三次【变形】（拿 orchestrator 举例）：
    #   ① nodes.py 里的原函数            orchestrator(state, llm_with_tools)     ← 要两个参数
    #              │  partial(orchestrator, llm_with_tools=llm_with_tools)
    #              │  把第二个参数【焊死】，因为 LangGraph 只会给节点函数传 state 一个参数
    #              ▼
    #   ② 焊完的新函数                    f(state)                                ← 只剩一个参数 ✅
    #              │  logged_node("agent.orchestrator", f)
    #              │  再套一层"打印壳"：进节点时打印输入，出节点时打印输出
    #              ▼
    #   ③ 包了壳的函数                    f'(state)                               ← 签名不变，行为多了打印
    #              │  add_node("orchestrator", f')
    #              ▼
    #   ④ 图里一个名叫 "orchestrator" 的节点
    #   ⚠️ ①②③ 三个都是"函数"，长得一样、签名不同——这是 partial 最容易看懵的地方。
    agent_builder.add_node("orchestrator",          logged_node("agent.orchestrator",          partial(orchestrator,          llm_with_tools=llm_with_tools)))
    agent_builder.add_node("tools",                 tool_node)   # ⚠️ 唯一不是自己写的节点：ToolNode 是框架现成的
    agent_builder.add_node("compress_context",      logged_node("agent.compress_context",      partial(compress_context,      llm=llm)))
    agent_builder.add_node("fallback_response",     logged_node("agent.fallback_response",     partial(fallback_response,     llm=llm)))
    agent_builder.add_node("should_compress_context", logged_node("agent.should_compress_context", should_compress_context))  # ⚠️ 没 partial：它不调模型，不需要 llm
    agent_builder.add_node("collect_answer",        logged_node("agent.collect_answer",        collect_answer))              # ⚠️ 同上

    # 固定边：A → B，总是走这条
    agent_builder.add_edge(START, "orchestrator")
    # 条件边：orchestrator 结束后，由 route_after_orchestrator_call 决定去哪。
    # 第3个参数那个字典 = "判断函数返回值 → 去哪个节点" 的对照表：
    #   路由函数返回 "tools" → 去名叫 "tools" 的节点，返回 "collect_answer" → 去 collect_answer。
    #   (键是"路由函数的返回字符串"，值是"节点名"，这里刚好同名，但概念是两回事。)
    #
    # 🔢 这个对照表怎么用的（走一遍）：
    #   route_after_orchestrator_call 返回 → 框架查表 → 去哪个节点
    #   ────────────────────────────────    ─────────    ──────────
    #   "tools"              （有工具、预算足）  {"tools": ─────────→ "tools"}
    #   "collect_answer"     （无工具、答完了）  {"collect_answer": → "collect_answer"}
    #   "fallback_response"  （预算耗尽）        {"fallback_response": → "fallback_response"}
    #   ⚠️ 键和值这里刚好同名，纯属巧合。改成 {"去搜": "tools"} 也能跑——
    #      只要路由函数返回 "去搜"。所以别把"路由返回值"和"节点名"当成一回事。
    agent_builder.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator_call,
        {"tools": "tools", "fallback_response": "fallback_response", "collect_answer": "collect_answer"}
    )
    agent_builder.add_edge("tools",            "should_compress_context")
    # 🅰️ ⚠️ 注意：should_compress_context 【没有】在这里连出边！因为它在 nodes.py 里返回的是
    #    Command(goto=…)，去向(compress_context 或 orchestrator)写在节点内部，不由这里的边决定。
    #    这正是 LangGraph 最容易看懵的地方：找不到某节点的出边时，去 nodes.py 看它是不是返回了 Command。
    #
    # 🔢 它返回的 Command 真实长这样（在 nodes.py 里）：
    #   Command(
    #       update={"retrieval_keys": {"search::上海住宿费报销上限"},          # ← 顺手改的状态
    #               "retrieved_contexts": ["Parent ID: 公司规章制度_p0\n…"]},
    #       goto="orchestrator",                                             # ← ⭐ 出边写在这里！
    #   )
    #   ⭐ 所以你在本文件里【永远搜不到】 add_edge("should_compress_context", …) 这一行。
    agent_builder.add_edge("compress_context", "orchestrator")   # 压缩完回去继续搜
    agent_builder.add_edge("fallback_response","collect_answer")
    agent_builder.add_edge("collect_answer",   END)

    # ── 内置方法讲解：.compile()（第一次见）──────────────────────────
    # compile() 把拼好的图【冻结】成一个可运行的对象。冻结后才能 .invoke()/.stream() 跑它。
    # ⭐ 记住 agent_subgraph 这个变量：它下面会作为"一个节点"塞进主图。
    #
    # 🔢 变形前后：
    #   agent_builder   StateGraph(…)          ← "拼装台"，还能加节点、加边，【不能跑】
    #        │  .compile()   冻结：检查图合法性（有没有孤立节点、有没有死循环入口）+ 记下所有 reducer
    #        ▼
    #   agent_subgraph  CompiledStateGraph(…)  ← 【能跑】了：有 .invoke() / .stream()
    #                                             但【不能再改结构】——加不了节点、连不了边
    agent_subgraph = agent_builder.compile()

    # ── 再建外层：主图（多轮对话 + 并行分发）────────────────────────
    # 这次传的是 State(主图状态)，和子图的 AgentState 是两套模板，互不相同。
    graph_builder = StateGraph(State)

    graph_builder.add_node("summarize_history",   logged_node("main.summarize_history",   partial(summarize_history,   llm=llm)))
    graph_builder.add_node("rewrite_query",       logged_node("main.rewrite_query",       partial(rewrite_query,       llm=llm)))
    graph_builder.add_node("request_clarification", logged_node("main.request_clarification", request_clarification))
    # 🅰️ ⭐ 最关键的一行：把上面编译好的【整张子图】当成主图的一个普通节点，取名 "agent"。
    # 主图不关心它内部有循环/工具/压缩，只当它是"喂进子问题、吐出答案"的黑盒方框。
    # (edges.route_after_rewrite 的 Send("agent", …) 里那个 "agent"，指的就是这个节点。)
    #
    # 🔢 这个"黑盒"的接口——喂进去什么、吐出来什么：
    #   喂进去（由 edges.route_after_rewrite 的 Send 传，见 edges.py）：
    #       {"question": "上海出差住宿费报销上限是多少", "question_index": 0, "messages": []}
    #                     ↓  子图内部跑 orchestrator ⇄ tools ⇄ compress 循环（主图完全不管）
    #   吐出来（子图 collect_answer 写的，结束时整体冒泡回主图）：
    #       {"agent_answers": [{"index": 0,
    #                           "question": "上海出差住宿费报销上限是多少",
    #                           "answer": "每天上限 500 元。",
    #                           "contexts": ["Parent ID: 公司规章制度_p0\n…"]}]}
    #   ⚠️ 子图的 messages 默认【也会】流回主图——靠 _name_internal_message 打的 name 标签
    #      被 _is_plain_conversation_message 过滤掉，才不污染真实对话（见 nodes.aggregate_answers）
    graph_builder.add_node("agent",               agent_subgraph)
    graph_builder.add_node("aggregate_answers",   logged_node("main.aggregate_answers",   partial(aggregate_answers,   llm=llm)))

    graph_builder.add_edge(START,                  "summarize_history")
    graph_builder.add_edge("summarize_history",    "rewrite_query")
    # 条件边：rewrite_query 后，route_after_rewrite 决定去 request_clarification 或并行启动 agent。
    # 这里没写第3个字典参数，因为该路由函数直接返回真实节点名或 Send 列表，不需要对照表。
    #
    # 🔢 ⭐ route_after_rewrite 的返回值有【两种完全不同的类型】——这就是不需要对照表的原因：
    #   ① 问题不清晰 → 返回【字符串】，就是真实节点名，框架直接照着走：
    #        "request_clarification"
    #   ② 问题清晰   → 返回【Send 列表】，列表里几个 Send 就同时启动几个 agent 实例：
    #        [Send("agent", {"question": "上海出差住宿费报销上限是多少", "question_index": 0, "messages": []}),
    #         Send("agent", {"question": "上海出差交通费怎么报销",     "question_index": 1, "messages": []})]
    #        ⭐ 每个 Send 的第二个参数 = 那个 agent 实例的【初始 AgentState】
    #           "messages": [] → 每个实例从一张空白草稿纸开始，和主图 messages 是两个独立列表
    #   💡 对比上面 orchestrator 那条：那个路由返回的是"暗号字符串"，所以要查表翻译成节点名；
    #      这个路由返回的直接就是节点名 / Send 对象，不用翻译。
    graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)
    graph_builder.add_edge("request_clarification","rewrite_query")  # 澄清后回来重新改写
    # 🅰️ ⭐ 注意第一个参数是【列表】["agent"] 而不是字符串 "agent"。
    # 列表形式 = fan-in(汇合)：等 "agent" 的【所有并行实例】都跑完，才走到 aggregate_answers。
    # (各实例的答案已由 collect_answer 冒泡进 State.agent_answers，靠 accumulate_or_reset 追加不覆盖。)
    #
    # 🔢 汇合时 State["agent_answers"] 真实长这样（2 个实例都跑完了）：
    #   [{"index": 1, "question": "上海出差交通费怎么报销",   "answer": "高铁二等座据实报销。", "contexts": [...]},
    #    {"index": 0, "question": "上海出差住宿费报销上限是多少", "answer": "每天上限 500 元。", "contexts": [...]}]
    #   ⚠️ 顺序是【乱的】——谁先跑完谁先进列表。所以 aggregate_answers 必须按 index 重排。
    #
    # ⚠️ 写成 add_edge("agent", …)（字符串）会怎样？
    #    → 变成普通边：【第一个】实例一跑完就立刻走 aggregate_answers，
    #      另一个还在搜，它的答案还没冒泡上来 → 最终回复缺一半。
    #    这一对方括号 [] 就是"等所有人"和"谁先到算谁"的全部差别。
    graph_builder.add_edge(["agent"],              "aggregate_answers")
    graph_builder.add_edge("aggregate_answers",    END)

    # ── 内置方法讲解：.compile(参数)（第一次见）──────────────────────
    # 冻结主图。这次传两个参数：
    #   checkpointer=checkpointer          → 装上存档器，每步存状态 → 多轮对话能接上
    #   interrupt_before=["request_clarification"]
    #       → 一个"节点名列表"，意思是"每次快到 request_clarification 节点【之前】先暂停"，
    #         把控制权交回给用户等他补充信息（这就是澄清机制/Human-in-the-loop）。
    #         暂停后用户再发消息，会从 rewrite_query 重新进入(见 request_clarification→rewrite_query 边)。
    #
    # 🔢 "暂停"到底是什么状态？——用户问了句含糊的"报销怎么弄"，图跑到这里停住：
    #   ① rewrite_query 判定不清晰，写进状态：
    #        {"questionIsClear": False,
    #         "pendingQuery": "报销怎么弄",              # 把原问题存着，免得暂停后丢了
    #         "pendingClarifications": [],
    #         "messages": [..., AIMessage("请问你想问哪类报销？差旅、办公用品还是其他？",
    #                                     name="clarification")]}   # ← 打了 name，不算真实对话
    #   ② 框架看到 interrupt_before 命中 → 【不执行 request_clarification 节点】，直接停
    #   ③ 整个状态被 checkpointer 按 thread_id 存下来，invoke() 返回，控制权回到你的代码
    #   ④ 用户补一句"差旅住宿" → 你再调一次 invoke，从 rewrite_query 重新进入，
    #      rewrite_query 读出 pendingQuery + pendingClarifications，拼成完整问题："差旅住宿报销怎么弄"
    #   ⚠️ 注意 request_clarification 这个节点函数【几乎什么都不干】（返回 {}）——
    #      它只是一块"路障牌"，用来给 interrupt_before 一个可以停靠的名字。
    agent_graph = graph_builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["request_clarification"]
    )

    print("✓ Agent 图编译成功。")

    # 🔢 返回的 agent_graph 是怎么被调用的（本文件之外，chat 入口那边）：
    #   config = {"configurable": {"thread_id": "user_123"}}    # ← checkpointer 靠它区分是谁的会话
    #   result = agent_graph.invoke(
    #       {"messages": [HumanMessage("上海出差住宿和交通能报多少")]},   # ← 只喂新消息，历史 checkpointer 自动接上
    #       config
    #   )
    #   🔢 result 真实长这样（一整份跑完的 State）：
    #     {"messages": [HumanMessage("上海出差住宿和交通能报多少"),
    #                   AIMessage("上海出差方面：住宿费每天上限 500 元；交通费按高铁二等座据实报销。")],
    #      "conversation_summary": "…", "agent_answers": [...], "originalQuery": "…", …}
    #   ⚠️ 没传 config / 没传 thread_id → 装了 checkpointer 的图会直接【报错】，跑不起来。
    return agent_graph