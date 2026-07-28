# 组装两张图：内层 Agent 子图(检索循环) + 外层主图(改写/并行/汇总)。
#

#     "在这张图里流动的数据长什么样"就由这里传的类决定(见 graph_state.py)。
#   · 子图是被主图当【一个普通节点】(取名 "agent")塞进去的——主图不关心它内部的
#     循环/工具/压缩，只当它"喂进子问题、吐出答案"。子问题靠 edges.py 的 Send 传进去，
#     答案靠 nodes.collect_answer 写进 agent_answers 冒泡回主图。


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

    #   [
    #       StructuredTool(

    #           description="在文档中搜索与用户问题相关的片段（子块）。这是检索的第一步…",
    #                                                    # ⭐ 这段就是 tools.py 里函数的 docstring！
    #                                                    #   @tool 自动把它抽出来当"给模型看的使用手册"


    #       ),

    #   ]

    # bind_tools 是模型对象的方法，作用 = 把工具"挂"到模型上，返回一个【新模型】。
    #   输入：工具列表   输出：一个"知道有哪些工具可调"的新模型对象

    # 没绑的原始 llm 只会说话、不会调工具。两个都保留：不需要工具的节点用 llm，
    # 只有 orchestrator 需要调工具，用 llm_with_tools。

    #



    #              ▼
    #   ② StructuredTool 对象          （就是上面 tools_list 里那个）
    #              │  bind_tools 翻译成 OpenAI 的函数调用格式
    #              ▼
    #   ③ 塞进每次 HTTP 请求体的 "tools" 字段，模型真正看到的是这串 JSON：
    #      "tools": [{
    #          "type": "function",
    #          "function": {
    #              "name": "search_child_chunks",

    #              "parameters": {"type": "object",
    #                             "properties": {"query": {"type": "string"},
    #                                            "limit": {"type": "integer"}},
    #                             "required": ["query"]}
    #          }
    #      }, { …retrieve_parent_chunks… }]

    llm_with_tools = llm.bind_tools(tools_list)

    # ToolNode 是 LangGraph 预制好的一个"节点对象"。加括号传入工具列表就造出来。
    # 它的本事：当上一步模型说"我要调 search_child_chunks(query='…')"，
    # 它就【真的去执行】那个工具函数，把结果包成 ToolMessage 返回。执行逻辑内置，不用自己写。


    # 🔢 ⭐【一问一答】bind_tools 和 ToolNode 是配对的两半，必须放一起看：
    #
    #   bind_tools 让模型能【说】"我要调"          ToolNode 【真的去】调

    #   ① llm_with_tools 回复（orchestrator 写进 messages）：
    #      AIMessage(

    #          tool_calls=[{



    #          }]
    #      )
    #                    │  图走到 "tools" 节点 → ToolNode 接手
    #                    │  它按 name 找到对应的 StructuredTool，把 args 拆开传进去真的执行
    #                    ▼
    #   ② ToolNode 执行完，把返回的字符串包成 ToolMessage 塞回 messages：
    #      ToolMessage(

    #                                                         # ⭐ 就是 tools.py 拼的那个固定格式

    #          name="search_child_chunks"
    #      )
    #
    #   💡 一句话记住：模型【只会说话】，它永远碰不到向量库。

    tool_node = ToolNode(tools_list)

    # InMemorySaver：把每次图运行的状态存在内存里，支持多轮对话（进程重启后消失）
    #
    # 🔢 它存的是什么？—— 按 thread_id 存一份完整状态快照：
    #   {


    #           "conversation_summary": "用户在问差旅报销规则…",

    #           "pendingQuery": "",  "pendingClarifications": [],
    #       },

    #   }
    # 🔢 空态：进程一重启 → {} → 所有会话历史全丢（这就是"内存版"的代价）
    #    要跨重启保存，得换成 SqliteSaver / PostgresSaver。
    checkpointer = InMemorySaver()

    print("正在编译 Agent 图...")
    # StateGraph 是"造图的模具"。加括号传入状态类型(这里 AgentState)，
    # 造出一个【空的拼装台】agent_builder。传 AgentState 是告诉它：
    # "在我这张图里流动的数据，长 AgentState 那个样子"。之后往台子上加节点、连边。
    #
    # 🔢 此刻 agent_builder 真实长这样（一个空拼装台）：

    #   ⭐ 注意 state_schema 里存的是【类本身】，不是实例——运行时真正流动的是普通 dict

    agent_builder = StateGraph(AgentState)

    # add_node：注册节点。logged_node 是调试装饰器，加不加都不影响功能。



    #
    # 🔢 一个节点被注册进图，中间经过三次【变形】（拿 orchestrator 举例）：


    #              │  把第二个参数【焊死】，因为 LangGraph 只会给节点函数传 state 一个参数
    #              ▼

    #              │  logged_node("agent.orchestrator", f)
    #              │  再套一层"打印壳"：进节点时打印输入，出节点时打印输出
    #              ▼

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

    # 第3个参数那个字典 = "判断函数返回值 → 去哪个节点" 的对照表：

    #   (键是"路由函数的返回字符串"，值是"节点名"，这里刚好同名，但概念是两回事。)
    #
    # 🔢 这个对照表怎么用的（走一遍）：
    #   route_after_orchestrator_call 返回 → 框架查表 → 去哪个节点




    #   ⚠️ 键和值这里刚好同名，纯属巧合。改成 {"去搜": "tools"} 也能跑——
    #      只要路由函数返回 "去搜"。所以别把"路由返回值"和"节点名"当成一回事。
    agent_builder.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator_call,
        {"tools": "tools", "fallback_response": "fallback_response", "collect_answer": "collect_answer"}
    )
    agent_builder.add_edge("tools",            "should_compress_context")


    # 🔢 它返回的 Command 真实长这样（在 nodes.py 里）：
    #   Command(



    #   )

    agent_builder.add_edge("compress_context", "orchestrator")   # 压缩完回去继续搜
    agent_builder.add_edge("fallback_response","collect_answer")
    agent_builder.add_edge("collect_answer",   END)


    # ⭐ 记住 agent_subgraph 这个变量：它下面会作为"一个节点"塞进主图。
    #
    # 🔢 变形前后：


    #        ▼

    #                                             但【不能再改结构】——加不了节点、连不了边
    agent_subgraph = agent_builder.compile()

    # 这次传的是 State(主图状态)，和子图的 AgentState 是两套模板，互不相同。
    graph_builder = StateGraph(State)

    graph_builder.add_node("summarize_history",   logged_node("main.summarize_history",   partial(summarize_history,   llm=llm)))
    graph_builder.add_node("rewrite_query",       logged_node("main.rewrite_query",       partial(rewrite_query,       llm=llm)))
    graph_builder.add_node("request_clarification", logged_node("main.request_clarification", request_clarification))
    # 主图不关心它内部有循环/工具/压缩，只当它是"喂进子问题、吐出答案"的黑盒方框。

    #
    # 🔢 这个"黑盒"的接口——喂进去什么、吐出来什么：



    #   吐出来（子图 collect_answer 写的，结束时整体冒泡回主图）：
    #       {"agent_answers": [{"index": 0,
    #                           "question": "上海出差住宿费报销上限是多少",
    #                           "answer": "每天上限 500 元。",
    #                           "contexts": ["Parent ID: 公司规章制度_p0\n…"]}]}


    graph_builder.add_node("agent",               agent_subgraph)
    graph_builder.add_node("aggregate_answers",   logged_node("main.aggregate_answers",   partial(aggregate_answers,   llm=llm)))

    graph_builder.add_edge(START,                  "summarize_history")
    graph_builder.add_edge("summarize_history",    "rewrite_query")

    # 这里没写第3个字典参数，因为该路由函数直接返回真实节点名或 Send 列表，不需要对照表。
    #

    #   ① 问题不清晰 → 返回【字符串】，就是真实节点名，框架直接照着走：
    #        "request_clarification"



    #        ⭐ 每个 Send 的第二个参数 = 那个 agent 实例的【初始 AgentState】


    #      这个路由返回的直接就是节点名 / Send 对象，不用翻译。
    graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)
    graph_builder.add_edge("request_clarification","rewrite_query")  # 澄清后回来重新改写


    # 🔢 汇合时 State["agent_answers"] 真实长这样（2 个实例都跑完了）：



    #
    # ⚠️ 写成 add_edge("agent", …)（字符串）会怎样？
    #    → 变成普通边：【第一个】实例一跑完就立刻走 aggregate_answers，
    #      另一个还在搜，它的答案还没冒泡上来 → 最终回复缺一半。
    #    这一对方括号 [] 就是"等所有人"和"谁先到算谁"的全部差别。
    graph_builder.add_edge(["agent"],              "aggregate_answers")
    graph_builder.add_edge("aggregate_answers",    END)

    # 冻结主图。这次传两个参数：

    #   interrupt_before=["request_clarification"]

    #         把控制权交回给用户等他补充信息（这就是澄清机制/Human-in-the-loop）。

    #
    # 🔢 "暂停"到底是什么状态？——用户问了句含糊的"报销怎么弄"，图跑到这里停住：
    #   ① rewrite_query 判定不清晰，写进状态：
    #        {"questionIsClear": False,

    #         "pendingClarifications": [],

    #                                     name="clarification")]}   # ← 打了 name，不算真实对话


    #   ④ 用户补一句"差旅住宿" → 你再调一次 invoke，从 rewrite_query 重新进入，


    #      它只是一块"路障牌"，用来给 interrupt_before 一个可以停靠的名字。
    agent_graph = graph_builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["request_clarification"]
    )

    print("✓ Agent 图编译成功。")

    # 🔢 返回的 agent_graph 是怎么被调用的（本文件之外，chat 入口那边）：

    #   result = agent_graph.invoke(

    #       config
    #   )
    #   🔢 result 真实长这样（一整份跑完的 State）：
    #     {"messages": [HumanMessage("上海出差住宿和交通能报多少"),
    #                   AIMessage("上海出差方面：住宿费每天上限 500 元；交通费按高铁二等座据实报销。")],


    return agent_graph