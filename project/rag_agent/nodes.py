# 图的节点函数：图上每个方框里的具体代码。
# 节点函数 = 接收当前状态(state) → 做一件事 → 返回要更新的字段(dict)
# 框架把返回的 dict 和旧状态合并，字段有 reducer 的走 reducer，没有的直接覆盖。

from typing import Literal, Set
from langchain_core.messages import (
    SystemMessage,   # 系统提示词消息
    HumanMessage,    # 用户消息
    RemoveMessage,   # 特殊消息：告诉框架"删掉这条"
    AIMessage,       # 模型回复消息
    ToolMessage,     # 工具返回结果消息
)
# ── 内置对象讲解：这 5 个"消息类"长什么样、怎么用（第一次见，务必看）──────
# 和大模型对话，历史是一个"消息列表"，列表里每一项是一个"消息对象"。
# 这些对象都来自 langchain_core，构造方式统一是：  类名(content="文本", 其它可选字段)
# 每种消息代表一个"角色"，模型靠角色区分谁说的话：
#   SystemMessage(content="你是…")          → 系统指令(提示词)，给模型定规矩
#   HumanMessage(content="报销多少？")       → 用户说的话
#   AIMessage(content="报销六百元")          → 模型回复的话
#       ↑ AIMessage 还可能带一个 .tool_calls 字段(列表)，表示"模型想调哪些工具"
#   ToolMessage(content="检索结果…", tool_call_id="…") → 工具执行后的返回结果
#   RemoveMessage(id="要删的那条消息的id")   → 不含正文，专门用来"删除某条历史消息"
# 构造好之后，常用它们的这几个属性(字段)：
#   .content    → 这条消息的正文文本
#   .tool_calls → 仅 AIMessage 有，形如 [{"name":"工具名","args":{...},"id":"…"}]
#   .id         → 每条消息的唯一编号(框架自动生成，删消息时要用)
#   .name       → 可选的名字标记(本项目用它给子图内部消息打标签，见下方 _name_internal_message)
from langgraph.types import Command     # 允许节点同时"改状态 + 决定去哪"
from .graph_state import State, AgentState
from .schemas import QueryAnalysis
from .prompts import *
from utils import estimate_context_tokens
from config import (
    BASE_TOKEN_THRESHOLD,       # 触发压缩的基础 token 阈值
    CHILD_CHUNK_SEPARATOR,      # 分隔多个子块结果的分隔符
    MAIN_HISTORY_MESSAGES_TO_KEEP,  # 主图保留多少条近期消息
    TOKEN_GROWTH_FACTOR,        # 压缩摘要允许增长的比例
)

# 启动时就检查配置合法性，而不是等到运行时才发现
if MAIN_HISTORY_MESSAGES_TO_KEEP < 2:
    raise ValueError("MAIN_HISTORY_MESSAGES_TO_KEEP must be at least 2.")

# 在收集答案之前，主图保留的消息条数（比总量少1，给最新的问题留位）
PRE_ANSWER_HISTORY_MESSAGES_TO_KEEP = max(MAIN_HISTORY_MESSAGES_TO_KEEP - 1, 0)


# ============================================================
# 🅱️ 辅助函数（一眼能懂，知道功能即可）
# ============================================================

def _is_plain_conversation_message(msg) -> bool:
    """判断一条消息是否是"普通对话消息"（用于历史摘要和保留）。

    普通对话 = 用户问题 或 助手回答，排除：带工具调用的 AI 消息、内部标记消息。
    输入：任意消息对象
    输出：True = 是普通对话消息，False = 不是
    """
    return (
        isinstance(msg, (HumanMessage, AIMessage))
        and not getattr(msg, "tool_calls", None)  # 排除"我要调工具"的 AI 消息
        and not getattr(msg, "name", None)         # 排除打了内部标记的消息
    )


def _name_internal_message(message, name):
    """🅰️（特殊）给消息打上内部标记，防止它被当成对话历史流回主图。

    这个函数只有 1 行，但它反证了一件重要的事：
    子图的消息默认会流回主图的 messages 字段。
    如果不打标记，Agent 子图里的"我要调工具""工具返回了什么"
    都会出现在主图的对话历史里，污染下一轮对话。
    打上 name 标记后，_is_plain_conversation_message 会把它们过滤掉。

    model_copy(update={...}) 是 pydantic 对象的"浅拷贝并修改字段"，
    不会改原对象，返回一个新对象。
    """
    return message.model_copy(update={"name": name})


def _retrieval_contexts(messages) -> list[str]:
    """从消息列表里提取所有有效的检索结果文本，去重保序。

    输入：消息列表（包含 ToolMessage）
    输出：去重后的文本块列表（用于 RAGAS 评测，不发给模型）

    dict.fromkeys 去重保序的用法同 append_unique（见 graph_state.py）。
    """
    contexts = []
    # 下面这些前缀是"没搜到"或"出错了"的标记，不算有效检索结果
    ignored_prefixes = (
        "NO_RELEVANT_CHUNKS",
        "NO_PARENT_DOCUMENT",
        "RETRIEVAL_ERROR:",
        "PARENT_RETRIEVAL_ERROR:",
    )
    for message in messages:
        if not isinstance(message, ToolMessage):  # 只看工具返回的消息
            continue
        content = str(message.content).strip()
        if content and not content.startswith(ignored_prefixes):
            # search_child_chunks 返回多个块，用分隔符拼在一起；这里拆开
            parts = content.split(CHILD_CHUNK_SEPARATOR) if message.name == "search_child_chunks" else [content]
            contexts.extend(part for part in parts if part)
    return list(dict.fromkeys(contexts))  # 去重保序（同 append_unique 的语法）


def _format_conversation(messages) -> str:
    """🅱️ 把消息列表格式化成"User: ...\nAssistant: ..."文本，给摘要模型看。"""
    lines = []
    for msg in messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def _remove_messages_not_in(messages, keep_ids) -> list:
    """🅱️ 生成一批 RemoveMessage，用于删除不在白名单里的消息。

    输入：
        messages : 当前全部消息
        keep_ids : 要保留的消息 id 集合
    输出：[RemoveMessage(id=...), ...] 列表，框架收到这些消息后会删除对应条目。
    """
    removals = []
    for msg in messages:
        msg_id = getattr(msg, "id", None)
        if isinstance(msg, SystemMessage) or not msg_id:
            continue                              # 系统消息和没有 id 的跳过
        if msg_id not in keep_ids:
            removals.append(RemoveMessage(id=msg_id))  # 标记为"待删除"
    return removals


def _recent_conversation(messages, pending_query="") -> list:
    """🅱️ 取当前用户消息之前的近期普通对话（排除子图内部消息）。

    澄清流程时额外去掉那个未解决的问题和对应的澄清请求，
    因为它们已经作为 pendingQuery / pendingClarifications 被显式传递了。

    输入：
        messages      : 当前全部消息
        pending_query : 等待澄清的问题文本（有则排除它之前的那段）
    输出：过滤后的近期消息列表
    """
    # 只保留普通对话（过滤掉工具调用、内部标记消息）
    plain_messages = [msg for msg in messages if _is_plain_conversation_message(msg)]
    # 去掉最后一条（那是当前用户消息，不算"之前的"）
    recent_messages = plain_messages[:-1]

    # if：处于澄清流程（有未解决问题）→ 要把那个问题及其之后的消息排除掉
    if pending_query:
        # 倒序找到 pendingQuery 对应的那条消息，之前的才算"近期上下文"
        for index in range(len(recent_messages) - 1, -1, -1):
            msg = recent_messages[index]
            if isinstance(msg, HumanMessage) and str(msg.content).strip() == pending_query:
                return recent_messages[:index]  # 切掉这条及之后的

    return recent_messages


# ============================================================
# 主图节点
# ============================================================

def summarize_history(state: State, llm):
    """🅱️ 历史摘要节点：压缩旧消息，避免 token 无限增长。

    每轮对话开始时自动触发。把"太老的"消息摘要成一段文字存到
    conversation_summary，再发 RemoveMessage 把原始消息删掉。
    同时发 __reset__ 标记清空上一轮的 agent_answers。

    输入：State（读 messages、conversation_summary）
    输出：dict，包含 conversation_summary 更新 + messages 删除列表 + agent_answers 重置
    """
    messages = state.get("messages", [])
    # 先准备"重置 agent_answers"的更新（每轮必做）
    updates = {"agent_answers": [{"__reset__": True}]}

    # if：这一轮压根没有消息 → 只做上面的重置，直接返回
    if not messages:
        return updates

    plain_messages = [msg for msg in messages if _is_plain_conversation_message(msg)]
    keep_count = PRE_ANSWER_HISTORY_MESSAGES_TO_KEEP

    # 要摘要的 = 全部普通消息里，保留最后 keep_count 条，其余的摘要
    messages_to_summarize = plain_messages[:-keep_count] if len(plain_messages) > keep_count else []
    # 要保留的消息 id（最近几条）
    keep_ids = {getattr(msg, "id", None) for msg in plain_messages[-keep_count:]}
    keep_ids.discard(None)

    # 生成删除指令（除了要保留的，其他全删）
    removals = _remove_messages_not_in(messages, keep_ids)
    # if：确实有要删的旧消息 → 写进更新（删除也是通过 messages 字段下发的）
    if removals:
        updates["messages"] = removals

    # if：没有够老、需要摘要的消息 → 到此为止（本轮只做了重置和删除）
    if not messages_to_summarize:
        return updates

    # 把"已有摘要 + 需要摘要的旧消息"一起发给模型，让它合并成新摘要
    existing_summary = state.get("conversation_summary", "").strip()
    conversation = "Existing summary:\n"
    conversation += f"{existing_summary or '(none)'}\n\n"
    conversation += "New messages to merge into the summary:\n"
    conversation += _format_conversation(messages_to_summarize)

    # ── 内置方法讲解：llm.invoke(消息列表)（第一次见）──────────────────
    # .invoke() 是模型对象的方法，作用 = "把这串消息发给大模型，等它回复"。
    #   入参：一个消息列表（就是上面那些 SystemMessage/HumanMessage 拼成的 list）
    #   返回：一个 AIMessage 对象（模型的回复）
    #   取回复文本：用它的 .content 属性
    # 这里发两条：System(压缩摘要的指令) + Human(要压缩的对话内容)。
    # ⚠️ 每次 .invoke() 就是一次真实的智谱 API 调用（花 token、有网络等待）。
    summary_response = llm.invoke([
        SystemMessage(content=get_conversation_summary_prompt()),
        HumanMessage(content=conversation),
    ])
    updates["conversation_summary"] = summary_response.content.strip()  # .content = 模型回复的正文
    return updates


def rewrite_query(state: State, llm):
    """🅱️ 问题改写节点：判断清晰度，改写成适合检索的形式，最多拆成 3 个子问题。

    输入：State（读 messages、conversation_summary、pendingQuery、pendingClarifications）
    输出：dict，包含 questionIsClear、rewrittenQuestions 等（清晰时），
          或 pendingQuery、clarification 消息（不清晰时）
    """
    last_message = state["messages"][-1]
    current_query = str(last_message.content).strip()
    conversation_summary = state.get("conversation_summary", "").strip()
    pending_query = state.get("pendingQuery", "").strip()
    pending_clarifications = state.get("pendingClarifications", [])
    recent_messages = _recent_conversation(state["messages"], pending_query)

    # 拼装给模型看的上下文：有哪块就加哪块（这就是"上下文"的真面目）
    context_parts = []
    if conversation_summary:                 # if：有滚动摘要 → 加进去
        context_parts.append(f"Conversation Summary:\n{conversation_summary}")
    if recent_messages:                      # if：有近期对话 → 加进去
        context_parts.append(f"Recent Conversation:\n{_format_conversation(recent_messages)}")

    if pending_query:
        # 澄清流程：当前输入是用户的补充，合并到原始问题里
        clarifications = [*pending_clarifications, current_query]
        clarification_text = "\n".join(
            f"{index}. {value}" for index, value in enumerate(clarifications, start=1)
        )
        context_parts.append(
            f"Unresolved User Query:\n{pending_query}\n\n"
            f"User Clarifications:\n{clarification_text}"
        )
        original_query = f"{pending_query}\nClarifications:\n{clarification_text}"
    else:
        clarifications = []
        context_parts.append(f"User Query:\n{current_query}")
        original_query = current_query

    context_section = "\n\n".join(context_parts)

    # json_mode 结构化输出：智谱不支持强制 tool_choice，改用 response_format=json_object
    llm_with_structure = llm.with_structured_output(QueryAnalysis, method="json_mode")
    response = llm_with_structure.invoke([
        SystemMessage(content=get_rewrite_query_prompt()),
        HumanMessage(content=context_section)
    ])

    # 如果这条是澄清回复，给它打上内部标记，避免被当成普通对话历史
    clarification_message_update = (
        [_name_internal_message(last_message, "clarification_response")]
        if pending_query else []
    )

    # if：模型判定清晰、且给出了改写后的问题 → 走"清晰"分支
    if response.questions and response.is_clear:
        # 问题清晰：写入改写后的问题列表，路由节点会据此并行发射 Agent
        return {
            "questionIsClear": True,
            "originalQuery": original_query,
            "pendingQuery": "",
            "pendingClarifications": [],
            "rewrittenQuestions": response.questions,
            "messages": clarification_message_update,
        }

    # 问题不清晰：写入澄清请求，等用户下一条消息
    clarification = (
        response.clarification_needed
        if response.clarification_needed and len(response.clarification_needed.strip()) > 10
        else "我需要更多信息才能理解你的问题。"
    )
    return {
        "questionIsClear": False,
        "originalQuery": "",
        "pendingQuery": pending_query or current_query,
        "pendingClarifications": clarifications,
        "rewrittenQuestions": [],
        "messages": clarification_message_update + [
            AIMessage(content=clarification, name="clarification")
        ],
    }


def request_clarification(state: State):
    """🅲 澄清节点：什么都不做，只是一个"暂停点"。

    LangGraph 在这个节点前设了 interrupt，图会暂停等待用户输入。
    用户输入后，从 rewrite_query 重新进入，带上新的澄清信息。
    """
    return {}


# ============================================================
# Agent 节点
# ============================================================

def orchestrator(state: AgentState, llm_with_tools):
    """🅰️ 编排节点：Agent while 循环的循环体本身。

    这是整个系统最核心的节点。它就是"让模型看手头的材料，决定下一步"。
    模型每次回复只能做两件事之一：
      ① 返回工具调用  → 说"我要搜这个"（route 会把它发给 tools 节点）
      ② 返回文本答案  → 说"我答完了"（route 会把它发给 collect_answer）

    这就是 Agent "while 循环"的循环体：
        while True:
            模型看材料 → 决定调工具 or 答完了
            if 调工具 → 执行工具，结果塞回消息 → 回到循环头
            if 答完了 → 跳出

    输入：AgentState（读 question、messages、context_summary）
    输出：dict（写 messages、tool_call_count、iteration_count）
    """
    # 取压缩上下文摘要（可能为空，第一次总是空）
    context_summary = state.get("context_summary", "").strip()

    # 系统提示词：告诉模型它的角色和行为规则
    sys_msg = SystemMessage(content=get_orchestrator_prompt())

    # 如果有压缩上下文，以"人类消息"的形式注入（让模型把它当已知背景）
    summary_injection = (
        [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}")]
        if context_summary else []
    )

    # ── 第一次进入（messages 为空）：强制要求先搜索 ──────────────────
    if not state.get("messages"):
        human_msg = HumanMessage(content=state["question"], name="agent_question")
        # 额外加一句强制指令，防止模型跳过检索直接用自己的知识回答
        force_search = HumanMessage(content="你必须先调用 'search_child_chunks' 工具作为回答这个问题的第一步。")

        # ── 内置方法讲解：llm_with_tools.invoke(...)（绑过工具的模型）──────
        # llm_with_tools 是"绑了工具的模型"(在 graph.py 里 llm.bind_tools(...) 得到)。
        # 它 .invoke() 后返回的 AIMessage 有两种可能：
        #   ① 模型决定调工具 → response.content 为空，response.tool_calls 里装着
        #      [{"name":"search_child_chunks","args":{"query":"…"},"id":"…"}] 这样的调用请求
        #   ② 模型直接作答   → response.content 是答案文本，response.tool_calls 为空列表
        # 把 [系统提示, 压缩上下文, 用户问题, 强制指令] 一起发给模型
        response = llm_with_tools.invoke([sys_msg] + summary_injection + [human_msg, force_search])

        # 打上内部标记（防止流回主图的对话历史）
        response = _name_internal_message(response, "agent_response")

        return {
            "messages": [human_msg, response],
            # response.tool_calls 是本次工具调用列表，len 就是本次调用数
            "tool_call_count": len(response.tool_calls or []),
            "iteration_count": 1,   # 第一轮
        }

    # ── 后续轮次：带上完整的历史消息继续 ────────────────────────────
    # state["messages"] 包含了之前所有的问题、工具调用、工具结果
    response = llm_with_tools.invoke([sys_msg] + summary_injection + state["messages"])
    response = _name_internal_message(response, "agent_response")

    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    return {
        "messages": [response],
        "tool_call_count": len(tool_calls) if tool_calls else 0,
        "iteration_count": 1,   # 每轮都 +1，框架自动累加（因为 reducer 是 operator.add）
    }


def fallback_response(state: AgentState, llm):
    """🅱️ 兜底节点：预算耗尽时，用现有材料强行给出最好的答案。

    输入：AgentState（读 messages、context_summary、question）
    输出：dict（写 messages，包含模型的最终回复）
    """
    # 从消息历史里收集所有工具返回结果，去重
    seen = set()
    unique_contents = []
    for m in state["messages"]:
        if isinstance(m, ToolMessage) and m.content not in seen:
            unique_contents.append(m.content)
            seen.add(m.content)

    context_summary = state.get("context_summary", "").strip()

    # 把"压缩上下文 + 当前检索数据"拼成给模型看的材料
    context_parts = []
    if context_summary:
        context_parts.append(f"## Compressed Research Context (from prior iterations)\n\n{context_summary}")
    if unique_contents:
        context_parts.append(
            "## Retrieved Data (current iteration)\n\n" +
            "\n\n".join(f"--- DATA SOURCE {i} ---\n{content}" for i, content in enumerate(unique_contents, 1))
        )

    context_text = "\n\n".join(context_parts) if context_parts else "No data was retrieved from the documents."

    prompt_content = (
        f"USER QUERY: {state.get('question')}\n\n"
        f"{context_text}\n\n"
        f"INSTRUCTION:\nProvide the best possible answer using only the data above."
    )
    response = llm.invoke([SystemMessage(content=get_fallback_response_prompt()), HumanMessage(content=prompt_content)])
    response = _name_internal_message(response, "agent_response")
    return {"messages": [response]}


def should_compress_context(state: AgentState) -> Command[Literal["compress_context", "orchestrator"]]:
    """🅰️ 压缩判断节点：同时改状态 + 决定去哪——LangGraph 最容易迷路的地方。

    普通节点只能返回 dict（更新状态），去哪由外部的边决定。
    但这个节点用 Command 同时做两件事：
      update = 要更新的状态字段
      goto   = 下一步去哪个节点

    为什么要这样？
    工具执行完之后，不能让框架直接回去找 orchestrator，
    因为需要先决定"token 够不够"——够了直接回去，不够先压缩再回去。
    这个"决定"本身就要修改状态（记录下已用的 retrieval_keys），
    所以必须用 Command 把两件事原子地做完。

    输入：AgentState（读 messages、retrieval_keys、context_summary）
    输出：Command（update=状态更新, goto=下一节点名）
    """
    messages = state["messages"]

    # ── 第一段：翻出「最近一次」模型调了哪些工具，记成一串"暗号" ──
    # 目的：把"这次搜过什么 / 取过哪个父块"记下来，之后写进摘要，防止模型重复搜同样的东西。
    new_ids: Set[str] = set()                    # 本轮新用到的检索键，先建一个空集合

    for msg in reversed(messages):               # reversed = 从最后一条往前翻
        # 只关心「模型发出的、且带工具调用的」那一条消息。
        # getattr(msg,"tool_calls",None)：安全取 tool_calls，没有这个属性就返回 None（假）。
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):

            for tc in msg.tool_calls:            # 一条 AI 消息里可能同时调了好几个工具，逐个看
                # 每个 tc 长这样：{"name":"工具名", "args":{参数字典}, "id":"..."}

                # 情况①：这次调的是「取父块」工具
                if tc["name"] == "retrieve_parent_chunks":
                    # 模型填的父块 ID 参数名不固定（parent_id / id / ids 都可能出现），
                    # 用 a or b or c or [] 依次兜底：谁先非空就用谁，全都没有就用空列表 []。
                    raw = tc["args"].get("parent_id") or tc["args"].get("id") or tc["args"].get("ids") or []
                    # raw 可能是单个字符串 "公司规章制度_p0"，也可能是一批 ["_p0","_p1"]
                    if isinstance(raw, str):     # 是单个字符串
                        new_ids.add(f"parent::{raw}")               # → 记成 "parent::公司规章制度_p0"
                    else:                        # 是一批 ID（列表）
                        new_ids.update(f"parent::{r}" for r in raw) # → 逐个加前缀后批量并入

                # 情况②：这次调的是「搜子块」工具
                elif tc["name"] == "search_child_chunks":
                    query = tc["args"].get("query", "")            # 取搜索关键词，没有就空字符串
                    if query:                                      # 关键词非空才记
                        new_ids.add(f"search::{query}")            # → 记成 "search::上海 住宿费 报销上限"

            break   # ★关键★ 只看「最近这一条」AI 消息就 break 停下。
                    # 更早的工具调用在之前的轮次里已经记过了，不必再往前翻。

    # 把本轮新记的键，并入历史已记的集合（| 是集合并集，重复的自动去掉）
    updated_ids = state.get("retrieval_keys", set()) | new_ids

    # ── 第二段：估算当前 token 数，决定要不要压缩 ──
    current_token_messages = estimate_context_tokens(messages)   # ① 消息历史占多少 token
    current_token_summary = estimate_context_tokens([HumanMessage(content=state.get("context_summary", ""))])  # ② 已有摘要占多少
    current_tokens = current_token_messages + current_token_summary          # 当前总量 = ① + ②
    # 允许的上限 = 固定阈值 + 摘要已有大小的一定比例（摘要越大，上限越高，给它留余地）
    max_allowed = BASE_TOKEN_THRESHOLD + int(current_token_summary * TOKEN_GROWTH_FACTOR)

    goto = "compress_context" if current_tokens > max_allowed else "orchestrator"

    # ── 用 Command 同时写状态 + 指定去向 ─────────────────────────────
    return Command(
        update={
            "retrieval_keys": updated_ids,                      # 更新已用检索键集合
            "retrieved_contexts": _retrieval_contexts(messages), # 更新检索到的文本块
        },
        goto=goto,  # "compress_context" 或 "orchestrator"
    )


def compress_context(state: AgentState, llm):
    """🅱️ 上下文压缩节点：把过长的消息历史压缩成摘要，腾出空间继续搜索。

    输入：AgentState（读 messages、context_summary、question、retrieval_keys）
    输出：dict（写 context_summary + messages 删除列表）

    完成后去 orchestrator 继续下一轮。
    """
    messages = state["messages"]
    existing_summary = state.get("context_summary", "").strip()

    if not messages:
        return {}

    # 拼出给压缩模型看的内容（问题 + 已有摘要 + 本轮消息历史）
    conversation_text = f"USER QUESTION:\n{state.get('question')}\n\nConversation to compress:\n\n"
    if existing_summary:
        conversation_text += f"[PRIOR COMPRESSED CONTEXT]\n{existing_summary}\n\n"

    for msg in messages[1:]:  # 跳过第一条（用户问题，已经在上面了）
        if isinstance(msg, AIMessage):
            tool_calls_info = ""
            if getattr(msg, "tool_calls", None):
                calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                tool_calls_info = f" | Tool calls: {calls}"
            conversation_text += f"[ASSISTANT{tool_calls_info}]\n{msg.content or '(tool call only)'}\n\n"
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "tool")
            conversation_text += f"[TOOL RESULT — {tool_name}]\n{msg.content}\n\n"

    summary_response = llm.invoke([
        SystemMessage(content=get_context_compression_prompt()),
        HumanMessage(content=conversation_text)
    ])
    new_summary = summary_response.content

    # 在摘要末尾追加"已执行过的操作清单"，防止模型重复检索
    retrieved_ids: Set[str] = state.get("retrieval_keys", set())
    if retrieved_ids:
        parent_ids = sorted(r for r in retrieved_ids if r.startswith("parent::"))
        search_queries = sorted(r.replace("search::", "") for r in retrieved_ids if r.startswith("search::"))

        block = "\n\n---\n**Already executed (do NOT repeat):**\n"
        if parent_ids:
            block += "Parent chunks retrieved:\n" + "\n".join(f"- {p.replace('parent::', '')}" for p in parent_ids) + "\n"
        if search_queries:
            block += "Search queries already run:\n" + "\n".join(f"- {q}" for q in search_queries) + "\n"
        new_summary += block

    return {
        "context_summary": new_summary,
        # 删掉除第一条以外的所有消息（已经摘要了，不需要原文了）
        "messages": [RemoveMessage(id=m.id) for m in messages[1:]],
    }


def collect_answer(state: AgentState):
    """🅰️ 收尾节点：子图 → 主图的唯一接口，两个 agent_answers 的差异在这里生效。

    这个函数是子图和主图之间的"交接点"：
      - 子图里：从 AgentState.messages 取出模型的最终回复，打包成 dict
      - 主图里：这个 dict 会被 accumulate_or_reset 追加到 State.agent_answers

    为什么两个 agent_answers 的 reducer 不同？
      AgentState.agent_answers 是普通 List（没有 reducer），子图内部直接赋值。
      State.agent_answers 用 accumulate_or_reset（有 reducer），多个并行子图
      各自 collect_answer → 各自的结果都被追加进主图的列表，不互相覆盖。

    输入：AgentState（读 messages、question_index、question、retrieved_contexts）
    输出：dict（写 final_answer + agent_answers）
    """
    last_message = state["messages"][-1]

    # 检查最后一条消息是否是有效的文本回复（不是工具调用、不是空的）
    is_valid = (
        isinstance(last_message, AIMessage)
        and last_message.content
        and not last_message.tool_calls
    )
    answer = last_message.content if is_valid else "无法生成答案。"

    return {
        "final_answer": answer,   # 这个字段实际上没有其他地方读，只是记录留档
        "agent_answers": [{
            "index": state["question_index"],   # 排序用的编号
            "question": state["question"],       # 对应的子问题
            "answer": answer,                    # 答案文本
            "contexts": state.get("retrieved_contexts", []),  # 检索到的原始块（评测用）
        }]
        # 这个列表会被主图的 accumulate_or_reset 追加到 State.agent_answers
        # 多个并行子图都会写这里，每个写入都被 reducer 追加，不覆盖
    }


def aggregate_answers(state: State, llm):
    """🅱️ 汇总节点：把所有并行 Agent 的答案整合成一个最终回复。

    输入：State（读 agent_answers、originalQuery、messages）
    输出：dict（写 messages，包含最终 AI 回复 + 旧消息删除列表）
    """
    messages = state.get("messages", [])
    plain_messages = [msg for msg in messages if _is_plain_conversation_message(msg)]
    # 只保留最近几条普通对话消息的 id，其余的生成删除指令
    keep_ids = {getattr(msg, "id", None) for msg in plain_messages[-PRE_ANSWER_HISTORY_MESSAGES_TO_KEEP:]}
    keep_ids.discard(None)
    removals = _remove_messages_not_in(messages, keep_ids)

    if not state.get("agent_answers"):
        return {"messages": removals + [AIMessage(content="没有生成任何答案。")]}

    # 按 index 排序，保证答案顺序和问题顺序一致
    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += (f"\nRetrieved response {i}:\n" f"{ans['answer']}\n")

    # 把所有子答案 + 原始问题一起发给模型，让它整合成最终回复
    user_message = HumanMessage(content=f"""Original user question: {state["originalQuery"]}\nRetrieved answers:{formatted_answers}""")
    synthesis_response = llm.invoke([SystemMessage(content=get_aggregation_prompt()), user_message])
    return {"messages": removals + [AIMessage(content=synthesis_response.content)]}
