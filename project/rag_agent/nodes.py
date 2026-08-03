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
    MAX_ITERATIONS,
    MAX_TOOL_CALLS,
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
        # ① 只接受 HumanMessage 或 AIMessage 这两种类型
        #    isinstance(msg, (A, B)) → msg 是 A 或 B 其中一种就返回 True
        isinstance(msg, (HumanMessage, AIMessage))

        # ② 排除"我要调工具"的 AI 消息
        #    AI 决定调用工具时，消息上会挂 tool_calls = [...]（非空列表）
        #    getattr(对象, "属性名", 默认值) → 安全读取属性，属性不存在时返回默认值而不报错
        #    not (...) → 只有 tool_calls 是 None 或 [] 时才通过
        and not getattr(msg, "tool_calls", None)

        # ③ 排除被 _name_internal_message 打过标记的消息
        #    子图里的消息（调工具、工具返回等）会被打上 name 标记，
        #    防止它们流回主图污染对话历史
        #    not (...) → 只有 name 是 None 或空字符串时才通过
        and not getattr(msg, "name", None)
    )


def _name_internal_message(message, name):
    """🅰️（特殊）给消息打上内部标记，防止它被当成对话历史流回主图。

    这个函数只有 1 行，但它反证了一件重要的事：
    子图的消息默认会流回主图的 messages 字段。
    如果不打标记，Agent 子图里的"我要调工具""工具返回了什么"
    都会出现在主图的对话历史里，污染下一轮对话。
    打上 name 标记后，_is_plain_conversation_message 会把它们过滤掉。
    """
    return message.model_copy(
        # model_copy() → 不改原对象，返回一个新的拷贝
        # 这样持有原对象的地方不受影响，只有拿到返回值的地方才有标记
        # update={"字段名": 新值} → 指定新对象里要改哪个字段
        # 这里把 name 字段改成传入的 name 参数，其余字段原样复制
        update={"name": name}
    )


def _retrieval_contexts(messages) -> list[str]:
    """从消息列表里提取所有有效的检索结果文本，去重保序。

    输入：消息列表（包含 ToolMessage）
    输出：去重后的文本块列表（用于 RAGAS 评测，不发给模型）
    """
    contexts = []

    # 这些前缀代表"没搜到内容"或"工具报错"，不是真实检索结果，跳过
    ignored_prefixes = (
        "NO_RELEVANT_CHUNKS",
        "NO_PARENT_DOCUMENT",
        "RETRIEVAL_ERROR:",
        "PARENT_RETRIEVAL_ERROR:",
    )

    for message in messages:
        # isinstance(对象, 类型) → 判断对象是不是这个类型，返回 True / False
        # not isinstance(...) + continue → 不是 ToolMessage 就跳过
        # 能走到下面的代码说明一定是 ToolMessage
        if not isinstance(message, ToolMessage):
            continue

        # str() 强制转成字符串防止报错，strip() 去掉首尾空格和换行
        content = str(message.content).strip()

        # 跳过空内容 和 以错误前缀开头的内容
        if content and not content.startswith(ignored_prefixes):

            # ToolMessage 上的 .name 是框架自动设置的工具名
            # search_child_chunks 一次返回多个文本块，用分隔符拼成一整段
            # 是它的结果 → split() 按分隔符切回多个独立块
            # 其他工具  → 整段就是一个块，包进列表保持格式统一
            parts = (
                content.split(CHILD_CHUNK_SEPARATOR)
                if message.name == "search_child_chunks"
                else [content]
            )

            # part for part in parts if part → 过滤掉切分后的空字符串
            # extend() → 把过滤后的块逐个追加进 contexts（不是整个列表追加）
            contexts.extend(part for part in parts if part)

    # dict.fromkeys(列表) → 以元素为键建字典，键天然不重复且保留插入顺序
    # list(...) → 转回列表
    # 效果：重复的文本块只保留第一次出现的，顺序不变
    return list(dict.fromkeys(contexts))


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
        # getattr(对象, "属性名", 默认值) → 安全读取属性
        # 属性存在返回真实值，不存在返回默认值而不报错
        # 这里：msg.id 存在返回 id 字符串，不存在返回 None
        msg_id = getattr(msg, "id", None)

        # 两种情况直接跳过，continue 后下面的代码不执行，直接取下一条 msg：
        # ① isinstance(msg, SystemMessage) → 是系统消息，不能删，它是给模型定规矩的提示词
        # ② not msg_id → msg_id 是 None 或空字符串，取不到 id 就没法定位，也跳过
        if isinstance(msg, SystemMessage) or not msg_id:
            continue

        # not in → 这条消息的 id 不在白名单里，说明它不需要保留
        if msg_id not in keep_ids:
            # RemoveMessage 是 LangChain 框架自带的消息类（文件顶部已导入）
            # 不是真的删除，而是生成一条"删除指令"交给框架
            # 框架收到后会去 messages 列表里找对应 id 的消息并删掉
            removals.append(RemoveMessage(id=msg_id))

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
    # 列表推导式：过滤掉工具调用消息和被 _name_internal_message 打过标记的消息
    # 只保留普通的用户问题和助手回答
    plain_messages = [msg for msg in messages if _is_plain_conversation_message(msg)]

    # [:-1] → 列表切片，去掉最后一条
    # 最后一条是当前用户刚发的消息，不算"之前的对话"
    recent_messages = plain_messages[:-1]

    # pending_query 非空，说明当前处于澄清流程，举例：
    #
    #   用户："帮我查一下"               ← 模糊问题，存入 state["pendingQuery"]
    #   助手："你想查什么？"             ← 模型反问
    #   用户："查上海的住宿报销标准"     ← 用户补充，存入 state["pendingClarifications"]
    #
    # 此时 state 里已经单独存了这段信息：
    #   state["pendingQuery"]          = "帮我查一下"
    #   state["pendingClarifications"] = ["查上海的住宿报销标准"]
    #
    # 但 messages 里也有这段对话的原始记录
    # 如果不排除，模型会同时看到：
    #   recent_messages 里：  "帮我查一下" + "你想查什么？"
    #   pendingQuery    里：  "帮我查一下"      ← 重复了
    #
    # 所以要把 pendingQuery 那条及之后的消息从 recent_messages 里切掉
    # 让模型只通过 pendingQuery / pendingClarifications 这两个字段看到澄清流程的内容
    if pending_query:
        # range(起点, 终点, 步长)：
        # 起点 = len-1（最后一条的下标，列表下标从0开始所以是长度-1）
        # 终点 = -1（到0为止，不包含-1）
        # 步长 = -1（每次-1，从后往前倒序遍历）
        # 倒序是因为 pendingQuery 对应的消息在列表靠后的位置，从后往前找更快
        for index in range(len(recent_messages) - 1, -1, -1):
            # recent_messages[index] → 取下标为 index 的那条消息
            msg = recent_messages[index]
            # 找到内容和 pending_query 完全一致的那条 HumanMessage
            if isinstance(msg, HumanMessage) and str(msg.content).strip() == pending_query:
                # [:index] → 切片，只保留这条之前的消息，这条及之后的全部丢弃
                return recent_messages[:index]

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
    # state 运行时就是一个普通字典，.get("键", 默认值) 是 Python 字典自带的方法
    # 取 messages 字段，没有就返回空列表（防止后面操作报错）
    messages = state.get("messages", [])

    # {"__reset__": True} 是 LangGraph 框架约定的特殊格式
    # 框架内部检测到这个标记，会把 agent_answers 清空为 []
    # 包一层列表是框架要求的格式：[{"__reset__": True}]
    # 每轮对话开始都必须清空，否则上一轮各并行 Agent 的答案会残留
    updates = {"agent_answers": [{"__reset__": True}]}

    # not messages → messages 是空列表时为 True
    # 没有任何消息，没什么可删、可摘要的，只返回上面的重置指令
    if not messages:
        return updates

    # 列表推导式：过滤掉工具调用消息和被 _name_internal_message 打过标记的消息
    # 只保留普通的用户问题和助手回答（开发者自己写的辅助函数）
    plain_messages = [msg for msg in messages if _is_plain_conversation_message(msg)]

    # 配置文件里的常量，决定"保留最近几条不压缩"
    # 比如值是 4，就保留最近 4 条，更早的才拿去压缩
    keep_count = PRE_ANSWER_HISTORY_MESSAGES_TO_KEEP

    # [:-keep_count] → 列表切片，取"倒数 keep_count 条之前"的所有消息
    # 这些才是"足够老、需要被压缩摘要"的消息
    # 如果总条数 <= keep_count，说明全部消息都还新，没什么需要压缩的，给空列表
    messages_to_summarize = plain_messages[:-keep_count] if len(plain_messages) > keep_count else []

    # { ... for msg in ... } → 集合推导式，写法和列表推导式一样，但用 {} 括起来，结果是集合
    # 集合天然不重复，正好用来存"要保留的消息 id 白名单"
    # plain_messages[-keep_count:] → 切片，取最后 keep_count 条（就是要保留的那些）
    # getattr(msg, "id", None) → 安全读取每条消息的 id 属性，没有就返回 None
    # 消息的 .id 是 LangChain 框架自动给每条消息分配的唯一标识符，开发者不用手动设置
    keep_ids = {getattr(msg, "id", None) for msg in plain_messages[-keep_count:]}

    # .discard(值) → Python 集合自带的方法，删除集合里的某个值，值不存在也不报错
    # 把 None 从白名单里移除：没有 id 的消息无法被精确定位，不应该放进白名单
    keep_ids.discard(None)

    # 开发者自己写的辅助函数：遍历全部消息，生成"不在白名单里的消息的删除指令列表"
    # 返回的是 [RemoveMessage(id=...), RemoveMessage(id=...), ...] 这样的列表
    # RemoveMessage 是 LangChain 框架自带的消息类（在文件顶部 import 进来的）
    # 它不是真的删除，而是一条"删除指令"，框架收到后才真正执行删除
    removals = _remove_messages_not_in(messages, keep_ids)

    # removals 非空，说明确实有需要删除的旧消息
    # 把删除指令列表写进 updates["messages"]
    # 框架收到 messages 字段里有 RemoveMessage，就会去真正执行删除
    if removals:
        updates["messages"] = removals

    # messages_to_summarize 是空列表时为 False
    # 说明没有够老的消息需要压缩，本轮只做了"重置 + 删除"就可以返回了
    if not messages_to_summarize:
        return updates

    # state.get("conversation_summary", "") → 取上一轮已有的摘要文本，没有就给空字符串
    # .strip() → Python 字符串自带方法，去掉首尾空格和换行
    existing_summary = state.get("conversation_summary", "").strip()

    # 下面四行在逐步拼出一段完整的文字，发给模型作为 HumanMessage 的内容
    # += 等价于 s = s + "..."，每行往同一个变量末尾追加一段
    # 拼完后 conversation 长这样：
    #   Existing summary:
    #   （上一轮的摘要）
    #
    #   New messages to merge into the summary:
    #   （需要压缩的旧消息）
    conversation = "Existing summary:\n"
    conversation += f"{existing_summary or '(none)'}\n\n"        # 没有已有摘要就显示 (none)
    conversation += "New messages to merge into the summary:\n"
    conversation += _format_conversation(messages_to_summarize)  # 开发者写的格式化辅助函数

    # llm.invoke() 是 LangChain 框架模型对象自带的方法
    # 作用：把消息列表发给大模型，等它回复，返回一个 AIMessage 对象
    # ⚠️ 每次调用就是一次真实的 API 请求（花 token、有网络等待）
    #
    # 传入的列表里有两条消息：
    #   SystemMessage → LangChain 框架自带的消息类，装"系统级指令"（告诉模型它的任务是压缩摘要）
    #   HumanMessage  → LangChain 框架自带的消息类，装"用户内容"（这里放要压缩的对话文本）
    #
    # get_conversation_summary_prompt() → 开发者写的函数（在 prompts.py），返回压缩摘要的提示词文本
    summary_response = llm.invoke([
        SystemMessage(content=get_conversation_summary_prompt()),
        HumanMessage(content=conversation),
    ])

    # summary_response 是框架返回的 AIMessage 对象
    # .content → AIMessage 自带的属性，存的是模型回复的正文文本（LangChain 框架定义的）
    # .strip() → Python 字符串方法，去掉首尾空格
    # 把新摘要写进 updates，框架会用它覆盖 State.conversation_summary
    updates["conversation_summary"] = summary_response.content.strip()
    return updates


def rewrite_query(state: State, llm):
    """🅱️ 问题改写节点：判断清晰度，改写成适合检索的形式，最多拆成 3 个子问题。

    输入：State（读 messages、conversation_summary、pendingQuery、pendingClarifications）
    输出：dict，包含 questionIsClear、rewrittenQuestions 等（清晰时），
          或 pendingQuery、clarification 消息（不清晰时）
    """
    # state["messages"] → 字典取值，拿到消息列表
    # [-1] → 列表下标，-1 表示最后一条（就是用户刚发的那条）
    last_message = state["messages"][-1]

    # last_message.content → LangChain 框架消息对象自带的属性，存的是消息正文
    # str(...) → 强制转成字符串防报错，.strip() → 去掉首尾空格和换行
    current_query = str(last_message.content).strip()

    # state.get("键", 默认值) → Python 字典自带方法，取不到就返回默认值
    conversation_summary = state.get("conversation_summary", "").strip()
    pending_query = state.get("pendingQuery", "").strip()
    pending_clarifications = state.get("pendingClarifications", [])

    # 开发者写的辅助函数：取"当前用户消息之前"的近期普通对话
    # 传入 pending_query 是为了澄清流程时把那段重复消息排除掉（详见该函数注释）
    recent_messages = _recent_conversation(state["messages"], pending_query)

    # 把有内容的上下文块逐个收集进列表，最后再拼成一段文字发给模型
    # 为什么不直接拼字符串？因为每块都是"有才加，没有跳过"，用列表收集再 join 更干净
    context_parts = []

    # if：有上一轮的滚动摘要 → 加进上下文
    if conversation_summary:
        context_parts.append(f"Conversation Summary:\n{conversation_summary}")

    # if：有近期对话消息 → 格式化后加进上下文
    # _format_conversation() → 开发者写的辅助函数，把消息列表转成"用户：xxx\n助手：xxx"这样的文本
    if recent_messages:
        context_parts.append(f"Recent Conversation:\n{_format_conversation(recent_messages)}")

    if pending_query:
        # ── 澄清流程分支：当前输入是用户的补充说明 ───────────────────────
        # [*pending_clarifications, current_query] → 列表展开语法
        # * 把 pending_clarifications 里的元素逐个展开，再加上 current_query
        # 等价于 pending_clarifications + [current_query]，但写法更简洁
        # 作用：把历次澄清 + 这次输入合并成完整的澄清列表
        clarifications = [*pending_clarifications, current_query]

        # enumerate(列表, start=1) → Python 内置函数，同时给出编号和值
        # start=1 表示编号从1开始（不是默认的0）
        # 结果示例：1. 查上海  2. 住宿标准
        # "\n".join(...) → Python 字符串自带方法，把列表里每项用换行拼成一整段
        clarification_text = "\n".join(
            f"{index}. {value}" for index, value in enumerate(clarifications, start=1)
        )

        # 把"未解决的原始问题 + 所有澄清"一起加进上下文，让模型知道完整情况
        context_parts.append(
            f"Unresolved User Query:\n{pending_query}\n\n"
            f"User Clarifications:\n{clarification_text}"
        )

        # 把原始问题和澄清列表拼成"完整问题"，清晰后会写进 State.originalQuery
        original_query = f"{pending_query}\nClarifications:\n{clarification_text}"
    else:
        # ── 正常流程分支：当前输入是普通问题 ─────────────────────────────
        clarifications = []
        context_parts.append(f"User Query:\n{current_query}")
        original_query = current_query

    # "\n\n".join(列表) → Python 字符串自带方法
    # 把 context_parts 里每一块用两个换行（空一行）拼成完整的上下文文本
    context_section = "\n\n".join(context_parts)

    # llm.with_structured_output() → LangChain 框架模型对象自带的方法
    # 作用：告诉模型"你必须返回符合 QueryAnalysis 这个结构的 JSON"
    # method="json_mode" → 用 JSON 模式（智谱不支持强制 tool_choice，所以走这个）
    # 返回一个新的模型对象，调用它时会自动把回复解析成 QueryAnalysis 实例
    # QueryAnalysis 是开发者在 schemas.py 里定义的 pydantic 数据结构
    llm_with_structure = llm.with_structured_output(QueryAnalysis, method="json_mode")

    # .invoke() → LangChain 框架模型对象自带的方法，发起一次真实 API 调用
    # 返回的 response 已经被自动解析成 QueryAnalysis 对象（不是原始字符串）
    # 可以直接用 response.is_clear / response.questions / response.clarification_needed 取值
    # SystemMessage / HumanMessage → LangChain 框架自带的消息类（文件顶部已 import）
    # get_rewrite_query_prompt() → 开发者写的函数（在 prompts.py），返回改写任务的提示词
    response = llm_with_structure.invoke([
        SystemMessage(content=get_rewrite_query_prompt()),
        HumanMessage(content=context_section)
    ])

    # 如果这条消息是澄清流程里的补充，给它打上内部标记
    # _name_internal_message() → 开发者写的辅助函数，给消息设置 .name 属性做标记
    # 打了标记后，_is_plain_conversation_message 会把它过滤掉，不污染对话历史
    #
    # 三元表达式：条件成立 → 包成单元素列表；不成立 → 空列表
    # 后面 + 拼接时，空列表不影响结果（相当于什么都不加）
    clarification_message_update = (
        [_name_internal_message(last_message, "clarification_response")]
        if pending_query else []
    )

    # response.questions / response.is_clear → QueryAnalysis 对象的属性（pydantic 解析后的值）
    # if：模型判定问题清晰、且给出了改写后的子问题列表 → 走清晰分支
    if response.questions and response.is_clear:
        # 写入改写结果，框架会用这个 dict 更新 State 里对应的字段
        # edges.py 的 route_after_rewrite 接下来读 questionIsClear 和 rewrittenQuestions 来决定去哪
        return {
            "questionIsClear": True,
            "originalQuery": original_query,
            "pendingQuery": "",           # 清空：澄清流程结束
            "pendingClarifications": [],  # 清空：澄清流程结束
            "rewrittenQuestions": response.questions,
            "messages": clarification_message_update,  # 空列表或打了标记的澄清消息
        }

    # ── 问题不清晰分支 ─────────────────────────────────────────────────
    # response.clarification_needed → QueryAnalysis 的属性，模型填的"还需要用户补充什么"
    # 太短（<=10字）说明模型没给出有效内容，用兜底话术代替
    clarification = (
        response.clarification_needed
        if response.clarification_needed and len(response.clarification_needed.strip()) > 10
        else "我需要更多信息才能理解你的问题。"
    )

    return {
        "questionIsClear": False,
        "originalQuery": "",                              # 不清晰：没有可用的完整问题
        "pendingQuery": pending_query or current_query,   # 存起来等下一轮用户补充
        "pendingClarifications": clarifications,          # 存起来等下一轮合并
        "rewrittenQuestions": [],
        # + 拼接两个列表：
        # 前半段：打了标记的澄清回复（或空列表）
        # 后半段：[AIMessage(...)] 一条新消息，发给用户让他补充信息
        # AIMessage → LangChain 框架自带的消息类，name="clarification" 是开发者打的内部标记
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
    # state.get("context_summary", "") → Python 字典自带方法，取不到就返回 ""
    # .strip() → Python 字符串自带方法，去掉首尾空格和换行
    # 第一次进来时总是 ""，只有 compress_context 节点跑过之后才有内容
    context_summary = state.get("context_summary", "").strip()

    # SystemMessage → LangChain 框架自带的消息类，装系统级指令
    # get_orchestrator_prompt() → 开发者写的函数（prompts.py），返回编排节点的提示词
    sys_msg = SystemMessage(content=get_orchestrator_prompt())

    # 三元表达式：有压缩上下文就包成单元素列表，没有就空列表
    # + 拼接时，空列表不影响结果（相当于什么都不加）
    #
    # content 里的内容实际长这样：
    #   [COMPRESSED CONTEXT FROM PRIOR RESEARCH]   ← 开发者写的标题标签，告诉模型"下面这段是什么"
    #                                                  方括号 [] 是提示词常见写法，让模型识别这是结构化标签
    #   报销上限为500元/天，需要发票原件...         ← context_summary 的真实内容
    #
    # 为什么要加标题？不加的话模型看到一段文字会困惑"这是用户说的还是搜到的材料"
    # 加了标题，模型明白"这是之前搜索研究的压缩结果，是背景材料"
    #
    # 包成 HumanMessage 而不是 SystemMessage：模型把 HumanMessage 当"材料"认真读，而不是"规矩"
    summary_injection = (
        [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}")]
        if context_summary else []
    )

    # state.get("messages") → 取 messages 字段
    # not (...) → 空列表 [] 是"假"，not 后变 True
    # Send 派发子图时初始 messages = []，所以第一轮进来这里为 True
    # 后续轮次 messages 里已有消息，not [...] == False，走下面的后续轮次分支
    if not state.get("messages"):

        # HumanMessage → LangChain 框架自带的消息类
        # content=state["question"] → 子图要回答的子问题，由 edges.py 的 Send 传进 AgentState
        # 用 [] 直接取键而不是 .get()：Send 一定传，缺了就该报错，早暴露 bug
        # name="agent_question" → 内部标记，防止这条消息流回主图被当成"用户真的说过这句话"
        human_msg = HumanMessage(content=state["question"], name="agent_question")

        # 额外一条"伪装成用户的强制指令"
        # 大模型有时会跳过工具直接用训练知识回答，这条消息强制它第一步必须去搜文档
        # 用 HumanMessage 而不是 SystemMessage：模型把 HumanMessage 当"正在对话的人说的话"，更倾向于遵守
        force_search = HumanMessage(content="你必须先调用 'search_child_chunks' 工具作为回答这个问题的第一步。")

        # llm_with_tools.invoke() → LangChain 框架方法，发起一次真实 API 调用
        # llm_with_tools 是 graph.py 里 llm.bind_tools(tools_list) 得到的"绑过工具的模型"
        # ⚠️ 绑了工具的模型回复有两种可能：
        #   ① response.tool_calls 非空 → 模型决定调工具，route 会把它发给 tools 节点继续搜
        #   ② response.tool_calls 为空 → 模型直接给出答案，route 会把它发给 collect_answer
        #
        # [sys_msg] + summary_injection + [human_msg, force_search] → 列表拼接，组成完整消息列表
        # invoke() 是"输入进去、输出出来"，输入的消息不会出现在 response 里
        # 所以下面 return 时要把 human_msg 和 response 都手动存进 messages
        response = llm_with_tools.invoke([sys_msg] + summary_injection + [human_msg, force_search])

        # 给模型回复打上内部标记，防止它流回主图的对话历史（开发者写的辅助函数）
        response = _name_internal_message(response, "agent_response")

        return {
            # human_msg 是问题，response 是模型回复，两条都要存
            # 只存 response 的话，下一轮模型看不到"当初问的是什么"，会失去上下文
            "messages": [human_msg, response],
            # response.tool_calls → AIMessage 自带属性（LangChain 框架定义）
            # 它是一个列表，每个元素代表一次工具调用，长这样：
            #   [{"name": "search_child_chunks", "args": {"query": "报销上限"}, "id": "call_abc"}]
            # 模型可以一轮同时调多个工具，所以列表里可能有多个元素
            # len() 数列表里有几个元素，就是本轮调了几次工具
            # or [] → tool_calls 可能是 None，None 不能传给 len()，用 or [] 兜底
            "tool_call_count": len(response.tool_calls or []),
            "iteration_count": 1,   # 第一轮写 1，框架用 operator.add 自动累加到计数器里
        }

    # ── 后续轮次：带上完整的历史消息继续 ────────────────────────────
    # state["messages"] 包含了之前所有的：问题、工具调用请求、工具返回结果
    # 全部传给模型，让它看着完整材料决定下一步
    response = llm_with_tools.invoke([sys_msg] + summary_injection + state["messages"])
    response = _name_internal_message(response, "agent_response")

    # ── 陌生语法讲解：hasattr(对象, "属性名") ────────────────────────
    # Python 内置函数，检查对象有没有这个属性，返回 True 或 False
    # 和 getattr 的区别：
    #   hasattr(obj, "tool_calls")          → 只判断"有没有"，返回 True/False
    #   getattr(obj, "tool_calls", None)    → 返回属性的值，没有就返回默认值
    # 这里用 hasattr 是为了安全判断：后续轮次 response 是框架返回的对象，
    # 理论上 AIMessage 都有 .tool_calls，但做一层防御更稳
    # 有 → 取真实值；没有 → 用空列表 [] 兜底，后面 len() 不会报错
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []

    return {
        "messages": [response],
        # tool_calls 是列表，每个元素是一次工具调用，len() 数有几个就是调了几次
        # if tool_calls else 0 → 空列表时直接给 0，不调 len()（效果一样，只是更明确）
        # 框架用 operator.add 把每轮的数累加成总次数（比如第1轮2次+第2轮1次=3次）
        "tool_call_count": len(tool_calls) if tool_calls else 0,
        "iteration_count": 1,   # 每轮都写 1，框架用 operator.add 累加成总次数
    }

def fallback_response(state: AgentState, llm):
    """🅱️ 兜底节点：预算耗尽时，用现有材料强行给出最好的答案。

    输入：AgentState（读 messages、context_summary、question）
    输出：dict（写 messages，包含模型的最终回复）
    """
    # ── 第一段：从消息历史里把所有工具返回结果捞出来，去重 ─────────────
    # 为什么要单独捞 ToolMessage，而不直接把整个 messages 给模型？
    # messages 里夹着 HumanMessage（问题）、AIMessage（模型想法）、ToolMessage（工具结果）
    # 三种类型。模型现在需要的是"搜到了哪些原文材料"，而不是它自己说过的话，
    # 所以这里只过滤出 ToolMessage，作为"手头的参考资料"传给模型。
    #
    # seen = set() → 空集合，用来记"哪些内容已经见过了"（原理见下方"为什么用 set"）
    # unique_contents = [] → 普通列表，按顺序存去重后的工具结果（set 无序，结果还是得用 list 存）
    seen = set()
    unique_contents = []

    # state["messages"] → AgentState 里整个循环跑下来的完整消息历史
    # 每一轮 orchestrator 发请求、tools 执行检索，结果都追加在这里
    # 用 [] 直接取键：messages 一定存在（是 MessagesState 的继承字段），取不到就该报错
    for m in state["messages"]:

        # isinstance(m, ToolMessage) → Python 内置函数，判断 m 是不是 ToolMessage 类型
        # ToolMessage → LangChain 框架自带，专门装工具执行后的返回结果
        # 只看工具结果；过滤掉 HumanMessage（用户问题）和 AIMessage（模型的调用指令）
        #
        # m.content not in seen → 利用 set 的 O(1) 查询判断这条内容是否已收集过
        #   为什么用 set 不用 list？
        #   list 的 not in 要从头扫到尾，消息越多越慢（O(n)）
        #   set  的 not in 靠哈希直接定位，无论有多少条消息速度始终一样（O(1)）
        #
        # and 短路：左边 isinstance 为 False 就不再判断右边，避免对非 ToolMessage 做无意义的查重
        #
        # ⚠️ m.content 必须是 str 才能放进 set（set 要求元素可哈希）
        #    ToolMessage.content 在多模态场景下可能是 list，那种情况会抛 TypeError
        #    当前项目工具只返回字符串，故安全；若将来扩展工具类型需注意
        if isinstance(m, ToolMessage) and m.content not in seen:
            unique_contents.append(m.content)   # 保留原始内容（顺序存进结果列表）
            seen.add(m.content)                 # 在 seen 里打"已见过"标记

    # ── 第二段：拿出压缩摘要（上一轮检索循环浓缩存下来的背景材料）─────
    # state.get("context_summary", "") → 安全取值：字段存在返回真实值，不存在返回 ""
    # compress_context 节点跑过之后才有内容；第一次进来（没跑过压缩）这里就是 ""
    # .strip() → 去掉首尾空格和换行，防止空白字符影响后面的 if 判断
    #   （纯空格字符串在 Python 里是"真"，strip() 后变成 "" 才是"假"，判断才可靠）
    #
    # 【读】state["context_summary"] ← 由 nodes.compress_context 写入
    context_summary = state.get("context_summary", "").strip()

    # ── 第三段：把"压缩摘要 + 当前检索结果"拼成给模型看的完整材料 ──────
    # 为什么分两块而不直接拼一个字符串？
    # context_summary 可能没有（第一次兜底、没跑过压缩）
    # unique_contents 可能没有（整个循环什么都没搜到）
    # 两个都可能缺席，用 list 先把"有的部分"收集起来，最后再 join 拼接，
    # 比用 if-else 逐字拼要干净得多，且顺序固定（摘要永远在前，当前数据在后）
    context_parts = []

    # 情况①：有压缩摘要 → 包上标题标签，告诉模型"这是历史搜索的浓缩"
    # 方括号 [] 是提示词惯用写法，让模型识别这是结构化的块标题，不是正文内容
    if context_summary:
        context_parts.append(f"## Compressed Research Context (from prior iterations)\n\n{context_summary}")

    # 情况②：有当前检索结果 → 逐条加编号标题，把多条结果并排展示给模型
    if unique_contents:
        context_parts.append(
            "## Retrieved Data (current iteration)\n\n" +
            # ── 陌生语法讲解：enumerate(iterable, start=数字) ──────────────
            # enumerate(列表) 默认编号从 0 开始
            # enumerate(列表, 1) 第二个参数 = 起始编号，这里从 1 开始，显示更自然（第1条、第2条）
            # 每次迭代同时拿到 i（编号）和 content（内容文本）
            # 例：enumerate(["a","b"], 1) → (1,"a"), (2,"b")
            #
            # f"--- DATA SOURCE {i} ---\n{content}" → 给每条加标题，模型看到 "DATA SOURCE 1" 知道这是独立的一条
            # "\n\n".join(...) → 每条之间空一行，视觉上清晰分隔
            "\n\n".join(f"--- DATA SOURCE {i} ---\n{content}" for i, content in enumerate(unique_contents, 1))
        )

    # ── 三元表达式：有材料就拼，没材料给一个兜底提示 ───────────────────
    # "\n\n".join(context_parts) → 把 context_parts 里的每一段用两个换行分隔拼成一整段
    #   只有摘要：只拼摘要那块
    #   只有检索结果：只拼检索结果那块
    #   两个都有：摘要在前 + 空行 + 检索结果在后
    # if context_parts else "No data..." → context_parts 是空列表时（什么都没搜到，也没摘要）
    #   给模型一个明确的"没有材料"说明，让它照实回答"没找到相关信息"，而不是凭空编造
    context_text = "\n\n".join(context_parts) if context_parts else "No data was retrieved from the documents."

    # ── 第四段：组装最终的 prompt，把问题 + 材料 + 指令一起发给模型 ─────
    # state.get("question") → 安全取子问题文本（由 Send 传入，一定有，用 .get 只是惯例）
    # 【读】state["question"] ← 由 edges.route_after_rewrite 的 Send 传入
    #
    # 三段结构：
    #   USER QUERY      → 让模型明确知道"你要回答的是这个问题"
    #   {context_text}  → 上面拼好的全部材料（摘要 + 检索结果）
    #   INSTRUCTION     → 强制约束"只用上面的材料作答"，防止模型编造或用训练知识替代
    prompt_content = (
        f"USER QUERY: {state.get('question')}\n\n"
        f"{context_text}\n\n"
        f"INSTRUCTION:\nProvide the best possible answer using only the data above."
    )

    # ── 第五段：调模型，拿到兜底回复 ────────────────────────────────────
    # llm.invoke([消息列表]) → LangChain 框架方法，向模型发起一次 API 调用
    # llm 是不带工具的原始模型（graph.py 里直接传过来）——这里不需要调工具，只需要"整合作答"
    # （和 orchestrator 的 llm_with_tools 相区别：那个绑了工具；这个没绑，只做文字合成）
    #
    # [SystemMessage(...), HumanMessage(...)] → 两条消息组成完整对话：
    #   ① SystemMessage(content=get_fallback_response_prompt())
    #      系统指令：get_fallback_response_prompt() 在 prompts.py 里定义，
    #      告诉模型"你现在是兜底模式，搜索已经结束，用手头材料尽力作答"
    #   ② HumanMessage(content=prompt_content)
    #      把问题 + 材料 + 指令包成"用户说的话"发进去
    #      用 HumanMessage 而不是 SystemMessage：模型把 HumanMessage 当"当前对话内容"更认真对待
    response = llm.invoke([SystemMessage(content=get_fallback_response_prompt()), HumanMessage(content=prompt_content)])

    # 给模型回复打上内部标记（上文辅助函数 _name_internal_message）
    # name="agent_response" → 防止这条消息流回主图后被当成真实对话历史
    # （_is_plain_conversation_message 会把带 name 的过滤掉，所以主图的对话历史不会被污染）
    response = _name_internal_message(response, "agent_response")

    # ── 返回：只更新 messages 字段，把兜底回复追加进去 ──────────────────
    # 【写】state["messages"] ← 框架把 [response] 追加到现有的 messages 列表
    # 返回的是 dict 而不是完整 AgentState：LangGraph 只更新 dict 里写到的字段，
    # 没有写的字段（question、context_summary 等）保持原样不变
    # 之后图会走向 collect_answer 节点（见 graph.py 的固定边：fallback_response → collect_answer）
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
    # ── 陌生语法讲解：-> Command[Literal["compress_context", "orchestrator"]] ──
    # 这是函数的"返回值类型标注"，告诉人和工具"这个函数返回什么"：
    #   Command     → LangGraph 自带对象，"同时更新状态 + 决定下一步"的容器（下面用到时细讲）
    #   Literal[..] → 括号里列的就是 goto 字段可能的全部取值，相当于"返回值的取值范围提示"
    # 这行只是标注，运行时不强制、不影响逻辑，看时当注释读即可

    # state["messages"] → 这个子图从启动到现在积累的完整消息历史
    # 包含：用户问题、每轮模型回复（AIMessage）、每次工具执行结果（ToolMessage）
    # 用 [] 直接取键：messages 是 MessagesState 继承来的字段，一定存在，取不到就该报错
    messages = state["messages"]


    # ── 第一段：翻出「最近一次」模型调了哪些工具，把它们记成暗号 ──────
    # 目的：把"这次搜过什么词 / 取过哪个父块 ID"记录下来
    #       之后写进压缩摘要，告诉下一轮模型"别再搜这些了"，防止重复检索

    # ── 陌生语法讲解：new_ids: Set[str] = set() ─────────────────────────
    # 这是"带类型标注的变量赋值"：
    #   new_ids          → 变量名
    #   : Set[str]       → 冒号后面是类型标注，意思是"这个变量装的是字符串集合"
    #                      纯提示，运行时 Python 不检查，但让人一眼知道这是集合、元素是字符串
    #   = set()          → 真正的赋值：初始化为空集合（这行才是真正执行的代码）
    # 对比上一段的 seen = set()：没写类型标注，但功能一样——这里显式写出来是为了更清晰
    new_ids: Set[str] = set()

    # ── 陌生语法讲解：reversed(iterable) ────────────────────────────────
    # Python 内置函数，返回一个"反向迭代器"——从最后一个元素往前遍历
    # 不产生新列表（不占额外内存），只是改变遍历方向
    # 为什么从后往前翻？
    #   messages 是按时间顺序追加的，最新的消息在末尾
    #   "最近一次模型调了哪些工具"就藏在最后那条 AIMessage 里
    #   从后往前找，遇到就记录下来，找到即 break——不用扫整个历史
    for msg in reversed(messages):

        # 双重判断，两个条件都满足才进入处理逻辑：
        # ① isinstance(msg, AIMessage)：是模型发出的消息（不是工具结果、不是用户问题）
        # ② getattr(msg, "tool_calls", None)：
        #    getattr(对象, "属性名", 默认值) → 安全取属性，属性不存在时返回默认值而不报错
        #    AIMessage 带 .tool_calls 字段：有工具调用时是非空列表（真），没有时是 [] 或 None（假）
        #    两个条件合在一起 = "找到模型发出的、且带工具调用的那条消息"
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):

            # msg.tool_calls 是个列表，一轮里模型可能同时调多个工具
            # 每个 tc（tool call）长这样：
            #   {"name": "search_child_chunks", "args": {"query": "报销上限"}, "id": "call_abc"}
            for tc in msg.tool_calls:

                # 情况①：这次调的是「取父块」工具
                if tc["name"] == "retrieve_parent_chunks":

                    # 模型填的参数名不固定（它是语言模型，不总是用同一个参数名）
                    # 用 a or b or c or [] 做"优先级兜底链"：
                    #   先试 "parent_id"，非空就用它；是 None/空就往后试 "id"；
                    #   还不行再试 "ids"；三个都没有就兜底用 []（空列表）
                    # Python 里空字符串 "" / None / [] 都是"假"，or 会跳过这些继续找下一个
                    raw = (
                        tc["args"].get("parent_id")
                        or tc["args"].get("id")
                        or tc["args"].get("ids")
                        or []
                    )

                    # raw 可能是单个字符串（"公司规章制度_p0"），也可能是一批 ID（["_p0","_p1"]）
                    if isinstance(raw, str):
                        # 单个字符串 → 加前缀 "parent::" 后直接 add 进集合
                        # 前缀的作用：区分"搜索关键词"和"父块ID"，两种暗号不混淆
                        # 例：→ "parent::公司规章制度_p0"
                        new_ids.add(f"parent::{raw}")
                    else:
                        # 一批 ID（列表）→ 逐个加前缀后批量并入集合
                        # ── 陌生语法讲解：set.update(可迭代对象) ──────────────────
                        # set.add(x)       → 加入单个元素
                        # set.update(iter) → 把可迭代对象里的每个元素逐个加入集合
                        # (f"parent::{r}" for r in raw) → 生成器表达式，惰性地逐个生成带前缀的字符串
                        # 效果 = 循环写 new_ids.add(f"parent::{r}")，但写法更简洁
                        new_ids.update(f"parent::{r}" for r in raw)

                # 情况②：这次调的是「搜子块」工具
                elif tc["name"] == "search_child_chunks":
                    # 取搜索关键词，没有 "query" 这个参数就兜底空字符串
                    query = tc["args"].get("query", "")
                    # 关键词非空才记（空字符串没有记录价值，也放不进 set 当有效暗号）
                    if query:
                        # 同样加前缀区分类型
                        # 例：→ "search::上海 住宿费 报销上限"
                        new_ids.add(f"search::{query}")

            # ★ 关键：找到最近这条带工具调用的 AIMessage 后立即 break 跳出循环 ★
            # 理由：更早的工具调用在之前的轮次里已经被记录过了（存在 retrieval_keys 里）
            # 不 break 的话会把历史上所有调用都重复收集一遍，下面的 | 合并虽然去重，但白费时间
            break

    # 把本轮新记的暗号，并入历史上已记录的集合
    # state.get("retrieval_keys", set()) → 安全取历史集合，没有就用空集合兜底
    # | → Python 集合的"并集"运算符：{a,b} | {b,c} = {a,b,c}（重复的自动去掉）
    # 【读】state["retrieval_keys"] ← 由本节点自己上一轮写入（通过 Command.update，见文末）
    # 【写】updated_ids 稍后通过 Command.update 写回 retrieval_keys
    updated_ids = state.get("retrieval_keys", set()) | new_ids


    # ── 第二段：估算当前 token 总量，和允许上限比较，决定要不要压缩 ─────
    # estimate_context_tokens(消息列表) → 项目工具函数（utils.py），
    # 根据消息内容估算 token 数（不是精确统计，是快速估算，够用即可）

    # ① 当前消息历史占了多少 token
    current_token_messages = estimate_context_tokens(messages)

    # ② 已有压缩摘要占了多少 token
    # 为什么要包成 [HumanMessage(...)]？
    # estimate_context_tokens 期望接收"消息列表"，所以把纯字符串包进 HumanMessage 再传进去
    # state.get("context_summary", "") → 没跑过压缩时是 ""，包进去就是空消息，估算结果是 0
    current_token_summary = estimate_context_tokens(
        [HumanMessage(content=state.get("context_summary", ""))]
    )

    # 总量 = 消息历史 + 压缩摘要（两部分都会被发给模型，所以要合计）
    current_tokens = current_token_messages + current_token_summary

    # 允许的 token 上限 = 固定基础阈值 + 摘要已有大小的一定比例
    # 为什么上限不是固定值？
    #   摘要本身也占 token。摘要越大，说明之前搜了很多材料，上下文本来就长。
    #   允许上限随摘要大小适度增加，避免"摘要一更新就立刻触发再次压缩"的死循环。
    # BASE_TOKEN_THRESHOLD → config.py 里定义的固定基线（比如 6000 token）
    # TOKEN_GROWTH_FACTOR  → config.py 里定义的增长比例（比如 0.3，即允许摘要再增长 30%）
    # int(...) → 乘出来的结果是浮点数，int() 截断取整（Python 不自动把 float 当 int 用）
    max_allowed = BASE_TOKEN_THRESHOLD + int(current_token_summary * TOKEN_GROWTH_FACTOR)

    # 三元表达式：超上限 → 去压缩；没超 → 直接回 orchestrator 继续搜
    # 这个字符串就是后面 Command(goto=goto) 里的目标节点名
    goto = "compress_context" if current_tokens > max_allowed else "orchestrator"


    # ── 第三段：用 Command 把"更新状态"和"去哪"打包成一个对象返回 ──────
    # ── 陌生对象讲解：Command(update={...}, goto="节点名") ──────────────
    # LangGraph 自带对象，专门给"需要同时改状态 + 自己决定去哪"的节点用。
    # 普通节点只能返回 dict，由图里预先连好的边决定去哪。
    # Command 打破了这个限制，让节点自己在运行时动态指定下一站。
    #   update → 等同于普通节点返回的 dict，框架会把里面的字段合并进 AgentState
    #   goto   → 直接写节点名，框架执行完 update 后立刻跳到那个节点
    #
    # ⚠️ 正因为如此，graph.py 里查不到 should_compress_context 的"出边"——
    #    它的去向写在这个 Command 里，不在 graph.py 的 add_edge / add_conditional_edges 里。
    #    这是最容易让人迷路的地方：找不到出边时，去看节点函数有没有返回 Command。
    return Command(
        update={
            # 【写】AgentState["retrieval_keys"] ← 历史集合 | 本轮新增
            # 有 set_union reducer（见 graph_state.py）：并行写入时求并集，不覆盖
            "retrieval_keys": updated_ids,

            # 【写】AgentState["retrieved_contexts"] ← 从消息历史里抽出有效检索原文
            # _retrieval_contexts(messages)：辅助函数（nodes.py），
            # 按 "Parent ID / File Name / Content" 格式解析 ToolMessage，
            # 过滤掉 NO_RELEVANT_CHUNKS 等暗号，去重保序后返回文本块列表
            # 有 append_unique reducer（见 graph_state.py）：追加 + 去重 + 保序，不覆盖
            # 这个字段最终由 collect_answer 读取，跟着答案一起冒泡到主图供 RAGAS 评测
            "retrieved_contexts": _retrieval_contexts(messages),
        },
        goto=goto,   # "compress_context" 或 "orchestrator"，由上面的 token 比较决定
    )

def compress_context(state: AgentState, llm):
    """🅱️ 上下文压缩节点：把过长的消息历史压缩成摘要，腾出空间继续搜索。

    输入：AgentState（读 messages、context_summary、question、retrieval_keys）
    输出：dict（写 context_summary + messages 删除列表）

    完成后去 orchestrator 继续下一轮。
    """
    # 【读】state["messages"] ← 由 orchestrator / tools / fallback_response 写入
    # 用 [] 直接取键：messages 是 MessagesState 继承字段，一定存在，取不到应该报错
    messages = state["messages"]

    # 【读】state["context_summary"] ← 由本节点上一轮自己写入（第一次进来时是 ""）
    # .strip() → 去掉首尾空白，防止空白字符串在后面的 if 判断里被误当成"有内容"
    # state.get(..., "") → 安全取值，字段不存在时兜底空字符串而不报错
    existing_summary = state.get("context_summary", "").strip()

    # 防御性兜底：理论上 orchestrator 每轮都至少有一条消息，但若列表为空就没有可压缩的材料
    # 直接返回空 dict → 框架什么都不更新，状态保持原样
    if not messages:
        return {}


    # ── 第一段：拼出给"压缩模型"看的完整材料 ────────────────────────────
    # 目的：把"这轮跑下来的全部对话（问题 + 模型思考 + 工具结果）"翻译成人类可读的文本，
    # 再让模型把这段文本压缩成精华摘要
    #
    # 为什么不直接把 messages 丢给模型？
    # messages 是 LangChain 的结构化对象列表（AIMessage / ToolMessage / HumanMessage），
    # 直接扔进去格式乱、token 浪费；整理成带标签的纯文本更让模型知道"谁说了什么"

    # 先写第一行：当前要回答的子问题
    # state.get("question") → 由 Send 传入，一定有，.get 是习惯写法
    # 【读】state["question"] ← 由 edges.route_after_rewrite 的 Send 传入
    conversation_text = f"USER QUESTION:\n{state.get('question')}\n\nConversation to compress:\n\n"

    # 如果有旧摘要（上一轮压缩留下来的精华），先把它拼进来
    # 让模型知道"前几轮已经知道了什么"，把旧摘要和新消息合并压缩成新摘要
    # 效果：每轮压缩都是"旧摘要 + 新一轮检索内容 → 更新的摘要"，像滚雪球一样
    if existing_summary:
        conversation_text += f"[PRIOR COMPRESSED CONTEXT]\n{existing_summary}\n\n"

    # ── 遍历 messages[1:] 把每条消息转成带标签的纯文本 ──────────────────
    # ── 陌生语法讲解：列表切片 messages[1:] ──────────────────────────────
    # messages[1:] = "从第二个元素开始到末尾"的新列表（原列表不变，切片产生副本）
    # messages[0] 是第一条消息——orchestrator 第一次进来时写入的用户问题（HumanMessage）
    # 为什么跳过？因为 conversation_text 开头已经手动写了 "USER QUESTION"，不需要再拼一遍
    for msg in messages[1:]:

        # ── 情况①：模型发出的消息（AIMessage）─────────────────────────────
        # ── AIMessage 和 ToolMessage 是一问一答的关系 ──────────────────────
        # 模型自己没有能力访问向量库，它只能"说话"，整个调用链是：
        #   模型说"我要搜"（AIMessage.tool_calls）
        #       → 框架 ToolNode 真的去执行
        #           → 结果返回给模型（ToolMessage）
        # 所以 AIMessage 是"意图/请求"，ToolMessage 是"执行结果"，两者靠 tool_call_id 对上
        #
        # AIMessage 有两种形态：
        #   a) 只说话（content 有文本，tool_calls 为空）→ 模型在思考/总结
        #   b) 调工具（tool_calls 非空，content 可能是空字符串）→ 模型决定去搜什么
        # 两种都要记录，但格式不同
        if isinstance(msg, AIMessage):

            # 先初始化工具调用信息为空字符串
            # 如果这条消息没有工具调用，标签就是 [ASSISTANT]；有的话追加工具信息
            tool_calls_info = ""

            # getattr(msg, "tool_calls", None) → 安全取属性，AIMessage 不一定有 tool_calls
            # or [] 兜底：None（没有工具调用）→ 空列表 → 假 → if 不进去
            #
            # 真实的 AIMessage 长这样（调工具时）：
            #   AIMessage(
            #       content="",                        # 只调工具不说话，content 是空的
            #       tool_calls=[{
            #           "name": "search_child_chunks",       # 要调哪个工具
            #           "args": {"query": "上海住宿费报销上限"}, # 传什么参数
            #           "id": "call_abc123"                  # 这次调用的唯一编号
            #       }]
            #   )
            if getattr(msg, "tool_calls", None):
                # ── 陌生语法讲解：", ".join(... for ...) ────────────────────
                # 生成器表达式（惰性）+ join 把每个工具调用格式化成
                # "工具名(参数字典)" 这样的字符串，多个用 ", " 拼接
                # 例：'search_child_chunks({"query": "报销上限"}), retrieve_parent_chunks({"parent_id": "_p0"})'
                calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                # 拼成" | Tool calls: xxx"附在标签末尾，让压缩模型知道"这条消息触发了哪些工具"
                tool_calls_info = f" | Tool calls: {calls}"

            # msg.content or '(tool call only)' → 三元兜底
            # 当模型只发了工具调用指令、content 是空字符串时（空字符串是"假"）
            # or 跳到右边，填一个占位符说明"此条是纯工具调用"，避免内容栏完全空白
            conversation_text += f"[ASSISTANT{tool_calls_info}]\n{msg.content or '(tool call only)'}\n\n"

        # ── 情况②：工具返回结果（ToolMessage）─────────────────────────────
        # ToolNode 执行完工具后，把结果包成 ToolMessage 写进 messages
        # .tool_call_id 和上面 AIMessage.tool_calls 里的 id 一一对应，
        # 模型下一轮进来看到 ToolMessage 就知道"这是我之前那次调用的结果"
        #
        # 真实的 ToolMessage 长这样：
        #   ToolMessage(
        #       content="Parent ID: 公司规章制度_p0\nFile Name: 规章.pdf\nContent: 上海出差住宿费上限500元/天",
        #       tool_call_id="call_abc123",   # 和上面 AIMessage 里的 id 对上
        #       name="search_child_chunks"
        #   )
        #
        # .content 里装的是检索工具返回的原始字符串（"Parent ID:…\nContent:…" 格式）
        # 这是最有价值的材料，压缩模型主要就是在浓缩这些原文
        elif isinstance(msg, ToolMessage):
            # getattr(msg, "name", "tool") → 安全取工具名；没有 name 属性就兜底 "tool"
            # ToolMessage.name 通常是 "search_child_chunks" 或 "retrieve_parent_chunks"
            # 放进标签里让压缩模型知道"这是哪个工具搜出来的结果"
            tool_name = getattr(msg, "name", "tool")
            conversation_text += f"[TOOL RESULT — {tool_name}]\n{msg.content}\n\n"
        # 注意：没有处理 HumanMessage 的分支
        # messages[0]（用户原始问题）已经在开头手动写了，后续不会再有 HumanMessage
        # 如果有其他类型（SystemMessage 等）就静默跳过，不拼进压缩材料


    # ── 第二段：调模型做压缩，得到新摘要 ────────────────────────────────
    # 【写】新摘要将写进 state["context_summary"]（通过最后的 return dict）
    #
    # 传两条消息：
    #   ① SystemMessage → 告诉模型"你是压缩助手，任务是提炼关键信息"
    #      get_context_compression_prompt() 在 prompts.py 定义，包含具体的压缩指令
    #   ② HumanMessage  → 上面拼好的完整对话文本（"原材料"）
    # llm → 不带工具的原始模型（这里只需要文本生成，不需要调工具）
    summary_response = llm.invoke([
        SystemMessage(content=get_context_compression_prompt()),
        HumanMessage(content=conversation_text)
    ])
    # summary_response 是 AIMessage 对象，.content 才是模型返回的纯文本摘要
    new_summary = summary_response.content


    # ── 第三段：在摘要末尾追加"已执行清单"，防止模型重复检索 ─────────────
    # 为什么要追加这段？
    # 压缩摘要会被注入下一轮 orchestrator 的上下文。如果不告诉模型"这些已经搜过了"，
    # 模型看到摘要里提到某个话题，可能再去搜一遍——白白浪费 token 和工具调用次数。
    # 这段清单就是给模型的"已执行禁止重复清单"，和 tools.py 的 docstring 里叮嘱的
    # "已出现过的父块ID不要再取"相互呼应（一个是提示词，一个是运行时的动态记录）

    # 【读】state["retrieval_keys"] ← 由 should_compress_context 写入（set_union reducer 累计的）
    # 里面存的是 "parent::xxx" / "search::xxx" 格式的暗号字符串（见上文 should_compress_context 注释）
    # Set[str] 类型标注：提示这是一个字符串集合（运行时不强制）
    retrieved_ids: Set[str] = state.get("retrieval_keys", set())

    # 有历史记录才追加；第一次压缩、retrieval_keys 为空时跳过这段
    if retrieved_ids:
        # ── 陌生语法讲解：sorted(生成器表达式 + if 过滤) ─────────────────
        # sorted(可迭代对象) → Python 内置函数，返回【排好序的新列表】（原集合不变）
        # 为什么要排序？set 本身无序，每次迭代顺序可能不一样
        # 排序后清单稳定、可读性好，也方便调试时对比两轮的区别
        #
        # r for r in retrieved_ids if r.startswith("parent::") → 生成器表达式：
        # 遍历集合，只取以 "parent::" 开头的那些，其余过滤掉
        parent_ids = sorted(r for r in retrieved_ids if r.startswith("parent::"))

        # search 类暗号：取出来的同时把 "search::" 前缀去掉，得到原始搜索词
        # r.replace("search::", "") → 把暗号还原成人类可读的词（方便模型理解"这个词搜过了"）
        search_queries = sorted(r.replace("search::", "") for r in retrieved_ids if r.startswith("search::"))

        # 开头是分隔线 + 标题，Markdown 加粗（** 包裹）让模型更容易识别这是特殊指令块
        block = "\n\n---\n**Already executed (do NOT repeat):**\n"

        # 有父块记录 → 逐条列出（去掉 "parent::" 前缀还原成原始 ID）
        if parent_ids:
            block += "Parent chunks retrieved:\n" + "\n".join(f"- {p.replace('parent::', '')}" for p in parent_ids) + "\n"

        # 有搜索记录 → 逐条列出（上面已经 replace 过了，这里直接用）
        if search_queries:
            block += "Search queries already run:\n" + "\n".join(f"- {q}" for q in search_queries) + "\n"

        # 追加到摘要末尾：模型每次看到摘要，都会同时看到这份禁止清单
        new_summary += block


    # ── 第四段：返回要更新的状态字段 ────────────────────────────────────
    # 返回 dict 而不是完整 AgentState：框架只更新 dict 里写到的字段，其余保持原样
    return {
        # 【写】AgentState["context_summary"] ← 新摘要（旧内容 + 本轮压缩 + 已执行清单）
        # 无 reducer（直接覆盖）：旧摘要已经被合并进 new_summary 了，直接替换
        # 下一轮 orchestrator 会读它，作为"已知背景"注入模型
        "context_summary": new_summary,

        # 【写】AgentState["messages"] ← RemoveMessage 删除指令列表
        # ── 陌生对象讲解：RemoveMessage(id=消息ID) ───────────────────────
        # LangGraph / LangChain 自带对象，不是真的"消息"，而是一条"删除指令"：
        # 框架看到 messages 列表里有 RemoveMessage，就把对应 id 的消息从历史里删掉
        # 这里对 messages[1:] 里的每条消息都造一个 RemoveMessage，效果 = 删掉除第一条以外的所有消息
        #
        # 为什么保留 messages[0]（第一条）？
        # 第一条是 orchestrator 第一次进来时写的用户问题（HumanMessage），
        # 这个子问题要始终留在 messages 里，让下一轮 orchestrator 一进来还能看到"我要回答什么"
        # 其余的（模型思考 + 工具结果）已经浓缩进摘要了，原文占着 token 没有价值，统统删掉
        #
        # m.id → LangChain 消息对象的唯一 ID（框架自动生成），RemoveMessage 靠它定位要删哪条
        "messages": [RemoveMessage(id=m.id) for m in messages[1:]],
    }
    # 完成后图走向 orchestrator（见 graph.py 的固定边：compress_context → orchestrator）
    # orchestrator 再进来时，messages 只剩第一条问题，但 context_summary 带着全部精华

# ============================================================
# 🅰️ collect_answer：子图 → 主图的【唯一出口】，两个同名 agent_answers 的差异在这里生效
# ============================================================
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
    # 【读】state["messages"] ← 由 orchestrator（正常答完的模型回复）
    #                        或 fallback_response（预算耗尽的兜底答案）写入
    # [-1] → 取列表最后一个元素，也就是"刚刚那条最终回复"
    #
    # 🔢 走到这里时，last_message 有三种真实形态（正好对应下面 is_valid 的三个条件）：
    #   ① 正常答完（从 orchestrator 来，模型不再调工具）：
    #      AIMessage(
    #          content="上海出差住宿费每天上限 500 元。",   # 有正文
    #          tool_calls=[],                             # 空列表 = 不再想搜了
    #          name="agent_response"                      # 被 _name_internal_message 打过标签
    #      )                                              # ← ✅ 有效
    #   ② 兜底答完（从 fallback_response 来）：
    #      AIMessage(content="根据现有材料，上海住宿费上限为 500 元/天。", tool_calls=[], name="agent_response")
    #                                                     # ← ✅ 有效
    #   ③ 异常态：模型只调工具、不说话（理论上路由不会把这种情况送到这，但要防）：
    #      AIMessage(content="", tool_calls=[{"name":"search_child_chunks", "args":{...}, "id":"call_abc123"}])
    #                                                     # ← ❌ 无效，content 是空的
    last_message = state["messages"][-1]

    # ── 检查最后一条消息是不是"有效的文本回复"：三个条件用 and 串起来，全中才算数 ──
    #   ① isinstance(last_message, AIMessage)
    #      是模型发的（不是 ToolMessage 工具结果、不是 HumanMessage 用户问题）
    #   ② last_message.content
    #      ⚠️ content 非空。空字符串 "" 在 Python 里是"假"，会被这一条挡掉。
    #      这条专门挡上面的形态③——模型只调工具时 content="", 那不是答案，是个动作指令。
    #   ③ not last_message.tool_calls
    #      没有工具调用。有 tool_calls = 模型还想继续搜 = 还没答完。
    #      （空列表 [] 是"假"，not [] → True，所以正常答完时这一条通过）
    #
    # ⚠️ 三个条件缺一不可，尤其②——如果只判 isinstance，形态③会被当成"答案是空字符串"混过去，
    #    最后用户看到一片空白，而不是"无法生成答案"。
    is_valid = (
        isinstance(last_message, AIMessage)
        and last_message.content
        and not last_message.tool_calls
    )

    # 三元表达式：有效 → 用模型的原话；无效 → 用一句写死的兜底文案
    # 为什么必须兜底？如果这里传空串给主图，aggregate_answers 拼提示词时会出现一段空洞，
    # 模型看到空白会自己瞎编。给一句明确的"无法生成答案。"，让主图能照实说。
    answer = last_message.content if is_valid else "无法生成答案。"

    return {
        # 【写】AgentState["final_answer"] ← 答案文本
        # ⚠️ 纯留档：整个项目里【没有任何函数读它】（见 graph_state.py 对 final_answer 的【读】标注：❗无人读）
        #    真正冒泡到主图、被下游用起来的是下面的 agent_answers。删掉这个字段不影响运行。
        "final_answer": answer,

        # 【写】AgentState["agent_answers"] ← 本子图的答案包
        #      → 子图结束后，这个值整体【冒泡】到主图 State["agent_answers"]
        #
        # ⭐⭐ 本函数最关键的一处，也是"两个同名 agent_answers"差异真正生效的地方：
        #
        #   子图 AgentState.agent_answers ：普通 List[dict]，【无 reducer】→ 直接赋值覆盖
        #       为什么不需要 reducer？因为子图内部只有 collect_answer 这一个写入点，不存在并行写。
        #
        #   主图 State.agent_answers      ：Annotated[List[dict], accumulate_or_reset]，【有 reducer】
        #       N 个并行子图各自冒泡上来时，reducer 把它们逐个【追加】进列表，互不覆盖。
        #       （见 graph_state.py 的 accumulate_or_reset）
        #
        #   ⚠️ 如果主图那个也没 reducer，N 个子图的结果会互相覆盖 → 最后只剩最后一个跑完的那个答案。
        #
        # 🔢 单个子图返回的真实样子（假设这是 0 号子问题）：
        #   [{
        #       "index": 0,
        #       "question": "上海出差住宿费报销上限是多少",
        #       "answer": "每天上限 500 元。",
        #       "contexts": ["Parent ID: 公司规章制度_p0\nFile Name: 规章制度.pdf\nContent: 上海出差住宿费上限500元/天"],
        #   }]
        #
        # 🔢 变形后 —— 冒泡到主图、被 accumulate_or_reset 合并完（2 个并行子图都跑完）：
        #   State["agent_answers"] = [
        #       {"index": 1, "question": "上海出差交通费怎么报销",   "answer": "高铁二等座据实报销。", "contexts": [...]},
        #       {"index": 0, "question": "上海出差住宿费报销上限是多少", "answer": "每天上限 500 元。", "contexts": [...]},
        #   ]
        #   ⚠️ 注意顺序是【乱的】——哪个子图先跑完哪个先进列表，跟问题顺序无关。
        #      所以 aggregate_answers 必须靠 index 重新排序（这就是 index 存在的全部理由）。
        "agent_answers": [{
            # 【读】state["question_index"] ← 由 edges.route_after_rewrite 的 Send 派发时传入
            # 唯一用途：给主图 aggregate_answers 排序用（并行完成顺序是乱的，见上面 🔢）
            "index": state["question_index"],

            # 【读】state["question"] ← 同样由 Send 传入
            # 🔢 例："上海出差住宿费报销上限是多少"（rewrite_query 拆出来的子问题之一）
            "question": state["question"],

            "answer": answer,

            # 【读】state["retrieved_contexts"] ← 由 should_compress_context 写入
            #      （带 append_unique reducer：每轮追加 + 去重 + 保序）
            #
            # 📎 跨文件隐形合同：这里装的是 tools.py 拼出来的 "Parent ID / File Name / Content"
            #    固定格式原文 —— 由 nodes._retrieval_contexts 按那个格式【反向解析】ToolMessage 得到，
            #    并且已经过滤掉了 NO_RELEVANT_CHUNKS / RETRIEVAL_ERROR: 这些"暗号"。
            #    改了 tools.py 的格式或暗号词，这里拿到的东西就会错乱。
            #
            # ⭐ 它【不发给模型】，只是随答案一起冒泡到主图，供 RAGAS 之类的评测工具算召回率。
            # 🔢 空态：整轮一次都没搜到有效原文（或工具全返回暗号被过滤掉）→ []
            #    用 .get(..., []) 而不是 [...]：字段可能压根没被写过（一次工具都没调），直接取会 KeyError。
            "contexts": state.get("retrieved_contexts", []),

            # Evaluation trace metadata. These fields only describe how the
            # sub-agent reached its answer; they are not fed back to the LLM.
            "iteration_count": state.get("iteration_count", 0),
            "tool_call_count": state.get("tool_call_count", 0),
            "retrieval_keys": sorted(state.get("retrieval_keys", set())),
            "used_fallback": state.get("iteration_count", 0) >= MAX_ITERATIONS
                or state.get("tool_call_count", 0) >= MAX_TOOL_CALLS,
        }]
    }
    # 之后子图走向 END（见 graph.py：collect_answer → END），
    # 整个 AgentState 的 agent_answers 冒泡回主图，触发 accumulate_or_reset 追加。

# ============================================================
# 🅱️ aggregate_answers：主图终点站，把 N 份并行答案缝成一段给人看的回复
# ============================================================
def aggregate_answers(state: State, llm):
    """🅱️ 汇总节点：把所有并行 Agent 的答案整合成一个最终回复。

    输入：State（读 agent_answers、originalQuery、messages）
    输出：dict（写 messages，包含最终 AI 回复 + 旧消息删除列表）
    """
    # ── 第一段：算出"这轮结束要删掉哪些消息" ────────────────────────────
    # 为什么要删？主图 messages 里此刻是一锅粥：真实对话 + N 个并行子图流回来的内部消息。
    # 不清理，下一轮 messages 会越滚越长，token 爆炸，而且模型会看到一堆"我要调工具"的内部噪音。

    # 【读】state["messages"] ← chat 入口写用户提问 / 各节点写入 / ⚠️ 并行子图的消息也会默认流回这里
    messages = state.get("messages", [])

    # ── _is_plain_conversation_message(msg)（辅助函数，nodes.py）────────
    # 判据只有一条：这条消息【有没有 name 标签】。没有 name = 真实对话；有 name = 内部消息。
    #
    # 📎 跨文件隐形合同（⭐ 全项目最容易看漏的一条）：
    #    子图里每一条消息都被 _name_internal_message 打过 name（如 name="agent_response"），
    #    澄清消息被 rewrite_query 打了 name="clarification"。
    #    所以"带 name" ≡ "内部消息" ≡ "该过滤掉"。
    #    ⭐ 这正是"并行子图的 messages 会流回主图、却不污染真实对话历史"的全部机制。
    #    改了 _name_internal_message 里的标签名，就必须同步改 _is_plain_conversation_message。
    #
    # 🔢 变形前后（用户问"上海出差住宿和交通能报多少"，跑完 2 个并行子图）：
    #   messages（原始，一锅粥）：
    #     [HumanMessage("上海出差住宿和交通能报多少"),                                    ← 无 name ✅
    #      AIMessage(content="", tool_calls=[{...}], name="agent_response"),           ← 子图流回，有 name ❌
    #      ToolMessage(content="Parent ID: 公司规章制度_p0\n…", name="search_child_chunks"), ← 子图流回 ❌
    #      AIMessage("每天上限 500 元。", name="agent_response"),                        ← 子图流回 ❌
    #      AIMessage("高铁二等座据实报销。", name="agent_response")]                      ← 子图流回 ❌
    #            │  列表推导式：只留【没有 name】的
    #            ▼
    #   plain_messages（干净的真实对话）：
    #     [HumanMessage("上海出差住宿和交通能报多少")]
    plain_messages = [msg for msg in messages if _is_plain_conversation_message(msg)]

    # ── 陌生语法讲解：{... for ... in ...} 集合推导式 ────────────────────
    # 和列表推导式 [x for x in ...] 长得一模一样，只是外面换成花括号 {} → 造出来的是 set。
    # 为什么用 set 不用 list？下面 _remove_messages_not_in 要对每条消息反复判断
    # "它的 id 在不在 keep_ids 里" —— set 的 in 是 O(1)（哈希直接定位），
    # list 的 in 要从头扫到尾（O(n)）。消息一多，差别很明显。
    #
    # ── 陌生语法讲解：负数切片 plain_messages[-N:] ─────────────────────
    # "取列表最后 N 个元素"。N = config.PRE_ANSWER_HISTORY_MESSAGES_TO_KEEP（保留几条历史）
    #   例：N=4，列表有 10 条 → 拿到第 7、8、9、10 条
    #   ⭐ 不够 N 条也【不会报错】：只有 2 条时 [-4:] 就把这 2 条全给你（切片自动截断，不越界）
    #
    # getattr(msg, "id", None) → 安全取 id；万一某条消息没有 id 属性，返回 None 而不是崩掉
    #
    # 🔢 keep_ids 真实长这样（LangChain 自动生成的 UUID 字符串）：
    #   {"a1b2c3d4-5e6f-7890-abcd-ef1234567890", "b2c3d4e5-6f70-8901-bcde-f12345678901"}
    keep_ids = {getattr(msg, "id", None) for msg in plain_messages[-PRE_ANSWER_HISTORY_MESSAGES_TO_KEEP:]}

    # ── 内置方法讲解：set.discard(x) ────────────────────────────────────
    # 从集合里删掉 x。⭐ 和 set.remove(x) 的关键区别：
    #   remove(x)  → x 不存在时【抛 KeyError 报错】
    #   discard(x) → x 不存在时【静默忽略】，不报错
    # 这里要删的是 None —— 上面 getattr 的兜底可能往集合里塞进一个 None（某条消息没有 id）。
    # 用 discard 就不必先写 if None in keep_ids 判断一遍，直接删，没有就算了。
    keep_ids.discard(None)

    # ── _remove_messages_not_in(messages, keep_ids)（辅助函数，nodes.py）──
    # 遍历 messages，凡是 id 不在 keep_ids 里的，都造一条 RemoveMessage(id=...) 删除指令。
    # 效果 = 删掉【全部子图流回来的内部消息】+【超出保留窗口的老对话】，只留最近 N 条真实对话。
    #
    # 🔢 removals 真实长这样（注意：它装的不是"消息"，是"删除指令"）：
    #   [RemoveMessage(id="c3d4e5f6-…"), RemoveMessage(id="d4e5f6a7-…"), RemoveMessage(id="e5f6a7b8-…")]
    # 🔢 空态：如果 messages 里每一条都该保留 → []（下面 removals + [新消息] 就等于"只追加，不删"）
    removals = _remove_messages_not_in(messages, keep_ids)

    # ── 第二段：兜底 —— 一个答案都没收到 ────────────────────────────────
    # 【读】state["agent_answers"] ← 由各并行子图 collect_answer 冒泡追加（accumulate_or_reset）
    # 🔢 空态：所有子图都异常、或 rewrittenQuestions 本来就是空的 → []（空列表在 Python 里是"假"）
    # 这时候【不调模型】（没材料可整合，调了也是浪费一次 API），
    # 直接返回一句写死的话，同时把该删的消息删掉（removals 照样要执行，否则历史还是脏的）。
    if not state.get("agent_answers"):
        return {"messages": removals + [AIMessage(content="没有生成任何答案。")]}

    # ── 第三段：按 index 排序 ───────────────────────────────────────────
    # ⭐ 为什么必须排序？并行子图【谁先跑完谁先冒泡】，agent_answers 里的顺序是乱的。
    #    但用户看到的答案必须按"子问题1 → 子问题2"的顺序来读才通顺。
    #    index 这个字段存在的【唯一理由】就是这一行。
    #
    # ── 陌生语法讲解：sorted(列表, key=lambda x: x["index"]) ────────────
    # sorted(可迭代对象, key=函数) → 返回一个【排好序的新列表】（原列表不动）
    #   key=… 回答"按什么排"：框架拿列表里的每个元素去调这个函数，用它的返回值来比大小。
    #   lambda x: x["index"] → "匿名函数"，就地定义一个没名字的小函数，等价于：
    #       def 取index(x):
    #           return x["index"]
    #   合起来 = "把每个 dict 的 index 值抠出来，按它从小到大排"。
    #
    # 🔢 变形前后：
    #   排序前（并行完成顺序 —— 1 号子问题先跑完）：
    #     [{"index": 1, "question": "上海出差交通费怎么报销",     "answer": "高铁二等座据实报销。", ...},
    #      {"index": 0, "question": "上海出差住宿费报销上限是多少", "answer": "每天上限 500 元。",  ...}]
    #            │  sorted(key=lambda x: x["index"])
    #            ▼
    #   排序后（回到用户原本的提问顺序）：
    #     [{"index": 0, "answer": "每天上限 500 元。", ...},
    #      {"index": 1, "answer": "高铁二等座据实报销。", ...}]
    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

    # ── 第四段：把各路子答案拼成给模型看的一段文本 ──────────────────────
    # enumerate(列表, start=1) → 同时拿到编号 i 和元素 ans；start=1 表示编号从 1 开始（显示更自然）
    # ⚠️ 注意这里【只取 answer】，不取 question 和 contexts：
    #    · question 不用给 —— 模型只需要"几段现成答案"，原问题在下面单独给一份完整的
    #    · contexts 不能给 —— 那是给评测工具用的原文，塞进提示词会白白烧掉几千 token
    #
    # 🔢 循环跑完，formatted_answers 真实长这样：
    #   （开头有个 \n，所以第一行是空的）
    #   Retrieved response 1:
    #   每天上限 500 元。
    #
    #   Retrieved response 2:
    #   高铁二等座据实报销。
    #
    # 🔢 空态：走不到这里（上面的 if not agent_answers 已经拦掉了），所以循环至少跑一次
    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += (f"\nRetrieved response {i}:\n" f"{ans['answer']}\n")
        # ⚠️ 两个相邻的 f-string 之间没有 + 号 —— Python 会【自动把相邻的字符串字面量拼起来】
        #    f"a" f"b" 等价于 f"ab"。这里纯粹是为了折行好看，不是两个值。

    # ── 第五段：把"原始问题 + 各路答案"发给模型，让它缝成一段回复 ────────
    # 【读】state["originalQuery"] ← 由 rewrite_query 写入
    #      （若走过澄清，它是"原问题 + 历次澄清"合并后的完整问题）
    #
    # ⭐ 为什么发 originalQuery，而不是发 rewrittenQuestions（拆分后的子问题）？
    #    因为要让模型知道【用户最初真正想问的是什么】，才能把几段零散答案组织成一段自然的回复。
    #    如果只发拆分后的子问题，模型会把回复也写成支离破碎的"关于问题1…关于问题2…"。
    #
    # ── 陌生语法讲解：f"""……""" 三引号 f-string ────────────────────────
    # 三个引号 = 多行字符串（可以直接按回车换行）；前面加 f = 里面的 {…} 会被替换成变量的值。
    # ⚠️ 这里虽然用了三引号（本来就能换行），却又手写了 \n —— 因为这句代码整个写在一行里，
    #    所以只能靠 \n 产生换行。两种换行方式混用，能跑，但可读性差，属于原代码的小瑕疵。
    #
    # 🔢 user_message.content 真实长这样（这就是发给模型的原材料）：
    #   Original user question: 上海出差住宿和交通能报多少
    #   Retrieved answers:
    #   Retrieved response 1:
    #   每天上限 500 元。
    #
    #   Retrieved response 2:
    #   高铁二等座据实报销。
    user_message = HumanMessage(content=f"""Original user question: {state["originalQuery"]}\nRetrieved answers:{formatted_answers}""")

    # llm.invoke([系统提示, 用户消息]) → 调模型
    # 用的是【不带工具】的原始 llm（不是 llm_with_tools）—— 这里只做文字整合，不需要再检索
    # get_aggregation_prompt()（prompts.py）：告诉模型"把下面几段答案缝成一段连贯回复，别重复、别编造"
    #
    # 🔢 synthesis_response 真实长这样：
    #   AIMessage(content="上海出差方面：住宿费每天上限 500 元；交通费按高铁二等座据实报销。")
    synthesis_response = llm.invoke([SystemMessage(content=get_aggregation_prompt()), user_message])

    # ── 返回：删旧消息 + 追加最终回复，一次搞定 ──────────────────────────
    # 【写】State["messages"]（reducer 是继承来的 add_messages）
    #
    # ⭐ removals + [AIMessage(...)] → 一个列表里同时装着"删除指令"和"新消息"。
    #    add_messages 这个 reducer 会自动分辨：
    #      看到 RemoveMessage → 按 id 从历史里删掉那条
    #      看到普通消息       → 追加到列表末尾
    #    所以一次 return 就把"清理 + 追加"两件事都干了。
    #
    # ⚠️ 这条最终回复【故意不打 name 标签】—— 因为它就是要给用户看的真实对话。
    #    （对比：子图里所有消息都被 _name_internal_message 打了 name，会被 _is_plain_conversation_message
    #     过滤掉。这条没 name，所以下一轮 summarize_history / aggregate_answers 会把它当真实历史保留。）
    #    ⭐ "打不打 name"这一个细节，就决定了一条消息是"用户能看见的对话"还是"内部噪音"。
    #
    # 之后主图走向 END（见 graph.py：aggregate_answers → END），本轮对话结束。
    return {"messages": removals + [AIMessage(content=synthesis_response.content)]}
