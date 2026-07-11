# 聊天流处理：把 LangGraph 的流式输出转成 Gradio 界面能显示的消息。
# C 级：知道 chat() 是生成器、yield 出消息列表就够，内部 handle_* 不必逐行。
import json
import re
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
from core.execution_logger import log_chat_end, log_chat_start, log_error

SYSTEM_NODES = {"summarize_history", "rewrite_query"}   # 这两个节点的输出显示为可折叠的"系统步骤"
FINAL_RESPONSE_NODES = {"aggregate_answers"}            # 这个节点的输出才是给用户看的正式答案

SYSTEM_NODE_CONFIG = {
    "rewrite_query":     {"title": "🔍 查询分析与改写"},
    "summarize_history": {"title": "📋 对话历史摘要"},
}

# --- 辅助函数 ---

def make_message(content, *, title=None, node=None):
    """造一条 Gradio 聊天消息字典。带 title 的会显示成可折叠卡片。"""
    msg = {"role": "assistant", "content": content}
    # if：传了 title 或 node → 加 metadata（Gradio 靠 title 折叠，靠 node 定位这条消息）
    if title or node:
        # 字典推导 + if v：只保留非空的字段，避免写入 None
        msg["metadata"] = {k: v for k, v in {"title": title, "node": node}.items() if v}
    return msg


def find_msg_idx(messages, node):
    """在消息列表里找到 metadata.node == node 的那条，返回它的下标；找不到返回 None。"""
    # next(生成器, None)：取第一个匹配项，没有就返回 None（避免 StopIteration 报错）
    return next(
        (i for i, m in enumerate(messages) if m.get("metadata", {}).get("node") == node),
        None,
    )


def parse_rewrite_json(buffer):
    """从流式累积的文本里，尽力抠出一个 JSON 对象并解析；抠不出或解析失败返回 None。"""
    match = re.search(r"\{.*\}", buffer, re.DOTALL)   # 匹配第一个 {...}（DOTALL 让 . 跨行）
    # if：没匹配到花括号 → 还没输出完整 JSON，返回 None
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:      # 用 Exception 而非裸 except，保证 Ctrl+C 仍能中断
        return None        # JSON 还没输出完整、暂时解析失败 → 返回 None，下一帧再试


def format_rewrite_content(buffer):
    """把 rewrite_query 输出的 JSON，格式化成给用户看的中文提示。"""
    data = parse_rewrite_json(buffer)
    # if：还没解析出 JSON → 显示"正在分析"占位
    if not data:
        return "⏳ 正在分析问题..."
    # if：问题清晰 → 显示"清晰" + 改写后的问题列表
    if data.get("is_clear"):
        lines = ["✅ **问题清晰**"]
        # if：有改写后的问题 → 逐条列出（列表推导给每个问题加"- "前缀）
        if data.get("questions"):
            lines += ["\n**改写后的检索问题：**"] + [f"- {q}" for q in data["questions"]]
    # else：问题不清晰 → 显示"不够清晰" + 需要用户补充什么
    else:
        lines = ["❓ **问题不够清晰**"]
        clarification = data.get("clarification_needed", "")
        # if：澄清内容非空且不是字面量"no" → 展示出来
        if clarification and clarification.strip().lower() != "no":
            lines.append(f"\n需要你补充：*{clarification}*")
    return "\n".join(lines)

# --- 辅助函数结束 ---

class ChatInterface:
    """把图的流式输出，逐帧翻译成 Gradio 聊天界面的消息列表。"""

    def __init__(self, rag_system):
        self.rag_system = rag_system

    def _handle_system_node(self, chunk, node, response_messages, system_node_buffer):
        """更新（或新建）可折叠的系统节点消息，需要时展示澄清提示。"""
        # 把这一帧的文本累加进对应节点的缓冲区（流式是一小段一小段来的）
        system_node_buffer[node] = system_node_buffer.get(node, "") + chunk.content
        buffer = system_node_buffer[node]
        title  = SYSTEM_NODE_CONFIG[node]["title"]
        # rewrite_query 的原始输出是 JSON，要格式化成人话；其他节点直接用原文
        content = format_rewrite_content(buffer) if node == "rewrite_query" else buffer

        idx = find_msg_idx(response_messages, node)
        # if：还没有这条消息 → 新建一条；else：已存在 → 更新它的内容
        if idx is None:
            response_messages.append(make_message(content, title=title, node=node))
        else:
            response_messages[idx]["content"] = content

        # if：是改写节点 → 额外检查要不要冒出一条"请补充"的澄清气泡
        if node == "rewrite_query":
            self._surface_clarification(buffer, response_messages)

    def _surface_clarification(self, buffer, response_messages):
        """如果问题不清晰，添加/更新一条澄清提示消息。"""
        data          = parse_rewrite_json(buffer) or {}   # 解析失败就用空字典兜底
        clarification = data.get("clarification_needed", "")
        # if：问题不清晰 且 澄清内容有效（非空、非"no"）→ 需要显示澄清气泡
        if not data.get("is_clear") and clarification.strip().lower() not in ("", "no"):
            cidx = find_msg_idx(response_messages, "clarification")
            # if：还没有澄清气泡 → 新建；else：已有 → 更新内容
            if cidx is None:
                response_messages.append(make_message(clarification, node="clarification"))
            else:
                response_messages[cidx]["content"] = clarification

    def _handle_tool_call(self, chunk, response_messages, active_tool_calls):
        """把新的工具调用登记为可折叠消息。"""
        for tc in chunk.tool_calls:
            # if：这个工具调用有 id 且没登记过 → 新建一条"正在运行 XX"的折叠卡片
            if tc.get("id") and tc["id"] not in active_tool_calls:
                response_messages.append(
                    make_message(f"正在运行 `{tc['name']}`...", title=f"🛠️ {tc['name']}")
                )
                # 记下"这个工具调用 id → 对应第几条消息"，等结果回来时好填进去
                active_tool_calls[tc["id"]] = len(response_messages) - 1

    def _handle_tool_result(self, chunk, response_messages, active_tool_calls):
        """把工具返回结果填进对应的可折叠消息里。"""
        idx = active_tool_calls.get(chunk.tool_call_id)   # 靠 id 找到之前那条"正在运行"的消息
        # if：找到了对应消息 → 把结果（截断到300字）填进去
        if idx is not None:
            preview = str(chunk.content)[:300]
            suffix  = "\n..." if len(str(chunk.content)) > 300 else "" # 超长就加省略号
            response_messages[idx]["content"] = f"```\n{preview}{suffix}\n```"

    def _handle_llm_token(self, chunk, node, response_messages):
        """把流式返回的模型 token 追加到最后一条助手消息上。"""
        last = response_messages[-1] if response_messages else None   # 取最后一条消息（可能没有）
        # if：最后一条不是"正在生成的正式答案气泡"（不存在/不是assistant/是折叠卡片）→ 新开一条空气泡
        if not (last and last.get("role") == "assistant" and "metadata" not in last):
            response_messages.append(make_message(""))
        response_messages[-1]["content"] += chunk.content   # 把这一帧 token 追加上去

    def chat(self, message, history):
        """生成器：以流式方式产出 Gradio 聊天消息。"""
        # if：系统还没初始化好 → 直接回一句提示并结束
        if not self.rag_system.agent_graph:
            yield "⚠️ 系统尚未初始化！"
            return

        config        = self.rag_system.get_config()
        current_state = self.rag_system.agent_graph.get_state(config)   # 取当前会话状态
        log_chat_start(message.strip(), self.rag_system.thread_id, bool(current_state.next))

        try:
            # if：current_state.next 非空 → 说明图正停在澄清中断点等用户回话
            if current_state.next:
                # 把用户这句补充塞进状态，然后用 None 恢复运行（不是新开一轮）
                self.rag_system.agent_graph.update_state(config, {"messages": [HumanMessage(content=message.strip())]})
                stream_input = None
            # else：正常新一轮提问 → 把问题作为新输入
            else:
                stream_input = {"messages": [HumanMessage(content=message.strip())]}

            response_messages  = []    # 要显示的消息列表
            active_tool_calls  = {}    # 工具调用 id → 消息下标
            system_node_buffer = {}    # 系统节点名 → 累积的流式文本

            # 逐帧接收图的流式输出（chunk=内容片段，metadata=它来自哪个节点）
            for chunk, metadata in self.rag_system.agent_graph.stream(stream_input, config=config, stream_mode="messages"):
                node = metadata.get("langgraph_node", "")

                # if：是系统节点(摘要/改写)的文本片段 → 更新折叠卡片
                if node in SYSTEM_NODES and isinstance(chunk, AIMessageChunk) and chunk.content:
                    self._handle_system_node(chunk, node, response_messages, system_node_buffer)
                # elif：是"模型要调工具"的片段 → 登记工具调用卡片
                elif hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    self._handle_tool_call(chunk, response_messages, active_tool_calls)
                # elif：是工具返回结果 → 填进对应卡片
                elif isinstance(chunk, ToolMessage):
                    self._handle_tool_result(chunk, response_messages, active_tool_calls)
                # elif：是汇总节点吐的正式答案 token → 追加到答案气泡
                elif isinstance(chunk, AIMessageChunk) and chunk.content and node in FINAL_RESPONSE_NODES:
                    self._handle_llm_token(chunk, node, response_messages)
                # else：其他片段（比如中间节点的内部输出）→ 不显示，跳过
                else:
                    continue

                yield response_messages   # 每处理一帧就吐出最新消息列表，界面实时刷新

            final_state = self.rag_system.agent_graph.get_state(config)
            log_chat_end(getattr(final_state, "values", final_state))

        except Exception as e:
            log_error("chat", e)
            yield f"❌ 出错了: {str(e)}"    # 出错也要给用户一个可见的提示

    def clear_session(self):
        """清空当前会话：重置 thread_id 开启新对话，并 flush 可观测性数据。"""
        self.rag_system.reset_thread()
        self.rag_system.observability.flush()
