# 执行日志：在终端彩色打印每个节点/工具/路由的输入输出，便于调试(默认关闭)。
from __future__ import annotations

from datetime import datetime
from pprint import pformat
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage

import config


COLORS = {
    "blue": "\033[94m",
    "cyan": "\033[96m",
    "green": "\033[92m",
    "magenta": "\033[95m",
    "red": "\033[91m",
    "yellow": "\033[93m",
    "dim": "\033[2m",
    "reset": "\033[0m",
}


def _enabled() -> bool:
    return bool(getattr(config, "EXECUTION_LOGGING_ENABLED", True))


def _color(text: str, color: str) -> str:
    # if：配置里关掉了彩色 → 原样返回文本（不加颜色转义码）
    if not getattr(config, "EXECUTION_LOG_USE_COLOR", True):
        return text
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def _truncate(value: Any, max_chars: int | None = None) -> str:
    text = "" if value is None else str(value)
    limit = max_chars or getattr(config, "EXECUTION_LOG_MAX_CHARS", 1200)
    # if：文本没超长 → 原样返回；否则截断并标注省略了多少字
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated {len(text) - limit} chars]"


def _message_role(message: Any) -> str:
    # 一串 if 按消息类型分派，返回对应的角色标签（human/ai/tool/system/remove）
    if isinstance(message, HumanMessage):
        return "human"
    if isinstance(message, AIMessage):
        return "ai"
    if isinstance(message, ToolMessage):
        return "tool"
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, RemoveMessage):
        return "remove"
    return message.__class__.__name__


def _message_preview(message: Any) -> dict[str, Any]:
    preview = {
        "type": _message_role(message),
        "id": getattr(message, "id", None),
    }

    # if：删除消息没有正文/工具调用等内容 → 直接返回基本信息
    if isinstance(message, RemoveMessage):
        return preview

    content = getattr(message, "content", "")
    # if：有正文才收进预览
    if content:
        preview["content"] = _truncate(content)

    tool_calls = getattr(message, "tool_calls", None)
    # if：有工具调用才收进预览
    if tool_calls:
        preview["tool_calls"] = [
            {
                "name": call.get("name"),
                "args": call.get("args"),
                "id": call.get("id"),
            }
            for call in tool_calls
        ]

    tool_name = getattr(message, "name", None)
    # if：有 name（工具名/内部标记）才收
    if tool_name:
        preview["name"] = tool_name

    tool_call_id = getattr(message, "tool_call_id", None)
    # if：有 tool_call_id（工具结果对应哪次调用）才收
    if tool_call_id:
        preview["tool_call_id"] = tool_call_id

    return preview


def _messages_preview(messages: list[Any]) -> dict[str, Any]:
    return {
        "count": len(messages),
        "last_messages": [_message_preview(message) for message in messages[-4:]],
    }


def state_preview(state: Any) -> dict[str, Any]:
    # if：不是字典（异常情况）→ 直接截断成字符串返回
    if not isinstance(state, dict):
        return {"value": _truncate(state)}

    preview: dict[str, Any] = {}

    # 按字段名分派怎么摘要：消息列表只留最近几条、长文本截断、集合排序、其余原样
    for key, value in state.items():
        if key == "messages":                    # 消息字段 → 只预览最近几条
            preview[key] = _messages_preview(value or [])
        elif key in {"conversation_summary", "context_summary", "final_answer"}:   # 长文本 → 截断
            preview[key] = _truncate(value)
            preview[key] = _truncate(value)
        elif key == "agent_answers":
            preview[key] = {
                "count": len(value or []),
                "items": [
                    {
                        "index": item.get("index"),
                        "question": _truncate(item.get("question", ""), 240),
                        "answer": _truncate(item.get("answer", ""), 500),
                    }
                    for item in (value or [])[:3]
                    if isinstance(item, dict)   # 只保留字典型的答案项，最多3条
                ],
            }
        elif isinstance(value, set):              # 集合 → 排序成列表好看
            preview[key] = sorted(value)
        else:                                    # 其余字段 → 原样
            preview[key] = value

    return preview


def update_preview(update: Any) -> Any:
    # if：不是字典 → 截断返回（同 state_preview 的套路）
    if not isinstance(update, dict):
        return _truncate(update)

    preview: dict[str, Any] = {}
    # 同样按字段名分派（这里是"节点返回的更新"，字段规则和 state_preview 一致）
    for key, value in update.items():
        if key == "messages":
            preview[key] = [_message_preview(message) for message in value]
        elif key in {"conversation_summary", "context_summary", "final_answer"}:
            preview[key] = _truncate(value)
        elif key == "agent_answers":
            preview[key] = state_preview({"agent_answers": value})["agent_answers"]
        elif isinstance(value, set):
            preview[key] = sorted(value)
        else:
            preview[key] = value
    return preview


def _print_block(title: str, payload: Any, color: str) -> None:
    # if：日志开关关着（EXECUTION_LOGGING_ENABLED=False）→ 什么都不打印，直接返回
    if not _enabled():
        return

    timestamp = datetime.now().strftime("%H:%M:%S")
    print(_color(f"\n[{timestamp}] {title}", color))
    print(_color("-" * 80, "dim"))
    print(pformat(payload, width=120, sort_dicts=False))


def log_chat_start(message: str, thread_id: str, has_pending_interrupt: bool) -> None:
    _print_block(
        "USER QUERY",
        {
            "thread_id": thread_id,
            "pending_interrupt": has_pending_interrupt,
            "message": _truncate(message),
        },
        "blue",
    )


def log_chat_end(state: Any) -> None:
    _print_block("FINAL GRAPH STATE", state_preview(state), "blue")


def log_node_start(name: str, state: Any) -> None:
    _print_block(f"NODE START: {name}", state_preview(state), "cyan")


def log_node_end(name: str, update: Any) -> None:
    _print_block(f"NODE OUTPUT: {name}", update_preview(update), "green")


def log_route(name: str, decision: Any, state: Any | None = None) -> None:
    payload = {"decision": _truncate(decision)}
    # if：传了 state 就一并打印（帮助看清路由当时依据的状态）
    if state is not None:
        payload["state"] = state_preview(state)
    _print_block(f"ROUTE: {name}", payload, "yellow")


def log_tool_start(name: str, args: dict[str, Any]) -> None:
    _print_block(f"TOOL START: {name}", args, "magenta")


def log_tool_end(name: str, output: Any) -> None:
    _print_block(f"TOOL OUTPUT: {name}", {"output": _truncate(output)}, "magenta")


def log_error(scope: str, error: Exception) -> None:
    _print_block(f"ERROR: {scope}", {"type": error.__class__.__name__, "message": str(error)}, "red")


def logged_node(name: str, fn):
    # 包一层：调用前打印输入，调用后打印输出，出错也打印后再抛出
    def _wrapped(state, *args, **kwargs):
        log_node_start(name, state)      # 调用前：打印这个节点收到的 state
        try:
            result = fn(state, *args, **kwargs)   # 真正执行原节点函数
        except Exception as exc:
            log_error(name, exc)         # 出错也记一笔再抛出（不吞异常）
            raise
        log_node_end(name, result)
        return result

    return _wrapped
