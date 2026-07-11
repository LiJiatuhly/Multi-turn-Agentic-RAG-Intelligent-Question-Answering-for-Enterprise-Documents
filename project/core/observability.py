# 可观测性（可选）：接入 Langfuse 追踪 Agent 的每一步，默认关闭。
# 开启方式：在 .env 里设置 LANGFUSE_ENABLED=true 并填入对应的 Key。

import logging
import config

logger = logging.getLogger(__name__)


class Observability:
    """C 级 Langfuse 可观测性的开关和初始化。

    默认关闭（LANGFUSE_ENABLED=false），不影响任何功能。
    开启后，Agent 的每次节点调用、工具调用都会被记录到 Langfuse 后台，
    可以可视化地看到 Agent 的推理链路，便于调试。

    get_handler() : 返回 callback handler（关闭时返回 None，调用方忽略即可）
    flush()       : 把缓冲的 trace 数据发送到 Langfuse（关闭会话时调用）
    """

    def __init__(self):
        self._enabled = config.LANGFUSE_ENABLED
        self._handler = None
        self._client  = None

        if not self._enabled:
            return   # 未启用，直接结束

        # if：开了 Langfuse 但没填 Key → 打个警告并关掉，不影响主流程
        if not config.LANGFUSE_PUBLIC_KEY or not config.LANGFUSE_SECRET_KEY:
            logger.warning("Langfuse enabled but API keys are missing — skipping")
            self._enabled = False
            return

        try:
            from langfuse import get_client
            from langfuse.langchain import CallbackHandler
            self._client = get_client()
            # if：认证通过 → 提示可用；else：认证失败 → 关掉
            if self._client.auth_check():
                print("Langfuse 客户端已认证，可以使用！")
            else:
                print("Langfuse 认证失败，请检查密钥和地址。")
                self._enabled = False
                return
            self._handler = CallbackHandler()
        except Exception as exc:
            logger.warning("Could not initialize Langfuse: %s", exc)
            self._enabled = False

    def get_handler(self):
        return self._handler

    def flush(self):
        # if：有客户端才 flush（未启用时 _client 是 None，跳过）
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:
                pass
