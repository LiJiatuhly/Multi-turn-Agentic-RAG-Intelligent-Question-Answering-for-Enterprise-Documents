# RAG 系统总装：创建向量库、聊天模型、工具，编译成 Agent 图。
# 这个文件是"装配厂"，把所有零件拼在一起，其他文件都是零件。

import uuid
from langchain_openai import ChatOpenAI   # 智谱走 OpenAI 兼容接口，直接用这个
import config
from db.vector_db_manager import VectorDbManager
from db.parent_store_manager import ParentStoreManager
from document_chunker import DocumentChunker
from rag_agent.tools import ToolFactory
from rag_agent.graph import create_agent_graph
from core.observability import Observability


class RAGSystem:
    """B 级 RAG 系统的总装类，持有所有子系统的引用。

    __init__  : 创建各子系统对象（还没初始化，省内存）
    initialize: 真正建向量库、建聊天模型、编译 Agent 图（耗时操作放这里）
    get_config: 返回 LangGraph 运行时需要的配置（thread_id + recursion_limit）
    reset_thread: 清除当前会话，开始新对话
    """

    def __init__(self, collection_name=config.CHILD_COLLECTION):
        self.collection_name = collection_name
        self.vector_db    = VectorDbManager()       # 向量库（含智谱嵌入）
        self.parent_store = ParentStoreManager()    # 父块本地存储
        self.chunker      = DocumentChunker()       # 文档切块器
        self.observability = Observability()        # 可观测性（默认关闭）
        self.agent_graph  = None                    # Agent 图，initialize 后才有值
        self.thread_id    = str(uuid.uuid4())       # 会话 ID（每次对话唯一）
        self.recursion_limit = config.GRAPH_RECURSION_LIMIT

    def initialize(self):
        """B 级 真正初始化系统：建向量集合 + 创建聊天模型 + 编译 Agent 图。

        分开 __init__ 和 initialize 的原因：
        创建向量集合会发 embedding 请求、编译图有一定开销，
        放在 __init__ 里会让"创建对象"变慢，调试时不方便。
        """
        # 建向量集合（已存在则检查维度匹配，见 vector_db_manager.py）
        self.vector_db.create_collection(self.collection_name)
        collection = self.vector_db.get_collection(self.collection_name)

        # if：.env 里没填 API Key → 报错提示（否则调模型时才失败，不好排查）
        if not config.LLM_API_KEY:
            raise ValueError(
                "未检测到大模型 API Key。请在 project/.env 文件中填写 API_KEY=你的智谱密钥。"
            )

        # 智谱 GLM 提供 OpenAI 兼容接口，所以用 ChatOpenAI，只需要指定 base_url
        llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
        )
        # 创建工具（绑定具体的向量库集合），编译 Agent 图
        tools = ToolFactory(collection).create_tools()
        self.agent_graph = create_agent_graph(llm, tools)

    def get_config(self):
        """C 级 生成 LangGraph 运行时配置，每次 invoke/stream 都需要传入。

        thread_id      : 区分不同对话（相同 id = 同一对话，状态会延续）
        recursion_limit: 防止图无限循环的安全阀
        callbacks      : Langfuse 追踪（关闭时为 None）
        """
        cfg = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": self.recursion_limit
        }
        handler = self.observability.get_handler()
        # if：开了 Langfuse（handler 非 None）才把它挂进回调
        if handler:
            cfg["callbacks"] = [handler]
        return cfg

    def reset_thread(self):
        """C 级 开始新对话：删除旧会话的 checkpoint，生成新的 thread_id。"""
        try:
            self.agent_graph.checkpointer.delete_thread(self.thread_id)
        except Exception as e:
            print(f"警告：无法删除会话线程 {self.thread_id}: {e}")
        self.thread_id = str(uuid.uuid4())   # 新 id = 新对话
