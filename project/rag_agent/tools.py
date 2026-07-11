# ============================================================
# 检索工具定义
# 这里定义 Agent 可以调用的两个工具，并用 @tool 装饰器包装。
# 注意：每个工具函数的文档字符串(docstring)会作为"工具说明"发送给大模型，
#      模型正是根据这些说明来决定何时、如何调用工具，所以这里用中文写清楚。
# ============================================================
from langchain_core.tools import tool
import config
from db.parent_store_manager import ParentStoreManager
from core.execution_logger import log_error, log_tool_end, log_tool_start


class ToolFactory:
    """工具工厂：把绑定了具体向量库集合的检索函数，打包成 LangChain 工具列表。"""

    def __init__(self, collection):
        self.collection = collection                      # 向量库集合（用于搜子块）
        self.parent_store_manager = ParentStoreManager()  # 父块存储（用于取大块）

    def _search_child_chunks(self, query: str, limit: int = config.DEFAULT_RETRIEVAL_K) -> str:
        """在文档中搜索与用户问题相关的片段（子块）。

        这是检索的第一步。返回结果里包含父块ID、文件名和一小段子块摘录。
        如果这些摘录相关但太零碎、不足以自信作答，再用返回的 parent_id 去调用
        retrieve_parent_chunks 取更完整的上下文。

        参数:
            query: 聚焦的搜索查询，用问题里的具体关键词。
            limit: 最多返回多少个子块。
        """
        log_tool_start("search_child_chunks", {"query": query, "limit": limit})
        try:
            results = self.collection.similarity_search(
                query,
                k=limit,
                score_threshold=config.RETRIEVAL_SCORE_THRESHOLD,
            )
            # if：一个都没搜到 → 返回"没相关内容"的暗号（供 nodes 里过滤掉）
            if not results:
                output = "NO_RELEVANT_CHUNKS"  # 没搜到相关内容的标记
                log_tool_end("search_child_chunks", output)
                return output

            # 把每个命中的子块拼成"父块ID + 文件名 + 内容"的文本返回给模型
            output = config.CHILD_CHUNK_SEPARATOR.join([
                f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
                f"File Name: {doc.metadata.get('source', '')}\n"
                f"Content: {doc.page_content.strip()}"
                for doc in results
            ])
            log_tool_end("search_child_chunks", output)
            return output

        except Exception as e:
            log_error("search_child_chunks", e)
            output = f"RETRIEVAL_ERROR: {str(e)}"
            log_tool_end("search_child_chunks", output)
            return output

    def _retrieve_parent_chunks(self, parent_id: str) -> str:
        """根据一个相关子块的 parent_id，取出它所属的完整父块（大块）。

        仅当 search_child_chunks 返回了相关的 parent_id、且子块摘录需要更多上下文时才调用。
        压缩上下文里已经出现过的父块ID，不要再调用本工具去取。

        参数:
            parent_id: search_child_chunks 返回的父块ID。
        """
        log_tool_start("retrieve_parent_chunks", {"parent_id": parent_id})
        try:
            parent = self.parent_store_manager.load_content(parent_id)
            # if：这个 parent_id 找不到对应父块 → 返回"没这父块"的暗号
            if not parent:
                output = "NO_PARENT_DOCUMENT"  # 找不到对应父块的标记
                log_tool_end("retrieve_parent_chunks", output)
                return output

            output = (
                f"Parent ID: {parent.get('parent_id', 'n/a')}\n"
                f"File Name: {parent.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {parent.get('content', '').strip()}"
            )
            log_tool_end("retrieve_parent_chunks", output)
            return output

        except Exception as e:
            log_error("retrieve_parent_chunks", e)
            output = f"PARENT_RETRIEVAL_ERROR: {str(e)}"
            log_tool_end("retrieve_parent_chunks", output)
            return output

    def create_tools(self) -> list:
        """把上面两个函数包装成 LangChain 工具并返回列表。"""
        search_tool = tool("search_child_chunks")(self._search_child_chunks)
        retrieve_tool = tool("retrieve_parent_chunks")(self._retrieve_parent_chunks)

        return [search_tool, retrieve_tool]
