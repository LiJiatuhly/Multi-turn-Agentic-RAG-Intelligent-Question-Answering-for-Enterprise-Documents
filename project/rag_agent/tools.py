from langchain_core.tools import tool
import config
from db.parent_store_manager import ParentStoreManager
from core.execution_logger import log_error, log_tool_end, log_tool_start


class ToolFactory:
    """工具工厂：封装检索工具。"""

    def __init__(self, collection):
        self.collection = collection
        self.parent_store_manager = ParentStoreManager()

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
            if not results:
                output = "NO_RELEVANT_CHUNKS"
                log_tool_end("search_child_chunks", output)
                return output

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
            if not parent:
                output = "NO_PARENT_DOCUMENT"
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
        """把检索函数包装成 LangChain 工具列表。"""
        search_tool = tool("search_child_chunks")(self._search_child_chunks)
        retrieve_tool = tool("retrieve_parent_chunks")(self._retrieve_parent_chunks)
        return [search_tool, retrieve_tool]
