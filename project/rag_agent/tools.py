# ============================================================
# 检索工具定义
# ┌─ 这个文件在整个项目里扮演什么角色？ ──────────────────────────────
# │ 定义 Agent（orchestrator 节点）能调用的两个检索工具，并包装成 LangChain 工具。
# │ 工具 = 模型不能自己"变"出文档内容，只能通过调用这些函数去向量库/父块库里查。
# │
# │ 两个工具是"两步检索"：
# │   ① search_child_chunks(query)     先用问题去搜【小块(子块)】，命中里带 parent_id
# │   ② retrieve_parent_chunks(pid)    觉得小块太零碎，再用 parent_id 取【大块(父块)】
# │
# │ ⚠️ 注意：每个工具函数的【文档字符串(docstring)】会被 @tool 抽出来，作为"工具说明"
# │    发给大模型。模型正是靠读这些说明，来决定"什么时候调、传什么参数"。所以下面每个
# │    docstring 都用中文把"何时用、参数是什么、别重复取"写清楚——它们不是给人看的注释，
# │    是给模型看的"使用手册"。(对照 prompts.py 的 get_orchestrator_prompt 里也反复叮嘱
# │    "不要重复检索同一个父块 ID"，和这里 docstring 是相互呼应的。)
# └──────────────────────────────────────────────────────────────────
#
# ★★ 本文件最关键的一条跨文件数据流（务必理解）★★
#   这两个函数【返回的都是字符串】，且格式是固定的：
#       "Parent ID: xxx\nFile Name: yyy\nContent: zzz"
#   多个子块之间用 config.CHILD_CHUNK_SEPARATOR 拼接。
#   ── 这串字符串去哪了？ ──
#     graph.py 里的 ToolNode 执行完工具后，把这串字符串包成 ToolMessage 塞进
#     AgentState.messages。然后 nodes.py 的 _retrieval_contexts() 会【按上面这个格式
#     反过来解析】它（还会用 CHILD_CHUNK_SEPARATOR 把多个子块拆开），抽出干净的检索
#     原文，最终随答案冒泡到主图供 RAGAS 评测。
#   ⭐ 所以"这里拼字符串的格式"和"nodes.py 里拆字符串的逻辑"是一份【隐形合同】：
#     改了这边的分隔符或前缀，就必须同步改 nodes._retrieval_contexts，否则解析错乱。
#
# ★ 几个"暗号"返回值，以及它们在别处怎么被识别 ★
#   NO_RELEVANT_CHUNKS / NO_PARENT_DOCUMENT / RETRIEVAL_ERROR: / PARENT_RETRIEVAL_ERROR:
#   这些不是真正的检索结果，而是"没搜到 / 没这块 / 出错了"的标记字符串。
#   nodes.py 的 _retrieval_contexts 里有一个 ignored_prefixes 元组，正是用来
#   把这些暗号过滤掉、不当成有效检索原文。改了这里的暗号词，也要同步那边。
# ============================================================

# ── 内置对象讲解：from langchain_core.tools import tool（第一次见）─────
# tool 是 LangChain 的一个【装饰器工厂】。它的作用 = 把一个"普通 Python 函数"
# 变成"大模型能识别、能调用的工具对象"。变身时它会自动做两件事：
#   1. 读取函数签名(参数名、类型标注) → 生成工具的"入参 schema"；
#   2. 读取函数的 docstring → 作为工具的"说明文字"发给模型。
# 用法有两种，本文件用的是 tool("名字")(函数) 这种——给工具显式起个名字(见文件底部)。
from langchain_core.tools import tool
import config
# ── 陌生对象讲解：ParentStoreManager ─────────────────────────────────
# 这是项目自带的"父块存储管理器"(在 db/parent_store_manager.py，本包未附带)。
# 向量库里存的是切碎的【子块】(小片段，利于精确匹配)；完整的【父块】(大段原文)另存一处。
# ParentStoreManager 就负责"给我一个 parent_id，我把对应的整段父块内容捞出来"。
from db.parent_store_manager import ParentStoreManager
from .reranker import ChunkReranker
# 三个 log_* 是日志函数(在 core/execution_logger.py)：只负责打印调试信息，
# 不影响检索逻辑本身。看代码时可以直接忽略它们，把注意力放在 try 里的检索动作。
from core.execution_logger import log_error, log_tool_end, log_tool_start


class ToolFactory:
    """🅱️ 工具工厂：把绑定了具体向量库集合的检索函数，打包成 LangChain 工具列表。

    为什么要做成"工厂类"而不是直接写两个函数？
      因为检索要用到一个具体的向量库 collection（哪批文档、哪个集合），这个 collection
      是运行时才传进来的。用类把 collection 存成 self.collection，两个检索方法就都能用它。
      最后 create_tools() 把两个方法包装成工具列表，交给 graph.py 的 bind_tools/ToolNode。
    """

    def __init__(self, collection):
        # collection：向量库集合对象，运行时由外部传入。它提供 similarity_search 方法(下面用)。
        self.collection = collection                      # 向量库集合（用于搜子块）
        self.parent_store_manager = ParentStoreManager()  # 父块存储（用于取大块）
        self.reranker = ChunkReranker(config.RERANKER_MODEL)

    def _search_child_chunks(self, query: str, limit: int = config.DEFAULT_RETRIEVAL_K) -> str:
        """在文档中搜索与用户问题相关的片段（子块）。

        这是检索的第一步。返回结果里包含父块ID、文件名和一小段子块摘录。
        如果这些摘录相关但太零碎、不足以自信作答，再用返回的 parent_id 去调用
        retrieve_parent_chunks 取更完整的上下文。

        参数:
            query: 聚焦的搜索查询，用问题里的具体关键词。
            limit: 最多返回多少个子块。
        """
        # ↑↑ 上面这段 docstring 会被 @tool 抽走发给模型，是模型的"使用手册"，不是普通注释。
        #    limit 的默认值来自 config.DEFAULT_RETRIEVAL_K（默认取几个子块，集中配置便于调参）。
        log_tool_start("search_child_chunks", {"query": query, "limit": limit})  # 只打日志，可忽略
        try:
            # ── 内置方法讲解：collection.similarity_search(...)（第一次见）──────
            # 这是向量库的核心检索方法。作用 = "把 query 转成向量，在库里找最相近的若干个子块"。
            #   query           → 要搜的文本(模型填的关键词)
            #   k=limit         → 最多返回几个(取最相近的 k 个)
            #   score_threshold → 相似度门槛(低于这个分数的太不相关，直接丢弃)，来自 config
            # 返回：一个"文档对象列表"，每个 doc 有 .page_content(正文) 和 .metadata(元数据字典)。
            final_limit = min(max(1, limit), config.DEFAULT_RETRIEVAL_K)
            candidate_limit = max(final_limit, config.RETRIEVAL_CANDIDATE_K)
            results = self.collection.similarity_search(
                query,
                k=candidate_limit,
                score_threshold=config.RETRIEVAL_SCORE_THRESHOLD,
            )
            # if：一个都没搜到 → 返回"没相关内容"的暗号（供 nodes._retrieval_contexts 过滤掉）
            if not results:
                output = "NO_RELEVANT_CHUNKS"  # 没搜到相关内容的标记
                log_tool_end("search_child_chunks", output)
                return output

            # 第一阶段是混合召回，第二阶段用 cross-encoder 精排后只保留 top-k。
            try:
                results = self.reranker.rerank(query, results, final_limit)
            except Exception as e:
                # 模型首次下载或本地推理失败时，保留原有召回结果，避免整个 Agent 中断。
                log_error("reranker", e)
                results = results[:final_limit]

            # ── 把每个命中的子块拼成固定格式的文本返回给模型 ──────────────────
            # ⭐ 这里的格式("Parent ID:…\nFile Name:…\nContent:…" + CHILD_CHUNK_SEPARATOR 分隔)
            #    就是文件顶部说的"隐形合同"——nodes._retrieval_contexts 会按这个格式反解析。
            #    doc.metadata.get('parent_id','') → 取父块ID(供模型下一步 retrieve_parent_chunks 用)
            #    doc.metadata.get('source','')    → 取来源文件名(供最终"参考来源"引用)
            #    doc.page_content.strip()         → 子块正文(去掉首尾空白)
            output = config.CHILD_CHUNK_SEPARATOR.join([
                f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
                f"File Name: {doc.metadata.get('source', '')}\n"
                f"Content: {doc.page_content.strip()}"
                for doc in results
            ])
            log_tool_end("search_child_chunks", output)
            return output

        except Exception as e:
            # 检索过程报错 → 不让异常炸穿整个图，而是返回"出错暗号"字符串
            # （同样会被 nodes._retrieval_contexts 的 ignored_prefixes 过滤掉）
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
        # ↑↑ docstring 里"已出现过的父块ID不要再取"这句是给模型的硬约束。它能生效，靠的是
        #    should_compress_context 把取过的父块记进 retrieval_keys、compress_context 再把
        #    清单写进摘要喂回模型（见 graph_state.py 对 retrieval_keys 的【写】【读】说明）。
        log_tool_start("retrieve_parent_chunks", {"parent_id": parent_id})  # 只打日志，可忽略
        try:
            # 用 parent_id 去父块库捞完整父块。load_content 返回一个字典(或 None)。
            parent = self.parent_store_manager.load_content(parent_id)
            # if：这个 parent_id 找不到对应父块 → 返回"没这父块"的暗号
            if not parent:
                output = "NO_PARENT_DOCUMENT"  # 找不到对应父块的标记
                log_tool_end("retrieve_parent_chunks", output)
                return output

            # 同样拼成"Parent ID / File Name / Content"固定格式(与子块检索保持一致)。
            # parent.get(...) / parent.get('metadata',{}).get('source',...) 都带默认值，
            # 是为了"字段缺失也不报错"——取不到就填 'n/a'/'unknown'/''。
            output = (
                f"Parent ID: {parent.get('parent_id', 'n/a')}\n"
                f"File Name: {parent.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {parent.get('content', '').strip()}"
            )
            log_tool_end("retrieve_parent_chunks", output)
            return output

        except Exception as e:
            log_error("retrieve_parent_chunks", e)
            output = f"PARENT_RETRIEVAL_ERROR: {str(e)}"   # 出错暗号(同样会被下游过滤)
            log_tool_end("retrieve_parent_chunks", output)
            return output

    def create_tools(self) -> list:
        """把上面两个函数包装成 LangChain 工具并返回列表。

        ── tool("名字")(函数) 这个写法怎么读 ─────────────────────────────
        tool("search_child_chunks") 先返回一个"装饰器"，再 (self._search_child_chunks)
        把方法传进去 → 得到一个【工具对象】，并强制它对模型显示的名字叫 "search_child_chunks"。
        (显式命名的好处：不受方法名前缀下划线 _ 影响，模型看到的就是干净的工具名。)

        返回的这个列表 [search_tool, retrieve_tool] 会一路交给 graph.py：
          · llm.bind_tools(这个列表) → 让模型"知道有这两个工具可调"；
          · ToolNode(这个列表)       → 真正执行模型点名的那个工具。
        """
        search_tool = tool("search_child_chunks")(self._search_child_chunks)
        retrieve_tool = tool("retrieve_parent_chunks")(self._retrieve_parent_chunks)

        return [search_tool, retrieve_tool]
