# 全局配置：模型、接口地址、检索/切块/Agent 预算等所有参数都在这里。
import os

# --- 目录配置 ---
# 用 abspath 保证无论从哪个目录启动，数据文件夹都固定生成在项目根目录下
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MARKDOWN_DIR = os.path.join(_BASE_DIR, "markdown_docs")     # PDF 转成的 markdown 存这里
PARENT_STORE_PATH = os.path.join(_BASE_DIR, "parent_store") # 父块（大块）纯文本存这里
QDRANT_DB_PATH = os.path.join(_BASE_DIR, "qdrant_db")       # 本地向量数据库存这里

# --- Qdrant 向量库配置 ---
CHILD_COLLECTION = "document_child_chunks"
SPARSE_VECTOR_NAME = "sparse"

# ============================================================
# --- 大模型配置（智谱 GLM，走 OpenAI 兼容接口）---
# 下面三个值从 .env 文件读取，改 .env 即可，不用改这里
# ============================================================
LLM_API_KEY = os.environ.get("API_KEY", "")
LLM_BASE_URL = os.environ.get("BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
LLM_MODEL = os.environ.get("MODEL_ID", "glm-4.7")

# temperature：智谱部分模型不接受 0，这里用一个很小的值保证输出稳定。
# 如果你的模型支持 0，可以改成 0。
LLM_TEMPERATURE = 0.1

# --- 嵌入模型（Embedding）配置 ---
# 稠密向量：用智谱的 embedding API（无需本地 torch，复用 .env 里同一个 API_KEY / BASE_URL）
# 可选：embedding-3（2048 维，推荐）、embedding-2（1024 维）
DENSE_MODEL = "embedding-3"
EMBEDDING_BATCH_SIZE = 16   # 每次发给智谱多少条文本（太大可能超接口单次上限）
# 稀疏向量：本地 BM25（fastembed，轻量，不依赖 torch）
SPARSE_MODEL = "Qdrant/bm25"

# 注意：JUDGE_MODEL 在当前代码里没有被使用，保留仅为兼容，可忽略。
JUDGE_MODEL = ""
# LLM_SEED 仅 Ollama 用，智谱不使用，保留仅为兼容。
LLM_SEED = 42

# --- 检索配置 ---
# 重要（中文优化）：混合检索(HYBRID)下这个阈值卡的是 RRF 融合分(上限约 1.0)，
# 不是余弦相似度。只被一路(dense 或 sparse)命中的块分数很低，阈值一高就会被全部过滤掉，
# 导致"搜不到"。中文场景 sparse(BM25)较弱、主要靠 dense，所以这里用很低的阈值。
# 如果你发现检索经常为空，可以把它再调低甚至设为 0。
RETRIEVAL_SCORE_THRESHOLD = 0.02
DEFAULT_RETRIEVAL_K = 7
CHILD_CHUNK_SEPARATOR = "\n\n<CHILD_CHUNK_BOUNDARY>\n\n"

# --- Agent 预算配置（防止死循环）---
MAX_TOOL_CALLS = 8
MAX_ITERATIONS = 10
GRAPH_RECURSION_LIMIT = 50
MAIN_HISTORY_MESSAGES_TO_KEEP = 4
BASE_TOKEN_THRESHOLD = 2000
TOKEN_GROWTH_FACTOR = 0.9

# --- 终端执行日志（调试用，想看 Agent 每一步在干嘛就设为 True）---
EXECUTION_LOGGING_ENABLED = False
EXECUTION_LOG_MAX_CHARS = 1200
EXECUTION_LOG_USE_COLOR = True

# --- 文本切块配置 ---
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100
MIN_PARENT_SIZE = 2000
MAX_PARENT_SIZE = 4000
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]

# --- Langfuse 可观测性（可选，默认关闭）---
LANGFUSE_ENABLED = os.environ.get("LANGFUSE_ENABLED", "false").lower() == "true"
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000")
