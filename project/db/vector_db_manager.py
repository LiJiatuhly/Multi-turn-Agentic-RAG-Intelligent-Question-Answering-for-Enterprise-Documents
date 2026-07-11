# 向量库管理：智谱词嵌入 + 本地 BM25 稀疏向量 + Qdrant 混合检索。
#
# 混合检索 = 稠密向量(语义) + 稀疏向量(关键词) 两路结果融合，
# 用 RRF(排序融合)算法综合打分，比单独用任意一路更准。

import time
import requests
import config
from langchain_core.embeddings import Embeddings    # LangChain 嵌入基类，子类实现两个方法即可
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


class ZhipuEmbeddings(Embeddings):
    """🅱️ 智谱 embedding API 的轻量封装，继承自 LangChain 的 Embeddings 基类。

    为什么不用 langchain_openai.OpenAIEmbeddings？
    OpenAIEmbeddings 默认发送 tiktoken 编码的 token 数组（不是原始字符串），
    且有 base64 编码等 OpenAI 特定行为，对第三方兼容接口会报错。
    这里直接发原始文本字符串、收浮点数向量，行为完全可控。

    核心逻辑：
        文本列表 → 分批 → POST /embeddings → 收向量列表
    """

    def __init__(self, model, api_key, base_url, batch_size=16, timeout=60, max_retries=2):
        """
        model      : 嵌入模型名，如 "embedding-3"
        api_key    : 智谱 API Key
        base_url   : 接口地址（拼上 /embeddings 后缀）
        batch_size : 每次 POST 发多少条文本（太大可能超接口限制）
        timeout    : 单次请求超时秒数
        max_retries: 失败后最多重试几次
        """
        # if：没填 API Key → 直接报错（早失败，别等到调用时才炸）
        if not api_key:
            raise ValueError("嵌入模型缺少 API Key，请在 project/.env 里填写 API_KEY。")
        self.model = model
        self.api_key = api_key
        self.url = base_url.rstrip("/") + "/embeddings"   # 拼出完整的接口 URL
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries

    def _post(self, batch):
        """🅲 发一批文本，拿回向量列表（带重试）。"""
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": batch}
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
                resp.raise_for_status()                      # 非 2xx 状态码抛异常
                data = resp.json()["data"]
                data.sort(key=lambda d: d.get("index", 0))  # 保证返回顺序与输入一致
                return [d["embedding"] for d in data]
            except Exception as e:
                last_err = e
                # if：还有重试次数 → 等一下再试（指数退避）；用完了就跳出循环去抛错
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))          # 指数退避：1.5s, 3s, ...
        raise RuntimeError(f"调用智谱 embedding 接口失败: {last_err}")

    def _embed(self, texts):
        """🅲 把文本列表分批发出去，收集所有向量。"""
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._post(texts[i:i + self.batch_size]))
        return vectors

    def embed_documents(self, texts):
        """🅲 LangChain 基类要求实现的方法：批量文本 → 向量列表（建索引时用）。"""
        return self._embed([str(t) for t in texts])

    def embed_query(self, text):
        """🅲 LangChain 基类要求实现的方法：单条查询文本 → 向量（检索时用）。"""
        return self._embed([str(text)])[0]


class VectorDbManager:
    """🅱️ 管理 Qdrant 向量库的整个生命周期：建库、删库、获取集合对象。

    __init__ 初始化三件事：
      1. QdrantClient：本地文件模式，数据存在 qdrant_db/ 目录（单进程）
      2. 稠密嵌入：ZhipuEmbeddings，调用智谱 API
      3. 稀疏嵌入：FastEmbedSparse（BM25），本地运行，不需要 GPU/API
    """

    __client: QdrantClient
    __dense_embeddings: Embeddings
    __sparse_embeddings: FastEmbedSparse

    def __init__(self):
        # path= 本地文件模式：数据存磁盘，单进程独占（两个进程同时访问会抢锁报错）
        self.__client = QdrantClient(path=config.QDRANT_DB_PATH)
        self.__dense_embeddings = ZhipuEmbeddings(
            model=config.DENSE_MODEL,
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
            batch_size=config.EMBEDDING_BATCH_SIZE,
        )
        # FastEmbedSparse：本地 BM25，首次用时下载模型（很小），之后离线
        self.__sparse_embeddings = FastEmbedSparse(model_name=config.SPARSE_MODEL)

    def _dense_vector_size(self):
        """🅲 发一条测试文本，看向量维度是多少（建表时需要）。"""
        return len(self.__dense_embeddings.embed_query("test"))

    @staticmethod
    def _collection_vector_size(collection_info):
        """🅲 从已有集合的配置里读出向量维度（用于检测维度是否匹配）。"""
        vectors_config = collection_info.config.params.vectors
        # if：单一向量配置 → 直接有 size 属性
        if hasattr(vectors_config, "size"):
            return vectors_config.size
        # if：命名向量配置(字典形式) → 取第一个向量的 size
        if isinstance(vectors_config, dict) and vectors_config:
            first_vector = next(iter(vectors_config.values()))
            return getattr(first_vector, "size", None)
        return None

    def create_collection(self, collection_name):
        """🅱️ 创建向量集合（表）。若已存在则检查维度是否匹配，不匹配就报错（防止静默出错）。

        输入：collection_name（集合名，如 "document_child_chunks"）
        输出：无（直接在数据库里创建）
        """
        expected_size = self._dense_vector_size()
        # if：集合还不存在 → 新建；else：已存在 → 校验维度是否一致
        if not self.__client.collection_exists(collection_name):
            print(f"正在创建向量集合: {collection_name}...")
            self.__client.create_collection(
                collection_name=collection_name,
                # 稠密向量：用余弦相似度（最适合语义搜索）
                vectors_config=qmodels.VectorParams(size=expected_size, distance=qmodels.Distance.COSINE),
                # 稀疏向量：BM25 是稀疏的，不需要固定维度
                sparse_vectors_config={config.SPARSE_VECTOR_NAME: qmodels.SparseVectorParams()},
            )
            print(f"✓ 向量集合已创建: {collection_name}")
        else:
            collection_info = self.__client.get_collection(collection_name)
            existing_size = self._collection_vector_size(collection_info)
            # if：已存在集合的维度 ≠ 当前模型维度（换了嵌入模型）→ 报错让用户清库重建
            if existing_size and existing_size != expected_size:
                # 换了嵌入模型就会维度不匹配，必须清空重建
                raise ValueError(
                    f"向量集合 '{collection_name}' 的维度是 {existing_size}，"
                    f"但当前嵌入模型 '{config.DENSE_MODEL}' 产出 {expected_size} 维。"
                    f"请删除 qdrant_db/ 目录后重新索引。"
                )
            print(f"✓ 向量集合已存在: {collection_name}")

    def delete_collection(self, collection_name):
        """🅲 删除集合（清空知识库时调用）。"""
        try:
            # if：集合存在才删，不存在就当作已删除（幂等）
            if self.__client.collection_exists(collection_name):
                print(f"正在删除已存在的向量集合: {collection_name}")
                self.__client.delete_collection(collection_name)
        except Exception as e:
            raise RuntimeError(f"无法删除向量集合 '{collection_name}'。") from e

    def get_collection(self, collection_name) -> QdrantVectorStore:
        """🅱️ 返回一个可以 add_documents / similarity_search 的向量库操作对象。

        输入：collection_name
        输出：QdrantVectorStore（带混合检索配置）
        """
        try:
            return QdrantVectorStore(
                client=self.__client,
                collection_name=collection_name,
                embedding=self.__dense_embeddings,
                sparse_embedding=self.__sparse_embeddings,
                retrieval_mode=RetrievalMode.HYBRID,      # 稠密+稀疏融合检索
                sparse_vector_name=config.SPARSE_VECTOR_NAME
            )
        except Exception as e:
            raise RuntimeError(f"无法初始化向量集合 '{collection_name}'。") from e
