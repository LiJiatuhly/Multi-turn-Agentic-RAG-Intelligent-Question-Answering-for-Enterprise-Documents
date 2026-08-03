# 智谱 embedding 的独立实现（评测环境专用）。
#
# 为什么不直接 import 你项目里的 db/vector_db_manager.ZhipuEmbeddings？
#   那个文件会连带 import langchain_qdrant / qdrant_client / fastembed，
#   评测环境不需要这些重依赖。所以这里照抄一份只依赖 requests 的最小实现。
# ⚠️ 逻辑与项目里那份保持一致（同一个模型、同样发原始字符串），改一边记得同步另一边。

import time

import requests
from langchain_core.embeddings import Embeddings


class ZhipuEmbeddings(Embeddings):
    """智谱 embedding API 的轻量封装（LangChain Embeddings 子类）。"""

    def __init__(self, model, api_key, base_url, batch_size=16, timeout=60, max_retries=2):
        if not api_key:
            raise ValueError("嵌入模型缺少 API Key，请在 project/.env 里填写 API_KEY。")
        self.model = model
        self.api_key = api_key
        self.url = base_url.rstrip("/") + "/embeddings"
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries

    def _post(self, batch):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": batch}
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()["data"]
                data.sort(key=lambda d: d.get("index", 0))   # 保证顺序与输入一致
                return [d["embedding"] for d in data]
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"调用智谱 embedding 接口失败: {last_err}")

    def _embed(self, texts):
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._post(texts[i:i + self.batch_size]))
        return vectors

    def embed_documents(self, texts):
        return self._embed([str(t) for t in texts])

    def embed_query(self, text):
        return self._embed([str(text)])[0]
