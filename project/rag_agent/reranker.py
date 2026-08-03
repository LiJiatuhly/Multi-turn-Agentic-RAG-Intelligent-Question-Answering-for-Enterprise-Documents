"""第二阶段检索：用 cross-encoder 对候选子块重新排序。"""

from fastembed.rerank.cross_encoder import TextCrossEncoder


class ChunkReranker:
    """惰性加载本地 reranker，避免启动 UI 时立即下载模型。"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def rerank(self, query: str, documents: list, limit: int) -> list:
        if len(documents) <= 1:
            return documents[:limit]
        if self._model is None:
            self._model = TextCrossEncoder(self.model_name, lazy_load=True)

        texts = [doc.page_content for doc in documents]
        scores = list(self._model.rerank(query, texts))
        ranked = sorted(zip(scores, documents), key=lambda pair: pair[0], reverse=True)
        return [doc for _score, doc in ranked[:limit]]
