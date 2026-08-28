"""bge-m3 向量化客户端。

封装对 Ollama /api/embed 的调用，对外只暴露：
- embed(texts: list[str]) -> list[list[float]]  批量向量化
- dimension: int                                向量维度（建表/计算用）
上层不关心 Ollama 细节。
"""

import atexit
import httpx

from config import get_settings

_SETTINGS = get_settings()
_MODEL = _SETTINGS.embedding_model              # 向量化模型
_DIMENSION = _SETTINGS.embedding_dimension      # 向量化模型输出维


class EmbeddingClient:
    """
    bge-m3 向量化客户端（同步，带连接池复用）。
    """

    def __init__(self, timeout: float = 300.0) -> None:
        self._client = httpx.Client(base_url=_SETTINGS.ollama_base_url, timeout=timeout)

    @property
    def dimension(self) -> int:
        return _DIMENSION

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        批量将文本转为向量，顺序与输入一致。空输入返回空列表。
        """
        if not texts:
            return []
        # keep_alive 让模型在批量任务期间常驻显存，避免反复冷启
        resp = self._client.post(
            "/api/embed",
            json={"model": _MODEL, "input": texts, "keep_alive": "10m"},
        )
        resp.raise_for_status()
        embeddings = resp.json().get("embeddings")
        if not embeddings or len(embeddings) != len(texts):
            raise ValueError("Ollama embed 返回结果与输入数量不一致")
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        单条查询向量（检索时用语义向量）。
        """
        return self.embed([text])[0]

    def close(self) -> None:
        self._client.close()


_embedding_client: EmbeddingClient | None = None


def get_embedding_client() -> EmbeddingClient:
    """
    返回全局唯一的 EmbeddingClient（懒加载）。
    """
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
        atexit.register(_embedding_client.close)
    return _embedding_client
