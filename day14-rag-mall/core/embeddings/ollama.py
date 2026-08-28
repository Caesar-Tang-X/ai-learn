"""
本地 Ollama 嵌入式模型。

通过 Ollama /api/embed 批量向量化；keep_alive 让模型常驻显存避免冷启。
同步实现，调用方在异步环境中用 asyncio.to_thread 包裹。
"""
import httpx

from config import get_settings
from core.embeddings.base import BaseEmbedding

_SETTINGS = get_settings()


class OllamaEmbedding(BaseEmbedding):
    def __init__(self, timeout: float = 300.0) -> None:
        self._model = _SETTINGS.ollama_embedding_model
        self._dimension = _SETTINGS.ollama_embedding_dimension
        self._client = httpx.Client(
            base_url=_SETTINGS.ollama_base_url, timeout=timeout
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.post(
            "/api/embed",
            json={"model": self._model, "input": texts, "keep_alive": "10m"},
        )
        resp.raise_for_status()
        embeddings = resp.json().get("embeddings")
        if not embeddings or len(embeddings) != len(texts):
            raise ValueError("Ollama embed 返回结果与输入数量不一致")
        return embeddings

    def close(self) -> None:
        self._client.close()
