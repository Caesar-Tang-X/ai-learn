"""
阿里百炼嵌入式模型。

通过百炼 OpenAI 兼容 /embeddings 端点向量化，需 Bearer 鉴权。
同步实现，调用方在异步环境中用 asyncio.to_thread 包裹。
"""
import httpx

from config import get_settings
from core.embeddings.base import BaseEmbedding

_SETTINGS = get_settings()


class AlibabaEmbedding(BaseEmbedding):
    def __init__(self, timeout: float = 300.0) -> None:
        self._model = _SETTINGS.alibaba_embedding_model
        self._dimension = _SETTINGS.alibaba_embedding_dimension
        api_key = _SETTINGS.alibaba_dashscope_api_key
        if not api_key:
            raise ValueError(
                "未配置阿里百炼 API Key：请在 .env 中设置 ALIBABA_DASHSCOPE_API_KEY="
                "(或在 settings 中填写)，否则无法调用 embedding/llm 服务。"
            )
        self._client = httpx.Client(
            base_url=_SETTINGS.alibaba_dashscope_base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    # 百炼 embeddings 端点单次 input 数组上限为 16 条，超出返回 400，故内部自动分片
    _MAX_BATCH = 16

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._MAX_BATCH):
            batch = texts[i:i + self._MAX_BATCH]
            resp = self._client.post(
                "/embeddings",
                json={"model": self._model, "input": batch},
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            vectors = [d["embedding"] for d in data]
            if len(vectors) != len(batch):
                raise ValueError("Alibaba embed 返回结果与输入数量不一致")
            embeddings.extend(vectors)
        return embeddings

    def close(self) -> None:
        self._client.close()
