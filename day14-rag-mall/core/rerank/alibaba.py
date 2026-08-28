"""
阿里百炼重排模型。

通过百炼 OpenAI 兼容 /v1/rerank 端点精排，需 Bearer 鉴权。
同步实现，调用方在异步环境中用 asyncio.to_thread 包裹。
"""
import httpx

from config import get_settings
from core.rerank.base import BaseReranker

_SETTINGS = get_settings()


class AlibabaReranker(BaseReranker):
    def __init__(self, timeout: float = 30.0) -> None:
        self._model = _SETTINGS.alibaba_rerank_model
        self._base_url = _SETTINGS.alibaba_dashscope_base_url
        api_key = _SETTINGS.alibaba_dashscope_api_key
        if not api_key:
            raise ValueError(
                "未配置阿里百炼 API Key：请在 .env 中设置 ALIBABA_DASHSCOPE_API_KEY="
                "(或在 settings 中填写)，否则无法调用 rerank 服务。"
            )
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        resp = self._client.post(
            "/rerank",
            json={
                "model": self._model,
                "query": query,
                "documents": documents,
                "return_documents": False,
            },
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        # 百炼返回: [{"index": int, "relevance_score": float}, ...] 已按分数降序
        ranked = [(r["index"], float(r["relevance_score"])) for r in results]
        if top_n is not None:
            ranked = ranked[:top_n]
        return ranked

    def close(self) -> None:
        self._client.close()
