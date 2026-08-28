"""
Ollama 本地 LLM，OpenAI 兼容 /v1/chat/completions 流式。

异步实现（httpx.AsyncClient），适配 FastAPI 异步环境。
默认 temperature=0 保证商品导购输出稳定。
"""
import json

import httpx
from collections.abc import AsyncIterator

from config import get_settings
from core.llm.base import BaseLLM

_SETTINGS = get_settings()


class OllamaLLM(BaseLLM):
    def __init__(self, timeout: float = 300.0) -> None:
        self._model = _SETTINGS.ollama_llm_model
        self._base_url = _SETTINGS.ollama_base_url
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)

    async def stream(self, messages: list[dict], **kwargs) -> AsyncIterator[str]:
        url = f"{self._base_url}/v1/chat/completions"
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }
        async with self._client.stream("POST", url, json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
