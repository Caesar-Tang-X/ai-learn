"""
阿里百炼 LLM，OpenAI 兼容 /chat/completions 流式。

异步实现（httpx.AsyncClient），Bearer 鉴权。
默认 temperature=0 保证输出稳定。
"""
import json

import httpx
from collections.abc import AsyncIterator

from config import get_settings
from core.llm.base import BaseLLM

_SETTINGS = get_settings()


class AlibabaLLM(BaseLLM):
    def __init__(self, timeout: float = 300.0) -> None:
        self._model = _SETTINGS.alibaba_llm_model
        self._base_url = _SETTINGS.alibaba_dashscope_base_url
        api_key = _SETTINGS.alibaba_dashscope_api_key
        if not api_key:
            raise ValueError(
                "未配置阿里百炼 API Key：请在 .env 中设置 ALIBABA_DASHSCOPE_API_KEY="
                "(或在 settings 中填写)，否则无法调用 embedding/llm 服务。"
            )
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    async def stream(self, messages: list[dict], **kwargs) -> AsyncIterator[str]:
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 2048),
            # qwen3 系列默认开启思考模式，会输出 <think:6124c78e>...</think:6124c78e> 干扰 JSON 解析，这里关闭
            "enable_thinking": False,
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
