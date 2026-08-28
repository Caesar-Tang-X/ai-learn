"""
LLM 统一抽象接口（工厂模式基类）。

约定：
- stream() 为异步生成器，逐块 yield 文本片段（SSE 流式输出用）。
- messages 为 OpenAI 格式：[{"role": "system"/"user"/"assistant", "content": str}]
上层 service 只依赖此抽象，不感知具体服务方（ollama/alibaba）。
"""
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class BaseLLM(ABC):
    @abstractmethod
    async def stream(self, messages: list[dict], **kwargs) -> AsyncIterator[str]:
        """
        流式生成。
        :param messages: 对话消息列表
        :param kwargs: 可选生成参数（如 temperature）
        :return: 异步迭代器，逐块产出文本片段
        """
        raise NotImplementedError

    async def aclose(self) -> None:
        """释放底层异步连接资源，子类按需重写（如关闭 httpx.AsyncClient）。"""
        pass
