"""
LLM 客户端工厂：按 provider 返回实例（对外只暴露 get_llm），实例按 provider 缓存为进程级单例以复用连接池。
新增服务方（如 openai）只需加子类 + 此处分支，上层零改动。
"""
from core.llm.base import BaseLLM
from core.llm.ollama import OllamaLLM
from core.llm.alibaba import AlibabaLLM

from config import get_settings

_llm_clients: dict[str, BaseLLM] = {}


def get_llm(provider: str | None = None) -> BaseLLM:
    """
    工厂方法。
    :param provider: "ollama" | "alibaba"，为 None 时读 settings.llm_provider
    :return: BaseLLM 实例（按 provider 缓存单例）
    """
    if provider is None:
        provider = _SETTINGS.llm_provider
    if provider not in _llm_clients:
        if provider == "ollama":
            _llm_clients[provider] = OllamaLLM()
        elif provider == "alibaba":
            _llm_clients[provider] = AlibabaLLM()
        else:
            raise ValueError(f"不支持的 llm provider: {provider}")
    return _llm_clients[provider]


async def close_all() -> None:
    """关闭所有缓存的 LLM 客户端，释放连接（进程退出时由 lifespan 调用）。"""
    for client in _llm_clients.values():
        await client.aclose()


__all__ = ["get_llm", "close_all"]
