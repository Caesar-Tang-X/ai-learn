"""
嵌入式模型工厂：按 provider 返回实例（对外只暴露 get_embedding_client），实例按 provider 缓存为进程级单例以复用连接池。
新增服务方（如 openai）只需加子类 + 此处分支，上层零改动。
"""
from core.embeddings.base import BaseEmbedding
from core.embeddings.ollama import OllamaEmbedding
from core.embeddings.alibaba import AlibabaEmbedding

from config import get_settings

_SETTINGS = get_settings()

_clients: dict[str, BaseEmbedding] = {}


def get_embedding_client(provider: str | None = None) -> BaseEmbedding:
    """
    工厂方法。
    :param provider: "ollama" | "alibaba"，为 None 时读 settings.embedding_provider
    :return: BaseEmbedding 实例（按 provider 缓存单例）
    """
    if provider is None:
        provider = _SETTINGS.embedding_provider
    if provider not in _clients:
        if provider == "ollama":
            _clients[provider] = OllamaEmbedding()
        elif provider == "alibaba":
            _clients[provider] = AlibabaEmbedding()
        else:
            raise ValueError(f"不支持的 embedding provider: {provider}")
    return _clients[provider]


def close_all() -> None:
    """关闭所有缓存的 embedding 客户端，释放连接（进程退出时由 lifespan 调用）。"""
    for client in _clients.values():
        client.close()


__all__ = ["get_embedding_client", "close_all"]
