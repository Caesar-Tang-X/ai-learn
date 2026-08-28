"""
重排模型工厂：按 provider 返回实例（对外只暴露 get_reranker），实例按 provider 缓存为进程级单例以复用连接池。
新增服务方（如 openai）只需加子类 + 此处分支，上层零改动。
"""
from core.rerank.base import BaseReranker
from core.rerank.alibaba import AlibabaReranker

from config import get_settings

_rerankers: dict[str, BaseReranker] = {}


def get_reranker(provider: str | None = None) -> BaseReranker:
    """
    工厂方法。
    :param provider: "alibaba"，为 None 时读 settings.rerank_provider
    :return: BaseReranker 实例（按 provider 缓存单例）
    """
    if provider is None:
        provider = _SETTINGS.rerank_provider
    if provider not in _rerankers:
        if provider == "alibaba":
            _rerankers[provider] = AlibabaReranker()
        else:
            raise ValueError(f"不支持的 reranker provider: {provider}")
    return _rerankers[provider]


def close_all() -> None:
    """关闭所有缓存的 rerank 客户端，释放连接（进程退出时由 lifespan 调用）。"""
    for client in _rerankers.values():
        client.close()


__all__ = ["get_reranker", "close_all"]
