"""
嵌入式模型统一抽象接口（工厂模式基类）。

约定：
- embed(texts) 返回的向量列表顺序与输入 texts 一一对应。
- embed_query(text) 是 embed([text])[0] 的便捷封装，供单条查询使用。
- dimension 为向量维度，建表/计算距离时使用。
上层（retrieval / ingest）只依赖此抽象，不感知具体服务方。
"""
from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度。"""
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量文本转向量，返回顺序与输入一致。空输入返回空列表。"""
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        """单条查询文本转向量。"""
        return self.embed([text])[0]

    def close(self) -> None:
        """释放底层连接资源，子类按需重写。"""
        pass
