"""
重排模型统一抽象接口（工厂模式基类）。

约定：
- rerank(query, documents) 接收查询文本 + 候选文档列表，
  返回按相关性降序排列的结果：[(index, score), ...]，index 对应输入 documents 下标。
- 重排是"精排"环节，在向量/全文召回之后调用，输出最相关的若干条。
上层（retrieval）只依赖此抽象，不感知具体服务方。
"""
from abc import ABC, abstractmethod


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[tuple[int, float]]:
        """
        对候选文档按与 query 的相关性重排。
        :param query: 查询文本
        :param documents: 候选文档列表
        :param top_n: 返回前 N 条；None 表示返回全部（按分数降序）
        :return: [(doc_index, score), ...] 降序
        """
        raise NotImplementedError
