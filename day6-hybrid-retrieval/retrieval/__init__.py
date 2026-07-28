"""Day6 检索优化包：MMR / BM25 / RRF 融合 / CrossEncoder 重排。

分层职责：
    rag_chain.build_vectorstore  -> 底层：建库/复用，返回 vectorstore
    retrieval.mmr_retriever      -> 策略1：MMR 多样性检索
    retrieval.bm25_retriever    -> 策略2：BM25 关键词检索
    retrieval.fusion            -> 策略3：语义 + BM25 的 RRF 融合
    retrieval.reranker          -> 策略4：CrossEncoder 精排（站在前三者肩上）

调用方通常只需：
    from retrieval import hybrid_search_reranked
"""

from .mmr_retriever import build_mmr_retriever
from .bm25_retriever import BM25Retriever
from .fusion import hybrid_search
from .reranker import Reranker, hybrid_search_reranked

__all__ = [
    "build_mmr_retriever",
    "BM25Retriever",
    "hybrid_search",
    "Reranker",
    "hybrid_search_reranked",
]
