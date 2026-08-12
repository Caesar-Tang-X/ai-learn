"""
混合检索：向量检索 + BM25 关键词检索，RRF 融合重排。

纯本地实现，不依赖额外重排模型。
"""
from collections import defaultdict

from rank_bm25 import BM25Okapi

from config import get_settings
from core.embeddings import get_embedding_client
from core.vectorstore import VectorStore
import jieba


def _tokenize(text: str) -> list[str]:
    """
    中文/英文分词，供 BM25 使用（依赖 jieba，见 requirements.txt）。

    - 用 jieba 做中文分词，质量最好；
    - 英文统一转小写，并只保留中英文词（剔除标点/符号）。
    """
    import re

    text = text.lower()
    raw = jieba.lcut(text)
    return [t for t in raw if re.fullmatch(r"[\w\u4e00-\u9fff]+", t)]


def _bm25_rank(query: str, corpus: list[dict]) -> list[int]:
    """
    对 corpus 建 BM25，返回按分数降序的真实 doc id 列表。
    """
    if not corpus:
        return []
    tokenized = [_tokenize(d["content"]) for d in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(_tokenize(query))
    ranked_idx = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)
    return [corpus[i]["id"] for i in ranked_idx]


def hybrid_retrieve(query: str, top_k: int | None = None, rerank_top_n: int | None = None) -> list[dict]:
    """
    混合检索并 RRF 融合重排。

    Args:
        query: 用户问题。
        top_k: 向量召回数量，缺省读 config.top_k。
        rerank_top_n: 最终返回数量，缺省读 config.rerank_top_n。
    Returns:
        重排后的候选块列表（含 content/source/score），按 RRF 分降序。
    """
    cfg = get_settings()
    k = top_k or cfg.top_k
    n = rerank_top_n or cfg.rerank_top_n

    client = get_embedding_client()
    qvec = client.embed_query(query)
    store = VectorStore()

    # 1) 向量候选（先做语义召回，拿到候选集）
    vector_hits = store.search(qvec, top_k=k)
    if not vector_hits:
        return []

    # 2) BM25 只在向量召回的候选集上建索引，避免全表扫描
    #    （数据量大时 fetch_all 会爆炸；且 BM25 仅在语义相关子集上做关键词互补）
    id_to_doc = {d["id"]: d for d in vector_hits}
    bm25_ranked_ids = _bm25_rank(query, vector_hits)

    # 3) RRF 融合：两路候选并集，按排名倒数加权
    rrf = defaultdict(float)
    RRF_K = 60
    for rank, doc in enumerate(vector_hits, start=1):
        rrf[doc["id"]] += 1.0 / (RRF_K + rank)
    for rank, doc_id in enumerate(bm25_ranked_ids, start=1):
        rrf[doc_id] += 1.0 / (RRF_K + rank)

    # 4) 按 RRF 分降序，取前 n 篇
    ranked_ids = sorted(rrf.keys(), key=lambda i: rrf[i], reverse=True)[:n]
    results = []
    for i in ranked_ids:
        d = id_to_doc[i]
        results.append({
            "id": d["id"],
            "content": d["content"],
            "source": d["source"],
            "score": round(rrf[i], 6),
        })

    return results
