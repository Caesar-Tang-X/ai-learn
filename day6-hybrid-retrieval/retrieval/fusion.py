"""RRF 倒数排名融合（步骤3）。

将"语义检索"与"BM25 关键词检索"的结果用 RRF 融合，兼顾语义相近与精确词命中。
依赖：..rag_chain.build_vectorstore, .bm25_retriever.BM25Retriever
"""

from langchain_core.documents import Document

from rag_chain import build_vectorstore
from .bm25_retriever import BM25Retriever


def hybrid_search(query: str, file_path: str,
                  semantic_k: int = 10, bm25_k: int = 10,
                  final_k: int = 3, rrf_k: int = 60):
    """语义 Top-N + BM25 Top-N → RRF 融合 → 返回 Top-K 的 Document。

    RRF 用"排名"而非"分数"融合，消除两种检索器量纲不一致的难题：
        fused_score(doc) = Σ 1/(rrf_k + rank)，rank 从 1 起；用 chunk_id 去重合并。

    Args:
        query: 查询串。
        file_path: 源文档路径。
        semantic_k: 语义检索候选池大小。
        bm25_k: BM25 检索候选池大小。
        final_k: 融合后最终返回条数。
        rrf_k: RRF 常数（默认 60），防止头部排名分数过大、掩盖其他信号。

    Returns:
        List[Document]：融合后排序的文档列表（按 chunk_id 去重）。
    """
    vectorstore = build_vectorstore(file_path)
    # 1) 语义检索：取较大候选池（similarity），排名基于向量距离
    semantic_docs = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": semantic_k},
    ).invoke(query)
    # 2) BM25 关键词检索：复用同一 vectorstore（共享，不重复建库）
    bm25_results = BM25Retriever(vectorstore).search(query, k=bm25_k)

    # 3) RRF 融合
    fused = {}
    for rank, doc in enumerate(semantic_docs, start=1):
        cid = doc.metadata["chunk_id"]
        fused.setdefault(cid, {"doc": doc, "score": 0.0})
        fused[cid]["score"] += 1.0 / (rrf_k + rank)

    for rank, (content, meta, _) in enumerate(bm25_results, start=1):
        cid = meta["chunk_id"]
        if cid not in fused:
            fused[cid] = {"doc": Document(page_content=content, metadata=meta), "score": 0.0}
        fused[cid]["score"] += 1.0 / (rrf_k + rank)

    # 按融合分数降序，取最终 Top-K
    ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:final_k]
    return [item["doc"] for item in ranked]


if __name__ == "__main__":
    docs = hybrid_search("如何安装依赖", "samples/README.md", final_k=3)
    print(f"\n===== [RRF 混合召回] {len(docs)} 条 =====")
    for i, d in enumerate(docs):
        print(f"--- {i+1} | chunk_id {d.metadata['chunk_id']} ---")
        print(d.page_content[:80], "...\n")
