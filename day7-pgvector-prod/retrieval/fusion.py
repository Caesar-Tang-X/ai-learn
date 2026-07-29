"""RRF 融合检索——Day7 适配版。

与 Day6 的核心差异：两路检索解耦。
- 语义：PGVector.as_retriever()（原来 Chroma）
- BM25：BM25Retriever.from_file()（原来依赖 vectorstore.get()）
- RRF 融合公式：完全不变
"""

from typing import List, Tuple
from langchain_core.documents import Document

from rag_chain import build_vectorstore
from retrieval.bm25_retriever import BM25Retriever

RRF_K = 60


def _semantic_search(
    question: str, file_path: str, k: int = 10
) -> List[Document]:
    """语义分支：PGVector similarity。"""
    vs = build_vectorstore(file_path)
    retriever = vs.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    return retriever.invoke(question)


def _bm25_search(
    question: str, file_path: str, k: int = 10
) -> List[Tuple[str, dict, float]]:
    """BM25 分支：独立加载文档，不依赖 vectorstore。"""
    bm25 = BM25Retriever.from_file(file_path)
    return bm25.search(question, k=k)


def rrf_fusion(
    question: str,
    file_path: str,
    semantic_k: int = 10,
    bm25_k: int = 10,
    final_k: int = 3,
) -> List[Document]:
    """RRF 双路融合：语义 + BM25 → 排名去量纲 → Top-K。

    RRF 公式：score = Σ 1/(RRF_K + rank)
    rank 从 1 开始（即排名第 1 的文档 rank=1）。
    """
    # ---- 1. 双路召回 ----
    sem_docs = _semantic_search(question, file_path, semantic_k)
    bm25_results = _bm25_search(question, file_path, bm25_k)

    # ---- 2. RRF 分数表（以 chunk_id 为 key 去重累加）----
    scores: dict = {}

    # 语义排名贡献
    for rank, doc in enumerate(sem_docs):
        cid = doc.metadata["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (RRF_K + rank + 1)

    # BM25 排名贡献
    for rank, (text, meta, _) in enumerate(bm25_results):
        cid = meta["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (RRF_K + rank + 1)

    # ---- 3. 按 RRF 分降序 ----
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_cids = [cid for cid, _ in ranked[:final_k]]

    # ---- 4. 用 chunk_id 找回完整 Document ----
    doc_map = {d.metadata["chunk_id"]: d for d in sem_docs}
    # 补上只出现在 BM25 结果中、语义结果没有的
    for text, meta, _ in bm25_results:
        cid = meta["chunk_id"]
        if cid not in doc_map:
            doc_map[cid] = Document(page_content=text, metadata=meta)

    return [doc_map[cid] for cid in top_cids if cid in doc_map]


# ========== 单测 ==========
if __name__ == "__main__":
    FILE = "samples/README.md"
    Q = "如何安装依赖"

    results = rrf_fusion(Q, FILE)
    print("===== [RRF 混合召回] =====")
    for i, doc in enumerate(results, 1):
        print(
            f"{i} | chunk_id {doc.metadata['chunk_id']} "
            f"| {doc.page_content[:60]}..."
        )
