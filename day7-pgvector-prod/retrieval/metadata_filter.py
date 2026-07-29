"""元数据过滤检索——利用 PGVector 的 JSONB 做 SQL 层精确过滤。

这是 PGVector 相对于 Chroma 的核心优势之一：
- Chroma: filter 只能精确匹配，后置 Python 侧再筛
- PGVector: filter → SQL WHERE（支持范围/组合），在向量检索前完成

langchain_postgres 的 filter 语法限制：
- 每个字段的字典只能有一个运算符（key），不能 {"$gte": 0, "$lte": 5}
- 范围查询需用 $and 拆分：[{"chunk_id": {"$gte": 0}}, {"chunk_id": {"$lte": 5}}]
"""

from typing import List, Optional, Dict
from langchain_core.documents import Document

from rag_chain import build_vectorstore


def metadata_filter_search(
    question: str,
    file_path: str,
    filters: Optional[Dict] = None,
    k: int = 3,
) -> List[Document]:
    """语义检索 + PGVector JSONB filter。

    Args:
        question: 用户问题。
        file_path: 源文档。
        filters: filter 字典（None 不过滤）。
            - {"chunk_id": 3}                                              → 精确
            - {"chunk_id": {"$gte": 3}}                                    → 单运算符
            - {"$and": [{"chunk_id": {"$gte": 0}}, {"chunk_id": {"$lte": 5}}]}  → 范围
        k: 返回条数。
    """
    vs = build_vectorstore(file_path)
    kwargs = {"k": k}
    if filters:
        kwargs["filter"] = filters
    return vs.similarity_search(question, **kwargs)


def hybrid_with_filter(
    question: str,
    file_path: str,
    filters: Optional[Dict] = None,
    fetch_k: int = 10,
    final_k: int = 3,
) -> List[Document]:
    """混合检索 + 元数据过滤。

    流程：PGVector 语义（带 filter）+ BM25 → RRF 融合。

    过滤能力的差异：
    - 语义分支：filter 推给 PGVector → SQL WHERE，精确过滤
    - BM25 分支：无数据库，只做简单精确匹配；运算符/范围条件放行，
      靠 RRF 融合时语义分支的精确结果兜底
    """
    from retrieval.bm25_retriever import BM25Retriever

    # ---- 语义分支（filter 推给 PGVector，SQL 层精确执行）----
    vs = build_vectorstore(file_path)
    kwargs = {"k": fetch_k}
    if filters:
        kwargs["filter"] = filters
    sem_docs = vs.similarity_search(question, **kwargs)

    # ---- BM25 分支 ----
    bm25 = BM25Retriever.from_file(file_path)
    bm25_results = bm25.search(question, k=fetch_k)

    # ---- RRF 融合 ----
    scores: dict = {}
    RRF_K = 60

    for rank, doc in enumerate(sem_docs):
        cid = doc.metadata["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (RRF_K + rank + 1)

    for rank, (text, meta, _) in enumerate(bm25_results):
        cid = meta["chunk_id"]
        # BM25 侧过滤：只做简单精确匹配；运算符/范围无条件放行
        # （BM25 无 SQL 的天然局限——语义分支的精确过滤会在 RRF 总分上兜底）
        skip = False
        if filters:
            for key, val in filters.items():
                if key.startswith("$"):
                    continue
                if isinstance(val, dict):
                    # 运算符形式（$gte/$lte），BM25 不支持，放行
                    continue
                if meta.get(key) != val:
                    skip = True
                    break
        if not skip:
            scores[cid] = scores.get(cid, 0) + 1.0 / (RRF_K + rank + 1)

    # ---- 排序 + 还原 Document ----
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_cids = [cid for cid, _ in ranked[:final_k]]

    doc_map = {d.metadata["chunk_id"]: d for d in sem_docs}
    for text, meta, _ in bm25_results:
        cid = meta["chunk_id"]
        if cid not in doc_map:
            doc_map[cid] = Document(page_content=text, metadata=meta)

    return [doc_map[cid] for cid in top_cids if cid in doc_map]


def build_filter_app():
    """交互式 demo：手动构造各种 filter 看召回变化。"""
    FILE = "samples/README.md"
    Q = "如何安装依赖"

    test_cases = [
        ("无过滤", None),
        ("chunk_id >= 3", {"chunk_id": {"$gte": 3}}),
        (
            "chunk_id 0~5",
            {"$and": [
                {"chunk_id": {"$gte": 0}},
                {"chunk_id": {"$lte": 5}},
            ]},
        ),
        (
            "chunk_id 10~20",
            {"$and": [
                {"chunk_id": {"$gte": 10}},
                {"chunk_id": {"$lte": 20}},
            ]},
        ),
    ]

    print("===== 语义检索 + PGVector JSONB filter =====")
    for label, f in test_cases:
        print(f"\n===== [{label}] =====")
        docs = metadata_filter_search(Q, FILE, filters=f, k=3)
        if not docs:
            print("  (无结果)")
        for i, doc in enumerate(docs, 1):
            print(
                f"  {i} | chunk_id {doc.metadata['chunk_id']} "
                f"| {doc.page_content[:60]}..."
            )

    # 对比：混合检索 + 过滤
    print("\n===== [对比：混合检索 + chunk_id 0~5 过滤] =====")
    hybrid_filter = {
        "$and": [
            {"chunk_id": {"$gte": 0}},
            {"chunk_id": {"$lte": 5}},
        ]
    }
    docs2 = hybrid_with_filter(Q, FILE, filters=hybrid_filter)
    for i, doc in enumerate(docs2, 1):
        print(
            f"  {i} | chunk_id {doc.metadata['chunk_id']} "
            f"| {doc.page_content[:60]}..."
        )


# ========== 单测 ==========
if __name__ == "__main__":
    build_filter_app()
