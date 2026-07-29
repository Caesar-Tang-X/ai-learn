"""CrossEncoder 重排——核心类照搬 Day6，辅助函数适配 Day7 融合接口。

与 Day6 的差异：
- Reranker 类：零修改（只操作 (query, doc_content) 文本对，不碰 vectorstore）
- hybrid_search_reranked()：调用 Day7 的 rrf_fusion 替代 Day6 的 hybrid_search
"""

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from retrieval.fusion import rrf_fusion


class Reranker:
    """CrossEncoder 精排器。只关心 (query, doc) 交互打分，不看文档来源。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates, top_k: int = 3, threshold=None):
        """对候选文档精排并过滤。

        Args:
            query: 查询文本。
            candidates: List[Document]。
            top_k: 返回条数。
            threshold: 相关性阈值（None=不过滤，0.3=过滤弱相关）。
        """
        if not candidates:
            return []
        pairs = [(query, d.page_content) for d in candidates]
        scores = self.model.predict(pairs)
        scored = sorted(
            zip(candidates, map(float, scores)),
            key=lambda x: x[1],
            reverse=True,
        )
        if threshold is not None:
            scored = [s for s in scored if s[1] >= threshold]
        return [(d, s) for d, s in scored[:top_k]]


def hybrid_search_reranked(
    query: str,
    file_path: str,
    fetch_k: int = 10,
    final_k: int = 3,
    threshold=None,
):
    """混合检索 → 精排。

    流程：RRF 融合取候选池 → CrossEncoder 精排 → threshold 过滤 → Top-K。

    这是 main.py 最终使用的统一入口。
    """
    candidates = rrf_fusion(
        query, file_path,
        semantic_k=fetch_k, bm25_k=fetch_k, final_k=fetch_k,
    )
    return Reranker().rerank(
        query, candidates, top_k=final_k, threshold=threshold
    )


# ========== 单测 ==========
if __name__ == "__main__":
    results = hybrid_search_reranked(
        "如何安装依赖", "samples/README.md", fetch_k=10, final_k=3
    )
    print(f"\n===== [混合 + Rerank 召回] {len(results)} 条 =====")
    for i, (doc, score) in enumerate(results):
        print(
            f"--- {i+1} | chunk_id {doc.metadata['chunk_id']} "
            f"| rerank分 {score:.3f} ---"
        )
        print(doc.page_content[:80], "...\n")
