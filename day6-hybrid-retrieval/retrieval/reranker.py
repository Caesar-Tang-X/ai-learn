"""CrossEncoder 重排（步骤4）。

对混合检索召回的候选做精排：将 query+doc 拼接进交叉编码器算相关性分，压低弱相关。
依赖：.fusion.hybrid_search
"""

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from .fusion import hybrid_search


class Reranker:
    """基于 CrossEncoder 的精排器，专注相关性打分，不关心候选从哪来。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """初始化 CrossEncoder。

        Args:
            model_name: 重排模型名，与 Day3 的 bge-m3 同家族；
                首次运行自动下载（约 2GB+，需联网）。
        """
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates, top_k: int = 3, threshold=None):
        """对候选文档精排。

        Args:
            query: 查询串。
            candidates: List[Document] 候选文档。
            top_k: 返回条数。
            threshold: 相关性分数阈值（None 不过滤；否则低于该值的候选被丢弃）。

        Returns:
            List[Tuple[Document, float]]：(文档, 重排分数)，按分数降序。
        """
        if not candidates:
            return []
        pairs = [(query, d.page_content) for d in candidates]
        scores = self.model.predict(pairs)          # 一维 array，越相关分越高
        scored = sorted(zip(candidates, map(float, scores)),
                        key=lambda x: x[1], reverse=True)
        if threshold is not None:
            scored = [s for s in scored if s[1] >= threshold]
        return [(d, s) for d, s in scored[:top_k]]


def hybrid_search_reranked(query: str, file_path: str,
                           fetch_k: int = 10, final_k: int = 3, threshold=None):
    """混合检索取候选 → CrossEncoder 精排 → 返回 Top-K。

    Args:
        query: 查询串。
        file_path: 源文档路径。
        fetch_k: 混合检索候选池大小（粗排阶段）。
        final_k: 精排后最终返回条数。
        threshold: 重排分数阈值，用于过滤弱相关。

    Returns:
        List[Tuple[Document, float]]：(文档, 重排分数)。
    """
    candidates = hybrid_search(query, file_path,
                               semantic_k=fetch_k, bm25_k=fetch_k, final_k=fetch_k)
    return Reranker().rerank(query, candidates, top_k=final_k, threshold=threshold)


if __name__ == "__main__":
    results = hybrid_search_reranked("如何安装依赖", "samples/README.md",
                                     fetch_k=10, final_k=3)
    print(f"\n===== [混合 + Rerank 召回] {len(results)} 条 =====")
    for i, (doc, score) in enumerate(results):
        print(f"--- {i+1} | chunk_id {doc.metadata['chunk_id']} | rerank分 {score:.3f} ---")
        print(doc.page_content[:80], "...\n")
