"""BM25 关键词检索（步骤2）。

对所有 chunk 建关键词索引，补语义检索对精确词（pip install / 依赖）的短板。
依赖：..rag_chain.build_vectorstore

设计：BM25Retriever 只专注 BM25 算法，接收已构建的 vectorstore，
不关心"如何建库"，从而与 fusion 共享同一个 vectorstore、避免重复入库。
"""

import jieba
from rank_bm25 import BM25Okapi

from rag_chain import build_vectorstore

jieba.setLogLevel(20)  # 静默 jieba 的 Building prefix dict 日志


def _tokenize(text: str):
    """中文用 jieba 分词，保留英文/数字词，过滤空串。

    Args:
        text: 待分词文本。

    Returns:
        List[str]：分词后的 token 列表。
    """
    return [t for t in jieba.lcut(text) if t.strip()]


class BM25Retriever:
    """基于 BM25 的关键词检索器，专注关键词匹配算法，不关心建库细节。"""

    def __init__(self, vectorstore):
        """由已构建的 vectorstore 初始化（复用其已写入的 chunk）。

        Args:
            vectorstore: rag_chain.build_vectorstore 返回的 Chroma 实例。
        """
        data = vectorstore.get()              # Chroma 原生结构：{ids, documents, metadatas, ...}
        self.corpus = data["documents"]       # List[str]，全部 chunk 正文
        self.metadatas = data["metadatas"]    # List[dict]，对应 chunk_id / source
        # 每个 chunk 分词后建 BM25 倒排索引
        self.bm25 = BM25Okapi([_tokenize(d) for d in self.corpus])

    @classmethod
    def from_file(cls, file_path: str):
        """便捷构造：由源文档路径构建 vectorstore 再初始化。

        Args:
            file_path: 源文档路径。

        Returns:
            BM25Retriever 实例。
        """
        return cls(build_vectorstore(file_path))

    def search(self, query: str, k: int = 3):
        """关键词检索，返回 Top-K。

        Args:
            query: 查询串。
            k: 返回条数（仅返回 BM25 分数 > 0 的命中）。

        Returns:
            List[Tuple[str, dict, float]]：(正文, metadata, BM25分数)。
        """
        scores = self.bm25.get_scores(_tokenize(query))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.corpus[i], self.metadatas[i], scores[i])
                for i in top_idx if scores[i] > 0]


if __name__ == "__main__":
    retriever = BM25Retriever.from_file("samples/README.md")
    results = retriever.search("如何安装依赖", k=3)
    print(f"\n===== [BM25 关键词召回] {len(results)} 条 =====")
    for i, (content, meta, score) in enumerate(results):
        print(f"--- {i+1} | chunk_id {meta['chunk_id']} | BM25分 {score:.2f} ---")
        print(content[:80], "...\n")
