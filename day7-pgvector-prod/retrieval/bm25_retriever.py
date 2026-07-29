"""BM25 关键词检索——Day7 适配版。

与 Day6 的关键差异：
- Day6: 构造函数接收 vectorstore，调用 vectorstore.get() 取文档
- Day7: 构造函数接收 List[Document]，不依赖任何 vectorstore

算法（BM25Okapi + jieba 分词）完全不变。
"""

from typing import List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import jieba

from loaders.document_loader import DocumentLoader
from cleaners.text_cleaner import TextCleaner
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _tokenize(text: str) -> List[str]:
    """中文分词（jieba 精确模式），Day6 同款。"""
    return jieba.lcut(text)


class BM25Retriever:
    """BM25 关键词检索器。

    Args:
        docs: Document 列表（page_content 为正文，metadata 为元数据）。
    """

    def __init__(self, docs: List[Document]):
        self.corpus = [d.page_content for d in docs]
        self.metadatas = [d.metadata for d in docs]
        self.bm25 = BM25Okapi([_tokenize(t) for t in self.corpus])

    @classmethod
    def from_file(
        cls,
        file_path: str,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ) -> "BM25Retriever":
        """便捷构造：从文件加载、清洗、分段，建立 BM25 索引。

        不依赖 vectorstore——这是 Day7 相对于 Day6 的设计改进。
        """
        raw = DocumentLoader().load(file_path)
        clean = TextCleaner().clean(raw)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_text(clean)
        docs = [
            Document(
                page_content=c,
                metadata={"source": file_path, "chunk_id": i},
            )
            for i, c in enumerate(chunks)
        ]
        return cls(docs)

    def search(
        self, query: str, k: int = 3
    ) -> List[Tuple[str, dict, float]]:
        """BM25 检索，返回 (正文, metadata, BM25分数)。

        Args:
            query: 查询文本。
            k: 返回条数。

        Returns:
            List[Tuple[str, dict, float]]：(page_content, metadata, score)。
        """
        tokenized = _tokenize(query)
        scores = self.bm25.get_scores(tokenized)
        top_idx = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        return [
            (self.corpus[i], self.metadatas[i], scores[i])
            for i in top_idx
            if scores[i] > 0
        ]


# ========== 单测 ==========
if __name__ == "__main__":
    FILE = "samples/README.md"
    bm25 = BM25Retriever.from_file(FILE)

    results = bm25.search("如何安装依赖", k=3)
    print("===== [BM25 关键词召回] 3 条 =====")
    for i, (text, meta, score) in enumerate(results, 1):
        print(
            f"{i} | chunk_id {meta['chunk_id']} | BM25分 {score:.2f}  "
            f"| {text[:50]}..."
        )
