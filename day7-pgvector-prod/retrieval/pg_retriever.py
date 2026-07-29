"""PGVector 基础检索——语义相似度 + MMR。

PGVector.as_retriever() 接口与 Chroma 完全一致，
这是 LangChain VectorStore 抽象层的价值——存储无关的检索。
"""

from typing import List
from langchain_core.documents import Document

from rag_chain import build_vectorstore


def pg_similarity_search(
    question: str,
    file_path: str,
    k: int = 3,
) -> List[Document]:
    """PGVector 纯语义检索（等价 Day5 Chroma similarity）。"""
    vs = build_vectorstore(file_path)
    retriever = vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return retriever.invoke(question)


def pg_mmr_search(
    question: str,
    file_path: str,
    k: int = 3,
    fetch_k: int = 10,
    lambda_mult: float = 0.5,
) -> List[Document]:
    """PGVector MMR 多样性检索。

    Args:
        k: 最终返回条数。
        fetch_k: 候选池大小（先取 fetch_k 条再 MMR 筛选）。
        lambda_mult: 1=只看相关性，0=只看多样性。
    """
    vs = build_vectorstore(file_path)
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )
    return retriever.invoke(question)


# ========== 单测 ==========
if __name__ == "__main__":
    FILE = "samples/README.md"
    Q = "如何安装依赖"

    print("===== [PGVector 语义检索] 3 条 =====")
    for i, doc in enumerate(pg_similarity_search(Q, FILE), 1):
        print(f"{i} | chunk_id {doc.metadata['chunk_id']} | {doc.page_content[:60]}...")

    print("\n===== [PGVector MMR 检索] 3 条 =====")
    for i, doc in enumerate(pg_mmr_search(Q, FILE), 1):
        print(f"{i} | chunk_id {doc.metadata['chunk_id']} | {doc.page_content[:60]}...")
