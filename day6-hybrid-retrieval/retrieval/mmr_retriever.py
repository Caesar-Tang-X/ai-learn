"""MMR 多样性检索（步骤1）。

在"相关性"与"多样性"间权衡，避免召回高度相似的冗余段。
依赖：..rag_chain.build_vectorstore
"""

from rag_chain import build_vectorstore


def build_mmr_retriever(file_path: str, k: int = 3, fetch_k: int = 10, lambda_mult: float = 0.5):
    """构建 MMR（最大边际相关）检索器。

    Args:
        file_path: 源文档路径。
        k: 最终返回条数。
        fetch_k: 候选池大小（先按相似度取多少条候选，池越大多样性空间越大）。
        lambda_mult: 平衡系数。1=只看重相关性，0=只看重多样性（与已选不相似）。

    Returns:
        VectorStoreRetriever：配置为 MMR 的检索器，调用 .invoke(query) 取结果。
    """
    vectorstore = build_vectorstore(file_path)
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )


if __name__ == "__main__":
    retriever = build_mmr_retriever("samples/README.md", k=3)
    docs = retriever.invoke("如何安装依赖")
    print(f"\n===== [MMR 召回] {len(docs)} 条 =====")
    for i, d in enumerate(docs):
        print(f"--- {i+1} | chunk_id {d.metadata['chunk_id']} ---")
        print(d.page_content[:80], "...\n")
