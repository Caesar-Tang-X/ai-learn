"""
核心编排：文档入库与向量检索闭环。

把 loaders/cleaners/chunkers/embeddings/vectorstore 串起来，
对上层（cli/api/agents）只暴露 ingest_file 与 retrieve。
"""
from config import get_settings

from core.chunkers import chunk_text
from core.cleaners import clean_text
from core.embeddings import get_embedding_client
from core.loaders import load_document
from core.vectorstore import VectorStore
from core.retrieval import hybrid_retrieve


def ingest_file(path: str, source: str | None = None) -> int:
    """
    将单个文档文件入库：加载→清洗→分块→向量化→写入 PGVector。

    Args:
        path: 文档路径（.txt/.md/.pdf）。
        source: 来源标记，缺省用文件名。
    Returns:
        入库的文本块数量。
    """
    raw = load_document(path)
    cleaned = clean_text(raw)
    chunks = chunk_text(cleaned)
    if not chunks:
        return 0
    store = VectorStore()
    src = source or path
    return store.add(chunks, source=src)


def retrieve(query: str, top_k: int | None = None, rerank_top_n: int | None = None) -> list[dict]:
    """混合检索：向量 + BM25，RRF 融合重排。"""
    return hybrid_retrieve(query, top_k=top_k, rerank_top_n=rerank_top_n)
