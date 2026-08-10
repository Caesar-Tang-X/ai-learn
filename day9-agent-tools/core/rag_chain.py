"""Day7 PGVector 建库/复用。

与 Day6 Chroma 版本的差异只有三点：
1. 存储介质：本地文件 → PostgreSQL 远程表
2. 复用判断：vectorstore.get()['ids'] → similarity_search("test", k=1)
3. 连接配置：persist_dir 路径 → PG_CONNECTION 连接串

加载、清洗、分段、元数据格式完全照搬 Day6。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector

from core.loaders.document_loader import DocumentLoader
from core.cleaners.text_cleaner import TextCleaner
from core.adapters.ollama_embeddings import OllamaEmbeddings

load_dotenv()

PG_CONNECTION = os.getenv(
    "PG_CONNECTION",
    "postgresql+psycopg://rag:rag123@localhost:5432/ragdb",
)

def _collection_name(file_path: str) -> str:
    """基于文件名生成 collection 名。

    "samples/README.md" → "docs_readme"
    """
    stem = Path(file_path).stem.lower()
    return f"docs_{stem}"

def _has_data(vs: PGVector) -> bool:
    """判断 collection 是否已有数据。

    用 similarity_search 试查 1 条——
    这是跨 VectorStore 后端的通用做法，不依赖 vectorstore.get()。
    """
    try:
        return len(vs.similarity_search("test", k=1)) >= 1
    except Exception:
        return False

def build_vectorstore(
    file_path: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    force_rebuild: bool = False,
) -> PGVector:
    """建库或复用，返回裸 PGVector 实例。

    与 Day6 接口完全一致，唯一区别是返回值类型从 Chroma 变成 PGVector。
    """
    embedding = OllamaEmbeddings(model="bge-m3")
    coll_name = _collection_name(file_path)

    # -- force_rebuild 分支：删旧建新 --
    if force_rebuild:
        print(f"[强制重建] 删除 collection: {coll_name}")
        try:
            PGVector(
                embeddings=embedding,
                collection_name=coll_name,
                connection=PG_CONNECTION,
            ).delete_collection()
        except Exception:
            pass

    # -- 复用分支 --
    try:
        vs = PGVector(
            embeddings=embedding,
            collection_name=coll_name,
            connection=PG_CONNECTION,
            use_jsonb=True,
        )
        if _has_data(vs):
            print(f"[复用] collection '{coll_name}' 已有数据")
            return vs
    except Exception:
        pass

    # -- 新建分支：加载 → 清洗 → 分段 → 包装 → 入库 --
    print("[新建] 加载 → 清洗 → 分段 → 嵌入 → 写入 PGVector ...")
    raw_text = DocumentLoader().load(file_path)
    clean_text = TextCleaner().clean(raw_text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(clean_text)

    docs = [
        Document(page_content=c, metadata={"source": file_path, "chunk_id": i})
        for i, c in enumerate(chunks)
    ]

    vs = PGVector.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=coll_name,
        connection=PG_CONNECTION,
        use_jsonb=True,
    )
    print(f"[入库] {len(docs)} 条 → collection '{coll_name}'")
    return vs


if __name__ == "__main__":
    import time

    t0 = time.time()
    vs = build_vectorstore("samples/README.md")
    elapsed = time.time() - t0

    results = vs.similarity_search("如何安装依赖", k=3)
    for i, doc in enumerate(results, 1):
        cid = doc.metadata.get("chunk_id", "?")
        print(f"  {i} | chunk_id {cid} | {doc.page_content[:60]}...")
    print(f"\n耗时: {elapsed:.1f}s")





