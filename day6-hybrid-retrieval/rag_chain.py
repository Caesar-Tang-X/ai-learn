"""Day6 向量库构建（底层基础设施）。

复用 Day2/Day3/Day4 的加载、清洗、Embedding，将文档分块后写入 Chroma，
对外统一返回 vectorstore，供上层各种检索策略（MMR/BM25/Rerank）复用。

- 库为空才写入，否则复用，根治"重复运行累积"问题。
- persist_dir 基于本文件绝对路径，摆脱"相对 cwd"导致的库散落。
"""

import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from loaders.document_loader import DocumentLoader
from cleaners.text_cleaner import TextCleaner
from adapters.ollama_embeddings import OllamaEmbeddings

# 基于本文件绝对路径定位 chroma_db，彻底摆脱"相对 cwd"导致的库散落/重复入库
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(_BASE_DIR, "chroma_db")


def build_vectorstore(file_path: str, collection_name: str = "rag_docs_lc",
                      persist_dir: str = PERSIST_DIR, chunk_size: int = 300,
                      chunk_overlap: int = 50):
    """构建（或复用）Chroma 向量库，返回 vectorstore。

    流程：加载 → 清洗 → 分块 → 包装为 Document（带 chunk_id/source 元数据）→ 入库。

    Args:
        file_path: 源文档路径。
        collection_name: Chroma 集合名。
        persist_dir: 持久化目录（默认基于本文件绝对路径，避免 cwd 影响）。
        chunk_size: 分块字符数。
        chunk_overlap: 分块重叠字符数。

    Returns:
        Chroma: 已入库（或复用）的 vectorstore 实例。
    """
    embedding = OllamaEmbeddings(model="bge-m3")
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=persist_dir,
    )
    # 库为空才写入，否则复用（根治"重复运行累积"）
    if len(vectorstore.get()["ids"]) == 0:
        raw_text = DocumentLoader().load(file_path)
        clean_text = TextCleaner().clean(raw_text)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(clean_text)
        docs = [Document(page_content=c, metadata={"source": file_path, "chunk_id": i})
                for i, c in enumerate(chunks)]
        vectorstore.add_documents(docs)
        print(f"[入库] Chroma 已写入 {len(docs)} 条")
    else:
        print(f"[复用] 库已有 {len(vectorstore.get()['ids'])} 条，跳过写入")
    return vectorstore
