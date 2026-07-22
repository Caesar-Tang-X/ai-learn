from loaders.document_loader import DocumentLoader
from cleaners.text_cleaner import TextCleaner
from splitters.recursive_splitter import RecursiveTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

"""
author: tang
description: 文档处理流水线入口：加载 → 清洗 → 分块
"""

def mine_run_pipeline(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    手写递归字符分层分块器
    """
    print(f"\n===== 手写递归字符分层分块器 =====")
    # 1. 加载
    loader = DocumentLoader()
    raw_text = loader.load(file_path)
    print(f"[加载] 原始字符数: {len(raw_text)}")
    # 2. 清洗
    cleaner = TextCleaner()
    clean_text = cleaner.clean(raw_text)
    print(f"[清洗] 清洗后字符数: {len(clean_text)}")
    # 3. 分块
    splitter = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split(clean_text)
    print(f"[分块] 共切出 {len(chunks)} 块 (chunk_size={chunk_size}, overlap={chunk_overlap})")
    # 4. 打印每块预览
    for i, c in enumerate(chunks):
        print(f"\n===== Chunk {i+1} (长度 {len(c)}) =====")
        print(c[:200])  # 只预览前 200 字，避免刷屏
    return chunks

def lc_run_pipeline(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    LangChain分层分块器
    """
    print(f"\n===== LangChain分层分块器 =====")
    # 1. 加载
    loader = DocumentLoader()
    raw_text = loader.load(file_path)
    print(f"[加载] 原始字符数: {len(raw_text)}")
    # 2. 清洗
    cleaner = TextCleaner()
    clean_text = cleaner.clean(raw_text)
    print(f"[清洗] 清洗后字符数: {len(clean_text)}")
    # 3. 分块
    lc = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ""],
        keep_separator=True,
    )
    chunks = lc.split_text(clean_text)
    # 4. 打印每块预览
    for i, c in enumerate(chunks):
        print(f"\n===== Chunk {i+1} (长度 {len(c)}) =====")
        print(c[:200])  # 只预览前 200 字，避免刷屏
    return chunks


if __name__ == "__main__":
    mine_run_pipeline("samples/README.md", chunk_size=300, chunk_overlap=50)
    lc_run_pipeline("samples/README.md", chunk_size=300, chunk_overlap=50)
