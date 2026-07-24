"""
author: tang
description: Chroma 向量库封装：增(upsert) / 删 / 改 / 查，复用 Day3 的 EmbeddingClient
"""

import chromadb
from embeddings.embedding_client import EmbeddingClient


class ChromaStore:
    def __init__(self, collection_name: str = "rag_docs", persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedder = EmbeddingClient(model="bge-m3")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # 余弦距离，和 Day3 一致
        )

    def upsert(self, chunks: list[str], metadatas: list[dict] = None, ids: list[str] = None):
        """增/改合一：id 不存在则插入，存在则更新向量+原文。反复运行 Day4 不再冲突。"""
        if ids is None:
            ids = [f"id_{i}" for i in range(len(chunks))]
        embeddings = [self.embedder.embed(c) for c in chunks]
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        print(f"[upsert] 完成，collection 现有 {self.collection.count()} 条")

    def add(self, chunks: list[str], metadatas: list[dict] = None, ids: list[str] = None):
        """严格新增：id 已存在会抛 UniqueConstraintError（用于需要强校验的场景）"""
        if ids is None:
            ids = [f"id_{i}" for i in range(len(chunks))]
        embeddings = [self.embedder.embed(c) for c in chunks]
        self.collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        print(f"[add] 已写入 {len(chunks)} 条，collection 现有 {self.collection.count()} 条")

    def update(self, id: str, document: str, metadata: dict = None):
        """改：更新某条内容，向量会重新计算"""
        emb = self.embedder.embed(document)
        self.collection.update(
            ids=[id], embeddings=[emb], documents=[document],
            metadatas=[metadata] if metadata else None,
        )
        print(f"[改] 已更新 {id}")

    def delete(self, ids: list[str]):
        """删：按 id 删除"""
        self.collection.delete(ids=ids)
        print(f"[删] 已删除 {ids}，collection 现有 {self.collection.count()} 条")

    def query(self, text: str, n_results: int = 3):
        """查：问题向量化后检索 Top-K"""
        q_emb = self.embedder.embed(text)
        return self.collection.query(query_embeddings=[q_emb], n_results=n_results)

    def get(self, ids: list[str] = None, limit: int = 10):
        """查看库状态：按 id 取或取前 limit 条"""
        return self.collection.get(ids=ids, limit=limit)
