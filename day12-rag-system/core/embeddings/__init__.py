"""
embeddings 包：文本向量化（bge-m3）。
"""
from core.embeddings.client import EmbeddingClient, get_embedding_client

__all__ = ["EmbeddingClient", "get_embedding_client"]
