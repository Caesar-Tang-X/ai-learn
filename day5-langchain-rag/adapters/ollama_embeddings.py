"""
author: tang
description: 把 Day3 的 EmbeddingClient 包装成 LangChain 的 Embeddings 组件。
             这样 LangChain 的 VectorStore / Retriever 就能直接复用 bge-m3（经 Ollama /api/embeddings）。
"""

from typing import List

from langchain_core.embeddings import Embeddings

# 复用 Day3 成果：本地 Embedding 客户端（已复制到 day5/embeddings/）
from embeddings.embedding_client import EmbeddingClient


class OllamaEmbeddings(Embeddings):
    """LangChain Embeddings 适配器：内部委托给 Day3 的 EmbeddingClient"""

    def __init__(self, model: str = "bge-m3"):
        # 内部持有 Day3 的客户端实例
        self.client = EmbeddingClient(model=model)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain 调它给一批文档（chunks）算向量，用于入库"""
        return [self.client.embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        """LangChain 调它给"用户问题"算向量，用于检索"""
        return self.client.embed(text)
