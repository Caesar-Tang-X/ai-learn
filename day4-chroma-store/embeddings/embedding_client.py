"""
author: tang
description: 本地 Embedding 客户端，调用 Ollama /api/embeddings 把文字转成向量
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

class EmbeddingClient:
    def __init__(self, model: str = "bge-m3"):
        # Embedding 模型，默认 bge-m3（中文效果好，维度 1024）
        self.model = model
        # Ollama 地址：优先读 .env 的 OLLAMA_BASE_URL，没有就兜底本地默认端口
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def embed(self, text: str) -> list:
        """
        输入一段文字，返回对应的向量（一个浮点数列表）。
        例：embed("你好") -> [0.012, -0.034, ..., 共 1024 个数]
        """
        # 注意接口是 /api/embeddings（带 s），和 Day1 对话的 /api/generate 不同
        url = f"{self.base_url}/api/embeddings"
        # 请求体：模型名字 + 要向量化的文字
        payload = {"model": self.model, "prompt": text}
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()  # HTTP 非 200 时抛异常
            # 返回 JSON 的 "embedding" 字段就是向量数组
            return resp.json()["embedding"]
        except requests.exceptions.ConnectionError:
            # Ollama 没启动，最常见的错误
            raise RuntimeError("Ollama 服务未启动或无法连接，请先执行 ollama serve")
        except Exception as e:
            raise RuntimeError(f"向量化失败: {str(e)}")
