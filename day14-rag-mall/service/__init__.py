"""
服务层门面：对外只暴露 answer 流式生成器。
"""
from service.rag_service import answer

__all__ = ["answer"]
