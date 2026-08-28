"""
检索模块门面：对外只暴露 retrieve 主入口（多路召回→融合→重排→截断）。
内部含 vector/fulltext（召回）、fusion（编排）、budget/count/filters（通用解析与过滤），新增召回路在内部扩展，上层零改动。
"""
from core.retrieval.fusion import retrieve

__all__ = ["retrieve"]
