"""KB 查库工具：让 Agent 能查询 Day7 建好的 PGVector 知识库（docs_readme）。

这是 Day9 的第一个自定义工具。它包装了 Day7 的 rag_chain.get_vectorstore()，
把"语义检索"暴露成一个 Agent 可调用的函数。
"""
from typing import List

from langchain_core.documents import Document
from langchain_core.tools import tool

from core.rag_chain import build_vectorstore

_KB = None  # 模块级缓存，避免每次调用都重建连接


def _get_kb():
    """懒加载并缓存 vectorstore 连接（复用 Day7 的 get_vectorstore）。"""
    global _KB
    if _KB is None:
        _KB = build_vectorstore("samples/all_days.md")
    return _KB


def _format(docs: List[Document]) -> str:
    """把 Document 列表拼成 Agent 能读的字符串，保留出处。"""
    if not docs:
        return "（知识库中没有找到相关内容）"
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "未知来源")
        parts.append(f"[片段{i}] 来源：{src}\n{d.page_content}")
    return "\n\n".join(parts)


@tool
def search_kb(query: str) -> str:
    """查询项目内部知识库（docs_readme）。

    当用户的问题涉及本项目 README、各 Day 的学习内容、技术方案、代码说明时，
    应使用此工具查找权威资料。输入：自然语言查询语句；返回：相关的文档片段（含来源）。
    """
    try:
        vs = _get_kb()
        docs = vs.similarity_search(query, k=3)
        return _format(docs)
    except Exception as e:
        return f"（查库工具出错：{e}）"


if __name__ == "__main__":
    # 自检：直接跑一下，确认能连上 Day7 的库
    print(search_kb.invoke("Day7 做了什么？PGVector 怎么用？"))
