"""Day10 步骤二：给工程师 Agent 挂的轻量知识库工具（读取本地文本，无额外依赖）"""
from autogen_core.tools import FunctionTool
from pathlib import Path


def search_project_notes(query: str) -> str:
    """根据关键词查询项目学习笔记。

    Args:
        query: 要查找的关键词，例如 "Day7"

    Returns:
        包含该关键词的笔记行；若未找到返回提示。
    """
    notes_path = Path(__file__).parent / "samples" / "project_notes.txt"
    if not notes_path.exists():
        return "知识库文件不存在。"
    text = notes_path.read_text(encoding="utf-8")
    matched = [line for line in text.splitlines() if query.lower() in line.lower()]
    if not matched:
        return f"未找到与 '{query}' 相关的笔记。"
    return "\n".join(matched)


# 用 FunctionTool 包装成 AutoGen 可用的工具
project_kb_tool = FunctionTool(
    func=search_project_notes,
    name="search_project_notes",
    description="查询项目学习笔记知识库，传入关键词（如 Day7）返回对应内容。",
)
