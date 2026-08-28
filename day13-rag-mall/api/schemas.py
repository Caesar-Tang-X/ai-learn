"""API 请求/响应模型。"""
from typing import Any

from pydantic import BaseModel


class AskRequest(BaseModel):
    """应用端调用：prompt 文本 + 结构化过滤条件。"""
    prompt: str
    filters: dict[str, Any] | None = None
    top_k: int = 20


class AskResponse(BaseModel):
    answer: list
