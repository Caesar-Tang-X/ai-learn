"""
API 请求与响应模型。
"""
from pydantic import BaseModel


class IngestRequest(BaseModel):
    path: str
    source: str | None = None


class IngestResponse(BaseModel):
    chunks: int


class AskRequest(BaseModel):
    query: str
    rerank_top_n: int | None = None


class AskResponse(BaseModel):
    answer: str
