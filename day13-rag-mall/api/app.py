"""
FastAPI 应用：暴露 /ask 接口，接收应用端 prompt + filters。
"""
import asyncio

from fastapi import FastAPI

from api.schemas import AskRequest, AskResponse
from query import ask

app = FastAPI(title="Mall RAG API", version="1.0.0")


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(req: AskRequest) -> AskResponse:
    """基于商品知识库回答（向量检索 + metadata 硬过滤 + LLM）。"""
    answer = await ask(req.prompt, req.filters, req.top_k)
    return AskResponse(answer=answer)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
