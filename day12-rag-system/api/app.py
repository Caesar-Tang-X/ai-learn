"""
FastAPI 应用：暴露 ingest 与 ask 接口。
"""
import asyncio

from fastapi import FastAPI

from agents import ask as agent_ask
from api.schemas import AskRequest, AskResponse, IngestRequest, IngestResponse
from core.pipeline import ingest_file

app = FastAPI(title="Private RAG System", version="1.0.0")


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    """将本地文档文件入库（加载→清洗→分块→向量化→PGVector）。"""
    chunks = await asyncio.to_thread(ingest_file, req.path, source=req.source)
    return IngestResponse(chunks=chunks)


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """基于私有知识库回答用户问题（混合检索 + 多智能体）。"""
    answer = await asyncio.to_thread(agent_ask, req.query, req.rerank_top_n)
    return AskResponse(answer=answer)


@app.get("/health")
async def health() -> dict:
    """健康检查。"""
    return {"status": "ok"}

