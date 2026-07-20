import os
import requests
from fastapi import APIRouter, Depends
from schemas.request import ChatRequest
from utils.ollama_client import ollama_client
from core.auth import check_auth
from core.exceptions import resp_format

"""
author: tang
description: ollama模型推理接口路由模块
"""

router = APIRouter(prefix="/v1/ollama", tags=["Ollama统一推理接口"])

@router.post("/chat")
async def ollama_chat(req: ChatRequest, auth=Depends(check_auth)):
    """统一对话推理接口"""
    req_params = req.model_dump()
    result = ollama_client.chat(req_params)
    return resp_format(200, "success", data=result)

@router.get("/models")
async def list_model(auth=Depends(check_auth)):
    """查询本地已拉取模型列表"""
    res = requests.get(f"{os.getenv('OLLAMA_BASE_URL')}/api/tags")
    return resp_format(200, "success", res.json())
