from pydantic import BaseModel, Field

"""
author: tang
description: ollama对话通用请求体模型
"""

# ollama对话通用请求体
class ChatRequest(BaseModel):
    model: str = Field(..., description="ollama模型名称，如qwen2.5:3b")
    prompt: str = Field(..., description="用户提问内容")
    temperature: float = Field(0.7, ge=0, le=1, description="生成温度，0确定性高，1创造力强")
    stream: bool = Field(False, description="是否开启流式输出")