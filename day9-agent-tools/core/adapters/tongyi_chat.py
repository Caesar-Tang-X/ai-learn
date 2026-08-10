"""阿里云百炼通义千问 ChatModel 封装（OpenAI 兼容端点，原生支持 tool calling）。"""
import os
from langchain_openai import ChatOpenAI


def build_tongyi_chat(model: str = "qwen3.7-plus",
                      temperature: float = 0.0) -> ChatOpenAI:
    """构造通义千问 ChatModel，供框架版 Agent（create_agent）使用。
    走百炼 OpenAI 兼容端点；依赖环境变量 DASHSCOPE_API_KEY（不写进代码）。"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到 DASHSCOPE_API_KEY，请在 .env 中配置")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=temperature,
    )
