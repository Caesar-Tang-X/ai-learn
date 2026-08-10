"""
把 Ollama 对话模型封装为 LangChain 的 ChatModel（BaseChatModel）。
走 Ollama /api/chat 协议，原生支持 bind_tools（工具调用），供 Agent 使用。
默认模型 qwen2.5:3b（本项目约定）。
"""
from langchain_ollama import ChatOllama


def build_chat_model(model: str = "qwen2.5:3b",
                     base_url: str = "http://localhost:11434",
                     temperature: float = 0.0) -> ChatOllama:
    # stream=False：避免 qwen2.5 在「tools + 流式」场景返回空响应（langchain-ollama 已知问题）
    return ChatOllama(model=model, base_url=base_url,
                     temperature=temperature, stream=False)
