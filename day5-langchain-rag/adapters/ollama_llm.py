"""
author: tang
description: 把 Day1 的 Ollama /api/generate 包装成 LangChain 的 LLM 组件。
             这样 LangChain 的 Chain / RetrievalQA 就能直接复用本地对话模型。
"""

import os
from typing import List, Optional

import requests
from dotenv import load_dotenv
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

load_dotenv()


class OllamaLLM(LLM):
    """LangChain LLM 适配器：内部委托给 Ollama 的 /api/generate（复用 Day1 思想）"""

    # 用 pydantic 字段声明，LangChain 的 LLM 基类基于 pydantic
    model: str = "qwen2.5:3b"          # 对话模型名，改成你实际 pull 的
    base_url: str = "http://localhost:11434"
    timeout: int = 180

    @property
    def _llm_type(self) -> str:
        # 给 LangChain 内部标识这个 LLM 类型，随便取个不重名的字符串
        return "ollama_llm_qwen2.5_3b"

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> str:
        """LangChain 调它生成回复，内部就是 Day1 的 /api/generate"""
        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()["response"]
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Ollama 服务未启动，请先执行 ollama serve")
        except Exception as e:
            raise RuntimeError(f"生成失败: {str(e)}")
