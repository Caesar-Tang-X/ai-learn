import requests
import json
from dotenv import load_dotenv
import os
from core.exceptions import BusinessException
from core.logger import log

"""
author: tang
description: ollama模型推理客户端，用于调用ollama模型推理接口
"""

load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL")

class OllamaClient:
    @staticmethod
    def chat(params: dict):
        url = f"{OLLAMA_URL}/api/generate"
        try:
            resp = requests.post(url, json=params, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            log.error("连接Ollama服务失败，请检查ollama是否启动")
            raise BusinessException(503, "Ollama推理服务未启动或无法连接")
        except Exception as e:
            log.error(f"调用ollama异常: {str(e)}")
            raise BusinessException(500, f"模型推理失败: {str(e)}")

    @staticmethod
    def chat_stream(params: dict):
        """流式对话：逐行 yield Ollama 返回的 NDJSON 片段"""
        url = f"{OLLAMA_URL}/api/generate"
        try:
            with requests.post(
                url, json={**params, "stream": True}, stream=True, timeout=120
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:                       # 跳过空行
                        yield json.loads(line)     # 每行是一个独立 JSON
        except requests.exceptions.ConnectionError:
            log.error("连接Ollama服务失败，请检查ollama是否启动")
            raise BusinessException(503, "Ollama推理服务未启动或无法连接")
        except Exception as e:
            log.error(f"流式调用ollama异常: {str(e)}")
            raise BusinessException(500, f"模型推理失败: {str(e)}")

# 全局单例客户端
ollama_client = OllamaClient()
