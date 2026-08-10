"""步骤四：外部公开 HTTP 接口调用工具。"""
from langchain_core.tools import tool
import requests


@tool
def call_api(url: str) -> str:
    """调用一个外部公开 HTTP 接口（GET 请求），返回响应内容。
    输入：完整的接口 URL（如 https://jsonplaceholder.typicode.com/posts/1）。
    返回：接口响应文本（最长 2000 字符），或出错原因说明。
    适用场景：需要实时联网获取外部数据、测试接口连通性时调用。"""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        text = resp.text
        if len(text) > 2000:
            text = text[:2000] + "\n...（已截断，原文过长）"
        return f"[接口 {resp.status_code}] {url}\n{text}"
    except Exception as e:
        return f"（调用接口出错：{e}）"


if __name__ == "__main__":
    print(call_api.invoke("https://jsonplaceholder.typicode.com/posts/1"))
