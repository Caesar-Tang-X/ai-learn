"""步骤五：调用本地 FastAPI 服务的工具。"""
from langchain_core.tools import tool
import requests


@tool
def local_status() -> str:
    """查询本地 FastAPI 服务（127.0.0.1:8000）的运行状态。
    返回本地服务的业务状态 JSON，或出错原因。
    适用场景：需要确认本地系统/服务是否在线、获取本地状态时调用。"""
    try:
        resp = requests.get("http://127.0.0.1:8000/status", timeout=5)
        resp.raise_for_status()
        return f"[本地服务] {resp.text}"
    except Exception as e:
        return f"（调用本地服务出错：{e}）"


if __name__ == "__main__":
    print(local_status.invoke(""))
