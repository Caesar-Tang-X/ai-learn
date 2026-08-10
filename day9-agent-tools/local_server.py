"""步骤五：本地 FastAPI 服务，供 Agent 通过 local_api_tool 调用。"""
from fastapi import FastAPI, Query
import uvicorn

app = FastAPI(title="Day9 本地服务")


@app.get("/status")
def status():
    """返回一个示例业务状态，模拟 Agent 可查询的本地系统状态。"""
    return {
        "service": "day9-local-demo",
        "online": True,
        "task_count": 3,
        "message": "本地服务运行正常",
    }


@app.get("/echo")
def echo(text: str = Query(..., description="要回显的内容")):
    """把传入文本原样返回，演示 Agent 向本地服务传参。"""
    return {"you_said": text}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
