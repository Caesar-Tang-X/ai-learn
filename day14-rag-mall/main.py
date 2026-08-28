"""
day14-rag-mall 统一启动入口。
用法（在项目根目录 day14-rag-mall/ 执行）：
    python main.py
或：
    uvicorn api.app:app --host 0.0.0.0 --port 8000
本文件等价于后者，但更方便直接 python 启动。
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
