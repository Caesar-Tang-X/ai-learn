from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv
import os
from core.exceptions import register_exception
from core.logger import log
from api.ollama_router import router as ollama_router

"""
author: tang
description: 主程序入口，初始化FastAPI应用、加载环境变量、挂载路由、启动服务
"""

load_dotenv()
HOST = os.getenv("SERVER_HOST")
PORT = int(os.getenv("SERVER_PORT"))

# 初始化FastAPI应用
app = FastAPI(
    title="Ollama统一推理网关服务",
    description="RAG项目底层推理统一封装服务，包含日志、鉴权、全局异常",
    version="1.0-Day1"
)

# 注册全局异常捕获
register_exception(app)
# 挂载路由
app.include_router(ollama_router)

# 健康检查接口（无需鉴权）
@app.get("/health")
async def health():
    log.info("健康检查访问成功")
    return {"status": "ok", "service": "ollama-gateway"}

if __name__ == "__main__":
    log.info(f"网关服务启动中 {HOST}:{PORT}")
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=True,  # 开发环境热更新，生产关闭
        log_level="info"
    )
