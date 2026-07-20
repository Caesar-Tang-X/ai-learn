from loguru import logger
import sys
from dotenv import load_dotenv
import os

"""
author: tang
description: 全局日志模块，用于记录服务运行时的全局日志
"""

load_dotenv()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# 移除默认控制台输出
logger.remove()
# 控制台日志格式
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=LOG_LEVEL,
    enqueue=True
)
# 文件日志，按天分割，保留7天
logger.add(
    "logs/gateway_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="7 days",
    encoding="utf-8",
    level=LOG_LEVEL,
    enqueue=True
)

# 对外导出日志对象
log = logger
