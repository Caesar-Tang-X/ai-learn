from fastapi import Depends
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv
import os
from core.exceptions import BusinessException

"""
author: tang
description: 全局鉴权模块，用于校验API密钥
"""

load_dotenv()
API_KEY = os.getenv("API_SECRET_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# 全局鉴权校验依赖
async def check_auth(api_key: str = Depends(api_key_header)):
    if not api_key or api_key != API_KEY:
        raise BusinessException(code=401, msg="非法访问，API密钥错误")
    return True