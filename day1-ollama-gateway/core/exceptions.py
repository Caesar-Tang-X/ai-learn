from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import RequestValidationError
from core.logger import log

"""
author: tang
description: 全局异常处理模块，用于处理服务运行时的异常情况
"""

# 自定义业务异常类
class BusinessException(Exception):
    def __init__(self, code: int, msg: str):
        self.code = code
        self.msg = msg

# 统一返回体格式化函数
def resp_format(code: int, msg: str, data=None):
    return {"code": code, "msg": msg, "data": data}

# 捕获自定义业务异常
async def business_exception_handler(request: Request, exc: BusinessException):
    log.error(f"业务异常: code={exc.code}, msg={exc.msg}, path={request.url.path}")
    return JSONResponse(content=resp_format(exc.code, exc.msg))

# 捕获参数校验异常
async def vali_exception_handler(request: Request, exc: RequestValidationError):
    err_info = exc.errors()[0]
    msg = f"参数错误：{err_info['loc']} {err_info['msg']}"
    log.error(f"参数校验异常: {msg}, path={request.url.path}")
    return JSONResponse(content=resp_format(400, msg))

# 捕获未知系统异常
async def global_exception_handler(request: Request, exc: Exception):
    log.exception(f"系统未知异常 path={request.url.path}")
    return JSONResponse(content=resp_format(500, "服务器内部错误，请联系管理员"))

# 批量注册所有异常处理器
def register_exception(app):
    app.add_exception_handler(BusinessException, business_exception_handler)
    app.add_exception_handler(RequestValidationError, vali_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)
