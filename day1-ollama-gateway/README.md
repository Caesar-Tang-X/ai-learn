# 一. 今日目标：
使用 FastAPI 搭建 Ollama 统一推理网关服务，完成三大工程化封装：**全局异常捕获 + 结构化日志 + 接口鉴权**，使程序直接具备工程可用规范。  

# 二、准备工作
##  步骤 1：安装前置软件  
###  1.1 安装 Python 3.11  
+ 官网下载 Python3.11 安装包：[https://www.python.org/downloads/release/python-3110/](https://link.wtturl.cn/?target=https%3A%2F%2Fwww.python.org%2Fdownloads%2Frelease%2Fpython-3110%2F&scene=im&aid=497858&lang=zh)
+ Windows 安装勾选底部 **Add Python to PATH**，一路下一步；Mac 用官网 pkg 或 brew 安装
+ 打开终端 / CMD 验证安装：

```powershell
python --version
```

### 1.2 安装 Ollama
+ 官网下载：[https://ollama.com/](https://link.wtturl.cn/?target=https%3A%2F%2Follama.com%2F&scene=im&aid=497858&lang=zh) 对应系统安装包
+ 安装完成后新开终端执行：

```powershell
ollama list
```

+  无报错后拉取测试模型：  

```powershell
ollama pull qwen2.5:3b
```

## 步骤 2：创建项目目录 + 虚拟环境
### 2.1 创建项目文件夹
在你方便存放代码的磁盘新建文件夹，命名 `ollama-gateway` 打开终端，cd 进入这个文件夹，示例（Windows cmd）：

```powershell
cd F:\ai-learn\ollama-gateway
```

### 2.2 创建 Python 虚拟环境
执行命令：

```powershell
python -m venv venv
```

### 2.3 激活虚拟环境
Windows CMD

```plain
venv\Scripts\activate
```

Windows PowerShell

```plain
.\venv\Scripts\activate
```

Mac / Linux

```plain
source venv/bin/activate
```

激活成功后，终端前缀会出现 `(venv)` 标识

### 2.4 批量安装依赖包
激活环境后执行安装指令：

```plain
pip install fastapi uvicorn python-multipart pydantic python-jose passlib loguru python-dotenv requests
```

等待全部依赖下载安装完成，无红色报错。

# 三、开发实操
## 步骤 1：设计目录结构
```plain
ollama-gateway/
├── .env
├── venv/
├── logs/
├── core/							# 核心公共模块：日志、异常、权限校验
│   └── __init__.py
├── api/							# 业务路由接口
│   └── __init__.py
├── schemas/					# Pydantic 请求参数校验模型
│   └── __init__.py
└── utils/						# 第三方工具封装（Ollama请求客户端）
    └── __init__.py

```

## 步骤 2：创建项目配置文件 .env
进入 `F:\ai-learn\ollama-gateway` 目录，新建文件，文件名严格为 `.env`（带点，无后缀） 写入下面全部内容：

```properties
# 服务配置
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
# 日志级别
LOG_LEVEL=INFO

# 鉴权密钥（生产务必更换长随机字符串）
API_SECRET_KEY=supersecretkey2026airaggateway

# Ollama原生地址
OLLAMA_BASE_URL=http://localhost:11434
```

## 步骤 3：编写 core/logger.py 全局日志模块
在 `core` 文件夹下新建 `logger.py`，复制下方全部代码保存：

```python
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

```

## 步骤 4：编写 core/exceptions.py 全局异常统一处理
core 目录新建 `exceptions.py`，粘贴下面代码并保存：

```python
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
```

## 步骤 5：编写 core/auth.py 接口鉴权依赖
core 文件夹新建 `auth.py`，复制代码保存：

```python
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
```

## 步骤 6：编写 schemas/request.py 请求参数校验模型
schemas 文件夹新建 `request.py`，复制下方代码保存：

```python
from pydantic import BaseModel, Field

"""
author: tang
description: ollama对话通用请求体模型
"""

# ollama对话通用请求体
class ChatRequest(BaseModel):
    model: str = Field(..., description="ollama模型名称，如qwen2.5:3b")
    prompt: str = Field(..., description="用户提问内容")
    temperature: float = Field(0.7, ge=0, le=1, description="生成温度，0确定性高，1创造力强")
    stream: bool = Field(False, description="是否开启流式输出")
```

## 步骤 7：编写 utils/ollama_client.py Ollama 底层请求封装
utils 文件夹新建 `ollama_client.py`，复制代码保存：

```python
import requests
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

# 全局单例客户端
ollama_client = OllamaClient()

```

## 步骤 8：编写 api/ollama_router.py 业务路由接口
api 文件夹新建 `ollama_router.py`，复制下面代码保存：

```python
import os
import requests
from fastapi import APIRouter, Depends
from schemas.request import ChatRequest
from utils.ollama_client import ollama_client
from core.auth import check_auth
from core.exceptions import resp_format

"""
author: tang
description: ollama模型推理接口路由模块
"""

router = APIRouter(prefix="/v1/ollama", tags=["Ollama统一推理接口"])

@router.post("/chat")
async def ollama_chat(req: ChatRequest, auth=Depends(check_auth)):
    """统一对话推理接口"""
    req_params = req.model_dump()
    result = ollama_client.chat(req_params)
    return resp_format(200, "success", data=result)

@router.get("/models")
async def list_model(auth=Depends(check_auth)):
    """查询本地已拉取模型列表"""
    res = requests.get(f"{os.getenv('OLLAMA_BASE_URL')}/api/tags")
    return resp_format(200, "success", res.json())

```

## 步骤 9：编写项目根目录 main.py 启动入口
在 `ollama-gateway` 根目录新建 `main.py`，复制全部代码保存：

```python
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

```

## 步骤 10：启动服务 + 分模块验证功能
### 10.1 启动服务
确认终端还在 `F:\ai-learn\ollama-gateway`，虚拟环境 `(venv)` 已激活，执行：

```plain
python main.py
```

 正常启动输出示例：  

```plain
2026-07-20 xx:xx:xx | INFO     | main:__main__ - 网关服务启动中 0.0.0.0:8000
INFO:     Will watch for changes in these directories: ['F:\\ai-learn\\ollama-gateway']
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

保持终端窗口不要关闭，新开浏览器访问接口文档：   
[http://127.0.0.1:8000/docs](https://link.wtturl.cn/?target=http%3A%2F%2F127.0.0.1%3A8000%2Fdocs&scene=im&aid=497858&lang=zh)

### 10.2 第一轮测试：健康接口（验证日志模块）
+ 在 docs 页面找到 `GET /health`，点击 Try it out → Execute
+ 返回内容：`{"status":"ok","service":"ollama-gateway"}`
+ 查看终端控制台，打印一条 INFO 日志；同时项目 logs 文件夹自动生成日志文件 日志模块验证通过

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/526089/1784538622888-24e2dd53-ee16-4f8c-9eba-09c707c2ed69.png)

### 10.3 第二轮测试：无鉴权访问模型接口（验证全局异常 + 鉴权）
+ 找到 `GET /v1/ollama/models`，直接 Execute，不带 X-API-Key 请求头
+ 返回结果：`{"code":401,"msg":"非法访问，API密钥错误","data":null}`
+ 终端打印 ERROR 异常日志 鉴权、全局异常捕获验证通过

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/526089/1784538727386-726b4382-8499-4d32-b749-1c699dd20ba5.png)

### 10.4 第三轮测试：携带正确密钥正常查询模型列表
+ 页面右上角点击 **Authorize**
+ 弹窗填入： Key: `X-API-Key` Value: `supersecretkey2026airaggateway` 点击 Authorize 授权
+ 再次执行 `/v1/ollama/models`，返回包含 `qwen2.5:3b`、`bge-m3:latest` 的模型列表 Ollama 客户端连通性验证通过

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/526089/1784538794430-4b2a92f3-513b-4b9b-823f-816911a8fc17.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/526089/1784538814818-51ec6967-f047-4e5a-945a-44363f2a62c8.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/526089/1784538855754-a7889750-f407-40f1-bde2-f6934d04fadd.png)

### 10.5 第四轮测试：对话推理接口
+ 找到 `POST /v1/ollama/chat` → Try it out，填入请求体：

```plain
{
  "model": "qwen2.5:3b",
  "prompt": "简单解释什么是RAG",
  "temperature": 0.7,
  "stream": false
}
```

+  Execute 执行，正常返回大模型回答内容：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/526089/1784538981564-fa24d181-49d7-41f1-ad05-3ce82078e03b.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/526089/1784539012160-4b3b6c42-7b35-40cb-93ae-ca19c374fe24.png)

### 10.6 模拟异常场景测试
+ 关闭本地 Ollama 程序，再次调用 `/chat` 接口，会返回 `code:503 Ollama推理服务未启动`，同时记录错误日志

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/526089/1784539098965-bc0d0fb6-aac5-44d7-87a6-dd96352f4959.png)

# 四. 总结：
从零实现了**工程化 Ollama 推理网关**，解决原生 Ollama 无鉴权、无日志、无统一异常返回、无分层架构的生产痛点，所有代码可直接复用在后续 RAG 项目。  

### 1. 技术栈
Python + FastAPI + Uvicorn + Loguru + Ollama HTTP API

### 2. 四大核心模块功能
1. **结构化日志模块 core/logger.py**
    - 替换原生 logging，彩色控制台输出
    - 日志按天自动切割文件，保留 7 天日志
    - 统一日志对象 `log` 全局复用，自动记录异常堆栈
2. **全局异常捕获 core/exceptions.py**
    - 自定义业务异常 BusinessException
    - 三层异常拦截：参数校验异常 / 业务异常 / 系统未知崩溃
    - 全局统一 JSON 返回格式，前端无需分别处理不同报错
3. **APIKey 统一鉴权 core/auth.py**
    - 基于 FastAPI 依赖注入全局拦截
    - 所有业务接口统一校验请求头 `X-API-Key`
    - 解耦鉴权逻辑，不用每个接口重复写判断
4. **Ollama 底层工具封装 utils/ollama_client.py**
    - 封装 `/api/generate` 对话接口、`/api/tags` 模型列表接口
    - 统一捕获连接失败、超时、请求异常，向上抛业务异常
    - 单例客户端，避免重复创建请求实例

### 3. 分层工程化思想  
```plain
配置(.env) → 核心中间件(日志/异常/鉴权) → 参数模型(schemas) → 工具类(ollama调用) → 路由接口 → 程序入口
```

+ 解耦：修改日志规则不用动接口代码；更换鉴权方式只改 auth.py
+ 可扩展：后续向量库、RAG、Agent 接口直接新增路由即可，不改动原有逻辑
+ 规范：统一返回体、统一日志、统一鉴权，企业私有化项目标准写法

# 三、关键知识点理解复盘
### Q1：为什么不能直接 `from loguru import log`？
loguru 内置变量叫 `logger`，`log` 是我们在 logger.py 自定义导出的全局对象，其他文件必须从 core.logger 导入，否则报导入错误。

### Q2：APIKey 鉴权依赖 `Depends(check_auth)` 的执行逻辑？
FastAPI 每次请求接口前，会优先执行 Depends 内函数；校验失败直接抛出 401 业务异常，被全局异常处理器捕获返回统一报错，接口函数不会执行。

### Q3：Ollama `ollama pull` / `ollama run` / HTTP 接口三者区别？
1. pull：仅下载模型文件到本地缓存
2. run：终端交互式命令行对话，开发调试用
3. 11434 HTTP 服务：后台常驻，代码程序调用专用，无需手动启动会话

### Q4：全局异常处理器的执行顺序？
请求参数校验异常（最先拦截）→ 自定义业务异常 → 通用 Exception 兜底捕获（服务器崩溃、第三方调用失败）。

# 五、Day1 拓展任务
### 1. 新增流式对话接口 `/v1/ollama/stream_chat`，返回 SSE 流式输出
#### 需求目标：
原 `/chat` 接口是一次性等大模型全部生成完再返回，用户等待时间长、体验差。本任务新增流式对话接口，基于 **SSE（Server-Sent Events）** 实现大模型「边生成边返回」，逐字输出，效果与 ChatGPT 打字机一致。

#### 实现原理：
+ Ollama 原生 `/api/generate` 接口在 `stream=true` 时，返回 **NDJSON**（每行一个独立 JSON，以换行分隔），逐帧下发生成内容，最后一帧带 `done=true`
+ 后端用 `requests` 的 `stream=True` + `iter_lines()` 逐行读取，用生成器 `yield` 出每一帧
+ FastAPI 用 `StreamingResponse` + `media_type="text/event-stream"` 将每帧包装成 SSE 格式 `data: {...}\n\n` 持续下发给前端

#### 实现步骤：
##### 步骤 1：utils/ollama_client.py 新增流式请求方法
在 `OllamaClient` 类中新增 `chat_stream`：

```python
import json
...

class OllamaClient:
    ...

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
```

说明：

+ `stream=True` 让 requests 不立即下载全部 body，而是建立可迭代长连接
+ `iter_lines()` 自动按行拆 NDJSON，`json.loads(line)` 把每片解析成字典（含 `response`、`done` 等字段）
+ 复用原有异常处理逻辑，连接失败抛 503，其他异常抛 500

##### 步骤 2：api/ollama_router.py 新增 SSE 路由接口
文件顶部新增导入：

```python
import json
from fastapi.responses import StreamingResponse
```

在路由中新增 `/stream_chat` 接口：

```python
import json
from fastapi.responses import StreamingResponse
...

@router.post("/stream_chat")
async def stream_chat(req: ChatRequest, auth=Depends(check_auth)):
    """流式对话推理接口（SSE 输出）"""
    def event_gen():
        for chunk in ollama_client.chat_stream(req.model_dump()):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    return StreamingResponse(event_gen(), media_type="text/event-stream")
```

说明：

+ `auth=Depends(check_auth)` 复用已有鉴权，逻辑零改动
+ `chat_stream` 内部强制 `stream=True`，不受请求体 stream 字段影响
+ `ensure_ascii=False` 保证中文不被转义；SSE 协议要求每条消息以 `\n\n` 结尾
+ `event_gen` 是同步生成器，Starlette 会自动放到线程池执行，不阻塞事件循环

##### 步骤 3：启动服务 + 联调验证
+ 找到 `POST /v1/ollama/stream_chat` → Try it out，填入请求体：

```plain
{
  "model": "qwen2.5:3b",
  "prompt": "简单解释什么是RAG",
  "temperature": 0.7,
  "stream": false
}
```

+ Execute 执行，正常返回大模型回答内容：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/526089/1784645506228-193cf8f4-82aa-4121-878f-fca40e6d9191.png)

#### 任务小结：
+ 复用工具层 + 鉴权 + 异常三大已有能力，仅新增一个生成器方法 + 一个路由，充分体现分层架构的可扩展性
+ 核心区别：`/chat` 用 `resp.json()` 一次性返回，`/stream_chat` 用 `iter_lines()` 逐帧 `yield`
+ 注意点：`requests` 需加 `stream=True`，curl 测试需加 `-N`，前端需用 fetch 流式读取而非 EventSource

