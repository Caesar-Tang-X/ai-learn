"""Day8 测试数据——从 Day7 samples/README.md 中取的真实段落。

设计原则：静态数据，不受检索波动影响。
这样 Prompt 优化的效果可以精确对比。
"""

# ── 段落 1：取自 samples/README.md 第 7~74 行 ──
# 内容：步骤 1 安装 Python 3.11 + 步骤 2 安装 Ollama + 步骤 2 创建虚拟环境 + 安装依赖
CONTEXT_1 = """###  步骤 1：安装前置软件  
####  1.1 安装 Python 3.11  
+ 官网下载 Python3.11 安装包：https://www.python.org/downloads/release/python-3110/
+ Windows 安装勾选底部 Add Python to PATH，一路下一步
+ 打开终端 / CMD 验证安装：python --version

#### 1.2 安装 Ollama
+ 官网下载：https://ollama.com/ 对应系统安装包
+ 安装完成后新开终端执行：ollama list
+ 无报错后拉取测试模型：ollama pull qwen2.5:3b

### 步骤 2：创建项目目录 + 虚拟环境
#### 2.1 创建项目文件夹
在你方便存放代码的磁盘新建文件夹，命名 ollama-gateway，cd 进入

#### 2.2 创建 Python 虚拟环境
执行命令：python -m venv venv

#### 2.3 激活虚拟环境
Windows CMD：venv\\Scripts\\activate
Windows PowerShell：.\\venv\\Scripts\\activate

#### 2.4 批量安装依赖包
激活环境后执行：pip install fastapi uvicorn python-multipart pydantic python-jose passlib loguru python-dotenv requests
等待全部依赖下载安装完成，无红色报错。"""

# ── 段落 2：取自 samples/README.md 第 211~234 行 ──
# 内容：步骤 6 编写 core/auth.py 接口鉴权依赖
CONTEXT_2 = """### 步骤 6：编写 core/auth.py 接口鉴权依赖
core 文件夹新建 auth.py，复制代码保存：

from fastapi import Depends
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv
import os
from core.exceptions import BusinessException

load_dotenv()
API_KEY = os.getenv("API_SECRET_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# 全局鉴权校验依赖
async def check_auth(api_key: str = Depends(api_key_header)):
    if not api_key or api_key != API_KEY:
        raise BusinessException(code=401, msg="非法访问，API密钥错误")
    return True"""

# ── 测试用例 ──
TEST_CASES = [
    {
        "id": "case1",
        "question": "如何安装依赖",
        "context": CONTEXT_1,
        "expect_keywords": ["pip install", "venv"],  # 答案中应包含的关键词
        "type": "multi_step"
    },
    {
        "id": "case2",
        "question": "鉴权是怎么实现的",
        "context": CONTEXT_2,
        "expect_keywords": ["APIKeyHeader", "Depends", "check_auth"],
        "type": "extraction"
    },
]

