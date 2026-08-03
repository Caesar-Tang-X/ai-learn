# 一. 今日目标

Day1~Day7 一直在优化「检索质量」——给 LLM 更好的上下文。Day8 转向优化「Prompt 质量」——让 LLM 更好地利用上下文。不改模型、不改数据，只改进「怎么问」。

三种 Prompt Engineering 核心策略 + 封装：

1. **CoT（Chain of Thought）思维链**：强制 LLM 先推理再回答，用更多 token 换更高准确率
2. **Few-Shot Prompting 少样本提示**：给范例教 LLM 输出格式，比口头描述格式更有效
3. **结构化 JSON 输出**：约束 schema + 三层容错解析 + 解析失败重试
4. **封装 Prompt 模板类**：把上述策略封装为可复用类 + Router 自动选择策略

最终 `main.py` 五组实验并行对比（同一 LLM、同一问题、同一 context），唯一变量是 Prompt 策略。

# 二、先想清楚几个问题

#### Q1：Day7 已经在优化检索了，Day8 为什么还要优化 Prompt？

Day7 做的事：从资料中找最相关的段落 → 拼进 context → 给 LLM。

但 context 再好，如果 prompt 没有引导 LLM 正确利用它，效果仍然打折。Day7 的 prompt 只说了「你是谁」和「参考资料」，没有告诉 LLM「怎么思考」。Day8 补这个缺口——检索负责「给什么」，Prompt 负责「怎么用」。

#### Q2：CoT 为什么有效？什么场景有效？

LLM 逐 token 生成。直接跳到答案时，只有第一个 token 的思考时间。先写推理步骤，每个推理 token 都在为后续铺路。

CoT 适合**多步骤推理**（计算、排序、比较、流程总结）。不适合**直接信息提取**（答案就在资料里躺着）——此时 CoT 的「分析」步骤反而浪费 token。

#### Q3：Few-Shot 的范例会不会限制 LLM？

会。范例不仅教格式，还教「关注什么」。如果范例关注的是命令执行，LLM 可能忽略环境准备步骤。Few-Shot 适合**格式要求严格**的场景（输出 JSON、步骤列表），不适合「答案内容完全由资料决定」的开放式问答。

#### Q4：3B 模型输出 JSON 为什么不稳定？

JSON 要求精确的语法（引号、逗号、花括号配对）。3B 模型参数少，对格式约束的遵循能力弱。`JsonPrompt` 做了三层容错解析 + 重试，但复杂 schema 的失败率仍然很高。简化 schema（少字段、简单类型）是最有效的提升手段。

#### Q5：为什么 `RouterPrompt` 只在 CoT 和 Simple 之间选，不选 Few-Shot 和 JSON？

Few-Shot 的副作用（范例限制注意力）让它不适合作为通用默认策略。JSON 的稳定性问题让它不适合所有场景。CoT 和 Simple 是覆盖面最广、副作用最小的两种策略——一个处理多步骤推理，一个处理直接信息提取。

# 三、准备工作

## 步骤 1：新建目录 + 复制复用包

```powershell
cd f:\ai-learn
mkdir day8-prompt-engineering
cd day8-prompt-engineering

mkdir adapters
mkdir prompts

Copy-Item ..\day7-pgvector-prod\adapters\ollama_llm.py adapters\
```

Day8 不需要向量库、不需要 embedding、不需要检索模块。只复制 `OllamaLLM` 适配器。

## 步骤 2：安装依赖

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install langchain-core python-dotenv requests
```

| 包 | 作用 |
|---|---|
| `langchain-core` | 提供 `PromptTemplate` 和 `StrOutputParser` |
| `requests` | `OllamaLLM` 底层调用 Ollama API |

## 最终目录结构

```plain
day8-prompt-engineering/
├── main.py                   # 步骤 7：五组实验并排对比
├── test_data.py              # 测试数据（静态 context + 标注答案）
├── cot_demo.py               # 步骤 3：CoT 单策略演示
├── few_shot_demo.py          # 步骤 4：Few-Shot 单策略演示
├── json_demo.py              # 步骤 5：JSON 输出单策略演示
├── prompt_runner.py          # 步骤 6：封装类验证
├── prompts/
│   ├── __init__.py            # 空文件
│   └── strategies.py          # 五种 Prompt 策略类
├── adapters/
│   └── ollama_llm.py          # 从 Day7 复制
└── .env                       # 可选：OLLAMA_BASE_URL
```

# 四、开发实操

## 步骤 0：`test_data.py`（测试数据）

从 Day7 `samples/README.md` 取两个真实段落作为静态测试数据。不受检索波动影响，Prompt 优化效果可精确对比。

两个测试用例的设计思路：

| case | 问题 | type | 为什么 |
|---|---|---|---|
| case1 | 如何安装依赖 | `multi_step` | 答案跨越 4 个小节（创建 venv → 激活 → pip install → 验证），需多步骤整合 |
| case2 | 鉴权是怎么实现的 | `extraction` | 答案就在「步骤 6」一个代码块里，直接提取即可 |

`expect_keywords` 用于客观衡量答案覆盖度——比肉眼判断更可靠。

```python
"""Day8 测试数据——从 Day7 samples/README.md 中取的真实段落。"""

# ── 段落 1：samples/README.md 第 7~74 行 ──
# 内容：安装 Python + Ollama + 创建虚拟环境 + 安装依赖
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

# ── 段落 2：samples/README.md 第 211~234 行 ──
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
        "expect_keywords": ["pip install", "venv"],
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

# ========== 自检 ==========
if __name__ == "__main__":
    print(f"加载 {len(TEST_CASES)} 个测试用例")
    for t in TEST_CASES:
        print(f"  {t['id']}: {t['question']}")
```

#### 0.1 运行结果

```plain
python test_data.py
```

```
加载 2 个测试用例
  case1: 如何安装依赖
  case2: 鉴权是怎么实现的
```

## 步骤 1：`cot_demo.py`（CoT 思维链演示）

同一 LLM、同一问题、同一 context，唯一变量是 Prompt 是否包含「逐步分析」指令。

关键设计：CoT Prompt 先【回答】再【分析】——答案不会被分析截断，分析作为补充说明。

```python
"""CoT（Chain of Thought）思维链演示。"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from adapters.ollama_llm import OllamaLLM
from test_data import TEST_CASES


# ── 普通 Prompt（对照，来自 Day7 main.py 第 26~34 行）──
BASELINE_PROMPT = PromptTemplate.from_template(
    """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。

【资料】
{context}

【问题】{question}

【回答】"""
)

# ── CoT Prompt ──
COT_PROMPT = PromptTemplate.from_template(
    """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。

【资料】
{context}

【问题】{question}

请按以下步骤回答：
1. 【回答】先给出完整答案
2. 【分析】简要说明答案的依据（不超过 80 字）

【回答】
【分析】"""
)


def run_comparison():
    llm = OllamaLLM()

    for case in TEST_CASES:
        question = case["question"]
        context = case["context"]

        print(f"\n{'='*60}")
        print(f"问题：{question}")
        print(f"{'='*60}")

        chain_baseline = BASELINE_PROMPT | llm | StrOutputParser()
        answer_baseline = chain_baseline.invoke({
            "context": context, "question": question
        })
        print(f"\n── 普通 Prompt ──")
        print(answer_baseline[:300])

        chain_cot = COT_PROMPT | llm | StrOutputParser()
        answer_cot = chain_cot.invoke({
            "context": context, "question": question
        })
        print(f"\n── CoT Prompt ──")
        print(answer_cot[:500])

        keywords = case.get("expect_keywords", [])
        qtype = case.get("type", "")
        if keywords:
            print(f"\n期望关键词: {keywords}")
            baseline_hit = [kw for kw in keywords if kw.lower() in answer_baseline.lower()]
            cot_hit = [kw for kw in keywords if kw.lower() in answer_cot.lower()]
            print(f"普通 Prompt 命中: {baseline_hit} ({len(baseline_hit)}/{len(keywords)})")
            print(f"CoT Prompt  命中: {cot_hit} ({len(cot_hit)}/{len(keywords)})")
            if qtype == "extraction":
                print("（注：此问题类型为「信息提取」，CoT 不一定优于普通 Prompt）")


if __name__ == "__main__":
    run_comparison()
```

#### 1.1 运行结果（本机实测）

```plain
python cot_demo.py
```

```
============================================================
问题：如何安装依赖
============================================================

── 普通 Prompt ──
为了在项目中安装所需的依赖包，请按照以下步骤操作：
1. 激活虚拟环境...
2. 执行 pip install fastapi uvicorn...

── CoT Prompt ──
【回答】
首先，您需要激活虚拟环境。在Windows CMD下执行 venv\Scripts\activate...
然后，在虚拟环境中安装依赖包：pip install fastapi uvicorn...
【分析】答案基于提供的资料中步骤2.4的内容。

期望关键词: ['pip install', 'venv']
普通 Prompt 命中: ['pip install', 'venv'] (2/2)
CoT Prompt  命中: ['pip install', 'venv'] (2/2)

============================================================
问题：鉴权是怎么实现的
============================================================

期望关键词: ['APIKeyHeader', 'Depends', 'check_auth']
普通 Prompt 命中: ['Depends', 'check_auth'] (2/3)
CoT Prompt  命中: ['Depends', 'check_auth'] (2/3)
（注：此问题类型为「信息提取」，CoT 不一定优于普通 Prompt）
```

> 解读：case1（multi_step）CoT 完整覆盖了 venv + pip install。case2（extraction）两者都漏了 `APIKeyHeader`（import 行被 LLM 当作非正文略过），CoT 的分析步骤没有帮助——印证了「CoT 不擅长直接信息提取」。

## 步骤 2：`few_shot_demo.py`（Few-Shot 演示）

范例选「如何启动服务」而非「如何安装依赖」——同类不同题，避免 LLM 直接抄答案。

```python
"""Few-Shot Prompting 演示。"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from adapters.ollama_llm import OllamaLLM
from test_data import TEST_CASES


ZERO_SHOT_PROMPT = PromptTemplate.from_template(
    """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。

【资料】
{context}

【问题】{question}

【回答】"""
)

FEW_SHOT_PROMPT = PromptTemplate.from_template(
    """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。

回答格式要求：按步骤编号列出，每步格式为「步骤X：操作内容（涉及的命令或工具）」

示例——
问题：如何启动服务
资料：...在项目根目录执行 uvicorn main:app --reload --host 0.0.0.0 --port 8000...
回答：
步骤1：确保虚拟环境已激活
步骤2：在项目根目录执行 uvicorn main:app --reload --host 0.0.0.0 --port 8000
步骤3：打开浏览器访问 http://localhost:8000/docs 验证

现在回答以下问题：
【资料】
{context}

【问题】{question}

【回答】"""
)


def run_comparison():
    llm = OllamaLLM()

    for case in TEST_CASES:
        question = case["question"]
        context = case["context"]
        qtype = case.get("type", "")

        print(f"\n{'='*60}")
        print(f"问题：{question}（类型：{qtype}）")
        print(f"{'='*60}")

        chain_zero = ZERO_SHOT_PROMPT | llm | StrOutputParser()
        answer_zero = chain_zero.invoke({
            "context": context, "question": question
        })
        print(f"\n── Zero-Shot（无范例）──")
        print(answer_zero[:300])

        chain_few = FEW_SHOT_PROMPT | llm | StrOutputParser()
        answer_few = chain_few.invoke({
            "context": context, "question": question
        })
        print(f"\n── Few-Shot（有范例）──")
        print(answer_few[:500])

        keywords = case.get("expect_keywords", [])
        if keywords:
            zero_hit = [kw for kw in keywords if kw.lower() in answer_zero.lower()]
            few_hit = [kw for kw in keywords if kw.lower() in answer_few.lower()]
            print(f"\n期望关键词: {keywords}")
            print(f"Zero-Shot 命中: {zero_hit} ({len(zero_hit)}/{len(keywords)})")
            print(f"Few-Shot  命中: {few_hit} ({len(few_hit)}/{len(keywords)})")


if __name__ == "__main__":
    run_comparison()
```

#### 2.1 运行结果（本机实测）

```plain
python few_shot_demo.py
```

```
============================================================
问题：如何安装依赖（类型：multi_step）
============================================================

── Zero-Shot（无范例）──
1. 激活虚拟环境...
2. 执行 pip install fastapi uvicorn...

── Few-Shot（有范例）──
步骤1：确保虚拟环境已激活
步骤2：在 ollama-gateway 文件夹中执行 pip install ...

期望关键词: ['pip install', 'venv']
Zero-Shot 命中: ['pip install', 'venv'] (2/2)
Few-Shot  命中: ['pip install'] (1/2)

============================================================
问题：鉴权是怎么实现的（类型：extraction）
============================================================

期望关键词: ['APIKeyHeader', 'Depends', 'check_auth']
Zero-Shot 命中: ['Depends', 'check_auth'] (2/3)
Few-Shot  命中: ['APIKeyHeader', 'Depends', 'check_auth'] (3/3)
```

> 解读：case1 中 Few-Shot 反而漏了 `venv`——范例的「步骤X」格式让 LLM 聚焦于命令执行，忽略了环境准备。case2 中 Few-Shot 的步骤拆解格式恰好迫使 LLM 逐行讲解代码，import 行里的 `APIKeyHeader` 也被提取出来了。**Few-Shot 的范例不仅教格式，还教「关注什么」——这是双刃剑。**

## 步骤 3：`json_demo.py`（JSON 输出演示）

核心机制：约束 schema → 三层容错解析（直接 JSON / markdown 提取 / 正则提取）→ 解析失败时反馈错误信息重试。

```python
"""结构化 JSON 输出演示。"""

import json, re
from typing import Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from adapters.ollama_llm import OllamaLLM
from test_data import TEST_CASES


FREE_TEXT_PROMPT = PromptTemplate.from_template(
    """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。

【资料】
{context}

【问题】{question}

【回答】"""
)

JSON_PROMPT = PromptTemplate.from_template(
    """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。

【资料】
{context}

【问题】{question}

请按以下 JSON 格式回答（仅输出 JSON，不要其他文字）：
{{
    "steps": ["步骤1描述", "步骤2描述", ...],
    "tools": ["用到的工具或命令", ...],
    "summary": "一句话总结"
}}

JSON："""
)

FIX_PROMPT = PromptTemplate.from_template(
    """你之前输出了以下内容，但它不是合法的 JSON：

{raw_output}

解析错误：{error}

请修正后重新输出合法的 JSON（仅输出 JSON，不要其他文字）：

JSON："""
)


def parse_json(text: str) -> Optional[dict]:
    """三层容错解析 JSON（与 Day7 self_query.py 第 64~83 行逻辑一致）。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def ask_json(llm, context: str, question: str, max_retries: int = 2) -> Optional[dict]:
    raw = (JSON_PROMPT | llm | StrOutputParser()).invoke({
        "context": context, "question": question,
    })
    result = parse_json(raw)
    if result is not None:
        return result
    for _ in range(max_retries):
        try:
            json.loads(raw)
        except json.JSONDecodeError as e:
            fix_text = FIX_PROMPT.format(raw_output=raw, error=str(e))
            raw = llm.invoke(fix_text)
            result = parse_json(raw)
            if result is not None:
                return result
    return None


def run_comparison():
    llm = OllamaLLM()

    for case in TEST_CASES:
        question = case["question"]
        context = case["context"]

        print(f"\n{'='*60}")
        print(f"问题：{question}")
        print(f"{'='*60}")

        chain_free = FREE_TEXT_PROMPT | llm | StrOutputParser()
        answer_free = chain_free.invoke({
            "context": context, "question": question
        })
        print(f"\n── 自由文本 ──")
        print(answer_free[:200])

        result_json = ask_json(llm, context, question)
        print(f"\n── JSON 输出 ──")
        if result_json:
            print(json.dumps(result_json, ensure_ascii=False, indent=2))
            for field in ["steps", "tools", "summary"]:
                print(f"  {field}: {'✓ 存在' if field in result_json else '✗ 缺失'}")
        else:
            print("（解析失败，所有重试均未产出合法 JSON）")


if __name__ == "__main__":
    run_comparison()
```

#### 3.1 运行结果（本机实测）

```plain
python json_demo.py
```

```
── JSON 输出 ──
（解析失败，所有重试均未产出合法 JSON）
```

> 解读：3B 模型对复杂 schema 的 JSON 输出不稳定。后续封装时简化为单字段 schema（只保留 `steps`），成功率更高。这是模型能力边界——换大模型后 JSON 输出会显著改善。

## 步骤 4：`prompts/strategies.py`（Prompt 策略模板类）

五种策略类 + Router 自动分发。设计要点：

- `BasePrompt` 基类定义 `build()` + `run()` 接口
- `JsonPrompt.run()` 返回 `dict`（而非 `str`），把解析逻辑封在类内部
- `RouterPrompt` 根据 `qtype` 自动选择：`multi_step` → CoT，`extraction` → Simple
- `FewShotPrompt.EXAMPLES` 是类属性，表示「策略定义的常量配置」

```python
"""Prompt 策略模板类。"""

import json
import re
from typing import Optional, List, Dict

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class BasePrompt:
    def __init__(self, name: str):
        self.name = name

    def build(self, context: str, question: str) -> str:
        raise NotImplementedError

    def run(self, llm, context: str, question: str) -> str:
        return llm.invoke(self.build(context, question))


class SimplePrompt(BasePrompt):
    """普通 Prompt，来自 Day7 main.py 第 26~34 行。"""

    def __init__(self):
        super().__init__("simple")
        self.template = PromptTemplate.from_template(
            """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。

【资料】
{context}

【问题】{question}

【回答】"""
        )

    def build(self, context: str, question: str) -> str:
        return self.template.format(context=context, question=question)


class CoTPrompt(BasePrompt):
    """CoT 思维链 Prompt——先回答再简要分析。"""

    def __init__(self):
        super().__init__("cot")
        self.template = PromptTemplate.from_template(
            """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。

【资料】
{context}

【问题】{question}

请按以下步骤回答：
1. 【回答】先给出完整答案
2. 【分析】简要说明答案的依据（不超过 80 字）

【回答】
【分析】"""
        )

    def build(self, context: str, question: str) -> str:
        return self.template.format(context=context, question=question)


class FewShotPrompt(BasePrompt):
    """Few-Shot Prompt——用范例教 LLM 输出格式。"""

    EXAMPLES = [
        {
            "question": "如何启动服务",
            "answer": (
                "步骤1：确保虚拟环境已激活\n"
                "步骤2：在项目根目录执行 uvicorn main:app --reload --host 0.0.0.0 --port 8000\n"
                "步骤3：打开浏览器访问 http://localhost:8000/docs 验证"
            ),
        },
    ]

    def __init__(self):
        super().__init__("few_shot")
        self.template = PromptTemplate.from_template(
            """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。

回答格式要求：按步骤编号列出，每步格式为「步骤X：操作内容（涉及的命令或工具）」

示例——
问题：{example_question}
回答：
{example_answer}

现在回答以下问题：
【资料】
{context}

【问题】{question}

【回答】"""
        )

    def build(self, context: str, question: str) -> str:
        ex = self.EXAMPLES[0]
        return self.template.format(
            example_question=ex["question"],
            example_answer=ex["answer"],
            context=context,
            question=question,
        )


class JsonPrompt(BasePrompt):
    """JSON 结构化输出——约束 schema + 容错解析 + 重试。"""

    def __init__(self, max_retries: int = 2):
        super().__init__("json")
        self.max_retries = max_retries
        self.template = PromptTemplate.from_template(
            """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。

【资料】
{context}

【问题】{question}

请按以下 JSON 格式回答（仅输出 JSON，不要其他文字）：
{{
    "steps": ["步骤1描述", "步骤2描述", ...]
}}

JSON："""
        )
        self.fix_template = PromptTemplate.from_template(
            """你之前输出了以下内容，但它不是合法的 JSON：

{raw_output}

解析错误：{error}

请修正后重新输出合法的 JSON（仅输出 JSON，不要其他文字）：

JSON："""
        )

    def build(self, context: str, question: str) -> str:
        return self.template.format(context=context, question=question)

    def run(self, llm, context: str, question: str) -> Optional[dict]:
        raw = super().run(llm, context, question)
        result = self._parse(raw)
        if result is not None:
            return result
        for _ in range(self.max_retries):
            try:
                json.loads(raw)
            except json.JSONDecodeError as e:
                fix_text = self.fix_template.format(raw_output=raw, error=str(e))
                raw = llm.invoke(fix_text)
                result = self._parse(raw)
                if result is not None:
                    return result
        return None

    @staticmethod
    def _parse(text: str) -> Optional[dict]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None


class RouterPrompt:
    """根据问题 type 自动选择 Prompt 策略。

    multi_step → CoT
    extraction → Simple
    """

    def __init__(self):
        self.strategies: Dict[str, BasePrompt] = {
            "multi_step": CoTPrompt(),
            "extraction": SimplePrompt(),
        }

    def run(self, llm, context: str, question: str, qtype: str) -> str:
        strategy = self.strategies.get(qtype, SimplePrompt())
        return strategy.run(llm, context, question)

    def get_strategy_name(self, qtype: str) -> str:
        strategy = self.strategies.get(qtype)
        return strategy.name if strategy else "simple"
```

#### 4.1 验证

```plain
python prompt_runner.py
```

```
问题：如何安装依赖（类型：multi_step）
Router 自动选择策略：cot
关键词命中: ['pip install', 'venv'] (2/2)

问题：鉴权是怎么实现的（类型：extraction）
Router 自动选择策略：simple
关键词命中: ['APIKeyHeader', 'Depends', 'check_auth'] (3/3)
```

## 步骤 5：`main.py`（五组实验对比）

同一 LLM、同一问题、同一 context，唯一变量是 Prompt 策略。

| 实验 | 策略 | 目的 |
|---|---|---|
| A | SimplePrompt | Day7 基线 |
| B | CoTPrompt | 思维链 |
| C | FewShotPrompt | 范例引导 |
| D | JsonPrompt | 结构化 JSON |
| E | RouterPrompt | 自动选择 |

```python
"""Day8 完整对比实验。"""

import json

from adapters.ollama_llm import OllamaLLM
from test_data import TEST_CASES
from prompts.strategies import (
    SimplePrompt, CoTPrompt, FewShotPrompt, JsonPrompt, RouterPrompt,
)


def main():
    llm = OllamaLLM()

    strategies = {
        "A. Simple": SimplePrompt(),
        "B. CoT": CoTPrompt(),
        "C. Few-Shot": FewShotPrompt(),
    }
    router = RouterPrompt()

    for case in TEST_CASES:
        question = case["question"]
        context = case["context"]
        qtype = case.get("type", "")
        keywords = case.get("expect_keywords", [])

        print(f"\n{'#'*60}")
        print(f"问题：{question}")
        print(f"类型：{qtype}  |  期望关键词：{keywords}")
        print(f"{'#'*60}")

        for label, strategy in strategies.items():
            answer = strategy.run(llm, context, question)
            hit = [kw for kw in keywords if kw.lower() in answer.lower()]
            print(f"\n── {label} ──")
            print(answer[:250])
            print(f"关键词命中: {hit} ({len(hit)}/{len(keywords)})")

        if qtype == "multi_step":
            json_prompt = JsonPrompt()
            result = json_prompt.run(llm, context, question)
            print(f"\n── D. JSON ──")
            if result:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print("解析失败（模型未输出合法 JSON）")

        router_answer = router.run(llm, context, question, qtype)
        strategy_name = router.get_strategy_name(qtype)
        hit = [kw for kw in keywords if kw.lower() in router_answer.lower()]
        print(f"\n── E. Router（自动选择：{strategy_name}）──")
        print(router_answer[:250])
        print(f"关键词命中: {hit} ({len(hit)}/{len(keywords)})")


if __name__ == "__main__":
    main()
```

#### 5.1 运行结果（本机实测）

```plain
python main.py
```

```
问题：如何安装依赖
类型：multi_step  |  期望关键词：['pip install', 'venv']

── A. Simple ──
关键词命中: ['pip install', 'venv'] (2/2)

── B. CoT ──
关键词命中: ['pip install', 'venv'] (2/2)

── C. Few-Shot ──
关键词命中: ['pip install'] (1/2)

── D. JSON ──
解析失败（模型未输出合法 JSON）

── E. Router（自动选择：cot）──
关键词命中: ['pip install', 'venv'] (2/2)

############################################################
问题：鉴权是怎么实现的
类型：extraction  |  期望关键词：['APIKeyHeader', 'Depends', 'check_auth']

── A. Simple ──
关键词命中: ['APIKeyHeader', 'Depends', 'check_auth'] (3/3)

── B. CoT ──
关键词命中: ['APIKeyHeader', 'Depends', 'check_auth'] (3/3)

── C. Few-Shot ──
关键词命中: ['APIKeyHeader', 'Depends', 'check_auth'] (3/3)

── E. Router（自动选择：simple）──
关键词命中: ['APIKeyHeader', 'Depends', 'check_auth'] (3/3)
```

> 观察：
> - case1（multi_step）：CoT 与 Simple 持平（2/2），Few-Shot 反而漏了 venv
> - case2（extraction）：三种策略全部命中 3/3
> - Router 正确选择了策略（multi_step → cot，extraction → simple）
> - JSON 输出不稳定是 3B 模型的能力边界

# 五. 总结

## 1. 技术栈

Python + langchain-core + Ollama(qwen2.5:3b) + 复用 Day7 的 `OllamaLLM` 适配器

## 2. 核心模块

| # | 模块 | 职责 |
|---|---|---|
| 1 | `test_data.py` | 静态测试数据（context + 标注答案），不受检索波动影响 |
| 2 | `cot_demo.py` | CoT vs 普通 Prompt 单策略对比 |
| 3 | `few_shot_demo.py` | Few-Shot vs Zero-Shot 单策略对比 |
| 4 | `json_demo.py` | JSON 约束输出 + 三层容错解析 + 重试 |
| 5 | `prompts/strategies.py` | 五种策略类封装 + Router 自动分发 |
| 6 | `main.py` | 五组实验并排对比 |

## 3. Prompt 优化思想（Day8 版）

```plain
问题 + 资料(context)
        ↓
    RouterPrompt（判断问题类型）
        ↓
   ┌────┴────┐
multi_step  extraction
   ↓           ↓
CoTPrompt   SimplePrompt
（分步推理） （直接提取）
   ↓           ↓
     LLM(qwen2.5:3b)
        ↓
      答案 / JSON
```

## 4. 策略选择指南

| 问题类型 | 特征 | 推荐策略 |
|---|---|---|
| 多步骤推理 | 答案分散在资料多处，需要整合 | CoT |
| 直接信息提取 | 答案就在资料某处，直接摘取 | Simple |
| 格式要求严格 | 需要步骤列表、特定结构 | Few-Shot |
| 程序消费 | 下游需要结构化数据 | JSON（需模型支持） |

# 六、关键知识点理解复盘

#### Q1：为什么 CoT 的【回答】放在【分析】前面？

早期版本先【分析】再【回答】。但 LLM 在分析阶段消耗大量 token，到回答时被截断。交换顺序后答案完整，分析作为补充——这是「Answer-First CoT」的思路。

#### Q2：Few-Shot 的范例为什么不能和真实问题是同一个？

范例和问题相同 → LLM 直接抄范例答案，不做推理。范例应该「同类不同题」——格式一致但内容不同，让 LLM 学会格式而非答案。

#### Q3：为什么 `json_demo.py` 和 `strategies.py` 的 JsonPrompt schema 不同？

`json_demo.py` 是三字段（steps/tools/summary），用于演示复杂 schema 下 3B 模型的失败率。`strategies.py` 的 `JsonPrompt` 简化为单字段（steps）——实际使用中以成功率优先。

#### Q4：为什么 Router 只在 CoT 和 Simple 之间选？

Few-Shot 的范例副作用和 JSON 的不稳定性让它们不适合作为默认策略。CoT 和 Simple 覆盖最广、副作用最小——一个处理多步骤，一个处理直接提取。

#### Q5：3B 模型做 JSON 输出为什么这么不稳定？

JSON 要求精确的标点配对和嵌套结构。3B 参数意味着模型的「格式记忆」容量有限。约束手段（简化 schema + 容错解析 + 重试）能缓解但不能根治——换大模型是最有效的解决方案。

# 七、踩坑记录

| # | 现象 | 根因 | 解决 |
|---|---|---|---|
| 1 | CoT 回答被截断 | 分析阶段消耗太多 token | 交换顺序：先回答再分析 |
| 2 | Few-Shot 漏关键词 | 范例限制了 LLM 注意力范围 | 明确 Few-Shot 适用场景（格式严格，非内容自由） |
| 3 | JSON 解析全部失败 | 3B 模型对复杂 schema 不稳定 | 简化为单字段 schema |
| 4 | case2 CoT 未命中 `APIKeyHeader` | import 行被 LLM 当作非正文 | 属于直接信息提取场景，Simple 策略更合适 |
| 5 | `prompt_runner.py` 报 `NameError: name 'json' is not defined` | 缺少 `import json` | 顶部加 `import json` |

# 八、回顾：从 Day1 到 Day8 的技术演进

```plain
Day1: Ollama 网关 → 调用本地 LLM
Day2: 文档加载 → PDF/Word/MD → 纯文本
Day3: Embedding → bge-m3 → 文本转向量
Day4: Chroma → 向量存储 + 相似度检索
Day5: LangChain RAG → 标准问答链（检索 + 生成）
Day6: 混合检索 → MMR + BM25 + RRF + Rerank
Day7: PGVector 生产化 → 存储层升级 + 元数据过滤 + Self-Query + HyDE
Day8: Prompt Engineering → CoT + Few-Shot + JSON + 策略封装
```

Day1~Day7 解决「检索质量」，Day8 解决「Prompt 质量」。两者互补：检索决定 LLM 看到什么，Prompt 决定 LLM 怎么处理看到的内容。

## 运行方式

```powershell
cd F:\ai-learn\day8-prompt-engineering

python test_data.py            # 测试数据自检
python cot_demo.py             # CoT vs 普通 Prompt
python few_shot_demo.py        # Few-Shot vs Zero-Shot
python json_demo.py            # JSON 输出演示
python prompt_runner.py        # 封装类验证
python main.py                 # 完整对比实验
```
