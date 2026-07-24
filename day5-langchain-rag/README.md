# 一. 今日目标
Day1–Day4 我们已经**从零手搓**出一条 RAG 链路：Ollama 服务 → 文档分块 → 本地 Embedding → Chroma 存查 → 问答 Demo。今天要做的是**用 LangChain 把这套链路"标准化重写"**，并重点搞清楚一个黑盒：**`RetrievalQA`（检索问答链）内部到底发生了什么**。

1. 今天要解决的是：
+ 把 Day3 的 `EmbeddingClient`、Day1 的 Ollama 生成，**包装成 LangChain 的标准组件**（Embeddings / LLM），复用而非重写
+ 用 LangChain 的标准组件（Document / TextSplitter / VectorStore / Retriever / PromptTemplate / LLM）把 Day2→Day4 的成果串起来
+ 弄懂"检索 + 拼上下文 + 调 LLM"这条链路的真实数据流
2. 今日两个核心能力：
+ **手写 LangChain 适配器**：`OllamaEmbeddings` / `OllamaLLM` 两个子类，把自搓成果接进 LangChain 生态
+ **看清 RetrievalQA 链路**：先手写四步链剖开黑盒，再对比官方封装（注：新版 LangChain 已弃用 `RetrievalQA`，改用 LCEL，详见第七节）

# 二、先想清楚几个问题

#### Q1：LangChain 到底帮我们干了啥？为什么不直接用 Day4 手搓的？
LangChain 不替你"算余弦、存向量"，它提供**统一抽象**：`Document` / `TextSplitter` / `Embeddings` / `VectorStore` / `Retriever` / `PromptTemplate` / `LLM` 全是标准接口。你 Day1–Day4 手搓的每个模块都能对上一个组件。价值在于"可替换、可组合"——换模型/换向量库只动一处，不用重写整条链。

#### Q2：为什么不直接 `RetrievalQA.from_chain_type` 一行出结果？
那行代码是黑盒，你看不到"检索到的 chunk 怎么进 prompt、prompt 长什么样"。Day5 标题是"弄懂链路"，所以**先手写**把 `retriever → prompt → llm` 拆开看，最后再对比官方封装，理解它只是语法糖。

#### Q3：为什么要自己写 `OllamaEmbeddings` / `OllamaLLM` 适配器？
为了**复用 Day3/Day1 成果**（bge-m3 经 `/api/embeddings`、对话经 `/api/generate`），也为了理解 LangChain "任何模型只要实现接口就能接入" 的设计——这比直接 `from langchain_ollama import OllamaLLM` 学得更深。

#### Q4：`chain_type="stuff"` 是什么意思？
把所有召回的 chunk 直接"塞进"(stuff) 一个 prompt。还有 `map_reduce` / `refine` 等分多篇策略（Day8 展开）。

# 三、准备工作
## 步骤 1：新建目录 + 新建虚拟环境（依赖必须重装）
> Day5 是新建目录 + 新 venv，Day2/Day3/Day4 装过的库这里一个都没有，必须连同 LangChain 全家桶一起重装。

```powershell
cd F:\ai-learn
mkdir day5-langchain-rag
cd day5-langchain-rag
python -m venv venv
.\venv\Scripts\activate
pip install langchain langchain-core langchain-community langchain-chroma langchain-ollama pypdf python-docx markdown beautifulsoup4 requests numpy python-dotenv
```

| 包 | 作用 |
| --- | --- |
| langchain / langchain-core | 核心抽象（Document、PromptTemplate、LLM 基类、Runnable/LCEL） |
| langchain-community | 社区集成（含 Chroma、文本加载器） |
| langchain-chroma | Chroma 的官方 LangChain 集成（`Chroma` 类） |
| langchain-ollama | Ollama 的官方封装（备用的捷径，本项目仍手写适配器） |
| pypdf / python-docx / markdown / beautifulsoup4 | 复用 Day2 加载多格式文档 |
| requests / numpy / python-dotenv | 支撑手写 EmbeddingClient / Ollama 调用 |

## 步骤 2：复制复用包（不重写，直接复用 Day4 成果）
- 复制 **Day4 的 `embeddings/`**（含 `__init__.py`）→ `day5-langchain-rag/`（提供 `EmbeddingClient`）
- 复制 **Day4 的 `loaders/`、`cleaners/`**（各含 `__init__.py`）→ `day5-langchain-rag/`（提供加载 + 清洗）
- 分块器**不用复制 Day4 的**：步骤 3 改用 LangChain 标准版 `RecursiveCharacterTextSplitter`
- `samples/README.md`：从 Day4 复制一份测试文档
- 可选 `.env`：`OLLAMA_BASE_URL=http://localhost:11434`

> 前置：本机 Ollama 已 `ollama pull bge-m3`（向量化用）；问答还需另 pull 一个**对话模型**（如 `ollama pull qwen2.5:3b`），因为 `bge-m3` 是 Embedding 模型、不能用来对话。

# 四、开发实操
## 步骤 1：写 `adapters/ollama_embeddings.py`（把 Day4 成果包装成 LangChain Embeddings）

```python
"""
author: tang
description: 把 Day4 的 EmbeddingClient 包装成 LangChain 的 Embeddings 组件
"""
from typing import List
from langchain_core.embeddings import Embeddings
from embeddings.embedding_client import EmbeddingClient

class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str = "bge-m3"):
        self.client = EmbeddingClient(model=model)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain 调它给一批文档（chunks）算向量，用于入库"""
        return [self.client.embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        """LangChain 调它给"用户问题"算向量，用于检索"""
        return self.client.embed(text)
```
> 练习要点：LangChain 只要求实现 `embed_documents` / `embed_query` 两个方法（签名在 `langchain_core.embeddings.Embeddings` 规定），内部委托 Day4 的 `EmbeddingClient.embed()`，Day4 成果即插即用。两方法分开是为兼容"文档/查询可能不同处理"（如 bge-m3 中文 query 加指令前缀效果更好）。

## 步骤 2：写 `adapters/ollama_llm.py`（把 Day1 成果包装成 LangChain LLM）

```python
"""
author: tang
description: 把 Day1 的 Ollama /api/generate 包装成 LangChain 的 LLM 组件
"""
import os
from typing import List, Optional
import requests
from dotenv import load_dotenv
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
load_dotenv()

class OllamaLLM(LLM):
    model: str = "qwen2.5:3b"          # 对话模型名，改成你实际 pull 的
    base_url: str = "http://localhost:11434"
    timeout: int = 180

    @property
    def _llm_type(self) -> str:
        return "ollama_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()["response"]
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Ollama 服务未启动，请先执行 ollama serve")
        except Exception as e:
            raise RuntimeError(f"生成失败: {str(e)}")
```
> 练习要点：自定义 LLM 标准套路 = 继承 `LLM` + 实现 `_call()`（发请求拿回复）+ `_llm_type` 属性（类型标识）。字段用 pydantic 注解声明，LangChain 自动做校验。`_call` 内部就是 Day1 的 `/api/generate` + `stream: False`。

## 步骤 3：写 `rag_chain.py`（RAG 前半段：构建 retriever）
用 LangChain 标准组件：`Document` + `RecursiveCharacterTextSplitter` + `OllamaEmbeddings` + `Chroma`，得到 `retriever`。

```python
"""
author: tang
description: Day5 —— 用 LangChain 标准组件组装 RAG 链路（文档→分块→向量→Chroma→retriever）
"""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from loaders.document_loader import DocumentLoader
from cleaners.text_cleaner import TextCleaner
from adapters.ollama_embeddings import OllamaEmbeddings

def build_retriever(file_path: str, collection_name: str = "rag_docs_lc",
                    persist_dir: str = "./chroma_db", chunk_size: int = 300,
                    chunk_overlap: int = 50, k: int = 3):
    embedding = OllamaEmbeddings(model="bge-m3")
    # 先加载（或创建）持久化 collection，避免每次 from_documents 都重复累积
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,    
        persist_directory=persist_dir,
    )
    # 库为空才构建写入，否则复用已有数据（根治"重复运行累积"）
    if len(vectorstore.get()["ids"]) == 0:
        raw_text = DocumentLoader().load(file_path)
        clean_text = TextCleaner().clean(raw_text)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(clean_text)
        docs = [Document(page_content=c, metadata={"source": file_path, "chunk_id": i})
                for i, c in enumerate(chunks)]
        vectorstore.add_documents(docs)
        print(f"[入库] Chroma 已写入 {len(docs)} 条")
    else:
        print(f"[复用] 库已有 {len(vectorstore.get()['ids'])} 条，跳过写入")
    return vectorstore.as_retriever(search_kwargs={"k": k})
```
> 练习要点：`Document` = `page_content` + `metadata`，metadata 塞 `source/chunk_id` 解决 Day2/Day3 无元信息债；`Chroma(...).add_documents` 内部调步骤1 适配器的 `embed_documents` 把每篇变向量并持久化；`as_retriever()` 把"库"变成"检索接口"，Day6 换检索策略只动这里。

#### 3.1 运行结果（本机实测，首次入库）
```plain
[入库] Chroma 已写入 29 条
```

## 步骤 4：写 `main.py`（手写 RAG 链，剖开黑盒 + 打印每步中间产物）

```python
"""
author: tang
description: Day5 步骤4 —— 手写最简 RAG 链，剖开 retriever | prompt | llm 的 LCEL 写法
"""
from langchain_core.prompts import PromptTemplate
from rag_chain import build_retriever
from adapters.ollama_llm import OllamaLLM

def handwritten_rag(question: str, retriever, llm):
    # 1) 检索：retriever.invoke 返回相关 Document 列表（新版统一用 invoke）
    docs = retriever.invoke(question)
    print(f"\n===== [步骤1 检索] 召回 {len(docs)} 条 =====")
    for i, d in enumerate(docs):
        print(f"--- 相关文档 {i+1} (来源 {d.metadata['source']} | chunk_id {d.metadata['chunk_id']}) ---")
        print(d.page_content[:100], "...\n")
    # 2) 拼上下文
    context = "\n\n".join(d.page_content for d in docs)
    # 3) 构造 Prompt
    prompt = PromptTemplate.from_template(
        """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。

【资料】
{context}

【问题】{question}

【回答】"""
    )
    final_prompt = prompt.format(context=context, question=question)
    print("===== [步骤3 拼好的 Prompt]（前 600 字）=====")
    print(final_prompt[:600], "...\n")
    # 4) 生成
    answer = llm.invoke(final_prompt)
    print("===== [步骤4 生成答案] =====")
    print(answer)

def main():
    retriever = build_retriever("samples/README.md", k=3)
    llm = OllamaLLM()   # 默认 qwen2.5:latest，改成你 pull 的对话模型
    handwritten_rag("如何安装依赖", retriever, llm)

if __name__ == "__main__":
    main()
```
> 练习要点：手写链就是四步——`retriever.invoke(q)` 召回 → 拼 `context` → `PromptTemplate.format()` 出最终 prompt → `llm.invoke(prompt)` 出答案。**这四步就是 `RetrievalQA` 内部做的全部事**，没有魔法。打印中间产物是"剖黑盒"的关键。

#### 4.1 运行结果（本机实测）
```plain
[入库] Chroma 已写入 29 条

===== [步骤1 检索] 召回 3 条 =====
--- 相关文档 1 (chunk_id 4) ---  等待全部依赖下载安装完成...步骤 3：创建项目配置文件 .env...
--- 相关文档 2 (chunk_id 3) ---  Windows CMD / venv\Scripts\activate ... 激活虚拟环境...
--- 相关文档 3 (chunk_id 1) ---  官网下载 Python3.11 安装包...Windows 安装勾选 Add Python to PATH...

===== [步骤4 生成答案] =====
根据提供的资料，安装依赖的具体步骤如下：
1. 激活虚拟环境：... venv\Scripts\activate ...
2. 批量安装依赖包：pip install fastapi uvicorn python-multipart pydantic python-jose passlib loguru python-dotenv requests
...（模型把 .env 路径、ollama 命令等弱相关片段也揉进了答案）
```
> 解读：链路全通——召回 3 个不同 chunk，prompt 正确拼入 `pip install` 段，模型给出主线答案。但模型把 chunk_id 4（.env 配置段）、chunk_id 3（激活环境段）等弱相关片段也综合进答案，出现 `.env\venv\Scripts\activate`、错误的 `ollama model download` 等杂糅。**这暴露了"召回精度"问题，正是 Day6（MMR/混合检索/Rerank）要解决的**。

## 步骤 5：对比官方封装 —— 但 `RetrievalQA` 已弃用，改用 LCEL
原计划用 `RetrievalQA.from_chain_type` 对比手写版，但**新版 LangChain 已从 `langchain.chains` 和 `langchain_community.chains` 中彻底移除 `RetrievalQA`**（`langchain-community` 进入 sunset）。官方推荐用 **LCEL**（`retriever | prompt | llm`）——而这恰恰把黑盒显式拆开，比 `RetrievalQA` 更契合"弄懂链路"。

新建 `lcel_rag_demo.py`：

```python
"""
author: tang
description: Day5 步骤4 —— 手写最简 RAG 链，剖开 retriever | prompt | llm 的 LCEL 写法
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rag_chain import build_retriever
from adapters.ollama_llm import OllamaLLM

def format_docs(docs):
    """把召回的 Document 列表拼成一段 context（对应手写版第2步'拼 context'）"""
    return "\n\n".join(d.page_content for d in docs)

def lcel_rag_demo(question: str, retriever, llm):
    prompt = PromptTemplate.from_template(
        """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。

【资料】
{context}

【问题】{question}

【回答】"""
    )
    # LCEL 链：retriever 召回 → format_docs 拼 context → prompt 格式化 → llm 生成 → 解析字符串
    # 这正是 RetrievalQA(chain_type="stuff") 的现代等价实现，且每一步可见
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = rag_chain.invoke(question)
    docs = retriever.invoke(question)
    print("===== [LCEL RAG 答案] =====")
    print(answer)
    print("\n===== [LCEL 命中来源] =====")
    for d in docs:
        print(f"  {d.metadata['source']} | chunk_id {d.metadata['chunk_id']} | {d.page_content[:60]}...")

def main():
    retriever = build_retriever("samples/README.md", k=3)
    llm = OllamaLLM()
    lcel_rag_demo("如何安装依赖", retriever, llm)

if __name__ == "__main__":
    main()
```
> 练习要点（LCEL 即"显式版 RetrievalQA"）：
> + `|` 是管道符：`retriever | format_docs` = `format_docs(retriever.invoke(x))`，**上游输出自动成为下游第一个参数**（所以 `format_docs(docs)` 的 `docs` 不用手动传，由管道喂入）。
> + `{"context": ..., "question": RunnablePassthrough()}` 是 `RunnableParallel`：每个 value 都是独立子链，共享原始输入 `question`；`RunnablePassthrough()` 表示"问题原样透传"。
> + `prompt | llm | StrOutputParser()`：格式化 → 调 LLM → 解析成字符串。四段拼起来 = 手写版四步 = `RetrievalQA(stuff)` 内部全部动作。

# 五. 总结
## 1. 技术栈
Python + langchain / langchain-core / langchain-community / langchain-chroma + requests + numpy + python-dotenv + Ollama(bge-m3 + 对话模型) + 复用 /Day1/Day2/Day3/Day4 包

## 2. 核心模块功能
1. **适配器 adapters/ollama_embeddings.py**：`OllamaEmbeddings` 继承 LangChain `Embeddings`，内部委托 Day3 `EmbeddingClient`，让向量库/检索能用 bge-m3
2. **适配器 adapters/ollama_llm.py**：`OllamaLLM` 继承 LangChain `LLM`，内部调 Day1 `/api/generate`，让 Chain 能用本地对话模型
3. **rag_chain.build_retriever**：`Document` + LangChain `RecursiveCharacterTextSplitter` + `OllamaEmbeddings` + `Chroma` 串成 retriever（含"库空才写、否则复用"防累积）
4. **手写 RAG 链 main.py**：四步打印中间产物，看清"召回→拼context→format→invoke"数据流
5. **LCEL 链 lcel_rag_demo.py**：`retriever | prompt | llm` 的等价实现，即新版 `RetrievalQA(stuff)`

## 3. RAG 链路思想（Day5 版）
```plain
Day2 加载→清洗→分块 → Day4 Embedding(经 OllamaEmbeddings) → Chroma 持久化
→ retriever.invoke(问题) 召回 Top-K → 拼 context → PromptTemplate → LLM 生成 → 答案(标注来源)
```
+ 复用优先：Day3 向量化、Day1 生成都通过适配器直接接入，没有重写
+ 可替换：换 Embedding 模型只改 `OllamaEmbeddings(model=...)`；换对话模型只改 `OllamaLLM(model=...)`；换向量库只动 `Chroma`
+ 透明：LCEL 把链路每一步显式写出，比 `RetrievalQA` 黑盒更易调试

# 六、关键知识点理解复盘
#### Q1：`RetrievalQA` 和 Day4 手写的问答 Demo 本质一样吗？
一样，都是"检索 Top-K → 拼上下文 → 调 LLM"。Day5 手写版把它拆开看；`RetrievalQA` 只是封装（新版已弃用，LCEL 是其等价显式写法）。

#### Q2：为什么自己写 `OllamaEmbeddings` / `OllamaLLM` 适配器？
为了复用 Day3/Day1 成果，也为了理解 LangChain "模型只要实现接口就能接入" 的设计。真实项目可直接 `from langchain_ollama import OllamaLLM` 省事。

#### Q3：LCEL 里 `format_docs(docs)` 的 `docs` 为什么没手动传值？
`|` 管道符等价于函数组合：`retriever | format_docs` = `format_docs(retriever.invoke(x))`，上游输出自动作为下游第一个参数。这是 LCEL "隐式传参" 机制，类比 Unix 管道。

#### Q4：为什么 `retriever` 要用 `invoke` 而不是 `get_relevant_documents`？
新版 LangChain 把所有组件统一为 `Runnable`，调用入口是 `invoke()`，`get_relevant_documents` 已移除（旧版才有）。

#### Q5：`chain_type="stuff"` 是什么意思？还有别的吗？
`stuff` = 把所有召回 chunk 塞进一个 prompt。另有 `map_reduce` / `refine` 等分多篇策略（Day8 展开）。

#### Q6：`bge-m3` 为什么不能用来做问答？
`bge-m3` 是 Embedding 模型，专门把文字编码成向量，不具备"接着生成文字"能力。问答需对话模型（如 qwen2.5），二者任务不同。

# 七、踩坑记录与遗留问题
## 7.1 本次踩坑（调试真事）
| # | 现象 | 根因 | 解决 |
| --- | --- | --- | --- |
| 1 | `'VectorStoreRetriever' object has no attribute 'get_relevant_documents'` | 新版 LangChain 把 Retriever 改为 Runnable，旧方法已移除 | 改用 `retriever.invoke(question)` |
| 2 | 重复运行 `main.py` 后，召回的 3 条全是同一 `chunk_id 4` | `Chroma.from_documents` 每次生成新 uuid 入库，数据累积（29→58），同一段多份副本占满 Top-K | `build_retriever` 改为先 `Chroma(...)` + 判断 `count==0` 才写入，否则复用 |
| 3 | `ModuleNotFoundError/ImportError: cannot import 'RetrievalQA' from langchain.chains / langchain_community.chains` | 新版 LangChain 弃用 `RetrievalQA`，`langchain-community` 进入 sunset，该类已从顶层导出移除 | 改用 LCEL `retriever \| prompt \| llm` 实现等价链 |
| 4 | 问答 Demo 调 `/api/generate` 报 404 | 用 `bge-m3`（Embedding 模型）去对话 | 另 `ollama pull` 对话模型（如 qwen2.5:3b） |

## 7.2 遗留技术债 / 待补强
| # | 问题 | 影响 | 计划解决时机 |
| --- | --- | --- | --- |
| 1 | 召回的 Top-K 含弱相关段（chunk_id 3/1），模型综合后跑偏、杂糅错误命令 | 答案准确性下降 | Day6 MMR + 关键词/语义混合检索 + Rerank 重排 |
| 2 | 用 Day2 原版 cleaner（未剥代码围栏），与 Day4 增强版不一致；分块 29 块 vs Day4 的 54 块 | 检索质量未达最优 | 复制 Day4 增强 `text_cleaner.py`、调小 `chunk_size` |
| 3 | `stuff` 链有上下文长度上限，chunk 多会超 token | 大文档无法直接塞入 | Day8 学 `map_reduce` / `refine` |
| 4 | 问答无"无命中"兜底（相关性阈值） | 资料没有答案时模型仍可能硬编 | 加 distance 阈值判断，超阈值回"未提及" |
| 5 | `langchain-community` sunset，部分旧 API（如 `RetrievalQA`）不可用 | 网上老教程代码跑不通 | 一律改用 LCEL（`\|` 管道）写法 |
