# 一. 今日目标
Day2 把文档切成了 chunk，Day3 把 chunk 变成了向量，但向量"算完即丢"，没法存、没法查。今天引入 **Chroma 向量库**，把向量持久化存起来，学会对它做**增 / 删 / 改 / 查**，并拼一个**简易问答 Demo**——这是 RAG "检索 + 生成"的第一次闭环：

```plain
Day2 文档→chunks → Day3 embed 变向量 → Day4 存入 Chroma → 用户提问 → Chroma 检索 Top-K → 拼上下文 → Day1 Ollama 生成答案
```

1. 今天要解决的是：
+ 把 Day2/Day3 的产物（chunks + 向量）持久化进向量库，重启不丢
+ 对向量库做增删改查，并理解每条操作背后的语义
+ 用"检索到的资料"喂给大模型生成答案，完成最小可用 RAG 问答
2. 今日两个核心能力：
+ 向量库封装（ChromaStore）：对外暴露 `upsert / update / delete / query / get`
+ 简易问答 Demo：检索 → 拼上下文 → 调 Day1 的 `/api/generate` 基于资料作答

# 二、先想清楚几个问题（动手前的理解）

#### Q1：Chroma 为什么不需要独立安装服务？
Chroma 是**嵌入式（in-process）向量数据库**：它本身就是个 Python 库，运行在你的程序进程里，没有独立服务端。持久化靠把数据写本地目录（`chroma_db/`，内部用 SQLite 存元数据 + HNSW 索引文件），不需要起一个后台 daemon。所以 `pip install chromadb` 即装即用。

#### Q2：Chroma 的 `query` 和 Day3 自己写循环算余弦，本质一样吗？
一样。`query` 内部就是把你的问题向量和库里每条向量算余弦相似度并排序取 Top-K，只是交给数据库高效批量完成。Day3 手写 `cosine_similarity` 是"懂原理"，Chroma 是"工程实现"。

#### Q3：为什么 `query` 返回的是"距离"而不是"相似度"？
步骤 1 建 collection 时设了 `hnsw:space: cosine`，检索返回**余弦距离** = `1 - 余弦相似度`。距离越小越像，相似度 = `1 - distance`。两种口径一致，只是反过来表示。

#### Q4：为什么工程上要用 `upsert` 而不是 `add`？
`add` 遇重复 id 的行为在不同环境不确定（本次实测：重复 add 既不报错也不写入，详见第七节坑 1）；`upsert` 语义明确——**存在则覆盖、不存在则插入**，跨版本一致，适合反复调试的入库场景。

# 三、准备工作
## 步骤 1：新建目录 + 新建虚拟环境（依赖必须重装）
> Day4 是新建目录 + 新 venv，Day2/Day3 装过的库这里一个都没有，必须连同 chromadb 一起重装。

```powershell
cd F:\ai-learn
mkdir day4-chroma-store
cd day4-chroma-store
python -m venv venv
.\venv\Scripts\activate
pip install chromadb pypdf python-docx markdown beautifulsoup4 langchain-text-splitters requests numpy python-dotenv
```

| 包 | 作用 |
| --- | --- |
| chromadb | 向量库本体（增删改查 + 持久化） |
| pypdf / python-docx / markdown / beautifulsoup4 | 复用 Day2 加载多格式文档 |
| langchain-text-splitters | 复用 Day2 对照版分块 |
| requests / numpy / python-dotenv | 复用 Day3 的 EmbeddingClient + 余弦相似度 |

## 步骤 2：复制复用包（不重写，直接复用 Day2/Day3 成果）
- 复制 Day3 的 `embeddings/`（含 `__init__.py`）→ `day4-chroma-store/`
- 复制 Day3 的 `utils/`（含 `__init__.py`）→ `day4-chroma-store/`
- 复制 Day2 的 `loaders/`、`cleaners/`、`splitters/`（各含 `__init__.py`）→ `day4-chroma-store/`
- `samples/README.md`：从 Day2 复制一份测试文档
- 可选 `.env`：`OLLAMA_BASE_URL=http://localhost:11434`

> 前置：本机 Ollama 已 `ollama pull bge-m3`（入库向量化要用）；问答 Demo 还需另 pull 一个**对话模型**（如 `ollama pull qwen2.5:3b`），因为 `bge-m3` 是 Embedding 模型、不能用来对话。

# 四、开发实操
## 步骤 1：封装 `store/chroma_store.py`
#### 1.1 Chroma 向量库封装：增(upsert) / 删 / 改 / 查，复用 Day3 的 EmbeddingClient
```python
"""
author: tang
description: Chroma 向量库封装：增(upsert) / 删 / 改 / 查，复用 Day3 的 EmbeddingClient
"""

import chromadb
from embeddings.embedding_client import EmbeddingClient


class ChromaStore:
    def __init__(self, collection_name: str = "rag_docs", persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedder = EmbeddingClient(model="bge-m3")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # 余弦距离，和 Day3 一致
        )

    def upsert(self, chunks: list[str], metadatas: list[dict] = None, ids: list[str] = None):
        """增/改合一：id 不存在则插入，存在则更新向量+原文。反复运行 Day4 不再冲突。"""
        if ids is None:
            ids = [f"id_{i}" for i in range(len(chunks))]
        embeddings = [self.embedder.embed(c) for c in chunks]
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        print(f"[upsert] 完成，collection 现有 {self.collection.count()} 条")

    def add(self, chunks: list[str], metadatas: list[dict] = None, ids: list[str] = None):
        """严格新增：id 已存在会抛 UniqueConstraintError（用于需要强校验的场景）"""
        if ids is None:
            ids = [f"id_{i}" for i in range(len(chunks))]
        embeddings = [self.embedder.embed(c) for c in chunks]
        self.collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        print(f"[add] 已写入 {len(chunks)} 条，collection 现有 {self.collection.count()} 条")

    def update(self, id: str, document: str, metadata: dict = None):
        """改：更新某条内容，向量会重新计算"""
        emb = self.embedder.embed(document)
        self.collection.update(
            ids=[id], embeddings=[emb], documents=[document],
            metadatas=[metadata] if metadata else None,
        )
        print(f"[改] 已更新 {id}")

    def delete(self, ids: list[str]):
        """删：按 id 删除"""
        self.collection.delete(ids=ids)
        print(f"[删] 已删除 {ids}，collection 现有 {self.collection.count()} 条")

    def query(self, text: str, n_results: int = 3):
        """查：问题向量化后检索 Top-K"""
        q_emb = self.embedder.embed(text)
        return self.collection.query(query_embeddings=[q_emb], n_results=n_results)

    def get(self, ids: list[str] = None, limit: int = 10):
        """查看库状态：按 id 取或取前 limit 条"""
        return self.collection.get(ids=ids, limit=limit)
```

#### 1.2 练习要点
+ `chromadb.PersistentClient(path=...)`：嵌入式 + 持久化，数据落 `./chroma_db`，无独立服务。
+ `get_or_create_collection` 相当于"建表/取表"，`hnsw:space: cosine` 让检索用余弦距离，和 Day3 同口径。
+ 复用 Day3：`upsert/add/update/query` 内部都用 `self.embedder.embed()`，Day3 的 `EmbeddingClient` 被直接拿来用，没有重写。
+ `add` 一次存四样：`ids`（主键）、`embeddings`（向量）、`documents`（原文）、`metadatas`（元数据），解决 Day2/Day3 遗留的"无 source/chunk_id"问题。

## 步骤 2：写 `main.py`（入库 + 检索验证）
```python
"""
author: tang
description: Day4 入口 —— Day2 分块 → 存入 Chroma → 检索验证（增 + 查）
"""

from pipelines.loaders.document_loader import DocumentLoader
from pipelines.cleaners.text_cleaner import TextCleaner
from pipelines.splitters.recursive_splitter import RecursiveTextSplitter

from store.chroma_store import ChromaStore


def build_chunks(file_path: str, chunk_size: int = 300, chunk_overlap: int = 50) -> list[str]:
    """复用 Day2 流水线：加载 → 清洗 → 分块，返回 chunk 列表"""
    raw_text = DocumentLoader().load(file_path)
    clean_text = TextCleaner().clean(raw_text)
    chunks = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split(clean_text)
    print(f"[分块] 共切出 {len(chunks)} 块")
    return chunks


def main():
    file_path = "samples/README.md"
    chunks = build_chunks(file_path, chunk_size=300, chunk_overlap=50)

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": file_path, "chunk_id": i, "length": len(c)}
        for i, c in enumerate(chunks)
    ]

    store = ChromaStore(collection_name="rag_docs", persist_dir="./chroma_db")
    store.upsert(chunks, metadatas=metadatas, ids=ids)  # 用 upsert，反复运行不冲突

    question = "如何安装依赖"
    print(f"\n===== 提问: {question} =====")
    result = store.query(question, n_results=3)
    for i, (doc, meta, dist) in enumerate(zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    )):
        print(f"\n--- Top {i+1} (距离 {dist:.4f}, 相似度 {1-dist:.4f}) ---")
        print(f"来源: {meta['source']} | chunk_id: {meta['chunk_id']}")
        print(f"内容: {doc[:120]}...")


if __name__ == "__main__":
    main()
```

#### 2.1 运行结果（本机实测）
```plain
[分块] 共切出 54 块
[upsert] 完成，collection 现有 54 条

===== 提问: 如何安装依赖 =====
--- Top 1 (距离 0.3578, 相似度 0.6422) ---
来源: samples/README.md | chunk_id: 4
内容: ...批量安装依赖包激活环境后执行安装指令：plainpip install fastapi uvicorn ...
--- Top 2 (距离 0.3695, 相似度 0.6305) ---
来源: samples/README.md | chunk_id: 1
内容: ...安装 Python 3.11官网下载 Python3.11 安装包...
--- Top 3 (距离 0.3716, 相似度 0.6284) ---
来源: samples/README.md | chunk_id: 3
内容: ...创建 Python 虚拟环境执行命令：powershellpython -m venv venv...
```
> 解读：问题"如何安装依赖"召回了含 `pip install` 的 chunk（Top1 chunk_id 4），且距离递减、排序正确，说明 Day2→Day3→Day4 检索链路打通。

## 步骤 3：简易问答 Demo（RAG 检索 + 生成闭环）
在 `main.py` 里加轻量 Ollama 生成调用（不依赖 Day1 文件），把检索到的 chunk 当上下文喂给模型。

#### 3.1 在 `main.py` 增加以下函数并调用 `answer_question`
```python
import os
import requests
from dotenv import load_dotenv
load_dotenv()

def call_ollama_generate(prompt: str, model: str = "qwen2.5:latest") -> str:
    """轻量封装 Day1 的 /api/generate：发 prompt 收回复（非流式）"""
    url = f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(url, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()["response"]
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Ollama 服务未启动，请先执行 ollama serve")
    except Exception as e:
        raise RuntimeError(f"生成失败: {str(e)}")

def build_qa_prompt(question: str, contexts: list[str]) -> str:
    """把检索到的资料拼成上下文，构造 RAG 提示词"""
    context_text = "\n\n".join(f"【资料 {i+1}】\n{c}" for i, c in enumerate(contexts))
    return (
        "你是一个严谨的助手。请仅根据下面提供的【资料】回答问题。"
        "如果资料中没有相关信息，请明确回答“资料中未提及”。\n\n"
        f"【资料】\n{context_text}\n\n"
        f"【问题】{question}\n\n"
        "【回答】"
    )

def answer_question(store: ChromaStore, question: str, n_results: int = 3):
    """RAG 问答：检索 → 拼上下文 → 调 Ollama 生成 → 标注来源"""
    result = store.query(question, n_results=n_results)
    contexts = result["documents"][0]
    metas = result["metadatas"][0]
    prompt = build_qa_prompt(question, contexts)
    answer = call_ollama_generate(prompt)
    print(f"\n===== 问答 Demo =====")
    print(f"问题: {question}\n回答: {answer}")
    print(f"\n引用来源 (Top {n_results}):")
    for i, meta in enumerate(metas):
        print(f"  [{i+1}] {meta['source']} | chunk_id: {meta['chunk_id']}")
```
> 在 `main()` 末尾调用 `answer_question(store, "如何安装依赖")`。`model` 需改成你自己 pull 的对话模型名。

#### 3.2 练习要点
+ `build_qa_prompt` 是 RAG 的灵魂：强制"仅根据资料回答，没有就说未提及"——这是 RAG **抑制幻觉**的核心，模型不会凭空编。
+ 引用来源从 `metadatas` 取 `source + chunk_id`，回答后附上，证明答案有据可查。
+ 需先 `ollama pull qwen2.5:latest`（或你的对话模型），否则 `/api/generate` 报 404。

## 步骤 4：补 Day2 遗留债——剥离 Markdown 代码围栏
`cleaners/text_cleaner.py` 增加 `_remove_code_fences`，去掉 ```` ```lang ... ``` ```` 噪声，避免源码/语言标记词污染 chunk、干扰 Embedding。

#### 4.1 增强后的 `cleaners/text_cleaner.py`
```python
import re

"""
author: tang
description: 文本清洗器，去噪声（含 Day4 新增：剥离 Markdown 代码围栏，提升 Embedding 质量）
"""

class TextCleaner:
    def clean(self, text: str) -> str:
        text = self._remove_code_fences(text)      # Day4 新增：先剥代码块
        text = self._remove_extra_whitespace(text)
        text = self._normalize_blank_lines(text)
        return text

    def _remove_code_fences(self, text: str) -> str:
        """剥离 Markdown 代码围栏 ```lang ... ```，避免源码/语言标记词污染 chunk"""
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)  # DOTALL 让 . 跨换行
        return text

    def _remove_extra_whitespace(self, text: str) -> str:
        text = text.replace("\t", " ")
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \u3000]+", " ", text)
        return text

    def _normalize_blank_lines(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        return text
```
> 顺序很重要：先剥整段代码块这个"大噪声"，再做细粒度空白归一。改完后删 `chroma_db/` 重跑 `main.py`，对比清洗前后检索质量。

## 步骤 5：用 `upsert` 彻底解决重复运行 + 增删改查完整闭环
#### 5.1 `chroma_store.py` 增加 `upsert`（见步骤 1.1 完整版），`main.py` 入库改调 `store.upsert(...)`。
#### 5.2 新建 `crud_demo.py` 演示增删改查（用独立 collection，不污染 rag_docs）
```python
"""
author: tang
description: Day4 步骤5 —— Chroma 增删改查完整闭环演示（独立 collection）
"""

from store.chroma_store import ChromaStore

store = ChromaStore(collection_name="crud_demo", persist_dir="./chroma_db")

# 增（upsert，可反复运行不冲突）
chunks = [
    "Python 虚拟环境用 python -m venv venv 创建",
    "FastAPI 用 pip install fastapi uvicorn 安装",
    "Ollama 提供本地大模型推理服务",
]
ids = ["doc_0", "doc_1", "doc_2"]
metadatas = [{"source": "demo", "chunk_id": i} for i in range(3)]
store.upsert(chunks, metadatas=metadatas, ids=ids)

# 查
print("\n=== 查询: 怎么装 FastAPI ===")
res = store.query("怎么安装 FastAPI", n_results=2)
for doc, dist in zip(res["documents"][0], res["distances"][0]):
    print(f"  相似度 {1-dist:.4f} | {doc}")

# 改：把 doc_1 换成另一种说法，向量会重算
print("\n=== 改 doc_1 的内容 ===")
store.update("doc_1", "用 pip 安装 fastapi 和 uvicorn 两个包", metadata={"source": "demo", "chunk_id": 1})
res2 = store.query("怎么安装 FastAPI", n_results=2)
for doc, dist in zip(res2["documents"][0], res2["distances"][0]):
    print(f"  相似度 {1-dist:.4f} | {doc}")

# 删
print("\n=== 删 doc_2 ===")
store.delete(["doc_2"])

# 查（get 看库状态）
print("\n=== 当前库剩余内容 ===")
data = store.get()
for idd, doc in zip(data["ids"], data["documents"]):
    print(f"  {idd}: {doc}")
print(f"\n[核对] collection 现有 {store.collection.count()} 条")
```
#### 5.3 运行
```powershell
python .\crud_demo.py
python .\main.py      # 用 upsert 后，反复运行不再卡 id
```
#### 5.4 预期输出（crud_demo 关键片段）
```plain
[upsert] 完成，collection 现有 3 条
=== 查询: 怎么装 FastAPI ===
  相似度 0.xxxx | FastAPI 用 pip install fastapi uvicorn 安装
=== 改 doc_1 的内容 ===
[改] 已更新 doc_1
  相似度 0.xxxx | 用 pip 安装 fastapi 和 uvicorn 两个包   ← 内容已变，证明"改"生效
=== 删 doc_2 ===
[删] 已删除 ['doc_2']，collection 现有 2 条
=== 当前库剩余内容 ===
  doc_0: Python 虚拟环境用 python -m venv venv 创建
  doc_1: 用 pip 安装 fastapi 和 uvicorn 两个包
[核对] collection 现有 2 条
```

# 五. 总结
## 1. 技术栈
Python + chromadb + requests + numpy + python-dotenv + Ollama(bge-m3 + 对话模型) + 复用 Day2/Day3 包

## 2. 核心模块功能
1. **向量库封装 store/chroma_store.py**
    - `upsert`：增/改合一，反复运行不冲突（解决步骤 2 悬案）
    - `query`：问题向量化后检索 Top-K，返回原文/元数据/距离
    - `update/delete/get`：管理单条数据、确认库状态
2. **文档流水线（复用 Day2）**：`loaders/cleaners/splitters` 产出 chunks
3. **Embedding（复用 Day3）**：`embeddings/embedding_client.py` 把文字变向量
4. **问答 Demo**：`build_qa_prompt` 约束"仅依资料回答"，`call_ollama_generate` 调 Day1 接口，输出答案 + 引用来源

## 3. RAG 闭环思想
```plain
Day2 分块(chunks) → Day3 Embedding(向量) → Day4 入库(Chroma)
→ 提问向量化 → Chroma 检索 Top-K → 拼上下文 → Day1 生成答案（标注来源）
```
+ 解耦：换向量库只动 `ChromaStore`；换检索策略只动 `query`；换生成模型只动 `call_ollama_generate`
+ 可溯源：每条向量带 `source/chunk_id`，命中即回链原文
+ 可调试：用 `upsert` + `get` 反复验证，不卡 id 冲突

# 六、关键知识点理解复盘
#### Q1：Chroma 为什么不用装服务就能用？
它是嵌入式库，运行在你的进程内，数据直接写本地目录（SQLite + HNSW 索引文件），没有独立服务端进程。

#### Q2：为什么检索返回"距离"而不是"相似度"？
建 collection 时设了 `hnsw:space: cosine`，返回余弦距离 = `1 - 余弦相似度`。距离越小越像，和 Day3 的余弦分数同口径、反向表示。

#### Q3：`upsert` 和 `add` 有什么区别？为什么工程上用 `upsert`？
`add` 遇重复 id 行为不确定（本次实测既报错风险也存在静默忽略）；`upsert` 明确"有则覆盖、无则插入"，跨版本一致，适合反复调试入库。

#### Q4：RAG 凭什么能"不瞎编"？
因为 `build_qa_prompt` 强制模型"仅根据提供的资料回答，没有就说未提及"，只把检索到的相关资料塞进 prompt，这是抑制幻觉的核心。

#### Q5：`update` 改的是内容还是 id？为什么改完检索结果会变？
改的是内容（id 不变）。Chroma 会用新内容重算向量，所以再检索时该条的向量方向和分数都变了——这正是"改"真正生效的证据。

#### Q6：为什么 `bge-m3` 不能用来做问答？
`bge-m3` 是 Embedding 模型，专门把文字编码成向量，不具备"接着生成文字"的能力。问答需要用对话模型（如 qwen2.5），二者任务不同。

# 七、踩坑记录与遗留问题
## 7.1 本次踩坑（调试真事）
| # | 现象 | 根因 | 解决 |
| --- | --- | --- | --- |
| 1 | 重复运行 `add` 既不报错、count 也不翻倍（始终 54） | 与标准 Chroma"add 重复 id 必报错"行为不符，疑似版本对重复 id 静默跳过或持久化未生效，未彻底定位 | 步骤 5 改用 `upsert` 彻底规避 |
| 2 | 问答 Demo 调 `/api/generate` 报 404 | 用 `bge-m3`（Embedding 模型）去对话 | 另 `ollama pull pipelines.` 对话模型 |

## 7.2 遗留技术债 / 待补强
| # | 问题 | 影响 | 计划解决时机 |
| --- | --- | --- | --- |
| 1 | `upsert` 每次全量重算 embedding，大文档慢 | 反复调试耗时长 | Day6 后接入批量 `/api/embed` 或并发 |
| 2 | 问答无"无命中"兜底（相关性阈值） | 资料没有答案时模型仍可能硬编 | 加 `distance` 阈值判断，超阈值回"未提及" |
| 3 | 检索仅纯语义（无 MMR / 关键词混合） | 召回可能重复/偏离 | Day6 MMR + 混合检索 + Rerank |
| 4 | Chroma 嵌入式不适合多进程高并发 | 生产场景瓶颈 | Day7 换 PostgreSQL + PGVector |
| 5 | 问答输出未高亮引用原文片段 | 用户体验弱 | 在 `answer_question` 里打印命中句子 |
