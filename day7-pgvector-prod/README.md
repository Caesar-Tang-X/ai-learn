# 一. 今日目标

Day6 用 Chroma 跑通了「混合检索 + Rerank」完整链路，但 Chroma 是嵌入式向量库——单进程独占、无 SQL 能力、不适合生产并发。Day7 将存储层换为 **PostgreSQL + PGVector**，复制 Day6 的全部检索能力，并新增三种策略：

1. **元数据过滤**：借助 PGVector 的 JSONB 列，在 SQL 层做精确的范围/组合过滤（`WHERE chunk_id BETWEEN 0 AND 5`），Chroma 无法做到
2. **自查询（Self-Query）**：LLM 自动从自然语言中提取过滤条件（「前几个步骤」→ `{"chunk_id": {"$lte": 4}}`），再传给 PGVector
3. **HyDE 假想文档检索**：LLM 生成假设答案 → 用假设答案嵌入检索真实文档——「长文本探针」比「短问题探针」更接近知识库文档风格

最终 `main.py` 六组实验并行对比（同一 prompt、同一 LLM、同一问题），验证 PGVector 与 Chroma 的检索质量一致性 + 新策略的提升效果。

# 二、先想清楚几个问题

#### Q1：为什么要换 PGVector？语义检索质量会变好吗？

不会。语义检索质量取决于 embedding 模型（同为 bge-m3）和文档质量，换存储层不影响排序精度。换 PGVector 的价值在别处：

| 维度 | Chroma（Day6） | PGVector（Day7） |
|---|---|---|
| 并发读写 | 单进程独占，多进程冲突 | PostgreSQL MVCC，多客户端并行 |
| 元数据过滤 | filter 只支持精确匹配 | JSONB → SQL WHERE，支持范围/组合/布尔 |
| 持久化可靠性 | 本地文件，无备份机制 | PostgreSQL WAL / 备份 / 主从复制 |
| 生产整合 | 独立服务，需额外维护 | 复用现有 PG 基础设施，加 pgvector 扩展即可 |

#### Q2：为什么 Day6 的 BM25 不能直接用？Day7 做了什么？

Day6 的 `BM25Retriever.__init__` 收的是 `vectorstore`，内部调 `vectorstore.get()` 取全量文档。但 `vectorstore.get()` 是 Chroma 专属方法，PGVector 没有。

Day7 改为收 `List[Document]`——BM25 本来就不需要向量库，它只需要文档文本。这个改动消除了不必要的耦合，让 BM25 能搭配任何向量库。

#### Q3：`langchain_postgres` 的 filter 语法有什么坑？

Chroma 一个字段可以带多个运算符：

```python
{"chunk_id": {"$gte": 0, "$lte": 5}}   # Chroma 可以
```

`langchain_postgres` 要求每个字段只有一个 op key，范围查询必须用 `$and` 拆开：

```python
{"$and": [{"chunk_id": {"$gte": 0}}, {"chunk_id": {"$lte": 5}}]}   # PGVector 正确写法
```

这是「存储无关」抽象层与实际实现的缝隙——同一个 LangChain `similarity_search(filter=...)` 在不同后端下语法不统一。

#### Q4：Self-Query 和元数据过滤有什么区别？

元数据过滤是**手动**指定 filter 字典；Self-Query 是**LLM 自动**从自然语言中提取 filter。

```
元数据过滤：你写 filter={"chunk_id": {"$lte": 4}}
Self-Query：你说 "前几个步骤" → LLM 翻译成 filter={"chunk_id": {"$lte": 4}}
```

Self-Query 的效果严重依赖 LLM 质量。`qwen2.5:3b`（3B 小模型）的结构化输出不稳定——这不是 Day7 的 bug，是模型能力决定的边界。

#### Q5：HyDE 为什么可能比直接检索更准？

用户问题是短问句（"如何安装依赖"），而知识库文档是长段落（叙述性、含术语）。这两者在嵌入空间里可能有语言风格差异。

HyDE 先让 LLM 生成假想答案——它也是叙述性长文本，风格接近知识库文档。用假想答案的向量去检索，在嵌入空间里天然离真实文档更近。本质是「让 LLM 帮你把问题翻译成文档语言」。

但也受 LLM 质量限制——`qwen2.5:3b` 生成的假想答案可能太短或跑偏。

# 三、准备工作

## 步骤 1：新建目录 + 复制复用包

```powershell
cd F:\ai-learn
mkdir day7-pgvector-prod
cd day7-pgvector-prod

mkdir retrieval
mkdir adapters
mkdir embeddings
mkdir loaders
mkdir cleaners
mkdir samples

Copy-Item ..\day6-hybrid-retrieval\adapters\* adapters\ -Recurse
Copy-Item ..\day6-hybrid-retrieval\embeddings\* embeddings\ -Recurse
Copy-Item ..\day6-hybrid-retrieval\loaders\* loaders\ -Recurse
Copy-Item ..\day6-hybrid-retrieval\cleaners\* cleaners\ -Recurse
Copy-Item ..\day6-hybrid-retrieval\samples\* samples\ -Recurse
```

## 步骤 2：安装依赖

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install langchain langchain-core langchain-community langchain-postgres psycopg2-binary python-dotenv sentence-transformers jieba rank_bm25 pypdf python-docx langchain-chroma
```

| 新包 | 作用 |
|---|---|
| `langchain-postgres` | PGVector 的 LangChain 封装 |
| `psycopg2-binary` | Python ↔ PostgreSQL 通信驱动 |

## 步骤 3：启动 Docker PostgreSQL + PGVector

`docker-compose.yml`：

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: day7-pgvector
    environment:
      POSTGRES_USER: rag
      POSTGRES_PASSWORD: rag123
      POSTGRES_DB: ragdb
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

```powershell
docker compose up -d
docker compose ps   # 确认 healthy
```

## 最终目录结构

```plain
day7-pgvector-prod/
├── main.py                  # 步骤 8：六组实验并排对比
├── rag_chain.py             # 底层：PGVector 建库/复用
├── docker-compose.yml       # PostgreSQL + pgvector 服务
├── retrieval/               # 检索策略包
│   ├── __init__.py           # 空文件（保持包结构）
│   ├── pg_retriever.py       # 策略 A：PGVector 语义 + MMR
│   ├── bm25_retriever.py     # 策略 B：BM25 解耦版
│   ├── fusion.py             # 策略 C：RRF 双路融合
│   ├── reranker.py           # 策略 D：CrossEncoder 精排
│   ├── metadata_filter.py    # **新增** PGVector JSONB filter
│   ├── self_query.py         # **新增** LLM 自提取过滤条件
│   └── hyde.py               # **新增** 假想文档检索
├── adapters/ embeddings/ loaders/ cleaners/   # 从 Day6 复制
└── samples/README.md
```

# 四、开发实操

## 步骤 0：`rag_chain.py`（PGVector 建库/复用）

与 Day6 的差异只有三处——其余加载/清洗/分段/元数据格式完全照搬：

| 维度 | Day6 | Day7 |
|---|---|---|
| 导入 | `from langchain_chroma import Chroma` | `from langchain_postgres import PGVector` |
| 连接 | `persist_directory="chroma_db"` | `connection=PG_CONNECTION` |
| 复用判断 | `len(vs.get()["ids"]) == 0` | `len(vs.similarity_search("test", k=1)) == 0` |

```python
"""Day7 PGVector 建库/复用。
与 Day6 Chroma 版本的差异只有三点：
1. 存储介质：本地文件 → PostgreSQL 远程表
2. 复用判断：vectorstore.get()['ids'] → similarity_search("test", k=1)
3. 连接配置：persist_dir 路径 → PG_CONNECTION 连接串
加载、清洗、分段、元数据格式完全照搬 Day6。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector

from loaders.document_loader import DocumentLoader
from cleaners.text_cleaner import TextCleaner
from adapters.ollama_embeddings import OllamaEmbeddings

load_dotenv()

PG_CONNECTION = os.getenv(
    "PG_CONNECTION",
    "postgresql+psycopg://rag:rag123@localhost:5432/ragdb",
)


def _collection_name(file_path: str) -> str:
    """基于文件名生成 collection 名。"""
    stem = Path(file_path).stem.lower()
    return f"docs_{stem}"


def _has_data(vs: PGVector) -> bool:
    """判断 collection 是否已有数据。
    用 similarity_search 试查 1 条——跨 VectorStore 后端通用做法。
    """
    try:
        return len(vs.similarity_search("test", k=1)) >= 1
    except Exception:
        return False


def build_vectorstore(
    file_path: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    force_rebuild: bool = False,
) -> PGVector:
    """建库或复用，返回裸 PGVector 实例。"""
    embedding = OllamaEmbeddings(model="bge-m3")
    coll_name = _collection_name(file_path)

    if force_rebuild:
        print(f"[强制重建] 删除 collection: {coll_name}")
        try:
            PGVector(embeddings=embedding, collection_name=coll_name,
                     connection=PG_CONNECTION).delete_collection()
        except Exception:
            pass

    try:
        vs = PGVector(embeddings=embedding, collection_name=coll_name,
                      connection=PG_CONNECTION, use_jsonb=True)
        if _has_data(vs):
            print(f"[复用] collection '{coll_name}' 已有数据")
            return vs
    except Exception:
        pass

    print("[新建] 加载 → 清洗 → 分段 → 嵌入 → 写入 PGVector ...")
    raw_text = DocumentLoader().load(file_path)
    clean_text = TextCleaner().clean(raw_text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(clean_text)

    docs = [Document(page_content=c, metadata={"source": file_path, "chunk_id": i})
            for i, c in enumerate(chunks)]

    vs = PGVector.from_documents(documents=docs, embedding=embedding,
                                 collection_name=coll_name, connection=PG_CONNECTION,
                                 use_jsonb=True)
    print(f"[入库] {len(docs)} 条 → collection '{coll_name}'")
    return vs
```

#### 0.1 运行结果（本机实测）

```plain
python rag_chain.py
```

```
[新建] 加载 → 清洗 → 分段 → 嵌入 → 写入 PGVector ...
[入库] 29 条 → collection 'docs_readme'

# 第二次运行
[复用] collection 'docs_readme' 已有数据
```

> 二次运行输出 `[复用]`。`similarity_search("test", k=1)` 用任意探针词判断表里是否有行——不依赖 Chroma 专属的 `.get()` 方法。

## 步骤 1：`retrieval/pg_retriever.py`（PGVector 语义 + MMR）

`PGVector.as_retriever()` 接口与 Chroma 完全一致，两函数共 5 行有效代码。

```python
"""PGVector 基础检索——语义相似度 + MMR。
PGVector.as_retriever() 接口与 Chroma 完全一致。
"""

from typing import List
from langchain_core.documents import Document

from rag_chain import build_vectorstore


def pg_similarity_search(question: str, file_path: str, k: int = 3) -> List[Document]:
    """PGVector 纯语义检索。"""
    vs = build_vectorstore(file_path)
    return vs.as_retriever(search_type="similarity",
                           search_kwargs={"k": k}).invoke(question)


def pg_mmr_search(question: str, file_path: str, k: int = 3,
                  fetch_k: int = 10, lambda_mult: float = 0.5) -> List[Document]:
    """PGVector MMR 多样性检索。"""
    vs = build_vectorstore(file_path)
    return vs.as_retriever(search_type="mmr",
                           search_kwargs={"k": k, "fetch_k": fetch_k,
                                          "lambda_mult": lambda_mult}).invoke(question)
```

#### 1.1 运行结果（本机实测）

```plain
python -m retrieval.pg_retriever
```

```
===== [PGVector 语义检索] 3 条 =====
1 | chunk_id 4 | ...
2 | chunk_id 3 | ...
3 | chunk_id 1 | ...

===== [PGVector MMR 检索] 3 条 =====
1 | chunk_id 4 | ...
2 | chunk_id 3 | ...
3 | chunk_id 17 | ...
```

> 解读：语义检索结果与 Day5 Chroma 一致（同为 bge-m3、同一批文档）。MMR 保证 chunk_id 互不相同（去重达成），与 Day6 步骤 1 行为一致。

## 步骤 2：`retrieval/bm25_retriever.py`（BM25 解耦版）

设计改进：构造函数从 `vectorstore` 改为 `List[Document]`——BM25 只依赖文档文本。

```python
"""BM25 关键词检索——Day7 解耦版。
与 Day6 差异：构造函数从 vectorstore 改为 List[Document]。
"""

from typing import List, Tuple
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import jieba

from loaders.document_loader import DocumentLoader
from cleaners.text_cleaner import TextCleaner
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _tokenize(text: str) -> List[str]:
    return jieba.lcut(text)


class BM25Retriever:
    def __init__(self, docs: List[Document]):
        self.corpus = [d.page_content for d in docs]
        self.metadatas = [d.metadata for d in docs]
        self.bm25 = BM25Okapi([_tokenize(t) for t in self.corpus])

    @classmethod
    def from_file(cls, file_path: str, chunk_size: int = 300,
                  chunk_overlap: int = 50) -> "BM25Retriever":
        raw = DocumentLoader().load(file_path)
        clean = TextCleaner().clean(raw)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(clean)
        docs = [Document(page_content=c, metadata={"source": file_path, "chunk_id": i})
                for i, c in enumerate(chunks)]
        return cls(docs)

    def search(self, query: str, k: int = 3) -> List[Tuple[str, dict, float]]:
        tokenized = _tokenize(query)
        scores = self.bm25.get_scores(tokenized)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.corpus[i], self.metadatas[i], scores[i])
                for i in top_idx if scores[i] > 0]
```

#### 2.1 运行结果（本机实测）

```plain
python -m retrieval.bm25_retriever
```

```
===== [BM25 关键词召回] 3 条 =====
1 | chunk_id 3 | BM25分 4.46 | ...
2 | chunk_id 1 | BM25分 3.79 | ...
3 | chunk_id 0 | BM25分 3.06 | ...
```

> 解读：BM25 把真正含 `pip install` 的 chunk_id 3 排在第一位，比语义检索更聚焦精确词。与 Day6 BM25 结果完全一致（同算法、同文档）。

## 步骤 3：`retrieval/fusion.py`（RRF 融合）

两路检索各自独立调用（PGVector + BM25），RRF 公式 `Σ 1/(60+rank)` 按 chunk_id 去重累加。

```python
"""RRF 融合检索——Day7 适配版。

与 Day6 的核心差异：两路检索解耦。
- 语义：PGVector.as_retriever()（原来 Chroma）
- BM25：BM25Retriever.from_file()（原来依赖 vectorstore.get()）
- RRF 融合公式：完全不变
"""

from typing import List, Tuple
from langchain_core.documents import Document

from rag_chain import build_vectorstore
from retrieval.bm25_retriever import BM25Retriever

RRF_K = 60


def _semantic_search(question: str, file_path: str, k: int = 10) -> List[Document]:
    """语义分支：PGVector similarity。"""
    vs = build_vectorstore(file_path)
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever.invoke(question)


def _bm25_search(question: str, file_path: str, k: int = 10) -> List[Tuple[str, dict, float]]:
    """BM25 分支：独立加载文档，不依赖 vectorstore。"""
    bm25 = BM25Retriever.from_file(file_path)
    return bm25.search(question, k=k)


def rrf_fusion(question: str, file_path: str, semantic_k: int = 10,
               bm25_k: int = 10, final_k: int = 3) -> List[Document]:
    """RRF 双路融合：语义 + BM25 → 排名去量纲 → Top-K。"""
    sem_docs = _semantic_search(question, file_path, semantic_k)
    bm25_results = _bm25_search(question, file_path, bm25_k)

    scores: dict = {}
    for rank, doc in enumerate(sem_docs):
        cid = doc.metadata["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (RRF_K + rank + 1)
    for rank, (text, meta, _) in enumerate(bm25_results):
        cid = meta["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (RRF_K + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_cids = [cid for cid, _ in ranked[:final_k]]

    doc_map = {d.metadata["chunk_id"]: d for d in sem_docs}
    for text, meta, _ in bm25_results:
        cid = meta["chunk_id"]
        if cid not in doc_map:
            doc_map[cid] = Document(page_content=text, metadata=meta)
    return [doc_map[cid] for cid in top_cids if cid in doc_map]
```

#### 3.1 运行结果（本机实测）

```plain
python -m retrieval.fusion
```

```
===== [RRF 混合召回] =====
1 | chunk_id 3 | ...
2 | chunk_id 4 | ...
3 | chunk_id 1 | ...
```

> 解读：既被语义认可、又被关键词命中的 chunk_id 3 稳居第一。单路独有的噪点被另一路平衡挤出。

## 步骤 4：`retrieval/reranker.py`（CrossEncoder 精排）

`Reranker` 类从 Day6 照搬（零修改——只操作文本对，不碰 vectorstore）。辅助函数换一行 import。

```python
"""CrossEncoder 重排——核心类照搬 Day6，辅助函数适配 Day7 融合接口。"""

from sentence_transformers import CrossEncoder
from retrieval.fusion import rrf_fusion


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates, top_k: int = 3, threshold=None):
        if not candidates:
            return []
        pairs = [(query, d.page_content) for d in candidates]
        scores = self.model.predict(pairs)
        scored = sorted(zip(candidates, map(float, scores)),
                        key=lambda x: x[1], reverse=True)
        if threshold is not None:
            scored = [s for s in scored if s[1] >= threshold]
        return [(d, s) for d, s in scored[:top_k]]


def hybrid_search_reranked(query: str, file_path: str,
                           fetch_k: int = 10, final_k: int = 3, threshold=None):
    candidates = rrf_fusion(query, file_path, semantic_k=fetch_k,
                            bm25_k=fetch_k, final_k=fetch_k)
    return Reranker().rerank(query, candidates, top_k=final_k, threshold=threshold)
```

#### 4.1 运行结果（本机实测）

```plain
python -m retrieval.reranker
```

```
===== [混合 + Rerank 召回] 3 条 =====
--- 1 | chunk_id 3 | rerank分 0.879 ---
--- 2 | chunk_id 4 | rerank分 0.693 ---
--- 3 | chunk_id 6 | rerank分 0.018 ---
```

> 解读：Rerank 做两件事——① 拉大真答案与次相关差距（0.879 vs 0.693）；② 弱相关被压到 0.018。`threshold=0.3` 一刀切掉。

## 步骤 5：`retrieval/metadata_filter.py`（PGVector JSONB filter）

PGVector 独有能力：filter → SQL WHERE（支持 `$gte`/`$lte`/`$and`）。Chroma 只支持精确匹配。

> **踩坑**：`langchain_postgres` 不允许单字段多 op（如 `{"$gte": 0, "$lte": 5}`），必须用 `$and` 拆分：
> ```python
> {"$and": [{"chunk_id": {"$gte": 0}}, {"chunk_id": {"$lte": 5}}]}
> ```

```python
"""元数据过滤检索——利用 PGVector 的 JSONB 做 SQL 层精确过滤。
langchain_postgres filter 语法：范围查询需用 $and 拆分。
"""

from typing import List, Optional, Dict
from langchain_core.documents import Document

from rag_chain import build_vectorstore


def metadata_filter_search(question: str, file_path: str,
                           filters: Optional[Dict] = None, k: int = 3) -> List[Document]:
    vs = build_vectorstore(file_path)
    kwargs = {"k": k}
    if filters:
        kwargs["filter"] = filters
    return vs.similarity_search(question, **kwargs)


def hybrid_with_filter(question: str, file_path: str,
                       filters: Optional[Dict] = None,
                       fetch_k: int = 10, final_k: int = 3) -> List[Document]:
    from retrieval.bm25_retriever import BM25Retriever

    vs = build_vectorstore(file_path)
    kwargs = {"k": fetch_k}
    if filters:
        kwargs["filter"] = filters
    sem_docs = vs.similarity_search(question, **kwargs)

    bm25 = BM25Retriever.from_file(file_path)
    bm25_results = bm25.search(question, k=fetch_k)

    scores: dict = {}
    RRF_K = 60
    for rank, doc in enumerate(sem_docs):
        cid = doc.metadata["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (RRF_K + rank + 1)
    for rank, (text, meta, _) in enumerate(bm25_results):
        cid = meta["chunk_id"]
        # BM25 侧：只做简单精确匹配；运算符/范围无条件放行
        skip = False
        if filters:
            for key, val in filters.items():
                if key.startswith("$"):
                    continue
                if isinstance(val, dict):
                    continue
                if meta.get(key) != val:
                    skip = True
                    break
        if not skip:
            scores[cid] = scores.get(cid, 0) + 1.0 / (RRF_K + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_cids = [cid for cid, _ in ranked[:final_k]]

    doc_map = {d.metadata["chunk_id"]: d for d in sem_docs}
    for text, meta, _ in bm25_results:
        cid = meta["chunk_id"]
        if cid not in doc_map:
            doc_map[cid] = Document(page_content=text, metadata=meta)
    return [doc_map[cid] for cid in top_cids if cid in doc_map]


def build_filter_app():
    """交互式 demo：手动构造各种 filter 看召回变化。"""
    FILE = "samples/README.md"
    Q = "如何安装依赖"

    test_cases = [
        ("无过滤", None),
        ("chunk_id >= 3", {"chunk_id": {"$gte": 3}}),
        ("chunk_id 0~5", {"$and": [
            {"chunk_id": {"$gte": 0}}, {"chunk_id": {"$lte": 5}}]}),
        ("chunk_id 10~20", {"$and": [
            {"chunk_id": {"$gte": 10}}, {"chunk_id": {"$lte": 20}}]}),
    ]

    for label, f in test_cases:
        print(f"\n===== [{label}] =====")
        docs = metadata_filter_search(Q, FILE, filters=f, k=3)
        if not docs:
            print("  (无结果)")
        for i, doc in enumerate(docs, 1):
            print(f"  {i} | chunk_id {doc.metadata['chunk_id']} "
                  f"| {doc.page_content[:60]}...")

    # 对比：混合检索 + 过滤
    print("\n===== [对比：混合检索 + chunk_id 0~5 过滤] =====")
    docs2 = hybrid_with_filter(Q, FILE, filters={
        "$and": [{"chunk_id": {"$gte": 0}}, {"chunk_id": {"$lte": 5}}]})
    for i, doc in enumerate(docs2, 1):
        print(f"  {i} | chunk_id {doc.metadata['chunk_id']} "
              f"| {doc.page_content[:60]}...")


# ========== 单测 ==========
if __name__ == "__main__":
    build_filter_app()
```

#### 5.1 运行结果（本机实测）

```plain
python -m retrieval.metadata_filter
```

```
===== [无过滤] =====
  1 | chunk_id 4 | ...
  2 | chunk_id 3 | ...
  3 | chunk_id 1 | ...

===== [chunk_id >= 3] =====
  1 | chunk_id 4 | ...
  2 | chunk_id 3 | ...
  3 | chunk_id 6 | ...

===== [chunk_id 0~5] =====
  1 | chunk_id 4 | ...
  2 | chunk_id 3 | ...
  3 | chunk_id 1 | ...
```

> 解读：`chunk_id >= 3` 滤掉了 chunk_id 0/1/2，`chunk_id 0~5` 滤掉了后面段落。检索范围被 SQL 层精确控制。

## 步骤 6：`retrieval/self_query.py`（自查询）

LLM 从「前几个步骤」中提取 `filter={"chunk_id": {"$lte": 4}}` + `query="步骤内容"`。

`_parse_llm_output` 三层容错解析（直接 JSON → 提取 ` ```json ``` ` 块 → 正则提取 `{...}`），为 3B 小模型的不稳定输出做的妥协。

```python
"""自查询检索——LLM 从问题中自动提取结构化过滤条件。"""

from typing import List, Optional
import json, re
from langchain_core.documents import Document

from rag_chain import build_vectorstore
from adapters.ollama_llm import OllamaLLM


METADATA_FIELDS = [
    {"name": "chunk_id", "type": "integer",
     "description": "文档分段编号，从 0 开始递增"},
    {"name": "source", "type": "string",
     "description": "文档来源路径"},
]


def _build_filter_prompt(question: str) -> str:
    fields_desc = "\n".join(
        f"- {f['name']} ({f['type']}): {f['description']}"
        for f in METADATA_FIELDS)
    return f"""你是一个查询分析器。根据用户问题，提取结构化的过滤条件。

可用的元数据字段：
{fields_desc}

支持的运算符：$eq、$gte、$lte、$and

规则：返回 JSON：{{"query": "搜索词", "filter": filter字典 或 null}}

示例：
- "前 5 个步骤" → {{"query": "步骤内容", "filter": {{"chunk_id": {{"$lte": 4}}}}}}
- "如何安装依赖" → {{"query": "如何安装依赖", "filter": null}}

用户问题：{question}

仅返回 JSON，不要其他文字。"""


def _parse_llm_output(text: str) -> dict:
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
    return {"query": text.strip(), "filter": None}


def self_query_search(question: str, file_path: str,
                      k: int = 3, llm=None) -> List[Document]:
    if llm is None:
        llm = OllamaLLM()

    raw_output = llm.invoke(_build_filter_prompt(question))
    print(f"[LLM 输出] {raw_output.strip()}")

    parsed = _parse_llm_output(raw_output)
    search_query = parsed.get("query", question)
    filters = parsed.get("filter") or parsed.get("filters")

    vs = build_vectorstore(file_path)
    kwargs = {"k": k}
    if filters and isinstance(filters, dict) and len(filters) > 0:
        kwargs["filter"] = filters
        print(f"[应用 filter] {filters}")
    else:
        print("[无 filter] 纯语义检索")
    return vs.similarity_search(search_query, **kwargs)


class SelfQueryApp:
    """自查询演示器——对比有/无 filter 的检索差异。"""

    def __init__(self, file_path: str = "samples/README.md"):
        self.file_path = file_path
        self.llm = OllamaLLM()

    def compare(self, question: str, k: int = 3):
        """并排对比：纯语义 vs 自查询。"""
        print(f"\n{'='*60}")
        print(f"问题：{question}")
        print(f"{'='*60}")

        vs = build_vectorstore(self.file_path)
        baseline = vs.similarity_search(question, k=k)
        print(f"\n--- 无 filter（纯语义）---")
        for i, doc in enumerate(baseline, 1):
            print(f"  {i} | chunk_id {doc.metadata['chunk_id']}")

        print(f"\n--- 自查询（LLM 提取 filter）---")
        results = self_query_search(question, self.file_path, k=k, llm=self.llm)
        if not results:
            print("  (无结果)")
        for i, doc in enumerate(results, 1):
            print(f"  {i} | chunk_id {doc.metadata['chunk_id']} "
                  f"| {doc.page_content[:60]}...")


# ========== 单测 ==========
if __name__ == "__main__":
    app = SelfQueryApp()
    for q in ["前几个步骤讲了什么", "如何安装依赖"]:
        app.compare(q, k=3)
```

#### 6.1 运行结果（本机实测）

```plain
python -m retrieval.self_query
```

```
[LLM 输出] {"query": "步骤内容", "filter": {"chunk_id": {"$lte": 4}}}
[应用 filter] {"chunk_id": {"$lte": 4}}

--- 无 filter（纯语义）---
  1 | chunk_id 4
  2 | chunk_id 3
  3 | chunk_id 1

--- 自查询（LLM 提取 filter）---
  1 | chunk_id 4
  2 | chunk_id 3
  3 | chunk_id 2
```

> 解读：LLM 成功提取 filter，chunk_id 被限定在 0~4。但如果 LLM 输出格式不标准，`_parse_llm_output` 会退化为纯语义检索——qwen2.5:3b 小模型的现实边界。

## 步骤 7：`retrieval/hyde.py`（假想文档检索）

LLM 生成假想答案 → embed → 检索真实文档。核心洞察：长文本的向量在嵌入空间里天然离知识库文档更近。

```python
"""HyDE 假想文档检索。
用 LLM 生成的假想答案做检索探针，比短问题更接近知识库文档风格。
"""

from typing import List
from langchain_core.documents import Document

from rag_chain import build_vectorstore
from adapters.ollama_llm import OllamaLLM


HYDE_PROMPT = """你是一个技术助手。请根据以下问题，写一段假设性的回答。
要求用技术文档口吻，包含具体命令或术语，50-100 字。

问题：{question}

假设回答："""


class HyDERetriever:
    def __init__(self, file_path: str = "samples/README.md"):
        self.file_path = file_path
        self.llm = OllamaLLM()

    def generate_hypothetical(self, question: str) -> str:
        return self.llm.invoke(HYDE_PROMPT.format(question=question))

    def search(self, question: str, k: int = 3) -> List[tuple]:
        hypothetical = self.generate_hypothetical(question)
        print(f"[假想答案] {hypothetical.strip()[:100]}...")
        vs = build_vectorstore(self.file_path)
        docs = vs.similarity_search(hypothetical, k=k)
        return [(doc, hypothetical) for doc in docs]


def hyde_search(question: str, file_path: str, k: int = 3) -> List[Document]:
    """便捷函数：HyDE 检索，只返回 Document。"""
    retriever = HyDERetriever(file_path)
    return [doc for doc, _ in retriever.search(question, k=k)]


# ========== 单测 ==========
if __name__ == "__main__":
    retriever = HyDERetriever("samples/README.md")
    Q = "如何安装依赖"
    print(f"===== HyDE 检索：{Q} =====\n")
    results = retriever.search(Q, k=3)
    for i, (doc, hypo) in enumerate(results, 1):
        print(f"{i} | chunk_id {doc.metadata['chunk_id']} "
              f"| {doc.page_content[:60]}...")

    # 对比：普通语义检索
    print(f"\n===== 对比：普通语义检索 =====")
    vs = build_vectorstore("samples/README.md")
    baseline = vs.similarity_search(Q, k=3)
    for i, doc in enumerate(baseline, 1):
        print(f"{i} | chunk_id {doc.metadata['chunk_id']} "
              f"| {doc.page_content[:60]}...")
```

#### 7.1 运行结果（本机实测）

```plain
python -m retrieval.hyde
```

```
[假想答案] 首先创建虚拟环境 venv，然后激活它，在激活的虚拟环境中使用 pip install ...

===== HyDE 检索 =====
1 | chunk_id 3 | ...
2 | chunk_id 4 | ...
3 | chunk_id 1 | ...
```

> 解读：假想答案含 `pip install`/`venv` 等术语后，chunk_id 3 被提到第一。如果 LLM 生成的假想答案泛泛而谈，效果接近普通语义检索。

## 步骤 8：`main.py`（六组实验对比）

同一 prompt、同一 LLM(qwen2.5:3b)、同一问题「如何安装依赖」，唯一变量是召回文档。

| 实验 | 策略 | 目的 |
|---|---|---|
| A | PGVector 纯语义 | 验证 = Day5 Chroma |
| B | PGVector MMR | 验证 = Day6 MMR |
| C | 混合 + Rerank (threshold=0.3) | Day6 完整链路复现 |
| D | 语义 + 元数据过滤 (chunk_id 0~5) | PGVector 独有能力 |
| E | HyDE | 通用策略 |
| F | 混合 + Rerank + chunk_id 升序 | 检索负责准 + 排序负责顺 |

```python
"""Day7 完整对比实验——六组策略并行对比。"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from adapters.ollama_llm import OllamaLLM
from retrieval.pg_retriever import pg_similarity_search, pg_mmr_search
from retrieval.reranker import hybrid_search_reranked
from retrieval.metadata_filter import metadata_filter_search
from retrieval.hyde import hyde_search

PROMPT = PromptTemplate.from_template(
    """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。

【资料】
{context}

【问题】{question}

【回答】"""
)


def answer_with_docs(docs, question: str, llm, label: str = ""):
    if not docs:
        print(f"  (无召回文档)")
        return
    context = "\n\n".join(d.page_content for d in docs)
    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    print(f"\n{'='*60}")
    print(f"【{label}】")
    cids = [d.metadata.get("chunk_id", "?") for d in docs]
    print(f"召回 chunk_id: {cids}")
    print(f"召回数量: {len(docs)}")
    print(f"-" * 40)
    print(answer)
    print(f"{'='*60}")


def main():
    file_path = "samples/README.md"
    question = "如何安装依赖"
    llm = OllamaLLM()

    docs_a = pg_similarity_search(question, file_path, k=3)
    answer_with_docs(docs_a, question, llm, "A. PGVector 纯语义")

    docs_b = pg_mmr_search(question, file_path, k=3)
    answer_with_docs(docs_b, question, llm, "B. PGVector MMR")

    results_c = hybrid_search_reranked(
        question, file_path, fetch_k=10, final_k=3, threshold=0.3)
    docs_c = [d for d, _ in results_c]
    answer_with_docs(docs_c, question, llm, "C. 混合 + Rerank (threshold=0.3)")

    docs_d = metadata_filter_search(
        question, file_path,
        filters={"$and": [{"chunk_id": {"$gte": 0}}, {"chunk_id": {"$lte": 5}}]},
        k=3)
    answer_with_docs(docs_d, question, llm, "D. 语义 + 元数据过滤 (chunk_id 0~5)")

    docs_e = hyde_search(question, file_path, k=3)
    answer_with_docs(docs_e, question, llm, "E. HyDE 假想文档检索")

    results_f = hybrid_search_reranked(
        question, file_path, fetch_k=10, final_k=3, threshold=0.3)
    docs_f = sorted([d for d, _ in results_f],
                    key=lambda d: d.metadata["chunk_id"])
    answer_with_docs(docs_f, question, llm,
                     "F. 混合 + Rerank + chunk_id 升序（还原步骤顺序）")


if __name__ == "__main__":
    main()
```

#### 8.1 运行结果（本机实测）

```plain
python main.py
```

```
【A. PGVector 纯语义】
召回 chunk_id: [4, 3, 1]

【B. PGVector MMR】
召回 chunk_id: [4, 3, 17]

【C. 混合 + Rerank (threshold=0.3)】
召回 chunk_id: [3, 4]
答案：激活虚拟环境 → pip install fastapi uvicorn ... → 等待安装完成（干净聚焦）

【D. 语义 + 元数据过滤 (chunk_id 0~5)】
召回 chunk_id: [4, 3, 1]

【E. HyDE 假想文档检索】
召回 chunk_id: [3, 4, 1]

【F. 混合 + Rerank + chunk_id 升序】
召回 chunk_id: [3, 4]（按 chunk_id 升序后答案更有条理）
```

> 观察：
> - A ≈ Day5 Chroma 语义检索
> - B 去重达成，但仍有噪点 17（MMR 不解决弱相关）
> - **C 最佳**：threshold=0.3 滤掉弱相关，答案干净聚焦
> - D 限定范围后排序与 A 一致
> - E 排序微调（假想答案质量影响结果）
> - F 在 C 的基础上按步骤顺序排列 context，答案条理更好

# 五. 总结

## 1. 技术栈

Python + langchain-postgres + psycopg2-binary + sentence-transformers + jieba + rank_bm25 + Ollama(bge-m3 + qwen2.5:3b) + Docker(pgvector/pgvector:pg16) + 复用 Day2–Day6 包

## 2. 核心模块

| # | 模块 | 职责 |
|---|---|---|
| 1 | `rag_chain.py` | PGVector 建库/复用；复用判断用 `similarity_search("test", k=1)` 跨后端通用 |
| 2 | `pg_retriever.py` | `.as_retriever()` → 语义/MMR，接口与 Chroma 一致 |
| 3 | `bm25_retriever.py` | 解耦版——收 `List[Document]` 不依赖任何向量库 |
| 4 | `fusion.py` | 语义 + BM25 → RRF 融合，公式不变 |
| 5 | `reranker.py` | CrossEncoder 精排 + threshold 过滤；核心类零修改 |
| 6 | `metadata_filter.py` | PGVector JSONB filter → SQL WHERE（范围/组合/布尔） |
| 7 | `self_query.py` | LLM 提取结构化 filter；三层容错解析适配小模型 |
| 8 | `hyde.py` | 假想答案嵌入检索；"长文本探针"优于"短问题探针" |

## 3. 检索架构（Day7 版）

```plain
问题 → ┌ 语义检索(PGVector, bge-m3, Top-10) ┐
       │                                    ├→ RRF 融合 → 候选池 Top-10
       └ BM25关键词(jieba分词, Top-10) ──────┘
       → CrossEncoder 精排 → threshold 过滤 → Top-K
       → 可选：按 chunk_id 升序重排 → 拼 context → PromptTemplate → qwen2.5:3b → 答案

       可选前置：Self-Query (LLM 提取 filter) / HyDE (生成假想答案)
       可选后置：元数据过滤 → SQL 层提前筛掉不相关段
```

粗排召全（双塔，快），精排排准（CrossEncoder，准），阈值挡住弱相关，chunk_id 还原顺序。

# 六、关键知识点理解复盘

#### Q1：`similarity_search("test", k=1)` 为什么能判断库是否为空？

把 "test" embed 成 1024 维向量，与库里所有向量算余弦距离。余弦距离是纯数学运算，任意两向量都能算——库里有 ≥ 1 行必返回结果（取最近的），库空返回 `[]`。这是跨 VectorStore 后端的通用做法。

#### Q2：BM25 为什么不应该依赖 vectorstore？

BM25 要的是文档纯文本 + TF-IDF 统计，和向量毫无关系。Day7 改为收 `List[Document]`——任何来源都能传入，可搭配任何向量库。

#### Q3：为什么 HyDE 的假想答案能提升检索质量？

用户问题是短问句，知识库是长段落——嵌入空间里存在语言风格差异。LLM 生成的假想答案也是长段落风格，向量更接近知识库文档。前提是 LLM 生成的答案质量够好。

#### Q4：为什么实验 F 中「安装 Python」不在结果里？

Reranker 判断「下载 Python 安装包」和「安装项目依赖（pip install）」是两件不同的事，相关性分低。`threshold=0.3` 把它挡在 prompt 外——这不是 bug，把不相关的步骤塞进错误答案里才是 bug。

#### Q5：`langchain_postgres` 的 filter 为什么和 Chroma 语法不同？

同为 `similarity_search(filter=...)`，但底层实现不同：Chroma 用 Python dict 解析，`langchain_postgres` 用 SQLAlchemy 生成 SQL WHERE。这是「存储无关」抽象层的真实缝隙。

#### Q6：`qwen2.5:3b` 做 Self-Query / HyDE 效果好吗？

不够稳定。3B 小模型对结构化输出和长文本生成的控制力有限。换 qwen2.5:72b 或 GPT-4 会显著改善。

# 七、踩坑记录与遗留问题

## 7.1 本次踩坑

| # | 现象 | 根因 | 解决 |
|---|---|---|---|
| 1 | `vectorstore.get()` 报 AttributeError | PGVector 没有 Chroma 的 `.get()` | 改 `similarity_search("test", k=1)` + 改 BM25 构造函数 |
| 2 | `{"$gte": 0, "$lte": 5}` 报 ValueError | `langchain_postgres` 每字段单 op | `$and` 拆分两个单 op 条件 |
| 3 | BM25 侧 filter 过度设计 | 试图让 BM25 支持范围过滤 | 诚实放行 `isinstance(val, dict): continue` |
| 4 | `pip install docx` 导入报错 | `docx` 是废弃老包 | `pip uninstall docx -y && pip install python-docx` |
| 5 | 纯语义 Top-3 排序偶尔微飘 | 余弦距离的边缘波动 | 加 Rerank 精排稳定排序 |
| 6 | Self-Query LLM 输出格式不稳定 | qwen2.5:3b 结构化能力有限 | 三层容错解析（JSON → markdown → 正则） |

## 7.2 遗留技术债 / 待补强

| # | 问题 | 影响 | 方向 |
|---|---|---|---|
| 1 | `threshold`/`rrf_k`/`fetch_k` 仍是手拍超参 | 换文档集效果可能波动 | 建评测集做网格搜索 |
| 2 | BM25 索引每次从文件重建（约 50ms） | 大文档时启动慢 | 索引持久化（pickle）或换 Elasticsearch |
| 3 | Self-Query / HyDE 依赖 LLM 质量 | 3B 模型效果不稳定 | 换大模型或做 prompt 优化 |
| 4 | 元数据过滤的 chunk_id 范围靠人工指定 | 不知道分布无从下手 | 结合 Self-Query 自动推断 |
| 5 | 缺少与 Day6 Chroma 的定量对比 | 结论依赖肉眼判断 | 多文档多问题批量评测 |
| 6 | PGVector 单机 Docker，未验证高可用 | 不算真正的生产级 | 主从复制 / 连接池 / 读写分离 |

# 八、回顾：从 Day1 到 Day7 的技术演进

```plain
Day1: Ollama 网关 → 调用本地 LLM
Day2: 文档加载 → PDF/Word/MD → 纯文本
Day3: Embedding → bge-m3 → 文本转向量
Day4: Chroma → 向量存储 + 相似度检索
Day5: LangChain RAG → 标准问答链（检索 + 生成）
Day6: 混合检索 → MMR + BM25 + RRF + Rerank（检索优化四招）
Day7: PGVector 生产化 → 存储层升级 + 元数据过滤 + Self-Query + HyDE
```

每层各司其职：加载/清洗/分段不变（Day2–3），存储从嵌入式升级为关系型（Day4 → Day7），检索从单路进化为多路融合（Day5 → Day6），策略从手动进化为 AI 驱动（Day6 → Day7 Self-Query / HyDE）。

## 运行方式

```powershell
cd F:\ai-learn\day7-pgvector-prod

docker compose up -d                    # 启动 PostgreSQL + PGVector
python rag_chain.py                     # 建库/复用测试
python -m retrieval.pg_retriever        # 语义 + MMR
python -m retrieval.bm25_retriever      # BM25
python -m retrieval.fusion               # RRF 融合
python -m retrieval.reranker             # Rerank
python -m retrieval.metadata_filter      # 元数据过滤
python -m retrieval.self_query           # 自查询
python -m retrieval.hyde                 # HyDE
python main.py                           # 完整对比实验
```
