# Day12：生产级私有 RAG 系统

# 一、今日目标

Day1~Day11 我们依次掌握了：Ollama 本地模型（Day1）、文档加载（Day2）、Embedding（Day3）、向量库 Chroma（Day4）、LangChain RAG（Day5）、混合检索（Day6）、PGVector 生产化（Day7）、提示工程（Day8）、Agent 工具（Day9）、AutoGen 多智能体（Day10）、Ollama 调优（Day11）。

Day12 主题：**把前面所有能力整合成一个高内聚、低耦合的生产级私有 RAG 系统**，做到：

1. **分层架构**：`api → agents → core → config` 单向依赖，每层单一职责、可独立测试。
2. **配置外置**：所有连接串、模型名、分块参数集中在 `config/settings.py`，从 `.env` 读取，只此一处来源。
3. **混合检索**：向量召回（PGVector 余弦）+ BM25 关键词召回（jieba 中文分词）+ RRF 融合重排，纯本地、无重排模型。
4. **多智能体作答**：AutoGen `RoundRobinGroupChat` 组建 `retriever_agent`（整理检索要点）+ `answer_agent`（基于要点作答），知识库无关问题诚实拒答。
5. **可运维**：CLI 提供 `init/ingest/ask`，API 提供 `/ingest /ask /health`（已异步化）。

> 核心约束：本机 **6G 显存**、Ollama 本地模型 `qwen2.5:3b`（对话）+ `bge-m3`（向量化，1024 维）、PostgreSQL + PGVector（`rag/rag123@localhost:5432/ragdb`）。

# 二、先想清楚几个问题

#### Q1：为什么 Day12 不直接用 Day5 的 LangChain RAG，而要自己分层重写？

A：Day5 是「用框架快速跑通」，重点是理解 RAG 链路。Day12 是「工程化落地」，重点是**可控、可维护、可替换**。LangChain 把很多东西藏在 Chain 里，出问题时不好定位；自己分层（loaders/cleaners/chunkers/embeddings/vectorstore/retrieval）每一层都能单独跑、单独测、单独换实现。生产系统更看重后者。

#### Q2：为什么 PGVector 用 psycopg 直连，而不用 SQLAlchemy ORM？

A：我们只有一张表 `documents`，且用到了 PGVector 专属的 `vector` 类型和 `<=>` 余弦算子。ORM 对这种自定义类型支持别扭，反而增加复杂度。单表 + 专属算子的场景，直连 SQL 最直接、性能最好、也最好调试。

#### Q3：混合检索为什么选 RRF 而不是训练一个重排模型？

A：RRF（Reciprocal Rank Fusion）是**无监督、无参数**的排名融合算法，公式 `score = Σ 1/(k+rank)`，只在排名上做加权，不依赖模型。在本地 6G 显存、数据量中等的私有场景下，RRF 足够好且零额外成本；训练重排模型（如 bge-reranker）要更多显存且收益边际，故不选。

#### Q4：AutoGen 为什么用 RoundRobinGroupChat，而不是 SelectorGroupChat？

A：Selector 需要 LLM 自己判断「下一步该谁说话」，这依赖模型的 function calling / 决策能力。我们用的 `qwen2.5:3b` 在 6G 上不够稳，Selector 容易选错或卡死。RoundRobin（轮流发言）是确定性调度——固定顺序 retriever → answer 各说一轮，简单可靠，不依赖模型的「自我调度」能力。

#### Q5：为什么 BM25 只在向量召回的候选集上建索引，而不是全表？

A：两点。（1）**性能**：全表 BM25 要 `fetch_all()` 把所有文档拉进内存建索引，文档多时会爆炸；（2）**语义合理性**：向量召回已筛出语义相关子集，BM25 只在这个子集上做关键词互补，既避免全表扫描，也符合「先语义粗排、再关键词精排」的混合检索直觉。向量无命中时直接返回空，不再走 BM25。

#### Q6：`init` 为什么需要 `--yes` 才真正清空？

A：`init` 语义是「重置」——`DROP TABLE IF EXISTS documents` 会清空全部已入库数据且不可恢复。加 `--yes` 二次确认，防止手滑把知识库清空。不加 `--yes` 只提示不执行。

# 三、准备工作

## 步骤 1：确认环境与模型

```powershell
ollama list                       # 应有 qwen2.5:3b、bge-m3
docker ps                         # Day7 的 Postgres+PGVector 容器在跑
# 或确认 5432 端口可连
```

本机应有：`qwen2.5:3b`（对话）、`bge-m3`（向量化，1024 维）。PGVector 库信息：库 `ragdb`、用户 `rag`、密码 `rag123`、端口 `5432`。

## 步骤 2：建目录与依赖

```powershell
cd f:\ai-learn
mkdir day12-rag-system
cd day12-rag-system
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`（精简版，仅含实际用到的包，附踩坑注释）：

```text
# Day12 私有 RAG 系统依赖（精简版，仅含实际用到的包）
# 安装：pip install -r requirements.txt

# ===== 配置 =====
pydantic-settings==2.15.0      # 从 .env 读取配置（BaseSettings）
python-dotenv==1.2.2           # .env 文件加载支持

# ===== 向量化 / 数据库 =====
httpx==0.26.0                  # 调用 Ollama /api/embed（注意：必须 0.26.0，
                               # 更高版本与 autogen-ext 的 proxies 参数冲突）
psycopg==3.2.3                 # PGVector 直连（不用 +psycopg 前缀，psycopg.connect 不支持）
pgvector==0.3.6                # vector 类型支持（建表/余弦算子）
pypdf==6.15.0                  # 加载 PDF 文档

# ===== 检索 =====
rank-bm25==0.2.2               # BM25 关键词召回
jieba==0.42.1                  # 中文分词（提升 BM25 质量；未装时自动回退字符级）

# ===== LLM / 多智能体 =====
openai==1.40.0                 # AutoGen 走 OpenAI 兼容接口连本地 Ollama
autogen-agentchat==0.7.5       # RoundRobinGroupChat 多智能体编排
autogen-core==0.7.5
autogen-ext==0.7.5

# ===== API / CLI =====
fastapi==0.115.0               # /ingest /ask /health 接口
uvicorn==0.30.6                # ASGI 服务
```

> 版本踩坑：① `httpx` 必须用 0.26.0，高版本与 autogen-ext 的 `proxies` 参数冲突；② `psycopg` 不要写 `postgresql+psycopg://`（SQLAlchemy 风格），`psycopg.connect` 只认 `postgresql://`；③ `openai` 用 1.40.0，AutoGen 0.7.5 的 OpenAI 客户端需传 `model_info` dict。

## 步骤 3：建目录结构与 .env

```powershell
mkdir config core core\loaders core\cleaners core\chunkers core\embeddings core\vectorstore core\retrieval agents api
```

`.env`（放在 `day12-rag-system/.env`）：

```ini
ollama_base_url=http://localhost:11434
embedding_model=bge-m3
embedding_dimension=1024
llm_model=qwen2.5:3b

postgres_host=localhost
postgres_port=5432
postgres_user=rag
postgres_password=rag123
postgres_db=ragdb

chunk_size=500
chunk_overlap=80
top_k=8
rerank_top_n=4
```

# 四、开发实操

## 步骤 4：配置层 config/settings.py

单一配置源，所有模块通过 `get_settings()` 读取，配置只此一处。

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "bge-m3"
    embedding_dimension: int = 1024
    llm_model: str = "qwen2.5:3b"

    # Postgres
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "rag"
    postgres_password: str = "rag123"
    postgres_db: str = "ragdb"

    # 检索 / 分块
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 8
    rerank_top_n: int = 4

    @property
    def database_url(self) -> str:
        # psycopg.connect 只认 postgresql://，不支持 +psycopg 前缀
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
```

## 步骤 5：向量化层 core/embeddings/client.py

封装 Ollama `/api/embed`，复用 httpx 连接，`atexit` 关闭，单例懒加载。

```python
import atexit
import httpx

from config import get_settings

_settings = get_settings()
_DIMENSION = _settings.embedding_dimension


class EmbeddingClient:
    def __init__(self) -> None:
        self._base_url = _settings.ollama_base_url.rstrip("/")
        self._model = _settings.embedding_model
        self._client = httpx.Client(timeout=300.0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        vectors = [item["embedding"] for item in data["embeddings"]]
        if len(vectors) != len(texts):
            raise ValueError("向量数量与输入文本数量不一致")
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def close(self) -> None:
        self._client.close()


_embedding_client: EmbeddingClient | None = None


def get_embedding_client() -> EmbeddingClient:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client


atexit.register(lambda: _embedding_client.close() if _embedding_client else None)
```

## 步骤 6：向量存储层 core/vectorstore/store.py

PGVector 直连，建表 + HNSW 索引 + 余弦检索 + 重置。

```python
import psycopg


class VectorStore:
    def __init__(self) -> None:
        from config import get_settings
        self._settings = get_settings()
        self._dim = self._settings.embedding_dimension

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self._settings.database_url)

    def init(self) -> None:
        """
        重置初始化：删表重建（清空全部文档）。可重复调用。
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute("DROP TABLE IF EXISTS documents;")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS documents (
                        id        BIGSERIAL PRIMARY KEY,
                        content   TEXT NOT NULL,
                        source    TEXT,
                        embedding vector({self._dim})
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS documents_embedding_idx
                    ON documents USING hnsw (embedding vector_cosine_ops);
                    """
                )
            conn.commit()

    def add(self, items: list[dict]) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO documents (content, source, embedding) "
                    "VALUES (%s, %s, %s::vector)",
                    [(it["content"], it.get("source"), it["embedding"]) for it in items],
                )
            conn.commit()

    def search(self, qvec: list[float], top_k: int = 8) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, content, source, 1 - (embedding <=> %s::vector) AS score "
                    "FROM documents ORDER BY embedding <=> %s::vector LIMIT %s;",
                    (qvec, qvec, top_k),
                )
                return [
                    {"id": r[0], "content": r[1], "source": r[2], "score": r[3]}
                    for r in cur.fetchall()
                ]

    def count(self) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM documents;")
                return cur.fetchone()[0]
```

> `<=>` 是 PGVector 的余弦距离算子，`1 - 距离` 即余弦相似度。`vector_cosine_ops` 是 HNSW 索引的余弦构建方式。

## 步骤 7：文档处理三件套（loaders / cleaners / chunkers）

`core/loaders/text_loader.py`：

```python
import os


def load_document(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"不支持的文档类型: {ext}")
```

`core/cleaners/text_cleaner.py`：

```python
import re


def clean_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)  # 去控制字符
    text = re.sub(r"[ \t]+", " ", text)                        # 合并空格
    text = re.sub(r"\n{3,}", "\n\n", text)                    # 合并空行
    return text.strip()
```

`core/chunkers/text_chunker.py`：

```python
from config import get_settings


def chunk_text(text: str) -> list[str]:
    s = get_settings()
    size, overlap = s.chunk_size, s.chunk_overlap
    if size <= overlap:
        raise ValueError("chunk_size 必须大于 chunk_overlap")
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += size - overlap
    return chunks
```

## 步骤 8：混合检索 core/retrieval/hybrid.py

向量召回（PGVector）+ BM25（候选集上）+ RRF 融合。

```python
from collections import defaultdict

from rank_bm25 import BM25Okapi

from config import get_settings
from core.embeddings import get_embedding_client
from core.vectorstore import VectorStore
import jieba

jieba.setLogLevel("ERROR")


def _tokenize(text: str) -> list[str]:
    """
    中文/英文分词，供 BM25 使用（依赖 jieba，见 requirements.txt）。

    - 用 jieba 做中文分词，质量最好；
    - 英文统一转小写，并只保留中英文词（剔除标点/符号）。
    """
    import re

    text = text.lower()
    raw = jieba.lcut(text)
    return [t for t in raw if re.fullmatch(r"[\w\u4e00-\u9fff]+", t)]


def _bm25_rank(query: str, corpus: list[dict]) -> list:
    if not corpus:
        return []
    tokenized = [_tokenize(d["content"]) for d in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)
    return [corpus[i]["id"] for i in ranked]


def hybrid_retrieve(query: str, k: int = 8, n: int = 4) -> list[dict]:
    store = VectorStore()
    client = get_embedding_client()
    qvec = client.embed_query(query)
    settings = get_settings()
    k = k or settings.top_k
    n = n or settings.rerank_top_n

    vector_hits = store.search(qvec, top_k=k)
    if not vector_hits:
        return []

    id_to_doc = {d["id"]: d for d in vector_hits}
    bm25_ranked_ids = _bm25_rank(query, vector_hits)

    rrf = defaultdict(float)
    RRF_K = 60
    for rank, doc in enumerate(vector_hits, start=1):
        rrf[doc["id"]] += 1.0 / (RRF_K + rank)
    for rank, doc_id in enumerate(bm25_ranked_ids, start=1):
        rrf[doc_id] += 1.0 / (RRF_K + rank)

    ranked_ids = sorted(rrf.keys(), key=lambda i: rrf[i], reverse=True)[:n]
    results = []
    for i in ranked_ids:
        d = id_to_doc[i]
        results.append({
            "id": d["id"],
            "content": d["content"],
            "source": d["source"],
            "score": round(rrf[i], 6),
        })
    return results
```

> RRF 公式 `1/(60+rank)`：排名越靠前权重越大；两路并集后取前 `n` 篇。BM25 只在 `vector_hits` 候选集上建索引（见 Q5）。

## 步骤 9：编排层 core/pipeline.py

```python
from config import get_settings
from core.chunkers.text_chunker import chunk_text
from core.cleaners.text_cleaner import clean_text
from core.embeddings import get_embedding_client
from core.loaders.text_loader import load_document
from core.retrieval.hybrid import hybrid_retrieve
from core.vectorstore import VectorStore


def ingest_file(path: str, source: str | None = None) -> int:
    raw = load_document(path)
    cleaned = clean_text(raw)
    chunks = chunk_text(cleaned)
    client = get_embedding_client()
    embeddings = client.embed(chunks)
    items = [
        {"content": c, "source": source or path, "embedding": e}
        for c, e in zip(chunks, embeddings)
    ]
    VectorStore().add(items)
    return len(items)


def retrieve(query: str, rerank_top_n: int | None = None) -> list[dict]:
    return hybrid_retrieve(query, n=rerank_top_n)
```

## 步骤 10：多智能体层 agents/

`agents/retriever_agent.py`：

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


def build_retriever_agent(client) -> AssistantAgent:
    return AssistantAgent(
        name="retriever_agent",
        model_client=client,
        system_message=(
            "你是检索整理助手。下面会给你一组检索到的文档片段，"
            "请从中提取与用户问题最相关的要点，用简洁的要点列表输出，"
            "不要作答，只整理事实要点。"
        ),
    )
```

`agents/answer_agent.py`（含诚实拒答）：

```python
from autogen_agentchat.agents import AssistantAgent


def build_answer_agent(client) -> AssistantAgent:
    return AssistantAgent(
        name="answer_agent",
        model_client=client,
        system_message=(
            "你是严谨的问答助手。请仅基于提供的检索要点回答用户问题，"
            "用简洁中文作答；若要点与用户问题无关或不足以回答"
            "（例如问候、自我介绍、闲聊），请明确说明"
            "'该问题不在我的知识库范围内'，不要强行用片段拼凑答案。"
        ),
    )
```

`agents/coordinator.py`：

```python
import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import get_settings
from core.retrieval.hybrid import hybrid_retrieve


def _build_client():
    s = get_settings()
    return OpenAIChatCompletionClient(
        model=s.llm_model,
        base_url=f"{s.ollama_base_url}/v1",
        api_key="ollama",
        model_info={
            "context_window": 8192,
            "function_calling": False,
            "json_output": False,
            "vision": False,
            "family": "unknown",
        },
    )


async def _run_team(query: str, rerank_top_n: int | None) -> str:
    client = _build_client()
    try:
        hits = hybrid_retrieve(query, n=rerank_top_n)
        context = "\n\n".join(
            f"[片段{i+1}] (来源: {h['source']}, 相似度: {h['score']:.3f})\n{h['content']}"
            for i, h in enumerate(hits)
        ) or "（未检索到相关文档）"

        retriever = build_retriever_agent(client)
        answer = build_answer_agent(client)

        team = RoundRobinGroupChat(
            [retriever, answer],
            termination_condition=TextMentionTermination("TERMINATE"),
            max_turns=4,
        )
        stream = team.run_stream(
            task=(
                f"用户问题：{query}\n\n"
                f"检索到的文档片段：\n{context}\n\n"
                "请 retriever_agent 先整理要点，再由 answer_agent 基于要点作答。"
            )
        )
        answer_text = ""
        last_any = ""
        async for message in stream:
            content = getattr(message, "content", None)
            if isinstance(content, str):
                last_any = content
                # 注意：speaker 取自消息的 source/sender，依赖 answer_agent 的 name 字段。
                # 若将来修改 answer_agent 的 name，此过滤会静默失效并退化为 last_any。
                speaker = getattr(message, "source", "") or getattr(message, "sender", "")
                if speaker == "answer_agent":
                    answer_text = content
        return answer_text or last_any or "（未生成回答）"
    finally:
        await client.close()


def ask(query: str, rerank_top_n: int | None = None) -> str:
    return asyncio.run(_run_team(query, rerank_top_n))
```

`agents/__init__.py`：

```python
from agents.coordinator import ask

__all__ = ["ask"]
```

> `run_stream` 返回的是一个**异步生成器**，必须用 `async for` 逐条消费，不能 `await`（会报 `async_generator can't be awaited`）。这是 Day12 实际踩过的坑。

## 步骤 11：API 层 api/

`api/schemas.py`：

```python
from pydantic import BaseModel


class IngestRequest(BaseModel):
    path: str
    source: str | None = None


class IngestResponse(BaseModel):
    chunks: int


class AskRequest(BaseModel):
    query: str
    rerank_top_n: int | None = None


class AskResponse(BaseModel):
    answer: str
```

`api/app.py`（已异步化）：

```python
import asyncio

from fastapi import FastAPI

from agents import ask as agent_ask
from api.schemas import AskRequest, AskResponse, IngestRequest, IngestResponse
from core.pipeline import ingest_file

app = FastAPI(title="Private RAG System", version="1.0.0")


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    """将本地文档文件入库（加载→清洗→分块→向量化→PGVector）。"""
    chunks = await asyncio.to_thread(ingest_file, req.path, source=req.source)
    return IngestResponse(chunks=chunks)


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """基于私有知识库回答用户问题（混合检索 + 多智能体）。"""
    answer = await asyncio.to_thread(agent_ask, req.query, req.rerank_top_n)
    return AskResponse(answer=answer)


@app.get("/health")
async def health() -> dict:
    """健康检查。"""
    return {"status": "ok"}
```

`api/__init__.py`：空或导出 app，按需。

## 步骤 12：CLI 入口 cli.py

```python
import argparse

from agents import ask as agent_ask
from core.pipeline import ingest_file
from core.vectorstore import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="私有 RAG 系统 CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="将文档入库")
    p_ingest.add_argument("path", help="文档路径（.txt/.md/.pdf）")
    p_ingest.add_argument("--source", default=None, help="来源标记")

    p_ask = sub.add_parser("ask", help="基于知识库问答")
    p_ask.add_argument("query", help="用户问题")
    p_ask.add_argument("--rerank-top-n", type=int, default=None, help="重排后返回数量")

    p_init = sub.add_parser("init", help="重置向量表（清空 documents 并重建）")
    p_init.add_argument("--yes", action="store_true", help="确认清空并重建，不加则只提示不执行")

    args = parser.parse_args()

    if args.cmd == "ingest":
        n = ingest_file(args.path, source=args.source)
        print(f"已入库文本块数：{n}")
    elif args.cmd == "ask":
        print(agent_ask(args.query, rerank_top_n=args.rerank_top_n))
    elif args.cmd == "init":
        if not args.yes:
            print("⚠️ 此操作会清空 documents 表全部数据。确认请加 --yes")
        else:
            VectorStore().init()
            print("向量表已重置（全部文档已清空并重建）")


if __name__ == "__main__":
    main()
```

> `ingest` 的 `path` 是**位置参数**，调用时直接 `python cli.py ingest "路径"`，不要写成 `path=路径`（会被当成未知参数导致 FileNotFoundError）。

# 五、运行验证

## 验证 1：初始化 + 入库 + 问答（CLI）

```powershell
cd f:\ai-learn\day12-rag-system
python cli.py init --yes
python cli.py ingest "F:\ai-learn\day7-pgvector-prod\README.md"
python cli.py ask "什么是向量数据库"
```

预期输出（问答部分，日志忽略）：

```
人工智能是计算机科学的一个分支。向量数据库用于存储高维向量。PGVector 是 PostgreSQL 的向量扩展，
并支持余弦相似度检索。
```

边界问题验证：

```powershell
python cli.py ask "介绍下你自己"
```

预期：`该问题不在我的知识库范围内`（诚实拒答，不强行拼凑）。

## 验证 2：API 服务

```powershell
# 终端 1
uvicorn api.app:app --host 0.0.0.0 --port 8000

# 终端 2
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"query\":\"什么是向量数据库\"}"
curl http://localhost:8000/health
```

预期：`/ask` 返回 JSON 回答，`/health` 返回 `{"status":"ok"}`。

# 六、踩坑记录（Day12 全部已解决）

1. **`database_url` 前缀**：`postgresql+psycopg://` 会报 `missing '=' after postgresql+psycopg`。`psycopg.connect` 只认 `postgresql://`。→ settings 里返回 `postgresql://`。

2. **httpx / autogen-ext 的 `proxies` 冲突**：高版本 httpx 移除/改了 `proxies` 参数，autogen-ext 调用时报 `TypeError: proxies`。→ 锁定 `httpx==0.26.0`。

3. **`model_info required`**：AutoGen 0.7.5 的 `OpenAIChatCompletionClient` 必须传 `model_info` dict（含 `context_window` 等），否则 `ValueError`。→ `_build_client` 补 `model_info`。

4. **`async_generator can't be awaited`**：`team.run_stream(...)` 返回异步生成器，不能 `await`，必须用 `async for` 消费。→ coordinator 改用 `async for message in stream`。

5. **消息提取取到 retriever 而非 answer**：多智能体两条消息都进了 stream，需按 `speaker == "answer_agent"` 过滤，并保留 `last_any` 兜底。→ coordinator 加过滤。

6. **jieba 加载日志刷屏**：`import jieba` 后首次分词会向 stderr 打印 `Building prefix dict...`。可在 `import jieba` 后加 `jieba.setLogLevel("ERROR")` 关闭（当前为忽略状态）。

# 七、已知约束与后续优化

- **jieba 日志**：当前每次 CLI 进程启动首次分词打印加载日志（已记录，未关闭）。
- **API 鉴权**：未实现（步骤 5 已跳过），生产需在中间件层补 API Key 校验。
- **全链路异步**：core 层为同步 I/O，API 用 `asyncio.to_thread` 隔离阻塞，未做 asyncpg / httpx.AsyncClient 全异步改造。
- **BM25 缓存**：每次 ask 在候选集上重建 `BM25Okapi`，候选集较小时可接受。
