# 一. 今日目标
Day5 跑通了 LangChain 标准 RAG 链，但暴露一个核心问题：**纯语义检索召回的 Top-3 里混有弱相关段（chunk_id 3/1），模型把它们当上下文"综合发挥"导致答案跑偏**（混入 Python/Ollama 安装等无关步骤）。Day6 用四招检索优化根治它：

1. **MMR 多样性检索**：惩罚与已选结果高度相似的候选，解决"召回冗余重复段"（Chroma 原生支持，零新依赖）。**仅作步骤 1 探索/对照，未进入最终 `main.py` 链路**——最终链路由 RRF 融合承担去重职责。
2. **BM25 关键词检索**：补语义向量对精确词（`pip install`、函数名、版本号）的短板
3. **RRF 倒数排名融合**：语义 + 关键词两路排名融合，互补短板、挤掉纯噪点
4. **CrossEncoder Rerank 精排**：对候选池算 (query, doc) 交互相关性分，配阈值过滤弱相关——检索质量提升最大的一招

最终 `main.py` 用**同一 prompt、同一 LLM(qwen2.5:3b)、同一问题**跑"Day5 基线 vs Day6 优化"并排对比，答案差异 100% 归因到检索优化。

# 二、先想清楚几个问题

#### Q1：MMR 是什么？
Maximal Marginal Relevance（最大边际相关）。先取最相关的，之后每选一条都**惩罚与已选高度相似的候选**，保证结果"相关且多样不重复"。三个旋钮：`fetch_k`（候选池）、`lambda_mult`（1=只看相关性，0=只看多样性）、`k`（最终条数）。

#### Q2：已有语义检索，为什么还要 BM25 关键词检索？
语义向量擅长"意思相近但用词不同"（"怎么装包"↔"安装依赖"），但对精确专有名词（版本号、命令串、报错码）可能漏召。BM25 只数词频/逆文档频率，**用词完全一致就命中**，两者正好互补。注意：中文必须先 jieba 分词，否则整段变一个 token，BM25 失效。

#### Q3：RRF 融合为什么用"排名"而不用"分数"？
语义检索给余弦相似度（0~1），BM25 给 TF-IDF 加权分（几到几十），**量纲完全不同，直接加权不公平**。RRF 把两者都转成排名，统一公式 `score = Σ 1/(rrf_k + rank)` 累加——量纲问题瞬间消失。`rrf_k=60` 是经典常数，防止头部排名分数过大掩盖其他信号。

#### Q4：Rerank 和召回有什么区别？
召回（向量/BM25）是"双塔"思路——query 和 doc 各自独立编码再比相似，快但粗。CrossEncoder 把 **query+doc 拼接**送进模型算交互相关性分，能捕捉细粒度匹配（如 `pip install` 是否真在 doc 里），准但慢——所以只对召回的少量候选（fetch_k=10）做精排，这是 RAG 标准的"粗排召回 → 精排重排"两阶段范式。

#### Q5：为什么 Day6 恰好治 Day5 的病？
Day5 跑偏的根因是弱相关段进了 prompt。最终 `main.py` 链路三层设防（MMR 仅步骤 1 探索，未堆叠进最终链路）：BM25 补精确词 → RRF 融合挤掉纯噪点、按 chunk_id 去重 → Rerank 阈值把弱相关直接挡在 prompt 之外，模型没有跑偏的原料。

# 三、准备工作
## 步骤 1：新建目录 + 新建虚拟环境

```powershell
cd F:\ai-learn
mkdir day6-hybrid-retrieval
cd day6-hybrid-retrieval
python -m venv venv
.\venv\Scripts\activate
pip install pypdf langchain docx markdown bs4 langchain-core langchain-community langchain-chroma requests numpy python-dotenv rank_bm25 jieba sentence-transformers
```

| 包 | 作用 |
| --- | --- |
| langchain 全家桶 | 同 Day5（Chroma、Retriever、LCEL） |
| rank_bm25 | BM25 关键词检索算法（轻量纯 Python） |
| jieba | 中文分词（BM25 前置，必须） |
| sentence-transformers | 提供 `CrossEncoder`，加载 `bge-reranker-v2-m3` 做 Rerank |

## 步骤 2：复制复用包（不重写）
- 复制 Day5 的 `adapters/`、`embeddings/`、`loaders/`、`cleaners/`、`samples/` → `day6-hybrid-retrieval/`
- `.env`（可选）：`OLLAMA_BASE_URL=http://localhost:11434`

> 前置：Ollama 已 `ollama pull bge-m3`（向量化）+ `qwen2.5:3b`（对话）。Rerank 模型 `BAAI/bge-reranker-v2-m3` 首次运行由 sentence-transformers 自动下载（约 2.27GB，需联网），缓存在 `C:\Users\<用户名>\.cache\huggingface\hub\`。

## 最终目录结构（重构后）

```plain
day6-hybrid-retrieval/
├── main.py                  # 步骤5：基线 vs 优化 并排对比
├── rag_chain.py             # 底层：建库/复用，返回 vectorstore
├── retrieval/               # 检索优化包（统一从 __init__.py 导出）
│   ├── __init__.py
│   ├── mmr_retriever.py     # 策略1：MMR 多样性检索
│   ├── bm25_retriever.py    # 策略2：BM25 关键词检索
│   ├── fusion.py            # 策略3：RRF 融合（语义+BM25）
│   └── reranker.py          # 策略4：CrossEncoder 精排
├── adapters/ embeddings/ loaders/ cleaners/   # 复制 Day5（成熟代码不动）
└── samples/README.md
```

分层依赖关系（高内聚低耦合）：

```plain
rag_chain(建库) ← mmr/bm25(单一策略) ← fusion(编排融合) ← reranker(精排) ← main(对比实验)
```

# 四、开发实操
## 步骤 0：改造 `rag_chain.py`（返回 vectorstore 而非 retriever）
相对 Day5 的关键改动：`return vectorstore.as_retriever(...)` → `return vectorstore`。Day5 把检索方式提前定死（similarity + k=3）；Day6 要玩 MMR/BM25/混合/重排四种策略，必须交出**裸 vectorstore**，由各策略自己决定怎么用（`as_retriever(search_type=...)` / `get()` 全量取 chunk / `similarity_search`）。

另一处关键：`persist_dir` 改为**基于本文件的绝对路径**（见 7.1 踩坑 2）：

```python
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(_BASE_DIR, "chroma_db")
```

入库逻辑与 Day5 一致：库空才写、否则复用（防重复累积），metadata 带 `source/chunk_id`。

## 步骤 1：`retrieval/mmr_retriever.py`（MMR 多样性检索）

```python
def build_mmr_retriever(file_path: str, k: int = 3, fetch_k: int = 10, lambda_mult: float = 0.5):
    vectorstore = build_vectorstore(file_path)
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )
```

#### 1.1 运行结果（本机实测）
```plain
python -m retrieval.mmr_retriever
===== [MMR 召回] 3 条 =====
1 | chunk_id 4   （.env 配置段）
2 | chunk_id 3   （激活环境 + pip install 依赖段）
3 | chunk_id 17  （解耦设计段 —— 弱相关噪点）
```
> 解读：对比 Day5 曾出现的"三条全是 chunk_id 4"，MMR 保证了 **chunk_id 互不相同**（去重达成）。但 chunk_id 17 说明 **MMR 只解决"冗余"，不解决"弱相关噪点"**——这正是后续步骤的意义。

## 步骤 2：`retrieval/bm25_retriever.py`（BM25 关键词检索）
设计要点：`BM25Retriever` **接收已构建的 vectorstore**（不自己建库），从 `vectorstore.get()` 取全部 chunk 分词建索引；`from_file()` 提供便捷构造。`search()` 返回 `(正文, metadata, BM25分数)`，带分数是为步骤 3 的 RRF 融合准备。

```python
class BM25Retriever:
    def __init__(self, vectorstore):
        data = vectorstore.get()
        self.corpus = data["documents"]
        self.metadatas = data["metadatas"]
        self.bm25 = BM25Okapi([_tokenize(d) for d in self.corpus])

    def search(self, query: str, k: int = 3):
        scores = self.bm25.get_scores(_tokenize(query))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.corpus[i], self.metadatas[i], scores[i])
                for i in top_idx if scores[i] > 0]
```

#### 2.1 运行结果（本机实测）
```plain
python -m retrieval.bm25_retriever
===== [BM25 关键词召回] 3 条 =====
1 | chunk_id 3 | BM25分 4.46  （激活环境 + pip install 依赖段 —— 真答案段）
2 | chunk_id 1 | BM25分 3.79  （下载 Python3.11，含"安装包"词）
3 | chunk_id 0 | BM25分 3.06  （FastAPI 目标，含"依赖"词）
```
> 解读：BM25 把**真正含 `pip install` 命令的 chunk_id 3 顶到第一**——比 Day5 语义检索（把 .env 段 chunk_id 4 排第一）更聚焦精确词。chunk_id 1/0 是词面命中的弱相关噪点，交给融合与重排处理。

## 步骤 3：`retrieval/fusion.py`（RRF 融合）
语义 Top-10 + BM25 Top-10 → 各自转排名 → `score = Σ 1/(60 + rank)` 按 chunk_id 去重累加 → 取融合 Top-3。**只 `build_vectorstore` 一次并共享给 BM25**，避免重复建库。

#### 3.1 运行结果（本机实测）
```plain
python -m retrieval.fusion
===== [RRF 混合召回] 3 条 =====
1 | chunk_id 3   （真答案段，稳居第一）
2 | chunk_id 4   （.env 配置段，弱相关）
3 | chunk_id 1   （Python 下载段，弱相关）
```
> 解读：既被语义认可、又被关键词命中的 chunk_id 3 总分最高；**纯噪点 chunk_id 17（语义独有）和 chunk_id 0（BM25 独有）都被挤出局**——单路短板被另一路平衡。但弱相关的 4/1 仍在 Top-3，需要步骤 4 精排压制。

## 步骤 4：`retrieval/reranker.py`（CrossEncoder 精排 + 阈值过滤）

```python
class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, top_k=3, threshold=None):
        pairs = [(query, d.page_content) for d in candidates]
        scores = self.model.predict(pairs)
        scored = sorted(zip(candidates, map(float, scores)), key=lambda x: x[1], reverse=True)
        if threshold is not None:
            scored = [s for s in scored if s[1] >= threshold]
        return [(d, s) for d, s in scored[:top_k]]
```

`hybrid_search_reranked()` = 混合检索取候选池（fetch_k=10）→ 精排 → Top-K。

#### 4.1 运行结果（本机实测）
```plain
python -m retrieval.reranker
===== [混合 + Rerank 召回] 3 条 =====
1 | chunk_id 3 | rerank分 0.879  （真答案段，高分自信）
2 | chunk_id 4 | rerank分 0.693  （次相关）
3 | chunk_id 6 | rerank分 0.018  （auth.py 鉴权段，弱相关被压到接近 0）
```
> 解读：Rerank 做了两件事——① 把真答案和次相关的差距**拉大**（0.879 vs 0.693），排序更自信；② 把弱相关分数**压到 0.018**，给出明确的"可过滤"信号。加 `threshold=0.3` 即可把它挡在 prompt 之外。

## 步骤 5：`main.py`（基线 vs 优化 并排对比）
LCEL 链 `PROMPT | llm | StrOutputParser()`，**prompt 模板、LLM(qwen2.5:3b)、问题三者完全相同，唯一变量是召回文档**——严格控制变量，差异 100% 归因检索优化。

- 基线：`baseline_docs()` = Day5 风格纯语义 similarity k=3
- 优化：`optimized_docs()` = 混合检索 + Rerank(threshold=0.3)

#### 5.1 运行结果（本机实测）
```plain
python main.py

########## 基线（Day5 风格）##########
召回 chunk_id: [4, 3, 1]
答案：激活环境 + pip install ...（主线对）
     但混入：下载安装 Python 3.11、安装 Ollama、ollama list 等跑偏步骤

########## 优化（Day6 混合+Rerank）##########
召回 chunk_id: [3, 4]      ← chunk_id 6 (0.018 < 0.3) 被阈值滤掉
答案：激活虚拟环境 → pip install fastapi uvicorn ... → 等待安装完成
     干净聚焦，零跑偏
```

#### 5.2 四种策略效果总表

| 阶段 | 召回 chunk_id | 效果 |
| --- | --- | --- |
| Day5 纯语义 | 4, 3, 1 | 含弱相关 1，答案跑偏 |
| Day6 MMR | 4, 3, 17 | 去重达成，但仍有噪点 17 |
| Day6 RRF 混合 | 3, 4, 1 | 真答案 3 顶第一，纯噪点 0/17 出局 |
| **Day6 混合+Rerank(0.3)** | **3, 4** | **弱相关全被滤掉，答案干净聚焦** |

> 注：`Day6 MMR` 为步骤 1 的独立探索/对照，**未进入最终 `main.py` 链路**；最终链路为「语义 + BM25 → RRF 融合 → Rerank(threshold=0.3)」三层。

## 运行方式（重构后统一）
重构后 retrieval 包内跨根引用用绝对导入，**必须在 day6 根目录用 `-m` 方式运行**：

```powershell
cd F:\ai-learn\day6-hybrid-retrieval

python -m retrieval.mmr_retriever    # 策略1 单测
python -m retrieval.bm25_retriever   # 策略2 单测
python -m retrieval.fusion           # 策略3 单测
python -m retrieval.reranker         # 策略4 单测
python main.py                       # 完整对比实验
```

# 五. 总结
## 1. 技术栈
Python + langchain 全家桶 + rank_bm25 + jieba + sentence-transformers(CrossEncoder) + Ollama(bge-m3 + qwen2.5:3b) + 复用 Day2–Day5 包

## 2. 核心模块功能
1. **rag_chain.build_vectorstore**：建库/复用（库空才写），返回裸 vectorstore 供上层策略复用；persist_dir 绝对路径防散落
2. **retrieval/mmr_retriever**：`search_type="mmr"`，fetch_k 候选池 + lambda_mult 权衡相关/多样
3. **retrieval/bm25_retriever**：jieba 分词 + BM25Okapi 倒排索引；接收 vectorstore（解耦建库）
4. **retrieval/fusion**：语义+BM25 双路召回 → RRF `Σ 1/(60+rank)` 按 chunk_id 去重融合
5. **retrieval/reranker**：CrossEncoder(bge-reranker-v2-m3) 精排 + threshold 过滤弱相关
6. **main.py**：控制变量对比实验（同 prompt/LLM/问题，只换召回文档）

## 3. 检索优化思想（Day6 版）
```plain
问题 → ┌ 语义检索(bge-m3, Top-10) ┐
       │                          ├→ RRF 融合(排名去量纲) → 候选池 Top-10
       └ BM25关键词(jieba分词, Top-10) ┘
       → CrossEncoder 精排(query+doc 交互打分) → threshold 过滤弱相关
       → 干净的 Top-K → 拼 context → PromptTemplate → qwen2.5:3b → 聚焦答案
```
+ 粗排（双塔，快）负责"召得全"，精排（交叉编码，准）负责"排得对"，阈值负责"挡得住"
+ 高内聚低耦合：每个策略文件只干一件事，fusion 编排、reranker 站在 fusion 肩上，main 只做实验

# 六、关键知识点理解复盘
#### Q1：为什么 `build_vectorstore` 返回 vectorstore 而不是 retriever？
Day5 提前 `.as_retriever()` 等于把检索方式定死。Day6 四种策略对 vectorstore 的用法各不相同（MMR 用 `as_retriever(search_type="mmr")`、BM25 用 `get()` 取全量、融合用 similarity 候选池），必须交出本体由策略层自己决定。

#### Q2：BM25Retriever 为什么改成接收 vectorstore 而不是 file_path？
职责单一：它只该关心"BM25 算法"，不该关心"如何建库"。这样 fusion 能把同一个 vectorstore 共享给语义检索和 BM25，避免重复建库；单测时用 `from_file()` 便捷构造。

#### Q3：RRF 的 `rrf_k=60` 调大调小有什么影响？
调小 → 头部排名(rank=1,2)的分数优势放大，更信任各检索器的第一名；调大 → 排名差距被抹平、融合更平滑。60 是论文经典值。

#### Q4：Rerank 分数怎么用？
它是 (query, doc) 的交互相关性（sigmoid 后 0~1）。除了排序，更大价值是**过滤**：本例弱相关段只有 0.018，`threshold=0.3` 一刀切掉；想只留真答案可用 0.7。这是"资料没答案时模型硬编"的防火墙（Day5 遗留债 4 的部分解法）。

#### Q5：为什么包内互引用相对导入 `from .xxx`，跨根引用绝对导入 `from rag_chain`？
`from ..rag_chain` 会"越过顶级包"报错（retrieval 已是顶级包，没有上层包可言）。跨根只能靠绝对导入 + `python -m` 把 cwd 加入 `sys.path`；包内互引用 `from .xxx` 保持内聚。

#### Q6：Rerank 模型下载到哪了？
HuggingFace 全局缓存 `C:\Users\<用户名>\.cache\huggingface\hub\models--BAAI--bge-reranker-v2-m3`（`.cache` 是隐藏目录）。跨项目复用，无需迁移；想跟项目走可设 `$env:HF_HOME`。

# 七、踩坑记录与遗留问题
## 7.1 本次踩坑（调试真事）
| # | 现象 | 根因 | 解决 |
| --- | --- | --- | --- |
| 1 | fusion 运行时先 `[入库] 29 条` 又 `[复用] 29 条`，库被重复建 | `persist_dir="./chroma_db"` 相对 cwd，在 `retrieval/` 子目录运行时库散落到 `retrieval/chroma_db` | `persist_dir` 改为基于 `rag_chain.py` 文件的绝对路径 `PERSIST_DIR` |
| 2 | `from ..rag_chain import ...` 报"attempted relative import beyond top-level package" | retrieval 是顶级包，`..` 越界 | 跨根引用改绝对导入 `from rag_chain import ...`，统一 `python -m` 运行 |
|3 | HF 下载警告：unauthenticated / symlinks 降级 | Windows 未开发者模式 + 未设 HF_TOKEN | 无害可忽略；可设 `HF_HUB_DISABLE_SYMLINKS_WARNING=1` 消警告 |
| 4 | 误以为 BM25 "没召回 chunk_id 4 就是错了" | chunk_id 4 实为 .env 配置段，真依赖安装段是 chunk_id 3 | BM25 把 3 排第一恰是正确行为，比语义检索更准 |

## 7.2 遗留技术债 / 待补强
| # | 问题 | 影响 | 计划解决时机 |
| --- | --- | --- | --- |
| 1 | Rerank 模型 2.27GB、CPU 推理慢 | 每次冷启动加载耗时 | 生产可换轻量 reranker 或常驻服务 |
| 2 | `rrf_k`/`lambda_mult`/`threshold`/`fetch_k` 均为手拍超参 | 换文档集效果可能波动 | 建评测集（问题+标注答案段）做网格搜索 |
| 3 | BM25 索引每次运行重建（`vectorstore.get()` 全量取出再分词） | 大库时启动慢 | 索引持久化（pickle）或换 Elasticsearch |
| 4 | 仅单文档、单问题验证 | 结论泛化性未知 | 多文档多问题批量评测（召回率/MRR） |
| 5 | Chroma 是嵌入式库，不适合生产并发 | 无法多进程共享 | Day7：PostgreSQL + PGVector 生产向量库 |
