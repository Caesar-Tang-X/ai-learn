# AI 学习计划 Day1–Day8 汇总知识库

本文档汇总 Day1 到 Day8 的核心内容，作为 Day9 知识库检索工具（search_kb）的底层语料。所有内容均来自各日真实的 README 学习记录。

---

# Day1：FastAPI 搭建 Ollama 统一推理服务

**目标**：使用 FastAPI 搭建 Ollama 统一推理网关，完成三大工程化封装——全局异常捕获 + 结构化日志 + 接口鉴权。

**技术栈**：Python + FastAPI + Uvicorn + Loguru + Ollama HTTP API

**四大核心模块**：
1. **结构化日志 core/logger.py**：彩色控制台输出，按天切割文件保留 7 天，统一 `log` 对象全局复用
2. **全局异常捕获 core/exceptions.py**：自定义 `BusinessException` + 三层异常拦截（参数校验 / 业务 / 系统未知），统一 JSON 返回体
3. **APIKey 鉴权 core/auth.py**：`Depends(check_auth)` 依赖注入，统一校验请求头 `X-API-Key`
4. **Ollama 客户端 utils/ollama_client.py**：封装 `/api/generate`、`/api/tags`，统一异常上抛，单例客户端

**分层思想**：配置(.env) → 核心中间件 → 参数模型 → 工具类 → 路由接口 → 程序入口

---

# Day2：多格式文档加载、清洗与分层分块

**目标**：把 PDF / Word / Markdown 读进来，清洗噪声，切成适合检索的"知识片段（chunk）"。

**技术栈**：pypdf + python-docx + markdown + beautifulsoup4 + langchain-text-splitters

**三大核心模块**：
1. **统一文档加载 loaders/document_loader.py**：按后缀路由三种解析器，统一输出纯文本
2. **文本清洗 cleaners/text_cleaner.py**：归一 `\t`/`\xa0`/全角空格，压缩空行保留单空行，Day4 新增剥离 Markdown 代码围栏
3. **分层分块 splitters/recursive_splitter.py**：从粗到细递归（段落→句子→字符兜底），相邻块保留 overlap 避免语义切断

**关键概念**：
- 分层分块优先自然结构分隔符，尽量保留完整语义单元
- `chunk_overlap < chunk_size` 是防死循环安全阀
- `separators` 末尾空字符串 `""` 提供字符兜底，保证递归必然终止

---

# Day3：Embedding 向量化与余弦相似度

**目标**：把文字变成向量，用余弦相似度衡量语义接近程度。

**技术栈**：requests + numpy + python-dotenv + Ollama(bge-m3)

**两大核心模块**：
1. **Embedding 客户端 embeddings/embedding_client.py**：调用 Ollama `/api/embeddings`，输出 1024 维向量
2. **余弦相似度工具 utils/cosine.py**：`cosine_similarity(a,b)` 输出 [-1,1]，看方向不看长度

**关键概念**：
- bge-m3 是 Embedding 模型，1024 维，与 Day1 对话模型（生成模型）任务不同
- 余弦相似度 = 点积 / (模长a × 模长b)，排除长短文本不公平
- 相似度是"接近程度"的相对分数，基线常 0.3~0.5，看相对高低而非绝对值
- 流水线：`Day2 chunks → embed() 变向量 → cosine_similarity() 排序 → 最相关 chunk`

---

# Day4：Chroma 向量库与最小 RAG 问答

**目标**：用嵌入式 Chroma 持久化向量，做增删改查，拼简易 RAG 问答 Demo（检索+生成第一次闭环）。

**技术栈**：chromadb + 复用 Day2/Day3 包 + Ollama(bge-m3 + qwen2.5:3b)

**核心模块**：
- **向量库封装 store/chroma_store.py**：`upsert`（增改合一、反复运行不冲突）、`query`（Top-K 检索，返回余弦距离=1-相似度）、`update/delete/get`
- **问答 Demo**：`build_qa_prompt` 强制"仅根据资料回答，没有就说未提及"（抑制幻觉核心），`call_ollama_generate` 调 Day1 接口

**RAG 闭环**：`Day2 分块 → Day3 Embedding → Day4 入库 → 提问向量化 → Chroma 检索 Top-K → 拼上下文 → Day1 生成答案（标注来源）`

---

# Day5：LangChain 标准化重写 RAG 链

**目标**：用 LangChain 标准组件重写 Day1–Day4 链路，弄清 `RetrievalQA` 黑盒内部数据流。

**技术栈**：langchain 全家桶 + 复用 Day2–Day4 包

**核心模块**：
1. **适配器 OllamaEmbeddings**：继承 `Embeddings`，委托 Day3 `EmbeddingClient`
2. **适配器 OllamaLLM**：继承 `LLM`，调 Day1 `/api/generate`（默认 qwen2.5:3b）
3. **rag_chain.build_retriever**：Document + RecursiveCharacterTextSplitter + OllamaEmbeddings + Chroma 串成 retriever（库空才写否则复用）
4. **手写 RAG 链**：四步——`retriever.invoke(q)` 召回 → 拼 context → PromptTemplate → llm.invoke
5. **LCEL 链**：`retriever | prompt | llm | StrOutputParser()`，等价新版 RetrievalQA(stuff)

**关键概念**：新版 LangChain 已弃用 `RetrievalQA`，统一用 `invoke()` 和 LCEL 管道符 `|`

---

# Day6：混合检索 + Rerank（检索优化）

**目标**：根治 Day5 纯语义检索召回弱相关段导致答案跑偏的问题。

**四种策略**：
1. **MMR 多样性检索**：惩罚与已选高度相似的候选，解决冗余重复（仅步骤1探索对照，未进最终链路）
2. **BM25 关键词检索**：补语义向量对精确词（版本号、命令串）的短板，中文需 jieba 分词
3. **RRF 倒数排名融合**：语义+BM25 双路排名 `score=Σ1/(60+rank)` 融合，按 chunk_id 去重
4. **CrossEncoder Rerank 精排**：bge-reranker-v2-m3 对 (query,doc) 交互打分 + threshold 过滤弱相关

**最终链路**：语义(bge-m3) + BM25(jieba) → RRF 融合 → Rerank(threshold=0.3) → 干净 Top-K → qwen2.5:3b

**效果**：基线召回 [4,3,1] 答案跑偏；优化后召回 [3,4]，答案干净聚焦

---

# Day7：PostgreSQL + PGVector 生产向量库

**目标**：将存储层从 Chroma（嵌入式、单进程）换成 PostgreSQL + PGVector，复制 Day6 检索能力并新增三种策略。

**为何换 PGVector**：语义检索质量不变（取决于 bge-m3），价值在并发（MVCC 多客户端）、元数据过滤（JSONB 支持范围/组合/布尔 SQL WHERE）、持久化可靠性（WAL/备份/主从）。

**新增三种策略**：
1. **元数据过滤**：借助 PGVector JSONB 列在 SQL 层做范围/组合过滤（`WHERE chunk_id BETWEEN 0 AND 5`），Chroma 无法做到
2. **自查询 Self-Query**：LLM 自动从自然语言提取过滤条件（"前几个步骤"→ `{"chunk_id":{"$lte":4}}`）再传给 PGVector
3. **HyDE 假想文档检索**：LLM 生成假设答案→用假设答案嵌入检索真实文档，"长文本探针"更接近知识库文档风格

**关键概念**：
- `langchain_postgres` 的 filter 语法坑：每个字段只有一个 op key，范围查询需 `$and` 拆开（与 Chroma 不同）
- Day6 的 BM25 改为收 `List[Document]`，消除对 Chroma `vectorstore.get()` 的耦合，可搭配任意向量库
- Self-Query 效果依赖 LLM 质量，qwen2.5:3b（3B 小模型）结构化输出不稳定

---

# Day8：Prompt Engineering 提示词工程

**目标**：Day1–Day7 优化"检索质量"（给 LLM 更好的 context），Day8 转向优化"Prompt 质量"（让 LLM 更好利用 context）。不改模型、不改数据，只改进"怎么问"。

**四种策略 + 封装**：
1. **CoT 思维链**：强制 LLM 先推理再回答，用更多 token 换更高准确率。适合多步骤推理，不适合直接信息提取
2. **Few-Shot 少样本提示**：给范例教 LLM 输出格式，比口头描述格式更有效。适合格式严格场景，范例会限制 LLM 注意力
3. **结构化 JSON 输出**：约束 schema + 三层容错解析 + 重试。3B 模型对复杂 JSON 遵循能力弱，简化 schema 最有效
4. **封装 Prompt 模板类**：上述策略封装为可复用类 + Router 自动选择（仅在 CoT 和 Simple 间选，覆盖面最广副作用最小）

**关键概念**：
- 检索负责"给什么"，Prompt 负责"怎么用"
- Day8 五组实验并行对比：同一 LLM、同一问题、同一 context，唯一变量是 Prompt 策略

---

# 跨日演进主线

| 阶段 | 日 | 核心突破 |
|---|---|---|
| 推理入口 | Day1 | Ollama 统一网关（日志/异常/鉴权） |
| 数据入口 | Day2 | 加载→清洗→分层分块 |
| 语义表示 | Day3 | 文字→向量（bge-m3）+ 余弦相似度 |
| 存储检索 | Day4 | Chroma 持久化 + 最小 RAG 闭环 |
| 标准化 | Day5 | LangChain 适配器 + LCEL 链路 |
| 检索优化 | Day6 | 混合检索 + Rerank |
| 生产存储 | Day7 | PGVector + 元数据过滤/Self-Query/HyDE |
| 提示优化 | Day8 | CoT/Few-Shot/JSON/Router |
| 智能编排 | Day9 | LangChain Agent 自定义工具（查库/调接口） |
