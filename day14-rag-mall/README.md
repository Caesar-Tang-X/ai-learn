# day14-rag-mall｜C 端 RAG 导购（多路召回 + 多轮改写 + 流式推荐）

一个面向 C 端的电商导购 RAG 服务：基于 **向量召回 + 全文召回** 的多路融合检索，
配合 **LLM 多轮查询改写 / 预算解析 / 指代解析 / 敏感品过滤 / 语义化硬过滤** 与 **LLM 精排**，
通过 SSE 流式接口输出图文一体的商品推荐。

> 设计原则：**通用，不针对单一场景硬编码**。所有"场景感知"（预算模糊词、送礼人群、
> 相对调价、商品指代）都由 LLM 在通用指令下理解，代码层只做确定的过滤/配置，
> 不枚举任何品类、关系或商品关键词。

---

## 功能特性

- **多路召回 + 融合重排**：`core/retrieval` 提供向量召回（pgvector）、全文召回（tsquery）两路，
  `fusion` 合并去重 → 相似度过滤 → rerank 精排 → 相关性截断（rerank 缺失/失败均自动降级）。
- **LLM 多轮查询改写**（`_rewrite_query`）：把多轮对话凝练成"可独立检索"的查询，
  - 一次 LLM 调用**同时**完成：场景/品类整合、相对调价识别（`cheaper`/`pricier`）、
    预算区间解析（元，`price_min`/`price_max`，由 LLM 理解模糊表述得出，失败回退正则）、
    场景上下文识别（`scene_context`，LLM 对任意需求场景/对象的自由自然语言概括，非关键词匹配、不枚举取值）。
  - **忠实度铁律**：只迁移用户明确说过的约束，禁止把笼统词脑补成具体品类/品牌。
  - 任意失败安全降级为"拼接所有用户消息 + 当前 query"。
- **通用预算解析**：优先用 LLM 理解模糊/量级/区间表述（"1千左右"→800~1200、"几百块"→200~900），
  失败回退正则（`core/retrieval/budget.py`，零延迟确定性兜底）；多轮间自动继承预算。
- **LLM 通用场景适用性过滤**：当改写 LLM 给出非空「场景上下文」（`scene_context`，任意自由文本，代码不枚举取值）时，
  用 LLM 通用判定「该商品是否适合此场景/对象」（不枚举任何词、不区分固定品类、不对取值做分支），
  确定性排除不适合者。例如私密/成人向商品在送长辈语境下会被排除，但在送伴侣/自购语境下可能放行——
  是否排除完全由 LLM 按场景语义决定。纯自购/无上下文语境（`scene_context` 为空）不触发过滤。
- **相对价格调整**：用户说"太贵了/便宜点"按上一轮最高价下压；"太便宜了"上提；
  带下限保护，避免召回极端低价垃圾品。
- **指代解析**：用户追问"刚才那个/第二款/第一个"时，通用识别其在上一轮推荐列表中的序号，
  直接复用已推荐卡片生成详情，跳过重新检索。
- **意图识别 + 引导**：`core/intent` 判断购物意图/需求是否明确，不明确时下发澄清引导文案。
- **流式输出**：`answer()` 异步生成器逐条 `yield` 事件（`clarify`/`intro`/`item_text`），
  API 层封装为 SSE（`text/event-stream`）。
- **多租户会话隔离**：`memory` 层按 `session_id` 分桶，保存上下文（压缩+滑动窗口）、
  上一轮推荐列表、预算约束。

---

## 项目结构

```
├── api/
│   └── app.py                 # FastAPI 入口：/chat (SSE)、/health、静态聊天页
├── config/
│   └── settings.py            # 全局配置（DB / LLM / rerank / 检索阈值 / 预算参数）
├── core/
│   ├── embeddings/            # 嵌入模型（alibaba / ollama，工厂 get_embedder）
│   ├── llm/                   # LLM 客户端（alibaba / ollama，工厂 get_llm）
│   ├── rerank/                # 重排模型（alibaba，工厂 get_reranker）
│   ├── retrieval/
│   │   ├── vector.py          # 向量召回（pgvector 余弦距离）
│   │   ├── fulltext.py        # 全文召回（PostgreSQL tsquery）
│   │   ├── fusion.py          # 多路融合 + 过滤 + rerank + 截断
│   │   ├── budget.py          # 正则预算解析（确定性兜底）
│   │   ├── count.py           # 数量词解析
│   │   └── filters.py         # 语义化硬过滤构造
│   ├── intent/
│   │   └── clarify.py         # 意图识别 + 澄清引导文案
│   └── memory/
│       ├── store.py           # 会话存储门面：Redis 读写 + 上下文组合 + 推荐/约束缓存
│       ├── compressor.py      # 历史压缩为摘要（超 token 阈值时触发）
│       └── window.py          # 滑动窗口截取最近 N 条
├── service/
│   └── rag_service.py         # 导购主流程（answer）：改写→检索→精排→流式推荐
├── scripts/
│   ├── export_products_from_mysql.py  # 从 MySQL 导出商品
│   ├── init_products_db.py            # 初始化商品库（建表/向量列/索引）
│   └── ingest_products_into_db.py     # 切分商品文案 + 生成向量 + 入库
├── static/                    # 前端聊天页（HTML/JS）
├── data/                      # 商品源数据（如 products.json）
├── main.py                    # 统一启动入口：python main.py 等价于 uvicorn api.app:app
├── requirements.txt           # 依赖清单（pip install -r requirements.txt）
├── .env.example               # 环境变量样例（API Key 等），使用时复制为 .env
└── README.md
```

---

## 环境准备

需要 **PostgreSQL（带 pgvector 扩展）** 与可访问的 **Embedding / LLM / Rerank** 服务。

1. 安装依赖（建议在虚拟环境）：

   ```bash
   python -m venv .venv && .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

2. 准备数据库并启用 pgvector：

   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

3. 复制并填写环境变量（`.env.example` 在**根目录**）：

   ```bash
   cp .env.example .env
   # 按需填写：DB 连接串、ALIBABA_API_KEY、各类模型名/端点等
   ```

4. 初始化商品库并导入数据（向量在导入时生成）：

   ```bash
   python -m scripts.init_products_db
   python -m scripts.ingest_products_into_db
   ```

---

## 运行

```bash
# 方式一：统一启动入口（推荐，等价于下方 uvicorn 命令）
python main.py

# 方式二：直接用 uvicorn
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

- 浏览器打开 `http://localhost:8000/` 即可使用内置聊天页。
- 健康检查：`GET http://localhost:8000/health`

---

## 接口

### `POST /chat`（SSE 流式）

请求体：

```json
{
  "session_id": "uuid-xxxx",
  "query": "推荐1千左右的女士礼物，5款",
  "filters": { "exclude_catalog_ids": [21] }
}
```

- `session_id`：会话 ID，前端每次会话生成唯一值，用于多轮隔离。
- `query`：用户问题（必填）。
- `filters`：语义化硬过滤（可选）。支持：
  - `include_catalog_ids` / `exclude_catalog_ids`：类目白/黑名单
  - `price_min` / `price_max`：价格上下限（**单位：分**）
  - 其他 metadata 字段（如 `spu_id` / `title` / `is_enable`）等值匹配
  - 未提供或 `null` 的键不参与过滤；后端始终附加 `is_enable=1` / `is_delete=0`。

响应：`text/event-stream`，逐事件推送一行 JSON，结束发送 `data: [DONE]`。事件类型：

| type | 含义 |
|------|------|
| `clarify` | 需求不明 / 非购物意图 / 0 命中时，下发引导文案 |
| `intro`   | 正常推荐流程开场白 |
| `item_text` | 单个商品（标题/价格/理由/图片/intro）图文合一 |

---

## 配置项（config/settings.py 摘要）

| 配置 | 说明 |
|------|------|
| `db_url` | PostgreSQL 连接串（含 pgvector） |
| `llm_provider` / `embedder_provider` / `rerank_provider` | 服务方选择（alibaba / ollama 等） |
| `rerank_top_n` | 精排后保留条数（默认推荐数量） |
| `retrieval_final_top_k` | 送 rerank 的候选上限 |
| `rerank_threshold` / `rerank_relative_threshold` | 相关性截断阈值（绝对 / 相对） |
| `vector_threshold` | 向量召回相似度下限 |
| `relative_price_factor` | "更便宜"时上限 = 上一轮最高价 × 该系数 |
| `relative_price_floor` / `relative_price_lift` | 相对降价下限保护 / 相对上提下限抬高 |
| `budget_expand_ratio` | 模糊/区间预算硬过滤时的外扩比例 |
| `budget_relax_ratios` | 候选不足时的渐进放宽系数序列（成对：下限倍率/上限倍率） |

---

## 检索与推荐主流程（`service/rag_service.py::answer`）

```
用户输入
  └─ 1. 取会话上下文 + 上一轮推荐列表
  └─ 2. LLM 多轮改写（场景整合 + 预算解析 + cheaper/pricier + scene_context）
  └─ 3. 预算解析优先级：LLM 预算 > 正则兜底 > 继承上一轮预算
  └─ 4. 指代解析：是否追问上一轮某个商品？
         ├─ 是 → 复用该卡片，生成详情，跳过检索
         └─ 否 → 意图识别
                       ├─ 不明确 → 返回 clarify 引导
                       └─ 明确 → 5. 检索 + 候选外扩(模糊预算) + LLM 精排
                                    （按 scene_context 通用判定排除不适合该场景/对象的商品）
                                    └─ 候选不足 → 按 budget_relax_ratios 渐进放宽
                                6. 流式 yield intro + item_text
```

所有"理解用户真实意图"的环节（预算、人群、相对调价、指代）均由 LLM 在**通用指令**下完成，
代码不写死任何具体品类 / 关系 / 商品关键词；阈值、放宽比例、过滤开关等均参数化于 `settings`。

---

## 开发实战：从零搭建这套系统的思维拆解


### 第 0 步：先想清楚"我们要解决什么"

接需求前先定义边界，避免一上来写代码：

- **场景**：C 端用户用自然语言找商品（"送爸爸的生日礼物""1千左右的女士手表"）。
- **痛点**：纯关键词搜索理解不了模糊语义；多轮对话会丢失上下文；没有预算/场景约束会导致推荐跑偏。
- **核心目标**：用 RAG 做到"理解意图 + 召回相关 + 精排 + 流式呈现"。
- **架构原则（贯穿全程）**：**通用优先，不硬编码场景词**。所有"场景理解"（预算、送礼人群、相对调价、指代）交给 LLM 在通用指令下完成，代码只做确定的过滤/配置，绝不写死品类或关键词。

由此确定技术选型：**PostgreSQL + pgvector**（向量+结构化过滤一体，省去额外向量库）、**FastAPI + SSE**（流式）、**阿里百炼 / Ollama 双 provider 可切换**（本地可跑、云端可量产）。

### 第 1 步：搭骨架 —— 配置中心与依赖

**目标**：让所有模块有一个唯一的"系统参数源"，且 provider 可插拔。

**涉及文件**：`config/settings.py`、`requirements.txt`、`.env.example`

**思考**：
- 参数散落在各处会灾难。用 `pydantic-settings` 的 `BaseSettings` 做单一配置源：默认值写在代码里，**敏感信息（API Key）从环境变量/`.env` 读**，不进版本库。
- 数据库、Redis 连接串用 `@property` 动态拼，避免重复拼接逻辑。
- 三个 provider（embedding / rerank / llm）各自独立开关，业务层只调用工厂拿"当前 provider 的实现"，切换模型零改业务代码。

**决策**：配置分三组——**基础设施**（DB/Redis）、**模型 provider**、**检索/预算阈值**。`settings.py:63-76` 的相对价格与预算参数属于"多轮价格约束"，与检索阈值并列但职责不同，故保持独立、不合并（见下方"设计问答"）。

### 第 2 步：接入模型能力 —— 三层抽象工厂

**目标**：让业务代码"只管调用模型"，不关心背后是阿里还是本地 Ollama。

**涉及文件**：`core/embeddings/`（base/alibaba/olibaba/__init__）、`core/llm/`、`core/rerank/`

**思考**：
- 每种模型能力（嵌入、生成、重排）都抽一个 **`Base` 接口 + 多个实现 + 一个 `get_xxx(provider)` 工厂**。
- 接口契约要稳：嵌入返回 `List[float]`；LLM 用**异步流式 `stream(messages) -> async iterator`**（为后面 SSE 打基础）；重排返回 `(index, score)` 列表。
- `__init__.py` 只暴露工厂函数，业务层 `from core.llm import get_llm`，**对具体实现零依赖**。

**决策**：模型调用统一走 OpenAI 兼容的 `/chat/completions` 与 `/embeddings` 协议（阿里百炼兼容模式 + Ollama 原生都支持），一套 HTTP 适配两种 provider。

### 第 3 步：建商品库 —— 数据建模与入库脚本

**目标**：把商品源数据（JSON/MySQL）变成"可被向量 + 结构化检索"的行。

**涉及文件**：`scripts/init_products_db.py`、`scripts/export_products_from_mysql.py`、`scripts/ingest_products_into_db.py`、`data/products.json`

**思考**：
- 表设计：`products` 存 `id / title / content / price（分）/ catalog_id / image / is_enable / is_delete / embedding（vector(1024)）`。
- 关键：**向量列与业务字段同表**，检索时一次 SQL 同时做"向量距离 + metadata 过滤"，避免"向量库召回后再回关系库 join"的额外开销。
- 索引：`ivfflat` 加速向量检索、`btree` 加速价格/类目过滤。
- 入库流水线：切分商品文案 → 调 embedding → 批量 upsert。拆成"导出/建表/导入"三个**独立脚本**，职责单一、可单独重跑。

**决策**：价格统一以**分**为单位存整数，避免浮点比较误差；对外接口同样用分，由前端展示时换算。

### 第 4 步：召回层 —— 多路检索与融合

**目标**：从商品库召回"可能相关"的候选，兼顾语义与关键词。

**涉及文件**：`core/retrieval/vector.py`、`fulltext.py`、`fusion.py`、`filters.py`、`budget.py`、`count.py`

**思考**：
- **向量召回**（`vector.py`）：`ORDER BY embedding <=> $query LIMIT k`，余弦距离转相似度；按 `vector_threshold` 过滤明显无关项（如"墨镜"下的"中药"）。
- **全文召回**（`fulltext.py`）：PostgreSQL `tsquery`，兜住向量召回不敏感的精确词（品牌名、型号）。
- **融合**（`fusion.py`）：两路按 `id` 去重、取高分；再送重排。
- **硬过滤**（`filters.py`）：把 `price_min/max`、`catalog_id` 等变成 SQL `WHERE`，与召回解耦。
- **预算/数量解析**（`budget.py`、`count.py`）：先做**确定性的正则兜底**（"300-600""5款"零延迟解析），LLM 解析失败时立刻回退，保证链路不挂。

**决策**：rerank 缺失/失败要**自动降级**为"融合分相对阈值截断"，检索链路不能是单点故障。这就是 `fusion.py` 里 `_fallback` 与 `try/except` 存在的必要性。

### 第 5 步：会话记忆 —— 多轮上下文管理

**目标**：让用户说"便宜点""刚才那个"时，系统知道指代什么。

**涉及文件**：`core/memory/store.py`、`compressor.py`、`window.py`、`__init__.py`

**思考**：
- 按 `session_id` 分桶存 Redis，天然 **TTL 自动过期**，无需手写清理。
- 上下文不能无限增长：先 `compress`（超 token 阈值时把早期对话交给 LLM 摘要）+ 再 `slide_window`（只留最近 N 条）。
- 额外缓存两类"跨轮状态"：**上一轮推荐卡片**（供"第一个/刚才那个"指代）+ **生效的价格约束**（供多轮继承预算）。

**决策**：`compressor` 用 `asyncio.run` 驱动异步 LLM，因此**必须在无运行事件循环的环境调用**——service 层用 `asyncio.to_thread` 包裹它。这是模块边界约束，注释已写明。

### 第 6 步：意图识别 —— 该推荐还是该反问

**目标**：用户表述不清或纯闲聊时，不下推荐，而是引导。

**涉及文件**：`core/intent/clarify.py`

**思考**：
- 每次请求先让 LLM 当"意图裁判"：返回 `{is_shoppable, is_clear, reason}`。
- 三种走向：非购物意图 / 需求不明 / **检索 0 命中** → 走引导；否则正常推荐。
- 引导文案用专用 system 指令**强制只围绕购物维度反问**（品类/场景/人群/预算/数量），禁止闲聊、禁止编造商品。

**决策**：纯 LLM 判定、不写关键词黑名单。这样既通用，也避免"用户换个说法就漏判"。

### 第 7 步：主流程编排 —— 把零件串成导购大脑

**目标**：用 `answer()` 异步生成器编排"改写→检索→精排→流式输出"，这是系统的核心。

**涉及文件**：`service/rag_service.py`

**思考**（这是最该讲清楚的模块）：
1. 取会话上下文 + 上一轮推荐列表。
2. **LLM 多轮改写** `_rewrite_query`：一次调用同时完成场景整合 + 预算解析 + cheaper/pricier 识别 + scene_context 判断（LLM 对任意需求场景/对象的自由概括，代码不枚举取值）。**忠实度铁律**：只迁移用户明确说过的约束，禁止把"实惠"脑补成"小米"。任意失败降级为"拼接所有用户消息 + 当前 query"。
3. **预算优先级**：LLM 预算 > 正则兜底 > 继承上一轮预算。
4. **指代解析**：追问"刚才那个/第二款"时，直接复用已推荐卡片，跳过检索。
5. 意图不明确 → `clarify`；明确 → 检索 + 外扩 + 重排（按 `scene_context` 通用判定排除不适合该场景/对象的商品）。
   **数量满足机制**：用户显式数量（如「10款」）经 `parse_count_max` 解析为 `want_n`，全程透传；
   `retrieve` 据此放大每路召回规模（`recall_top_k = max(默认, want_n)`），保证召回池 ≥ 需求数量；
   候选仍不足时 `_search_candidates` 先放大召回规模重试，再按 `budget_relax_ratios` 渐进放宽预算，尽量满足「要N款」而非固定返回少量。
6. 流式 `yield` 事件：`intro` → 多个 `item_text`。

**关键子能力设计**：
- **通用场景适用性过滤** `_is_unsuitable`：改写 LLM 给出 `scene_context`（任意场景/对象的自由自然语言概括，代码不枚举取值）后，
  对每个候选用 LLM 通用判定"该商品是否适合此场景/对象"（不枚举任何词、不固定品类、不对取值做分支），
  确定性排除不适合者。这是"硬约束"，不能用纯 LLM 重排代替（否则会回归漏判）；
  但判定随场景语义自适应——送长辈排除私密品、送伴侣或自购则可能放行，避免了"送礼=全排除"的单一场景简化。
- **异步桥接** `_run_sync`：唯一把同步 DB/Redis 操作塞进 `to_thread` 的入口，避免阻塞事件循环；LLM 流式调用本身已是 async，直接 `await`。

**决策**：相对调价、预算、指代、人群理解**全部通用 LLM 化**，代码零场景词。这就是为何它能"换个品类照样工作"。

### 第 8 步：接口层 —— 把能力对外暴露

**目标**：用 HTTP + SSE 把 `answer()` 的流式事件推给前端。

**涉及文件**：`api/app.py`、`main.py`、`static/`

**思考**：
- `POST /chat` 接收 `{session_id, query, filters}`，调用 `answer()`，把每个 `yield` 的事件包成 `data: {json}\n\n` 的 SSE 帧。
- 结束发 `data: [DONE]`。前端用 `EventSource`/`fetch` 流式渲染，体验接近"打字机"。
- `main.py` 作为统一启动入口（等价 `uvicorn api.app:app`），降低使用者心智负担。
- `static/` 内置一个零依赖的聊天页，开箱即用，方便演示与调试。

**决策**：`filters` 做成"语义化硬过滤"（白/黑名单、价格区间、等值匹配），后端始终附加 `is_enable=1/is_delete=0`，调用方只关心业务维度。

### 第 9 步：跑通端到端

按 README「环境准备 + 运行」章节：装依赖 → 建 pgvector → 配 `.env` → 初始化并导入商品 → `python main.py` → 打开 `http://localhost:8000/`。

逐一验证：多轮预算继承、相对调价、指代、送礼过滤、意图引导、SSE 流式、Redis 会话隔离。

### 第 10 步：打磨通用性与健壮性（复盘优化）

这一步是"从能跑到生产级"的关键，对应本项目的重构主线：

- **去硬编码**：审计所有"为解决单一场景写的特定代码/关键词"，一律改为 LLM 通用能力或参数化配置。
- **解耦**：检索、记忆、意图、模型、服务层各司其职，靠工厂与 `settings` 通信，无循环依赖。
- **降级链**：rerank 失败 → 融合分兜底；LLM 改写失败 → 拼接降级；摘要失败 → 仅滑窗。每个外部依赖都有"最坏情况不影响主链路"的退路。
- **注释即文档**：每个文件顶部一句话说明职责、依赖与降级策略，冗余注释全部清理。

### 第 11 步：可复刻 / 可面试的要点清单

如果你想**凭这个项目去面试**，重点讲清以下"为什么"：

1. 为什么选 pgvector 而不是独立向量库？（向量+结构化过滤一体，召回即过滤，省一次 join）
2. 多路召回为什么必要？（向量补语义、全文补精确词，互补）
3. rerank 为什么可降级？（外部依赖，单点故障要兜底）
4. 多轮上下文怎么不爆 token？（压缩+滑窗两级）
5. 怎么做到"通用不硬编码"？（预算/人群/调价/指代全部 LLM 化，零场景词）
6. 场景适用性过滤为什么用"LLM 通用场景判定 + 确定性排除"而不是关键词黑名单或一刀切？（黑名单永远漏；一刀切"送礼=全排除成人品"会误伤送伴侣等合理场景；`scene_context` 是任意自由文本、代码不枚举取值，是否排除完全由 LLM 按场景语义决定，才真正通用）
7. SSE 怎么和 async 生成器自然衔接？（`answer()` yield 事件，API 逐帧推送）

---

### 模块与文件职责速查表

| 文件 | 作用一句话 | 关键设计点 |
|------|-----------|-----------|
| `config/settings.py` | 唯一配置源，provider 可切换 | pydantic-settings，敏感信息走 `.env` |
| `core/embeddings/*` | 文本→向量（alibaba/ollama） | 工厂 `get_embedder`，统一返回 `List[float]` |
| `core/llm/*` | 流式对话生成 | 工厂 `get_llm`，接口为 `async stream(messages)` |
| `core/rerank/*` | 候选精排 | 工厂 `get_reranker`，返回 `(index, score)`；缺失/失败降级 |
| `core/retrieval/vector.py` | 向量召回（pgvector 余弦） | `vector_threshold` 过滤无关项 |
| `core/retrieval/fulltext.py` | 全文召回（tsquery） | 兜精确词，与向量互补 |
| `core/retrieval/fusion.py` | 融合+重排+截断门面 | 去重→过滤→rerank（放大召回去噪）→保留候选池（≥top_n，供下游 LLM 精排精选）→降级 |
| `core/retrieval/budget.py` | 正则预算解析 | 确定性兜底，零延迟 |
| `core/retrieval/count.py` | 数量词解析 | "5款"→top_n |
| `core/retrieval/filters.py` | 语义化硬过滤构造 | 把 price/catalog 变成 SQL WHERE |
| `core/intent/clarify.py` | 意图裁判 + 引导文案 | 纯 LLM 判定，不写关键词黑名单 |
| `core/memory/store.py` | Redis 会话存储 + 上下文组合 | 按 session 分桶，TTL 自动过期 |
| `core/memory/compressor.py` | 历史压缩为摘要 | 超 token 阈值触发，失败降级原样 |
| `core/memory/window.py` | 滑动窗口截取最近 N 条 | 与压缩配合 |
| `service/rag_service.py` | 导购主流程 `answer()` | 改写→检索→精排→流式；LLM 通用化所有意图理解 |
| `api/app.py` | FastAPI + SSE 接口 | `/chat` 流式推送，`/health` 探活 |
| `main.py` | 统一启动入口 | 等价 `uvicorn api.app:app` |
| `scripts/init_products_db.py` | 建表/向量列/索引 | 向量与业务字段同表 |
| `scripts/export_products_from_mysql.py` | 从 MySQL 导出 | 数据源可替换 |
| `scripts/ingest_products_into_db.py` | 切分→嵌入→入库 | 流水线职责单一 |
| `static/` | 零依赖聊天页 | 开箱演示 |
| `data/products.json` | 商品源数据样例 | 入库输入 |

---

### 设计问答：为什么这些配置"不合并"

**Q：`settings.py:63-76` 的相对价格与预算参数能不能合并成一个？**
A：不能，且刻意保持最小必要集：
- **A 组（relative_price_factor/floor/lift）**：作用于"相对调价意图"（cheaper/pricier），输入是上一轮推荐价，跨轮次动态推导。
- **B 组（budget_expand_ratio / budget_relax_ratios）**：作用于"绝对模糊预算"（"1千左右"），输入是本轮预算区间，单轮静态放宽。
- 二者**触发条件、作用阶段、调用函数完全不同**，合并会引入"一个参数多语义"的反模式。B 组内 `expand_ratio`（单值对称外扩）与 `relax_ratios`（多档非对称序列）结构也不同（float vs tuple），合并会丢失渐进多档能力。当前拆法已是最小必要参数集，符合"简洁严谨、无冗余"。
