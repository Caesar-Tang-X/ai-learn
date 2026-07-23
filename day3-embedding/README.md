# 一. 今日目标
Day2 我们已经把文档切成了适合检索的"知识片段（chunk）"。但 chunk 目前只是**文字**，计算机没法直接比较两段文字"意思有多像"。今天要解决 RAG 流水线的第二步——**把文字变成向量（Embedding），并学会用余弦相似度衡量语义接近程度**。这是后面 Day4（向量库检索）的数学基础，今天产出的 `EmbeddingClient` 和 `cosine_similarity` 会被 Day4 直接复用。

1. 今天要解决的是：
+ 把任意一段中文文字，通过本地模型变成一串固定长度的数字（向量）
+ 用余弦相似度，量化"两段文字语义有多接近"
2. 今日两个核心能力：
+ 本地 Embedding 客户端：调用 Ollama 的 `/api/embeddings` 接口，把文字 → 高维向量
+ 余弦相似度工具：输入两个向量，输出一个 [-1, 1] 的相似度分数

# 二、先想清楚几个问题

#### Q1：为什么文字要先变成"向量"才能比较？
计算机只认数字。一段文字"猫喜欢在阳光下睡觉"对人来说是语义，对机器只是一串字符。Embedding 模型会把语义"映射"到一个高维空间里：意思相近的文字，它们的向量方向也接近。于是"比较语义"就变成了"比较向量方向"，这是机器擅长的事。

#### Q2：什么是 Embedding 模型？为什么用 `bge-m3`？
Embedding 模型是专门训练来"把文字编码成向量"的模型（和 Day1 对话用的生成模型不是一回事）。`bge-m3` 是中文效果很好的开源模型，输出 **1024 维** 向量。也可以用 `nomic-embed-text`（768 维），但中文场景 bge-m3 更稳。

#### Q3：余弦相似度为什么"看方向、不看长度"？
两段文字一长一短，直接比点积会对长文本不公平。余弦相似度 = 点积 / (模长a × 模长b)，分子分母都除以各自的"长度"，相当于把向量统一压到单位长度再比方向。所以 chunk 长短不影响打分公平性。

# 三、准备工作
## 步骤 1：Day3 是新建目录 + 新建虚拟环境，依赖必须重装
> 这是 Day3 踩过的坑：新建目录后新建的 venv 是空的，**Day2 装过的库这里一个都没有**，跑代码前必须重新 install。

激活 day3 的虚拟环境后，执行：

```powershell
pip install requests numpy python-dotenv
```

| 包 | 作用 |
| --- | --- |
| requests | 调用 Ollama 的 HTTP 接口 |
| numpy | 向量运算（点积、模长） |
| python-dotenv | 读取 `.env` 里的 `OLLAMA_BASE_URL` |

> 前置条件：本机已安装 Ollama 且已拉取模型 `ollama pull bge-m3`。若 Ollama 没启动，运行时会报 `Ollama 服务未启动或无法连接，请先执行 ollama serve`。

# 四、开发实操
## 步骤 1：设计目录结构
```plain
day3-embedding/
├── main.py                 # 入口：向量化测试句子 + 余弦相似度验证
├── embeddings/             # Embedding 层：文字 → 向量
│   ├── __init__.py
│   └── embedding_client.py
├── utils/                  # 工具层：余弦相似度
│   ├── __init__.py
│   └── cosine.py
└── .env（可选）            # 存 OLLAMA_BASE_URL
```

## 步骤 2：编写 embeddings/embedding_client.py
#### 2.1 本地 Embedding 客户端，调用 Ollama `/api/embeddings` 把文字转成向量
```python
"""
author: tang
description: 本地 Embedding 客户端，调用 Ollama /api/embeddings 把文字转成向量
"""

import os
import requests
from dotenv import load_dotenv

# 加载 .env（若存在），这样 os.getenv 才能读到 OLLAMA_BASE_URL
# 若 day3 目录没有 .env，这行不会报错，下面会用默认地址兜底
load_dotenv()


class EmbeddingClient:
    def __init__(self, model: str = "bge-m3"):
        # 用哪个 Embedding 模型，默认 bge-m3（中文效果好，维度 1024）
        self.model = model
        # Ollama 地址：优先读 .env 的 OLLAMA_BASE_URL，没有就兜底本地默认端口
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def embed(self, text: str) -> list:
        """
        输入一段文字，返回对应的向量（一个浮点数列表）。
        例：embed("你好") -> [0.012, -0.034, ..., 共 1024 个数]
        """
        # 注意接口是 /api/embeddings（带 s），和 Day1 对话的 /api/generate 不同
        url = f"{self.base_url}/api/embeddings"
        # 请求体：模型名字 + 要向量化的文字
        payload = {"model": self.model, "prompt": text}
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()  # HTTP 非 200 时抛异常
            # 返回 JSON 的 "embedding" 字段就是向量数组
            return resp.json()["embedding"]
        except requests.exceptions.ConnectionError:
            # Ollama 没启动，最常见的错误
            raise RuntimeError("Ollama 服务未启动或无法连接，请先执行 ollama serve")
        except Exception as e:
            raise RuntimeError(f"向量化失败: {str(e)}")
```

#### 2.2 练习要点
+ 接口是 `/api/embeddings`（带 s），别和 Day1 对话用的 `/api/generate` 搞混。
+ `load_dotenv()` 不能漏：否则 `.env` 里的 `OLLAMA_BASE_URL` 读不到，会回退到默认 `localhost:11434`。
+ `timeout=120`：向量化大段文字可能慢，给足超时避免误报。
+ 异常做了分层：`ConnectionError` 单独提示"服务没启动"，其余异常包装成 `RuntimeError` 带原因——和 Day1 的"统一异常"思想一致。

#### 2.3 遗留问题
+ 每次 `embed()` 单独发一次 HTTP 请求，多个 chunk 串行调用会很慢。可 后续改为批量接口 `/api/embed`（一次传多段文字）或并发。
+ 模型名硬编码默认 `bge-m3`，若没 `ollama pull` 会报 404。可 在 `__init__` 里加一步 `ollama list` 校验模型是否存在。

## 步骤 3：编写 utils/cosine.py
#### 3.1 余弦相似度工具，衡量两段向量的语义接近程度
```python
"""
author: tang
description: 余弦相似度工具，衡量两段向量的语义接近程度
"""

import numpy as np


def cosine_similarity(a: list, b: list) -> float:
    """
    余弦相似度 = 点积 / (模长a × 模长b)，结果范围 [-1, 1]
    - 越接近 1：语义越像
    - 接近 0：基本无关
    - 接近 -1：语义相反
    """
    # 把普通列表转成 numpy 数组，方便向量运算
    va = np.array(a)
    vb = np.array(b)

    # 点积 A·B：对应位置相乘再求和
    dot = np.dot(va, vb)
    # 向量模长（长度）：各元素平方求和再开方
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)

    # 除模长 = "只看方向、不看长度"，避免长文本天然分数高
    # 分母为 0 兜底（极端情况两向量全 0）
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))
```

#### 3.2 练习要点
+ `np.dot(va, vb)` 是点积 `Σ aᵢ·bᵢ`；`np.linalg.norm(v)` 是模长 `√(Σ vᵢ²)`。
+ 公式本质：`cos = (A·B) / (|A|·|B|)`，把两段向量都归一化到单位长度后再比"夹角余弦"。夹角为 0° → 相似度 1；90° → 0；180° → -1。
+ 分母为 0 兜底：`if norm_a == 0 or norm_b == 0: return 0.0`，避免除零崩溃（极端全零向量）。

## 步骤 4：编写 main.py（验证"语义越近得分越高"）
```python
"""
author: tang
description: 把文字向量化，并用余弦相似度验证"语义越近得分越高"
"""

from embeddings.embedding_client import EmbeddingClient
from utils.cosine import cosine_similarity


def main():
    # 1. 创建本地 Embedding 客户端（默认 bge-m3）
    client = EmbeddingClient(model="bge-m3")

    # 2. 测试句子：0 和 1 意思相近；0 和 2、3 不相关
    sentences = [
        "猫喜欢在阳光下睡觉",
        "小猫爱晒太阳打盹",
        "今天股市大跌了",
        "苹果公司发布了新手机",
    ]

    # 3. 每句话向量化（文字 -> 高维向量）
    vectors = [client.embed(s) for s in sentences]
    print(f"向量化完成，每个向量的维度: {len(vectors[0])}")

    # 4. 两两计算余弦相似度
    print("\n===== 余弦相似度矩阵 =====")
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = cosine_similarity(vectors[i], vectors[j])
            print(f"[{i}] {sentences[i]}")
            print(f"[{j}] {sentences[j]}")
            print(f"  → 相似度: {sim:.4f}\n")

    # 5. 验证核心结论
    sim_related = cosine_similarity(vectors[0], vectors[1])
    sim_unrelated = cosine_similarity(vectors[0], vectors[2])
    print(f"相关句对(猫/小猫)相似度:   {sim_related:.4f}")
    print(f"不相关句对(猫/股市)相似度: {sim_unrelated:.4f}")
    assert sim_related > sim_unrelated, "Embedding 应让相近语义得分更高"
    print("\n✅ 验证通过：语义越接近，余弦相似度越高")


if __name__ == "__main__":
    main()
```

#### 4.1 运行结果（本机实测）
```plain
向量化完成，每个向量的维度: 1024
相关句对(猫/小猫)相似度:   0.8644
不相关句对(猫/股市)相似度: 0.3784
✅ 验证通过：语义越接近，余弦相似度越高
```
> 解读：相关句对（猫/小猫）0.8644 远高于不相关句对（猫/股市）0.3784，说明 Embedding 成功把"语义接近"编码进了向量方向。注意相似度不是 0/1 的"是否同类"，而是"接近程度"——通用模型给的是相对分数，基线大多在 0.3~0.5，看的是**相对高低**而非绝对值。

# 五. 总结：与 Day2 衔接（把 chunk 变成向量）
Day2 产出的 `chunks`（文字列表）正是 Day3 的输入。进阶玩法：把 Day2 的 `loaders / cleaners / splitters` 三个包复制到 day3，复用 `build_chunks()` 得到 chunks 后，逐个 `client.embed(c)` 变成向量，再拿用户的问题向量和每个 chunk 向量算余弦相似度——**取分数最高的 chunk，就是 RAG 检索的答案来源**。这一步的完整代码与运行方式已在对话中给出，核心流水线即：

```plain
Day2 加载→清洗→分块(chunks) → Day3 embed() 变向量 → cosine_similarity() 排序 → 最相关 chunk
```

#### 1. 技术栈
Python + requests + numpy + python-dotenv + Ollama(bge-m3)

#### 2. 两大核心模块功能
1. **Embedding 客户端 embeddings/embedding_client.py**
    - 封装 Ollama `/api/embeddings` 调用，文字 → 1024 维向量
    - 读 `.env` 的 `OLLAMA_BASE_URL`，未配置则兜底本地默认地址
    - 异常分层：连接失败提示"启动 Ollama"，其余包装为带原因的错误
2. **余弦相似度工具 utils/cosine.py**
    - 实现了 `cosine_similarity(a, b)`，输出 [-1, 1] 语义接近分数
    - 看方向不看长度，保证 chunk 长短不影响打分公平
    - 分母为 0 兜底，避免全零向量除零崩溃

#### 3. 分层工程化思想
```plain
embeddings(Embedding) → utils(余弦相似度) → main(编排验证)
```
+ 解耦：换 Embedding 模型只改 `EmbeddingClient` 的 `model` 参数；换相似度算法只动 `cosine.py`
+ 可复用：这两个模块是 Day4 向量库检索的直接底座（向量库本质就是帮我们高效算全量余弦并排序）

# 六、关键知识点理解复盘
#### Q1：Embedding 到底把文字变成了什么？
一串固定长度的数字（bge-m3 是 1024 个浮点数）。语义相近的文字，这串数字的方向也接近；语义无关的文字，方向差别大。

#### Q2：为什么用专门的 Embedding 模型，而不是 Day1 的生成模型？
两者任务不同：生成模型负责"接着往下写"，Embedding 模型专门训练来"把语义编码进向量空间"。用错模型要么拿不到向量，要么语义质量差。

#### Q3：余弦相似度为什么范围是 [-1, 1]？
公式里 A·B 最大等于 |A|·|B|（两向量同向），此时 cos=1；反向时 A·B = -|A|·|B|，cos=-1；垂直时 A·B=0，cos=0。所以天然落在 [-1, 1]。

#### Q4：为什么余弦相似度"看方向不看长度"对 RAG 很重要？
chunk 长短不一，如果直接比点积，长 chunk 天然分数高、会霸榜。除以各自模长后，所有向量被压到单位长度，只比"方向（语义）"，长短不再影响公平性。

#### Q5：相似度 0.86 很高、0.37 较低，是不是意味着 0.37 就"无关"？
不是非黑即白。通用 Embedding 给的是"接近程度"的相对分数，基线常在 0.3~0.5。判断相关性要看**相对高低**：0.86 比 0.37 高一大截，说明前者语义明显更近。RAG 检索也是取 Top-K 最高分，不靠绝对阈值。

#### Q6：为什么 `embed()` 一次只处理一段文字？多个 chunk 会慢吗？
当前是逐段串行发 HTTP 请求。chunk 多（几十上百）时确实慢，这是本地推理的常态。优化方向：用批量接口 `/api/embed` 一次传多段，或多线程并发——Day4 接入向量库后，写入是一次批量、检索是一次查询，效率高很多。

# 七、遗留问题与待补强技术债
| # | 问题 | 影响 | 计划解决时机 |
| --- | --- | --- | --- |
| 1 | `embed()` 串行调用，多 chunk 慢 | 大文档向量化耗时长 | Day4 接向量库批量写入；或改 `/api/embed` 批量接口 |
| 2 | 模型名硬编码默认 `bge-m3` | 未 `pull` 会 404 难排查 | `__init__` 里加 `ollama list` 校验 |
| 3 | 未做向量维度/空文本的校验 | 传入空字符串可能得到异常向量 | `embed()` 入口加 `if not text.strip()` 兜底 |
| 4 | 余弦相似度无批量/矩阵计算 | 多对多比较时重复调用低效 | 用 `numpy` 矩阵一次性算相似度矩阵 |
| 5 | Day2 的 chunk 未带 source/chunk_id 元信息 | 检索命中后无法回链原文 | Day4 写入向量库时附 `metadata` |
| 6 | 仅验证了短句，未在大段 chunk 上验证 | 长文本相似度分布未知 | 与 Day2 衔接后跑真实文档验证 |
