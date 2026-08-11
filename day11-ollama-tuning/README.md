# 一. 今日目标

Day1~Day10 我们一直在「用模型」：单 Agent、多 Agent、RAG、工具调用。但有一个工程问题一直没正面对付过——**本地 Ollama 怎么跑得更省显存、支持更多模型同时在线、并发更稳？**

Day11 主题：**Ollama 量化调优、多模型并发、显存优化**。今天不做算法，做「资源与性能工程」：

1. **量化 / 尺寸对比**：同一问题用 1.5b / 3b / 7b 三个尺寸跑，记录显存占用、响应速度、回答质量，看清「精度 ↔ 显存 ↔ 速度 ↔ 质量」的取舍。
2. **服务级参数**：弄懂 `keep_alive`（常驻 vs 卸载）、`num_ctx`（上下文窗口对显存的影响）、`num_gpu`（层上 GPU 比例）这些 Ollama 的服务旋钮。
3. **监控 + 并发压测脚本**：写一个 Python 小工具 `monitor.py`，调 Ollama HTTP API，自动测速、读显存、做并发压测，输出对比表。

> 核心约束：本机显存 **6G**。这是今天所有实验的真实边界——它决定了 7b 无法常驻并发，而小模型才是并发主力。

# 二、先想清楚几个问题

#### Q1：为什么 Day11 要专门搞「量化调优」？模型不是下载就能用吗？

A：下载即用没错，但默认配置不一定适合你的硬件。qwen2.5:7b 在 6G 显存上会**部分 offload 到 CPU**（变慢），且占满显存无法并发。量化/尺寸调优就是回答：「在我这台机器上，选哪个模型、配什么参数，才能既够用又不爆显存」。这是部署前必做的资源规划。

#### Q2：Ollama 的量化 tag（如 q8_0）为什么拉不到？

A：Ollama 官方对每个模型尺寸**只发布一个默认量化**（qwen2.5 系列是 Q4_K_M）。尝试 `ollama pull qwen2.5:3b-q8_0` 或 `-q3_K_M` 都会报 `file does not exist`——因为官方没做这些 tag。要更细的量化级别，得去 llama.cpp 用原始 GGUF 重新量化，超出 Ollama CLI 范围。因此本实验用「**不同尺寸**」(1.5b/3b/7b) 来演示资源取舍，而非「同尺寸多量化」。

#### Q3：`ollama ps` 的 SIZE 和模型文件大小为什么不一样？

A：`ollama ps` 的 SIZE（`/api/ps` 里的 `size_vram`）是**运行时实际占显存的量**，包括模型权重 + KV cache + 上下文 buffer，所以比 `ollama list` 的文件大小略大（约 ×1.1~1.2）。更精细地，`/api/ps` 还返回 `size`（总大小）和 `size_vram`（纯显存部分）——当 `size_vram < size` 时，说明有层被 offload 到 CPU 了。

#### Q4：`keep_alive` 到底控制什么？CLI 为什么没这个参数？

A：`keep_alive` 是 **Ollama API 的 `options` 参数**（不是命令行 flag）。它决定「请求结束后模型在显存里保留多久」：
- `keep_alive: 0` → 跑完立刻卸载（省显存，下次冷启慢）
- `keep_alive: "30m"` → 常驻 30 分钟（热启快，占显存）
- 默认值是 5 分钟（`ollama run` 后 `ollama ps` 显示 `4 minutes from now`）

CLI 的 `ollama run` 不直接暴露该 flag（0.31.1 实测 `--keep_alive` 报错），要通过 HTTP API 设置——这正是我们写 `monitor.py` 的意义。

#### Q5：6G 显存下到底能并发几个模型？

A：实测结论——**小模型能并发，大模型不能**。1.5b(1.14G) 并发 4 个墙钟仅 4.69s（≈单请求冷启），GPU 吃得消；7b(3.98G 显存) 并发 2 个就到 8.96s 且显存逼近极限。所以 6G 上提升吞吐的正确策略是「多开小模型」，而非堆 7b。

# 三、准备工作

## 步骤 1：确认环境与模型

```powershell
ollama --version    # 0.31.1
ollama ps           # 当前常驻模型（初始应为空）
ollama list         # 已有模型
```

本机已有：`qwen2.5:7b`(4.7G)、`qwen2.5:3b`(1.9G)、`bge-m3`(1.2G)。需补充一个小模型用于并发演示：

```powershell
ollama pull qwen2.5:1.5b    # 986MB，拉取成功
```

最终模型清单：

| 模型 | 文件大小 | 角色 |
|---|---|---|
| qwen2.5:1.5b | 986 MB | 最小，并发主力 |
| qwen2.5:3b | 1.9 GB | 中间档 |
| qwen2.5:7b | 4.7 GB | 最大，占满 6G |
| bge-m3 | 1.2 GB | embedding（不参与对话实验） |

## 步骤 2：建目录与依赖

```powershell
cd f:\ai-learn
mkdir day11-ollama-tuning
cd day11-ollama-tuning
python -m venv venv
.\venv\Scripts\activate
pip install requests
```

> 仅依赖 `requests`（调 Ollama HTTP API）。每日目录独立，不引入历史依赖。

# 四、开发实操

## 步骤 3：手动冷启对比（建立直觉）

用 `ollama run` + `ollama stop` 做真正冷启，每次跑完卸载，避免默认 5 分钟保活干扰：

```powershell
ollama run qwen2.5:1.5b "用一句话解释什么是向量数据库"
ollama ps
ollama stop qwen2.5:1.5b
# 同样方式跑 3b、7b
```

`ollama ps` 关键输出：

```
qwen2.5:1.5b   1.2 GB   100% GPU     4096   4 minutes from now
qwen2.5:3b     2.2 GB   100% GPU     4096   4 minutes from now
qwen2.5:7b     5.2 GB   18%/82% CPU/GPU   4096   4 minutes from now
```

> **重要发现**：7b 出现 `18%/82% CPU/GPU`——6G 放不下 7b 全层，Ollama 自动把 18% 层 offload 到 CPU（CPU 推理慢）。这是 Day10 多智能体卡顿的隐藏根因之一。

回答质量（肉眼对比）：三者都答对，但 1.5b 最朴素、7b 表述最完整。简单任务用小模型即可，省下显存给并发。

## 步骤 4：服务级参数 via API（keep_alive / num_ctx）

CLI 不便设细参，用 Ollama HTTP API 演示。注意 PowerShell 里用**单引号**包裹 JSON，避免双引号转义问题：

```powershell
# 大上下文常驻，再看显存
curl.exe -s http://localhost:11434/api/generate -H "Content-Type: application/json" `
  -d '{"model":"qwen2.5:3b","prompt":"hi","options":{"num_ctx":8192},"keep_alive":"30m"}' > $null
ollama ps
```

> 踩坑：用 `> $null` 吞掉 curl 输出时，若连接被提前关闭，`keep_alive` 可能不生效、`ollama ps` 查不到。生产/脚本里用 Python `requests` 精确控制更可靠——见步骤 5。

`num_ctx` 越大 → KV cache 越大 → 显存越高。6G 上 `num_ctx` 是比模型尺寸更隐蔽的爆显存点（长文本对话尤其注意）。

## 步骤 5：写监控 + 压测脚本 `monitor.py`

脚本做四件事：单模型冷/热启测速、读 `/api/ps` 的真实显存(`size_vram`)、并发压测、跑完自动清理。

```python
"""Day11 实验三：Ollama 本地模型监控 + 并发压测小工具
功能：
  1. 单模型基准：冷启/热启耗时 + 显存占用(size_vram)
  2. 并发压测：同时发 N 个请求，测吞吐与稳定性
  3. 跑完自动卸载模型，不常驻显存
模型：本地 Ollama（默认 http://localhost:11434）
依赖：pip install requests
"""
import time
import requests
from concurrent.futures import ThreadPoolExecutor

BASE = "http://localhost:11434"
PROMPT = "用一句话解释什么是向量数据库"


def generate(model: str, prompt: str, keep_alive: str = "30m", num_ctx: int = 4096) -> float:
    """发一次生成请求，返回耗时（秒）。stream:false 保证拿到完整结果。"""
    t0 = time.perf_counter()
    resp = requests.post(
        f"{BASE}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"num_ctx": num_ctx},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return time.perf_counter() - t0


def get_ps() -> list:
    """调 /api/ps，返回当前加载的模型列表（含 size_vram / context_length）。"""
    resp = requests.get(f"{BASE}/api/ps", timeout=10)
    resp.raise_for_status()
    return resp.json().get("models", [])


def find_model(ps_models: list, name: str) -> dict | None:
    for m in ps_models:
        if m["name"] == name:
            return m
    return None


def benchmark(model: str, num_ctx: int = 4096):
    print(f"\n=== 基准测试：{model} (num_ctx={num_ctx}) ===")
    dur = generate(model, PROMPT, keep_alive="30m", num_ctx=num_ctx)
    print(f"  首次(冷启)响应耗时：{dur:.2f}s")
    ps = get_ps()
    info = find_model(ps, model)
    if info:
        vram_gb = info.get("size_vram", 0) / (1024 ** 3)
        total_gb = info.get("size", 0) / (1024 ** 3)
        print(f"  显存占用(size_vram)：{vram_gb:.2f} GB | 模型总大小：{total_gb:.2f} GB | CONTEXT：{info.get('context_length')}")
    dur2 = generate(model, PROMPT, keep_alive="30m", num_ctx=num_ctx)
    print(f"  热启响应耗时：{dur2:.2f}s  (加速 {dur/dur2:.1f}×)")


def concurrency_test(model: str, n: int = 4, num_ctx: int = 4096):
    """并发发 n 个请求，记录总耗时与每个请求耗时。"""
    print(f"\n=== 并发压测：{model}  同时 {n} 个请求 ===")
    with ThreadPoolExecutor(max_workers=n) as ex:
        t0 = time.perf_counter()
        futures = [ex.submit(generate, model, PROMPT, "30m", num_ctx) for _ in range(n)]
        durs = [f.result() for f in futures]
        total = time.perf_counter() - t0
    ok = sum(1 for d in durs if d > 0)
    print(f"  成功 {ok}/{n} | 墙钟总耗时：{total:.2f}s | 平均单请求：{sum(durs)/len(durs):.2f}s | 吞吐：{n/total:.2f} req/s")


def cleanup(models: list):
    for m in models:
        requests.post(f"{BASE}/api/generate", json={"model": m, "prompt": "", "keep_alive": 0}, timeout=30)
        print(f"  已卸载 {m}")


if __name__ == "__main__":
    models = ["qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b"]
    for m in models:
        benchmark(m)
    # 并发演示：小模型适合并发，大模型受显存限制
    concurrency_test("qwen2.5:1.5b", n=4)
    concurrency_test("qwen2.5:7b", n=2)
    print("\n--- 清理常驻模型 ---")
    cleanup(models)
```

运行 `python monitor.py`，输出示例：

```
=== 基准测试：qwen2.5:1.5b (num_ctx=4096) ===
  首次(冷启)响应耗时：3.75s
  显存占用(size_vram)：1.14 GB | 模型总大小：1.14 GB | CONTEXT：4096
  热启响应耗时：0.49s  (加速 7.6×)
=== 基准测试：qwen2.5:3b (num_ctx=4096) ===
  首次(冷启)响应耗时：2.47s
  显存占用(size_vram)：2.08 GB | 模型总大小：2.08 GB | CONTEXT：4096
  热启响应耗时：0.71s  (加速 3.5×)
=== 基准测试：qwen2.5:7b (num_ctx=4096) ===
  首次(冷启)响应耗时：7.22s
  显存占用(size_vram)：3.98 GB | 模型总大小：4.85 GB | CONTEXT：4096
  热启响应耗时：1.67s  (加速 4.3×)

=== 并发压测：qwen2.5:1.5b  同时 4 个请求 ===
  成功 4/4 | 墙钟总耗时：4.69s | 平均单请求：4.14s | 吞吐：0.85 req/s
=== 并发压测：qwen2.5:7b  同时 2 个请求 ===
  成功 2/2 | 墙钟总耗时：8.96s | 平均单请求：7.87s | 吞吐：0.22 req/s

--- 清理常驻模型 ---
  已卸载 qwen2.5:1.5b
  已卸载 qwen2.5:3b
  已卸载 qwen2.5:7b
```

# 五、实验四：Modelfile 调优实操（真正的"调优"动作）

前三个实验是"观测对比"，本实验动手**修改模型配置并实测效果**——这才是 Day11 标题里"调优"二字的落地。

## 步骤 6：用 Modelfile 创建调优模型

在项目目录建 `Modelfile`（基于已有 3b，不重新下载权重）：

```
FROM qwen2.5:3b
PARAMETER num_ctx 2048
PARAMETER num_gpu 28
PARAMETER temperature 0.3
```

创建调优模型：

```powershell
ollama create qwen3b-tuned -f Modelfile
# gathering model components ... success
```

> `ollama create` 复用已有层（显示 `using existing layer`），秒级完成。新模型 `qwen3b-tuned` 出现在 `ollama list`。

## 步骤 7：第一次对比（发现负优化）

`ollama run` + `ollama ps` 实测：

| 模型 | SIZE | PROCESSOR | CONTEXT | 配置 |
|---|---|---|---|---|
| qwen2.5:3b | 2.2 GB | 100% GPU | 4096 | 默认 |
| qwen3b-tuned | **2.4 GB** | **30%/70% CPU/GPU** | 2048 | num_ctx=2048 + num_gpu=28 |

**反直觉发现**：调优版显存不降反升，且 30% 层被 offload 到 CPU。
根因：`num_gpu 28` 主动把本可放下的层推到 CPU——3b 在 6G 上本就能 100% GPU，强行 offload 是**负优化**。`num_ctx` 省下的 KV cache 被这个副作用抵消。

## 步骤 8：修正 Modelfile，做干净对比

删掉 `num_gpu` 行，只保留真正有用的降上下文：

```
FROM qwen2.5:3b
PARAMETER num_ctx 2048
PARAMETER temperature 0.3
```

重建为干净版：

```powershell
ollama create qwen3b-ctx -f Modelfile
```

再次对比：

| 模型 | SIZE | PROCESSOR | CONTEXT | 配置 |
|---|---|---|---|---|
| qwen2.5:3b | 2.2 GB | 100% GPU | 4096 | 默认 |
| qwen3b-ctx | **2.1 GB** | 100% GPU | 2048 | 仅 num_ctx=2048 |

**正确结论**：只降 `num_ctx`（4096→2048）且保持全 GPU，显存从 2.2G→2.1G，方向正确、无 offload 副作用。量化级别虽不可改（官方只给 Q4_K_M），但**上下文窗口是可用 Modelfile 调优的有效旋钮**。

## 步骤 9：踩坑——API options 传 num_gpu 不可靠

用 Python `requests` 在 `options` 里传 `num_gpu` 调 `/api/generate` 时，模型经常**未被加载**（`/api/ps` 查不到，`size_vram=0.00GB`）。但**同样的参数写在 Modelfile 里 `ollama create` 却完全生效**（30%/70% CPU/GPU 可见）。

结论：**调优参数优先用 Modelfile + `ollama create` 固化，不要依赖运行时 API options 动态传 `num_gpu`**——后者行为不稳定。

# 六、数据解读与结论

## 1. 尺寸 ↔ 资源 ↔ 速度对照表

| 模型 | 冷启 | 热启 | 加速比 | size_vram | 总大小 | offload |
|---|---|---|---|---|---|---|
| 1.5b | 3.75s | 0.49s | 7.6× | 1.14G | 1.14G | 无 |
| 3b | 2.47s | 0.71s | 3.5× | 2.08G | 2.08G | 无 |
| 7b | 7.22s | 1.67s | 4.3× | **3.98G** | 4.85G | **0.87G 在 CPU** |

- **7b 的 `size_vram`(3.98G) < `size`(4.85G)**：铁证——6G 放不下 7b 全层，0.87G offload 到 CPU。
- **热启加速最高的是 1.5b(7.6×)**：小模型加载开销占比大，常驻收益最明显。

## 2. 并发可行性（6G 显存）

| 模型 | 并发数 | 墙钟总耗时 | 平均单请求 | 吞吐 |
|---|---|---|---|---|
| 1.5b | 4 | 4.69s | 4.14s | 0.85 req/s |
| 7b | 2 | 8.96s | 7.87s | 0.22 req/s |

- **1.5b 并发 4 个墙钟 ≈ 单请求冷启(3.75s)**：小模型并发几乎不抢资源，GPU 吃得消。
- **7b 并发 2 个就 8.96s**：大模型并发立刻受显存/算力掣肘。
- **工程结论**：6G 上提升吞吐的正确策略是「多开小模型(1.5b/3b)」，而非堆 7b。

## 3. 服务级参数速查

| 参数 | 作用 | 6G 建议 |
|---|---|---|
| `keep_alive` | 模型保活时长（API options） | 常用模型 `"30m"` 常驻热启；省显存用 `0` |
| `num_ctx` | 上下文窗口，越大 KV cache 越大 | 默认 4096 足够；长文本按需调大但要盯显存 |
| `num_gpu` | 上 GPU 的层数比例 | 单卡默认全上；爆显存时调小让其 offload CPU |

# 七、今日踩坑表

| # | 坑 | 现象 | 根因 | 解决 |
|---|---|---|---|---|
| 1 | `ollama pull qwen2.5:3b-q8_0` 失败 | `file does not exist` | 官方只发默认 Q4_K_M tag | 改用不同尺寸(1.5b/3b/7b)对比 |
| 2 | `ollama run --keep_alive` 报错 | `unknown flag: --keep_alive` | 该参数是 API options，非 CLI flag | 用 HTTP API 的 `keep_alive` 字段 |
| 3 | curl + `> $null` 后 `ollama ps` 空 | keep_alive 未生效 | 管道关闭致连接中断，模型被释放 | 改用 Python `requests` 精确控制 |
| 4 | `ollama ps` 无 PROCESSOR 字段 | 脚本读 `processor` 得 None | 字段名是 `size_vram`/`context_length` | 改读正确字段 |
| 5 | 7b 显存占用 < 文件大小 | size_vram 3.98G < size 4.85G | 6G 放不下，部分 offload CPU | 用 `size_vram` 看真实显存 |
| 6 | API options 传 num_gpu 模型未加载 | `/api/ps` 查不到，size_vram=0 | 运行时动态传 num_gpu 行为不稳定 | 改用 Modelfile + `ollama create` 固化 |
| 7 | 调小 num_gpu 反而负优化 | 3b 显存 2.2→2.4G 且 30% offload | 本可全 GPU 却主动 offload 到 CPU | 仅降 num_ctx，num_gpu 保持默认全上 |

# 八、Day1~Day11 演进图

```text
Day1  Ollama 网关(单模型服务)
Day2  文档加载/清洗/分块
Day3  本地 Embedding(bge-m3)
Day4  Chroma 向量库
Day5  LangChain 最简 RAG
Day6  混合检索 + Rerank
Day7  PGVector 生产向量库
Day8  Prompt Engineering(CoT/Few-Shot/JSON)
Day9  LangChain Agent + 工具(本地 tool calling)
Day10 AutoGen 多智能体(RoundRobin / Selector)
Day11 Ollama 量化调优 / 多模型并发 / 显存优化  ← 今天
```

从「用模型」(Day1~10) 到「调模型资源」(Day11)，我们开始关注**部署侧的性能与成本**——这是从玩具走向生产的必经一步。

# 九、运行方式

```powershell
cd f:\ai-learn\day11-ollama-tuning
.\venv\Scripts\activate
python monitor.py
```

脚本会自动：加载三模型基准测试 → 并发压测 → 卸载清理。如需改并发数，调 `concurrency_test("qwen2.5:1.5b", n=4)` 的 `n`。

# 十、关键收获

1. **Ollama 官方每个尺寸只给一个默认量化(Q4_K_M)**，细粒度量化需 llama.cpp 自行处理。
2. **6G 显存下 7b 会 offload 到 CPU**（size_vram < size），这是性能隐患。
3. **热启比冷启快 3~7 倍**——`keep_alive` 常驻是降延迟的关键旋钮。
4. **并发要选小模型**：1.5b 并发 4 个无压力，7b 并发 2 个就吃紧。
5. **`num_ctx` 是隐蔽的显存杀手**，长上下文比大模型更易爆显存。
6. **`/api/ps` 的 `size_vram` 才是真实显存占用**，比 `ollama ps` 的 SIZE 更精确（能看出 offload）。
7. **脚本优于手动 curl**：API 参数（尤其 keep_alive）在 Python `requests` 下行为最可靠、可复现。
8. **Modelfile + `ollama create` 是调优的正道**：可固化 `num_ctx`/`num_gpu`/`temperature`，且比运行时 API options 更稳定。
9. **调优要"先测后信"**：`num_gpu` 主动调小在 6G 上对 3b 是负优化（offload 反增显存）；正确做法是只降 `num_ctx`，显存 2.2G→2.1G 且保持全 GPU。
