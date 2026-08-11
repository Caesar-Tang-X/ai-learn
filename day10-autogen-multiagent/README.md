# 一. 今日目标

Day1~Day9 一直是「单 Agent + 工具」的范式——一个模型、一个流程、一次问答。Day10 进入 **多智能体协作（Multi-Agent）**：把复杂任务拆给多个有不同职责的 Agent，让它们按某种规则轮流或按需发言、互相接力，最终得到一个更可靠的产出。

今天用 **AutoGen v0.4**（autogen-agentchat 0.7.5）落地两种最经典的多智能体编排范式，并拿我们本地的 **Ollama qwen2.5:7b** 实跑对比：

1. **RoundRobinGroupChat（轮流群聊）**：机械地一个接一个发言，确定性强、可预测。
2. **SelectorGroupChat（主持人选人）**：由一个 LLM「主持人」根据对话历史智能挑选下一个最该发言的 Agent，更灵活但更依赖模型能力。

此外还要掌握多智能体的三个关键机制：
- **角色分工**：用 `system_message` 把「项目经理 / 工程师 / 审查员」的职责写死，避免一人包办、胡编乱造。
- **工具挂载**：让工程师 Agent 持有 `search_project_notes` 工具，只能基于真实知识库回答。
- **终止条件**：`TextMentionTermination("TERMINATE")` 与 `MaxMessageTermination(N)` 的组合（`|` 运算符），既能在任务完成时优雅收尾，又能在失控时兜底。

> 核心结论预告：**本地 7b 模型「当演员」绰绰有余，「当裁判（主持人）」力不从心**。RoundRobin 稳稳跑通，Selector 在 7b 下会陷入空转。这是今天最重要的工程认知。

# 二、先想清楚几个问题

#### Q1：为什么需要多智能体？一个强 Agent 不能搞定一切吗？

A：单 Agent 容易「既当裁判又当运动员」——自己查资料、自己总结、自己检查，中间任何一步出错都无人纠正，而且 prompt 越堆越长、职责越混越乱。多智能体把任务**分工**：有人专门查资料（工程师）、有人专门拆解（经理）、有人专门挑错（审查员）。职责单一 → 每个 prompt 更聚焦 → 出错概率更低，且审查员能独立把关，产出更可靠。

#### Q2：RoundRobin 和 Selector 到底差在哪？

A：
- **RoundRobinGroupChat**：按 `participants` 列表顺序机械轮流（manager→engineer→reviewer→manager→…）。不依赖模型做决策，**确定性 100%**，但可能浪费轮次（比如该让工程师连发两次时它却切走了）。
- **SelectorGroupChat**：额外需要一个 `model_client` 当「主持人」。每轮主持人读历史、选下一个发言者。更聪明、更少废话，但**选人质量完全取决于主持模型**。弱模型（如 7b）选人会失灵，导致死循环。

#### Q3：终止条件为什么要「组合」？

A：单一条件都有短板：
- 只靠 `TextMentionTermination("TERMINATE")`：如果模型忘了写 TERMINATE（弱模型常犯），对话永不结束。
- 只靠 `MaxMessageTermination(N)`：可能还没完成任务就被硬切断。

用 `A | B`（逻辑「或」）组合后，**任一条件满足即终止**——正常靠 TERMINATE 优雅收尾，异常靠最大轮数兜底。这是生产级多智能体的标配。

#### Q4：Selector 的「主持人」用的是哪个模型？能换成更强的吗？

A：在我们的代码里，主持人复用同一个 `OllamaChatCompletionClient(model="qwen2.5:7b")`，也就是**和三个演员是同一个 7b 模型**。实验证明 7b 当演员 OK、当裁判不行。若你有 14b+ 模型，把 Selector 的 `model_client` 指向更强的模型即可显著改善选人质量（注意：本实验环境只有 7b，故 Selector 仅作对比演示，不强行跑通）。

#### Q5：`FunctionTool` 从哪导入？为什么不在 `autogen_agentchat` 里？

A：在 AutoGen v0.4 里，Agent/Team 编排在 `autogen_agentchat`，而**工具（Tool）属于更底层的 `autogen_core`**。所以正确导入是：

```python
from autogen_core.tools import FunctionTool
```

这是一个常见的坑——新手会去 `autogen_agentchat.tools` 找，结果 ImportError。

# 三、准备工作

## 步骤 1：目录与文件骨架

```plaintext
day10-autogen-multiagent/
├── samples/
│   └── project_notes.txt        # 项目学习笔记（Day1~Day10 一句话摘要），作为知识库源
├── tools.py                     # 步骤二：FunctionTool 包装的 search_project_notes 工具
├── multiagent_round.py          # 步骤三：RoundRobin 基础版（TextMentionTermination）
├── multiagent_round_ortermination.py  # 步骤四实验二：RoundRobin + 组合终止条件
├── multiagent_selector.py       # 步骤三对比：Selector 初版（偶发兜底）
├── multiagent_selector_designate.py   # 步骤四实验一：Selector 精简 prompt 版（仍空转）
└── venv/                        # 虚拟环境（autogen-agentchat 0.7.5 等）
```

## 步骤 2：依赖（本环境已装，仅记录）

```text
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

> 注意：今日只用 AutoGen 三件套，**不需要** langchain、pgvector、chroma 等历史依赖。Day10 目录相互独立。

## 步骤 3：知识库样本 `samples/project_notes.txt`

每行是一条 Day 摘要，工具按关键词（如 "Day7"）做子串匹配返回。内容见 `samples/project_notes.txt`，关键一行：

```text
Day7：PostgreSQL + PGVector 搭建生产向量库，替换 Chroma，解决并发、元数据过滤、持久化可靠性。
```

# 四、开发实操

## 步骤 4：封装知识库工具 `tools.py`

把「查笔记」封装成 AutoGen 可用的 `FunctionTool`。重点是工具函数本身是**纯函数**，与 Agent 调度解耦。

```python
"""Day10 步骤二：给工程师 Agent 挂的轻量知识库工具（读取本地文本，无额外依赖）"""
from autogen_core.tools import FunctionTool
from pathlib import Path


def search_project_notes(query: str) -> str:
    """根据关键词查询项目学习笔记。

    Args:
        query: 要查找的关键词，例如 "Day7"

    Returns:
        包含该关键词的笔记行；若未找到返回提示。
    """
    notes_path = Path(__file__).parent / "samples" / "project_notes.txt"
    if not notes_path.exists():
        return "知识库文件不存在。"
    text = notes_path.read_text(encoding="utf-8")
    matched = [line for line in text.splitlines() if query.lower() in line.lower()]
    if not matched:
        return f"未找到与 '{query}' 相关的笔记。"
    return "\n".join(matched)


# 用 FunctionTool 包装成 AutoGen 可用的工具
project_kb_tool = FunctionTool(
    func=search_project_notes,
    name="search_project_notes",
    description="查询项目学习笔记知识库，传入关键词（如 Day7）返回对应内容。",
)
```

> 讲解：函数签名里写清 `Args` / `Returns` 的 docstring 不是装饰——AutoGen 会把 `description` 和函数签名一起发给模型，模型据此决定「要不要调、传什么参数」。描述写得越清楚，模型越不容易乱传参。

## 步骤 5：RoundRobin 基础版 `multiagent_round.py`

三个角色用 `system_message` 锁死职责，RoundRobin 让它们机械轮流，终止条件是「谁写出 TERMINATE 谁收尾」。

```python
"""Day10 步骤一：AutoGen 多智能体基础 Demo
三个角色（项目经理/工程师/审查员）用 RoundRobinGroupChat 轮流发言协作。
模型：本地 Ollama qwen2.5:7b
"""
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.ollama import OllamaChatCompletionClient

from tools import project_kb_tool


async def main() -> None:
    # 1. 模型客户端：所有 Agent 共用本地 Ollama 的 qwen2.5:7b
    model_client = OllamaChatCompletionClient(model="qwen2.5:7b")

    # 2. 三个角色 Agent
    planner = AssistantAgent(
        name="manager",
        model_client=model_client,
        system_message="你是项目经理。你本人不掌握具体资料，严禁凭空编造任何内容。"
                       "处理用户问题时：第一步，先指示工程师调用 search_project_notes 工具查询真实资料；"
                       "第二步，等工程师回报查到的真实内容后，你再基于这些真实内容做任务拆解与分派。"
                       "如果工程师尚未查资料，你只需说『请工程师先查询相关资料』，不要自行回答。",
    )
    executor = AssistantAgent(
        name="engineer",
        model_client=model_client,
        tools=[project_kb_tool],
        system_message="你是工程师。只响应项目经理的分派。"
                       "涉及项目历史内容时，必须先调用 search_project_notes 工具查真实资料，"
                       "再基于返回内容回答，并明确写道『回报给项目经理：……』。不要凭空编造。",
    )
    reviewer = AssistantAgent(
        name="reviewer",
        model_client=model_client,
        system_message="你是审查员。检查项目经理的拆解和工程师的方案是否完整、正确。"
                       "如果没问题，在回复的【最后一行】单独写 TERMINATE；"
                       "否则指出遗漏，且不要写 TERMINATE。",
    )

    # 3. 终止条件：有人说出「TERMINATE」就结束
    termination = TextMentionTermination("TERMINATE")

    # 4. 群聊：轮流发言，最多 12 轮
    team = RoundRobinGroupChat(
        participants=[planner, executor, reviewer],
        termination_condition=termination,
        max_turns=12,
    )

    # 5. 启动群聊
    task = "我们项目 Day7 学了什么？请基于知识库回答，并说明它解决了什么问题。"
    stream = team.run_stream(task=task)
    async for event in stream:
        source = getattr(event, "source", None)
        content = getattr(event, "content", None)
        if source and content and source != "user":
            print(f"\n[{source}]：\n{content}")


if __name__ == "__main__":
    asyncio.run(main())
```

**运行实录（稳定收敛）**：

```
[manager]：
请工程师先调用 search_project_notes 工具，查询我们项目 Day7 学了什么。

[engineer]：
回报给项目经理：Day7 学了 PostgreSQL + PGVector 搭建生产向量库，替换 Chroma，
解决并发、元数据过滤、持久化可靠性。
Day7 解决的核心问题是把演示级 Chroma 换成可生产部署的向量库……

[manager]：
已基于工程师查到的真实资料做任务拆解与分派：……（要点归纳）

[reviewer]：
项目经理拆解完整，工程师基于真实资料回答，无编造。
TERMINATE
```

> 讲解：注意顺序 manager→engineer→reviewer 完全由 RoundRobin 决定，与对话内容无关。这正是它「确定性」的体现——不挑模型、不挑运气，永远按列表顺序走。

## 步骤 6：Selector 对比版 `multiagent_selector.py`

把 `RoundRobinGroupChat` 换成 `SelectorGroupChat`，由 LLM 主持人选下一个发言者。其余角色定义完全相同。

```python
# 与 multiagent_round.py 的唯一区别：
from autogen_agentchat.teams import SelectorGroupChat
# ...
team = SelectorGroupChat(
    participants=[manager, engineer, reviewer],
    model_client=model_client,  # 主持人也需要模型来决策下一个谁发言
    selector_prompt=(
        "你是一个对话主持人。请根据当前对话历史，从参与者中选择下一个最应该发言的人。\n"
        "规则：\n"
        "1. 如果用户刚提出任务，选择 manager（项目经理）先分派。\n"
        "2. 如果 manager 已分派但工程师尚未查资料，选择 engineer（工程师）执行。\n"
        "3. 如果 engineer 已回报真实资料/方案，选择 reviewer（审查员）审查。\n"
        "4. 如果 reviewer 已确认通过并写了 TERMINATE，对话应结束，不要再选人。\n"
        "5. 避免让同一个人连续发言两轮。\n"
    ),
    termination_condition=termination,
    max_turns=12,
    allow_repeated_speaker=False,
)
```

**运行实录（偶发兜底）**：多数时候也能收敛，但偶尔看到：

```
Model failed to select a speaker, falling back to the next speaker in the list.
```

> 讲解：这是 Selector 在 7b 下的典型失灵——主持人没能从候选名里解析出合法发言者，框架只好退回「轮到谁算谁」的兜底逻辑。基础版 prompt 较为笼统，失灵概率偏高，于是有了下面的实验一。

# 五、两个实验

## 实验一：优化 Selector prompt，消除兜底 / 空转

**目标**：把 selector_prompt 写得更「指令化」，明确每一步该选谁，试图让 7b 主持人不再失灵。

把 prompt 改为极简强指令版（文件 `multiagent_selector_designate.py`）：

```python
selector_prompt = """你是主持人，从 manager、engineer、reviewer 中选下一个发言者。

规则：
1. 若工程师尚未调用过 search_project_notes，选 engineer 去查询。
2. 工程师已回报真实资料后，选 manager 做拆解分派。
3. manager 完成分派后，必须选 reviewer 做最终审查。
4. reviewer 审查后，下一轮必须结束（reviewer 会在末行写 TERMINATE）。
5. 禁止同一人连续发言两轮。
"""
# 去掉 allow_repeated_speaker=False，避免弱模型选人失败时直接死锁
termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(30)
```

**实测结果**：运行后仍**无限空转**。即使去掉 `allow_repeated_speaker=False`（消除硬死锁），7b 主持人依旧反复乱选（如 engineer 复述角色、manager 回退），`MaxMessageTermination(30)` 只是兜底截断，并未真正「智能收敛」。

**结论**：问题不在 prompt 措辞，而在**模型本身的天花板**——7b 缺乏稳定的「元推理 / 自我监控」能力，无法可靠地扮演「裁判」角色。换更强的模型（14b+）才能根治，而非靠 prompt 修补。

## 实验二：RoundRobin 加组合终止条件（`|` 运算符）

**目标**：给 RoundRobin 同时上「TERMINATE 优雅结束」+「最大轮数兜底」两道保险。

先验证本版本有没有 `OrTermination` 类：

```powershell
python -c "from autogen_agentchat.conditions import OrTermination; print('ok')"
# 报错：ImportError: cannot import name 'OrTermination'
```

查 `conditions` 模块的实际导出，确认组合靠**运算符重载** `|`：

```powershell
python -c "from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination; \
print(type(TextMentionTermination('TERMINATE') | MaxMessageTermination(20)))"
# <class 'autogen_agentchat.base._termination.OrTerminationCondition'>  ok
```

于是把 `multiagent_round.py` 改造成 `multiagent_round_ortermination.py`：

```python
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
# ...
text_term = TextMentionTermination("TERMINATE")
max_msg = MaxMessageTermination(20)
termination = text_term | max_msg

team = RoundRobinGroupChat(
    participants=[planner, executor, reviewer],
    termination_condition=termination,
)  # 不再单独传 max_turns=12（与 MaxMessageTermination 语义重复，留一个即可）
```

**运行实录（4 轮收敛）**：

```
[manager]：请工程师先查询 Day7 相关资料。
[engineer]：回报给项目经理：Day7 学了 PostgreSQL + PGVector 生产向量库……
[manager]：已基于真实资料做拆解分派……
[reviewer]：拆解完整、无编造。TERMINATE
```

> 讲解：`|` 是 Python 的 `__or__` 重载，`A | B` 返回一个 `OrTerminationCondition` 对象——只要 A 或 B 任一满足，团队就停止。这正是「正常走 TERMINATE、异常走 MaxMessage」的工程写法。另外 `max_turns` 与 `MaxMessageTermination` 二选一即可，重复设置反而语义混乱，故此处删去 `max_turns`。

# 六、今日踩坑表

| # | 坑 | 现象 | 根因 | 解决 |
|---|---|---|---|---|
| 1 | `OrTermination` 导入失败 | `ImportError: cannot import name 'OrTermination'` | 本版本无该类，组合靠运算符重载 | 改用 `A \| B` 返回 `OrTerminationCondition` |
| 2 | Selector 偶发兜底 | `Model failed to select a speaker` | 7b 主持人解析发言者失败 | 框架自动 fallback 到顺序；属预期 |
| 3 | Selector 无限空转 | 反复互述、不写 TERMINATE | `allow_repeated_speaker=False` + 弱模型选人死锁 | 去掉该参数 + 加 `MaxMessageTermination` 兜底 |
| 4 | 去掉参数仍空转 | 30 轮截断但无收敛 | 7b 缺乏元推理，非 prompt 问题 | 确认模型边界：7b 不适任裁判 |
| 5 | prompt 自相矛盾 | manager 说「第一步指示工程师查」vs selector「第一步选 engineer」 | 角色 system_message 与 selector 规则冲突 | 统一为「未查资料先选 engineer」 |
| 6 | `max_turns` 与 `MaxMessageTermination` 重复 | 语义冗余 | 两者都限制轮数 | 删 `max_turns`，只留组合终止条件 |
| 7 | `FunctionTool` 导入位置错 | 去 `autogen_agentchat.tools` 找 | 工具在 `autogen_core.tools` | `from autogen_core.tools import FunctionTool` |

# 七、Day1~Day10 演进图

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
Day10 AutoGen 多智能体(RoundRobin / Selector)  ← 今天
```

从 Day8 的「优化怎么问」、Day9 的「让模型能动手」，到 Day10 的「让多个模型分工协作」——我们把单点能力逐步升级为**协作系统**。

# 八、运行方式

```powershell
cd f:\ai-learn\day10-autogen-multiagent
.\venv\Scripts\activate

# 1. RoundRobin 基础版
python multiagent_round.py

# 2. RoundRobin + 组合终止条件（推荐）
python multiagent_round_ortermination.py

# 3. Selector 初版（对比用，可能兜底）
python multiagent_selector.py

# 4. Selector 精简 prompt 版（对比用，7b 下会空转，靠 MaxMessage 兜底）
python multiagent_selector_designate.py
```

# 九、关键收获

1. **多智能体的价值在「分工 + 审查」**，而非模型本身更强。
2. **RoundRobin = 确定性编排**，本地 7b 即可稳定跑通，适合生产。
3. **Selector = 智能编排**，质量取决于主持模型；7b 当裁判力不从心，需 14b+。
4. **终止条件务必组合**（`TextMentionTermination | MaxMessageTermination`），永远留一道兜底。
5. **工具与调度解耦**：`FunctionTool` 纯函数封装，换模型/换团队零改动。
6. **本地小模型的定位**：优秀「演员」，勉强「工具调用者」，不胜任「裁判/主持人」。选架构时要据此扬长避短。
