"""Day10 步骤三：SelectorGroupChat 版多智能体（对比 RoundRobin 版）

与 multiagent_round.py 的唯一区别：用 SelectorGroupChat 替代 RoundRobinGroupChat，
由"主持人"（一个 LLM 决策）智能选择下一个最该发言的 Agent，避免无效轮次，消除 Model failed to select a speaker 兜底。
模型：本地 Ollama qwen2.5:7b
"""
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core.tools import FunctionTool
from autogen_ext.models.ollama import OllamaChatCompletionClient
from tools import project_kb_tool


async def main() -> None:
    # 1. 模型客户端（7b，强指令遵循更适合角色协同）
    model_client = OllamaChatCompletionClient(model="qwen2.5:7b")

    # 2. 三个角色 Agent（与 RoundRobin 版相同的 system_message）
    manager = AssistantAgent(
        name="manager",
        model_client=model_client,
        system_message="你是项目经理。你本人不掌握具体资料，严禁凭空编造。"
                    "当工程师已经回报了查到的真实资料后，你基于这些真实内容做任务拆解与分派，并明确说『已分派，请审查员审查』。"
                    "如果工程师尚未回报资料，你只需说『请工程师查询相关资料』，不要自行回答。",
    )
    engineer = AssistantAgent(
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

    # 3. 终止条件
    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(30)

    # 4. 群聊：用 SelectorGroupChat 智能选人（对比 RoundRobin 的死板轮流）
    team = SelectorGroupChat(
        participants=[manager, engineer, reviewer],
        model_client=model_client,  # 主持人也需要模型来决策下一个谁发言
        selector_prompt = """你是主持人，从 manager、engineer、reviewer 中选下一个发言者。

        规则：
        1. 若工程师尚未调用过 search_project_notes，选 engineer 去查询。
        2. 工程师已回报真实资料后，选 manager 做拆解分派。
        3. manager 完成分派后，必须选 reviewer 做最终审查。
        4. reviewer 审查后，无论结果如何，下一轮必须结束（reviewer 会在末行写 TERMINATE）。
        5. 禁止同一人连续发言两轮。
        """,
        termination_condition=termination
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
