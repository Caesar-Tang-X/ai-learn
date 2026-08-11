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

    # 4. 群聊：轮流发言，最多 9 轮（3 角色 × 3 轮）
    team = RoundRobinGroupChat(
        participants=[planner, executor, reviewer],
        termination_condition=termination,
        max_turns=12,
    )

    # 5. 启动群聊
    task = "我们项目 Day7 学了什么？请基于知识库回答，并说明它解决了什么问题。"
    stream = team.run_stream(task=task)
    async for event in stream:
        # TaskResult 等事件没有 source 属性，用 getattr 避免崩溃
        source = getattr(event, "source", None)
        content = getattr(event, "content", None)
        if source and content and source != "user":
            print(f"\n[{source}]：\n{content}")


if __name__ == "__main__":
    asyncio.run(main())
