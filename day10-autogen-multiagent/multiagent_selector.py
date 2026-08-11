"""Day10 步骤三：SelectorGroupChat 版多智能体（对比 RoundRobin 版）

与 multiagent_round.py 的唯一区别：用 SelectorGroupChat 替代 RoundRobinGroupChat，
由"主持人"（一个 LLM 决策）智能选择下一个最该发言的 Agent，避免无效轮次。
模型：本地 Ollama qwen2.5:7b
"""
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
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
        system_message="你是项目经理。你本人不掌握具体资料，严禁凭空编造任何内容。"
                       "处理用户问题时：第一步，先指示工程师调用 search_project_notes 工具查询真实资料；"
                       "第二步，等工程师回报查到的真实内容后，你再基于这些真实内容做任务拆解与分派。"
                       "如果工程师尚未查资料，你只需说『请工程师先查询相关资料』，不要自行回答。",
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
    termination = TextMentionTermination("TERMINATE")

    # 4. 群聊：用 SelectorGroupChat 智能选人（对比 RoundRobin 的死板轮流）
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
            "5. 避免让同一个人连续发言两轮。"
        ),
        termination_condition=termination,
        max_turns=12,
        allow_repeated_speaker=False,
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
