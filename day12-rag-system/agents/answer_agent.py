"""
回答智能体：基于检索整理后的要点，生成最终回答。
"""
from autogen_agentchat.agents import AssistantAgent


def build_answer_agent(client) -> AssistantAgent:
    """构建一个负责最终回答的 Agent。"""
    sys_msg = (
        "你是严谨的问答助手。请仅基于提供的检索要点回答用户问题，"
        "用简洁中文作答；若要点与用户问题无关或不足以回答"
        "（例如问候、自我介绍、闲聊），请明确说明"
        "'该问题不在我的知识库范围内'，不要强行用片段拼凑答案。"
    )
    return AssistantAgent(
        name="answer_agent",
        model_client=client,
        system_message=sys_msg,
    )
