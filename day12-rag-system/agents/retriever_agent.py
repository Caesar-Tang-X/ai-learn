"""
检索智能体：整理检索到的上下文，提炼与问题相关的要点。
"""
from autogen_agentchat.agents import AssistantAgent

from config import get_settings


def build_retriever_agent(client) -> AssistantAgent:
    """构建一个负责整理检索结果的 Agent。"""
    sys_msg = (
        "你是检索整理助手。你会收到用户问题和一段检索到的文档片段。"
        "请从中提炼与问题直接相关的要点，用要点列表输出，不要编造文档之外的内容。"
    )
    return AssistantAgent(
        name="retriever_agent",
        model_client=client,
        system_message=sys_msg,
    )
