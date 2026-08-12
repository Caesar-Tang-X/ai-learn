"""
协调者：预检索 + 组建 RoundRobin 多智能体团队，返回最终回答。
"""
import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from agents.retriever_agent import build_retriever_agent
from agents.answer_agent import build_answer_agent

from config import get_settings
from core.retrieval import hybrid_retrieve


def _build_client():
    """构造指向本地 Ollama 的 OpenAI 兼容客户端。"""
    cfg = get_settings()
    return OpenAIChatCompletionClient(
        model=cfg.llm_model,
        base_url=f"{cfg.ollama_base_url}/v1",
        api_key="ollama",
        timeout=300,
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "structured_output": False,
            "family": "unknown",
            "context_window": 8192,
        },
    )


def _format_context(hits: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] {h['content']}")
    return "\n".join(lines)


async def _run_team(query: str, context: str) -> str:
    client = _build_client()
    retriever = build_retriever_agent(client)
    answer = build_answer_agent(client)

    team = RoundRobinGroupChat(
        participants=[retriever, answer],
        termination_condition=MaxMessageTermination(max_messages=4),
    )

    prompt = (
        f"用户问题：{query}\n\n"
        f"检索到的文档片段：\n{context}\n\n"
        "请按顺序：检索助手先整理要点，回答助手再据此作答。"
    )
    stream = team.run_stream(task=prompt)
    # 收集最终回答（取 answer_agent 的最后一条消息）
    last_any = ""
    answer_text = ""
    async for message in stream:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            last_any = content
            # 注意：speaker 取自消息的 source/sender，依赖 answer_agent 的 name 字段。
            # 若将来修改 answer_agent 的 name，此过滤会静默失效并退化为 last_any。
            speaker = getattr(message, "source", "") or getattr(message, "sender", "")
            if speaker == "answer_agent":
                answer_text = content
    await client.close()
    return answer_text or last_any or "（未生成回答）"


def ask(query: str, rerank_top_n: int | None = None) -> str:
    """端到端：检索 → 多智能体整理与回答。同步入口。"""
    cfg = get_settings()
    hits = hybrid_retrieve(query, rerank_top_n=rerank_top_n or cfg.rerank_top_n)
    context = _format_context(hits)
    return asyncio.run(_run_team(query, context))
