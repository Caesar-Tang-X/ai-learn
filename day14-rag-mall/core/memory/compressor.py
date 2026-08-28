"""
上下文压缩：当历史 token 超阈值时，将早期对话交给 LLM 生成摘要，
用「摘要 + 近期窗口」替代完整历史，控制上下文长度。

注意：本模块为同步函数，内部用 asyncio.run 驱动异步 LLM 调用，
因此必须在「无运行中的事件循环」环境执行（如线程池 worker，
service 层通过 _run_sync 在 to_thread 中调用 get_context）。
摘要失败自动降级为「不压缩，仅滑窗」。
"""
import asyncio

from config import get_settings
from core.llm import get_llm

_SETTINGS = get_settings()


def _estimate_tokens(text: str) -> int:
    """粗略估算 token：中文约 1.5 字/token 的混合近似。"""
    return max(1, int(len(text) / 1.5))


def _build_summary_prompt(early_turns: list[dict]) -> str:
    convo = "\n".join(f"{t['role']}: {t['content']}" for t in early_turns)
    return (
        "请将以下对话压缩为一段简洁的要点摘要（保留用户需求、已确认商品、偏好等关键事实），"
        "不超过 200 字：\n" + convo
    )


async def _async_summarize(early_turns: list[dict]) -> str | None:
    """异步调用 LLM 流式生成摘要，聚合为完整文本。"""
    llm = get_llm(_SETTINGS.llm_provider)
    messages = [{"role": "user", "content": _build_summary_prompt(early_turns)}]
    chunks: list[str] = []
    async for chunk in llm.stream(messages):
        chunks.append(chunk)
    return "".join(chunks).strip() or None


def _summarize(early_turns: list[dict]) -> str | None:
    """同步入口：驱动异步摘要；任意异常返回 None（降级）。"""
    try:
        return asyncio.run(_async_summarize(early_turns))
    except Exception:
        # 本地无 ollama / 网络不通 / 模型缺失 / 方法异常 等，均降级
        return None


def compress(history: list[dict]) -> list[dict]:
    """
    压缩历史：若总 token 超 memory_max_tokens，将早期部分摘要，保留近期窗口。
    摘要失败则降级返回原历史（由 window 负责截取）。
    :param history: 完整历史
    :return: 压缩后的上下文列表（含一条 summary 系统消息 + 近期窗口）
    """
    total = sum(_estimate_tokens(t["content"]) for t in history)
    if total <= _SETTINGS.memory_max_tokens:
        return history

    keep = _SETTINGS.memory_keep_recent
    recent = history[-keep:] if keep > 0 else []
    early = history[:-keep] if keep > 0 else history
    if not early:
        return recent

    summary = _summarize(early)
    if not summary:
        return history  # 降级：不压缩，原样返回（window 会截最近）

    summary_msg = {"role": "system", "content": f"[历史摘要] {summary}"}
    return [summary_msg] + recent
