"""
滑动窗口：从（已压缩的）历史中截取最近 N 条（memory_keep_recent）作为上下文窗口，
丢弃更早的对话，防止上下文无限增长。与 compressor 配合：先压缩早期、再滑窗保近期。
"""
from config import get_settings

_SETTINGS = get_settings()


def slide_window(history: list[dict], keep_recent: int | None = None) -> list[dict]:
    """
    取最近 keep_recent 条消息作为窗口。
    :param history: 完整会话历史
    :param keep_recent: 窗口条数，默认 settings.memory_keep_recent
    :return: 窗口内的消息列表（保持原顺序）
    """
    keep_recent = keep_recent or _SETTINGS.memory_keep_recent
    if keep_recent <= 0:
        return history
    return history[-keep_recent:]
