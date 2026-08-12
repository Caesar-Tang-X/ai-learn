"""
文本分块：按字符数滑动窗口切分（中文友好）。

chunk_size 与 chunk_overlap 来自 config。
"""
from config import get_settings


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """
    将文本切成带重叠的块。

    Args:
        text: 待切分文本。
        size: 每块最大字符数，缺省读 config.chunk_size。
        overlap: 相邻块重叠字符数，缺省读 config.chunk_overlap。
    Returns:
        文本块列表（每块非空）。
    """
    cfg = get_settings()
    size = size or cfg.chunk_size
    overlap = overlap or cfg.chunk_overlap

    if size <= 0:
        raise ValueError("chunk_size 必须为正整数")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk_overlap 必须 >=0 且 < chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap   # 滑动窗口，重叠部分
    return chunks
