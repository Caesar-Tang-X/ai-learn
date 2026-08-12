"""
文本清洗：去除噪声、统一空白。

只做"质量"，不做"分块"。
"""
import re


def clean_text(text: str) -> str:
    """
    清洗纯文本：去控制字符、合并多余空白、去首尾空行。
    """
    # 去掉非打印控制字符（保留换行 \n 与制表 \t）
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    # 多个连续空行 -> 一个空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 行内多个空格 -> 一个空格
    text = re.sub(r"[ \t]+", " ", text)
    # 去每行首尾空白
    lines = [line.strip() for line in text.splitlines()]
    
    return "\n".join(lines).strip()
