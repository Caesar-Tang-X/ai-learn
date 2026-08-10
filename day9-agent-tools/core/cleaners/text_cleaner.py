import re

"""
author: tang
description: 文本清洗器，去噪声（含 Day4 新增：剥离 Markdown 代码围栏，提升 Embedding 质量）
"""

class TextCleaner:
    def clean(self, text: str) -> str:
        # 顺序很重要：先剥掉整段代码块这个"大噪声"，再做细粒度空白归一
        text = self._remove_code_fences(text)      # Day4 新增
        text = self._remove_extra_whitespace(text)
        text = self._normalize_blank_lines(text)
        return text

    def _remove_code_fences(self, text: str) -> str:
        """剥离 Markdown 代码围栏 ```lang ... ```，避免源码/语言标记词污染 chunk、干扰 Embedding"""
        # DOTALL 让 . 能跨换行匹配整段代码块；.*? 非贪婪，保证每块独立匹配
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        return text

    def _remove_extra_whitespace(self, text: str) -> str:
        text = text.replace("\t", " ")
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \u3000]+", " ", text)
        return text

    def _normalize_blank_lines(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        return text
