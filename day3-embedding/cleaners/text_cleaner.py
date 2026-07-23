import re

"""
author: tang
description: 文本清洗器，去除多余空行、制表符、不可见字符等噪声
"""

class TextCleaner:
    def clean(self, text: str) -> str:
        """清洗主流程：依次做多步正则处理"""
        text = self._remove_extra_whitespace(text)
        text = self._normalize_blank_lines(text)
        return text

    def _remove_extra_whitespace(self, text: str) -> str:
        """把制表符、不间断空格、多余空格统一处理"""
        # 1. 把制表符 \t 换成普通空格
        text = text.replace("\t", " ")
        # 2. 把不间断空格 \xa0（PDF/HTML 常见）换成普通空格
        text = text.replace("\xa0", " ")
        # 3. 把任意连续多个空格（含全角空格）压缩成 1 个普通空格
        text = re.sub(r"[ \u3000]+", " ", text)
        return text

    def _normalize_blank_lines(self, text: str) -> str:
        """把连续 3 个以上的空行压缩成 1 个空行"""
        # 把 3 个及以上连续换行 (\n) 压成 2 个（即 1 个空行）
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 再去掉每行首尾的空白（清掉"空行里残留的空格"）
        text = "\n".join(line.strip() for line in text.split("\n"))
        return text
