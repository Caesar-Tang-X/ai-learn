import re

"""
author: tang
description: 手写递归字符分层分块器，理解 chunking 原理
"""

class RecursiveTextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        # overlap 必须小于 chunk_size，否则下一箱永远带着上一箱全部内容，死循环
        assert chunk_overlap < chunk_size, "chunk_overlap 必须小于 chunk_size"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 分隔符按优先级从粗到细排列：先按段落，再按句子，最后按字符
        self.separators = ["\n\n", "\n", "。", "！", "？", "", ]

    def split(self, text: str) -> list[str]:
        """入口：先递归切成不超长的小片段，再合并成带重叠的 chunk"""
        pieces = self._split_recursive(text, self.separators)
        return self._merge_pieces(pieces)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """递归切分：优先用粗分隔符，超长段才降到更细粒度继续切"""
        # 取当前层分隔符，剩余的分隔符留给下一层递归（从粗到细逐级降级）
        sep = separators[0]
        rest = separators[1:]
        # 到最底层：按字符逐个切（保证任何超长内容最终都能被拆开）
        if sep == "":
            parts = list(text)
        else:
            parts = text.split(sep)
        result = []
        for part in parts:
            # 当前片段够短 -> 直接收下
            if len(part) <= self.chunk_size:
                result.append(part)
            else:
                # 仍超长 -> 用更细的分隔符继续递归切（rest 已去掉当前层）
                result.extend(self._split_recursive(part, rest))
        return result

    def _merge_pieces(self, pieces: list[str]) -> list[str]:
        """把小片段按顺序装箱，装满 chunk_size 就封箱，并在下一箱开头保留 overlap 个字符"""
        chunks = []
        current = ""
        for piece in pieces:
            # 加上这个 piece 会超出 -> 先把 current 封箱
            if len(current) + len(piece) > self.chunk_size and current:
                chunks.append(current)
                # 新箱开头 = 上一箱尾部 overlap 个字符（实现"重叠"）
                current = current[-self.chunk_overlap:] + piece
            else:
                current += piece
        # 循环结束，把最后一箱也收进去
        if current:
            chunks.append(current)
        return chunks
