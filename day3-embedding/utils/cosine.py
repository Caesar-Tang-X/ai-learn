"""
author: tang
description: 余弦相似度工具，衡量两段向量的语义接近程度
"""

import numpy as np


def cosine_similarity(a: list, b: list) -> float:
    """
    余弦相似度 = 点积 / (模长a × 模长b)，结果范围 [-1, 1]
    - 越接近 1：语义越像
    - 接近 0：基本无关
    - 接近 -1：语义相反
    """
    # 把普通列表转成 numpy 数组，方便向量运算
    va = np.array(a)
    vb = np.array(b)

    # 点积 A·B：对应位置相乘再求和
    dot = np.dot(va, vb)
    # 向量模长（长度）：各元素平方求和再开方
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)

    # 除模长 = "只看方向、不看长度"，避免长文本天然分数高
    # 分母为 0 兜底（极端情况两向量全 0）
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))
