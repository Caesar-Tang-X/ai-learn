"""
从用户 query 中解析期望的商品数量（通用，不依赖具体品类）。
支持阿拉伯数字与中文数字，以及「款/个/件/支/瓶/盒/种」等商品量词。
用户未明确要求数量时返回 None，由上层回退到默认数量。
"""
import re

_CN_NUM = {
    "零": 0, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _cn_to_int(s: str) -> int | None:
    """将简单中文数字（不含百千万单位）转 int；无法解析返回 None。"""
    if s in _CN_NUM:
        return _CN_NUM[s]
    # 处理「十X」「X十」「X十Y」等两位数
    if "十" in s:
        parts = s.split("十")
        if len(parts) == 2:
            tens = parts[0]
            ones = parts[1]
            t = _CN_NUM.get(tens, 1) if tens else 1
            o = _CN_NUM.get(ones, 0) if ones else 0
            return t * 10 + o
    return None


def parse_count_max(query: str) -> int | None:
    """
    解析 query 中期望的商品数量。
    例：'推荐3款' -> 3；'来三款' -> 3；'几款' 不解析（返回 None）。
    命中量词（款/个/件/支/瓶/盒/种）前的数字视为数量；否则返回 None。
    """
    q = query.strip()
    # 阿拉伯数字 + 量词
    m = re.search(r"(\d+)\s*(?:款|个|件|支|瓶|盒|种|件)", q)
    if m:
        v = int(m.group(1))
        return v if v > 0 else None
    # 中文数字 + 量词
    m = re.search(r"([零一二两三四五六七八九十]+)\s*(?:款|个|件|支|瓶|盒|种|件)", q)
    if m:
        v = _cn_to_int(m.group(1))
        return v if v and v > 0 else None
    return None
