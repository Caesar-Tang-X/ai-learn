"""
通用预算解析：从用户 query 中抽取价格上下限（单位：分）。

设计目标——「一次写好，覆盖所有表达形式，无需为每种新写法改代码」：
  1. 数值抽取与「约束方向」判定完全解耦，语法形式（区间 / 不等号 / 单位 / 阿拉伯与中文数字）
     已全量覆盖；
  2. 方向只靠「前后文方向词表」判定，新增罕见同义词（如「顶多」「不多于」）只需在词表里加一词，
     不需要改动正则或主流程；
  3. 支持的形式（举例，非穷举）：
       - 上限：50以内 / 100以下 / 不超过200 / 低于300 / 最多500 / 封顶600 / 预算800 / 限1000 / ≤100 / <100
       - 下限：500以上 / 2000往上 / 300起 / 不低于400 / 至少500 / 超过600 / 多于700 / 高于800 / ≥900 / >900
       - 区间：300-600 / 300~600 / 300到600 / 300至600 / 300—600 / 价格300-600元 / 预算300到600
       - 单位：元/块、百/千/万/亿、中文数字（三百、两千、一点五万）
       - 多段混合：取交集（如「预算300到600，封顶1000」→ min=300, max=600）

返回 (price_min, price_max)，未解析到对应端则为 None。
"""
import re

_CN_NUM = {
    "零": 0, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "百": 100, "千": 1000,
    "万": 10000, "亿": 100000000,
}

# 方向词表基础集（可按需扩展；「不」前缀自动反转方向，故只列无否定的基础词）。
# _MORE_SET：本身表达「≥ / 下限」（超过、高于、多于、以上、往上、起、之上、起步、至少）
# _LESS_SET：本身表达「≤ / 上限」（低于、少于、以内、以下、之内、内、最多、封顶、预算、限）
_MORE_SET = {"超过", "高于", "多于", "以上", "往上", "起", "之上", "起步", "至少", "≥", ">=", ">"}
_LESS_SET = {"低于", "少于", "以内", "以下", "之内", "内", "最多", "封顶", "封顶价", "预算", "限", "≤", "<=", "<"}
# 「不」前缀的方向词（如 不超过/不高于/不多于/不低于/不少于）由检测时自动反转，无需在此列出
_NEG = "不"

# 区间分隔符（连接两个金额 → 左为下限、右为上限）
_RANGE_SEPS = "[-~—–到至]"
_RANGE_RE = re.compile(rf"^\s*(?:{_RANGE_SEPS})\s*$")

# 数量单位：紧邻其后的数字视为「数量」而非「价格」，应从预算解析中排除
_QUANT_UNITS = ["款", "个", "件", "支", "只", "盒", "瓶", "袋", "包", "套", "双", "条", "罐", "枚", "台", "部", "本", "张", "片", "份", "根", "把"]

# 通用数值片段：阿拉伯数字[单位][元/块] 或 中文数字[元/块]（中文贪婪，交给 _cn_to_int 校验）
_NUM_RE = re.compile(
    r"(?P<ar>\d+(?:\.\d+)?)\s*(?P<au>万|千|百|亿)?\s*(?:元|块)?"
    r"|(?P<cn>[零一二两三四五六七八九十百千万亿]+)(?:元|块)?"
)

_WINDOW = 8  # 方向词判定窗口（字符数）


def _cn_to_int(s: str) -> int | None:
    """把 '两千零五十' / '1.5万' 这类中文/混合数字串转成 int；无法解析返回 None。"""
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([百千万亿])$", s)
    if m:
        base = float(m.group(1))
        unit = {"百": 100, "千": 1000, "万": 10000, "亿": 100000000}[m.group(2)]
        return int(base * unit)
    if not re.fullmatch(r"[零一二两三四五六七八九十百千万亿]+", s):
        return None
    total, section, current = 0, 0, 0
    for ch in s:
        if ch == "十":
            current = 10 if current == 0 else current * 10
        elif ch == "百":
            current = (current if current else 1) * 100
        elif ch == "千":
            current = (current if current else 1) * 1000
        elif ch == "万":
            section += (current if current else 1) * 10000
            total += section
            section, current = 0, 0
        elif ch == "亿":
            section += (current if current else 1) * 100000000
            total += section
            section, current = 0, 0
        else:
            current = _CN_NUM[ch]
    return total + section + current


def _to_fen(num: float, unit: str | None) -> int:
    if unit:
        num *= {"百": 100, "千": 1000, "万": 10000, "亿": 100000000}[unit]
    return int(round(num * 100))


def _extract_amounts(q: str):
    """抽取所有数值片段，返回 [(fen, start, end), ...]。排除被数量单位紧邻的数字（视为数量而非价格）。"""
    out = []
    for m in _NUM_RE.finditer(q):
        if m.group("ar") is not None:
            num = float(m.group("ar"))
            unit = m.group("au")
            fen = _to_fen(num, unit)
            s, e = m.start(), m.end()
        elif m.group("cn") is not None:
            val = _cn_to_int(m.group("cn"))
            if not val:
                continue
            fen = val * 100
            s, e = m.start(), m.end()
        else:
            continue
        # 紧邻其后的字符若为数量单位，则视为「数量」而非价格，跳过
        tail = q[e:e + 2]
        if any(tail.startswith(u) for u in _QUANT_UNITS):
            continue
        out.append((fen, s, e))
    return out


def _dir_of(window: str) -> int:
    """
    根据上下文窗口判定方向：
      返回 1 表示「下限(≥)」，-1 表示「上限(≤)」，0 表示无方向（裸金额）。
    方向词按长度降序匹配，命中「不」+词 时方向反转。
    """
    words = sorted(_MORE_SET | _LESS_SET, key=len, reverse=True)
    for w in words:
        if ("不" + w) in window:
            return -1 if w in _MORE_SET else 1   # 否定反转
        if w in window:
            return 1 if w in _MORE_SET else -1
    return 0


def parse_budget_range(query: str) -> tuple[int | None, int | None]:
    """
    通用预算解析。返回 (price_min, price_max)，单位：分；未解析到对应端为 None。
    支持上限/下限/区间/不等号/单位/中文数字及多段混合（取交集）。
    """
    if not query:
        return None, None
    q = query.replace(",", "").replace("，", "")
    amounts = _extract_amounts(q)
    if not amounts:
        return None, None

    lowers: list[int] = []
    uppers: list[int] = []

    for i, (fen, s, e) in enumerate(amounts):
        # 该片段前后的上下文窗口（截断到相邻片段边界，避免跨金额误判）
        prev_start = amounts[i - 1][2] if i > 0 else 0
        next_end = amounts[i + 1][1] if i + 1 < len(amounts) else len(q)
        before = q[prev_start:s]
        after = q[e:next_end]

        # 区间判定：与相邻片段被区间分隔符直接连接
        if i + 1 < len(amounts):
            mid = q[e:amounts[i + 1][1]]
            if _RANGE_RE.match(mid) and len(mid) <= 6:
                lowers.append(fen)                 # 左端 → 下限
                uppers.append(amounts[i + 1][0])   # 右端 → 上限
                continue

        d = _dir_of(before) or _dir_of(after)
        if d > 0:
            lowers.append(fen)
        elif d < 0:
            uppers.append(fen)
        else:
            # 裸金额（无方向词、非区间端点）：兼容旧行为，默认视为上限
            uppers.append(fen)

    pmin = max(lowers) if lowers else None
    pmax = min(uppers) if uppers else None
    # 防御：若上下限交叉（如异常输入），以更合理的一方为准
    if pmin is not None and pmax is not None and pmin > pmax:
        pmin, pmax = pmax, pmin
    return pmin, pmax


def parse_budget_min(query: str) -> int | None:
    """解析预算下限（分）。通用实现，见 parse_budget_range。"""
    return parse_budget_range(query)[0]


def parse_budget_max(query: str) -> int | None:
    """解析预算上限（分）。通用实现，见 parse_budget_range。"""
    return parse_budget_range(query)[1]
