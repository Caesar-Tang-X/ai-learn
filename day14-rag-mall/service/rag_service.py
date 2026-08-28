"""
RAG 编排服务：检索增强 + 会话记忆 + 生成。

标准流程（通用 RAG 范式）：
  1. 硬过滤（预算/价格/类目/上架状态）+ 向量相似度阈值，从向量库召回相关候选；
  2. 重排模型精排并按重排分数截断，仅保留真正相关的商品；
  3. 调用 LLM 为相关商品逐一撰写推荐理由（基于真实字段，不编造）；
  4. 以 SSE 事件流返回：intro 开场白 + 每个商品的文本（标题/价格/理由）与图片。

所有商品的标题、价格、图片均来自知识库检索结果；推荐理由由 LLM 基于真实
商品信息生成。解析按数组顺序对应候选，避免依赖标题精确匹配。
"""
import asyncio
import json as _json
import re as _re

from config import get_settings
from core.retrieval import retrieve
from core.retrieval.budget import parse_budget_range
from core.retrieval.count import parse_count_max
from core.memory import (save_turn, get_context, load_last_recommendations,
                         save_last_recommendations, save_constraints, load_constraints)
from core.intent import judge_intent, build_clarify_prompt
from core.llm import get_llm

_SETTINGS = get_settings()


def _build_cards(docs: list[dict]) -> list[dict]:
    """从检索结果构造商品结构化信息（id/标题/价格/图片）。价格单位为「分」，转「元」展示。"""
    cards = []
    for d in docs:
        m = d.get("metadata") or {}
        price_fen = m.get("price")
        # 商品 id 使用导入时的 spu_id（metadata.spu_id），而非数据库自增主键
        spu_id = m.get("spu_id")
        cards.append({
            "id": spu_id if spu_id is not None else d.get("id"),
            "title": m.get("title") or "(未命名)",
            "price_yuan": round(price_fen / 100, 2) if isinstance(price_fen, (int, float)) else None,
            "image": m.get("thumbnail_img"),
            "brief": m.get("brief") or "",
            "intro": m.get("intro") or "",
        })
    return cards


_RERANK_SYSTEM = (
    "你是购物助手的「候选精排器」。给定【用户需求】和一批【召回候选商品】，"
    "请挑选出最贴合用户需求（品类/场景/人群/预算/偏好）的商品。\n"
    "判断原则（优先级从高到低，全部基于商品的真实属性与用户需求推理，不要依赖关键词匹配）：\n"
    "  1) 人群/场景契合是硬约束：若某商品明显不适合该人群或场景（例如投其所好错位、"
    "私密/敏感类商品不适合作为给长辈或亲属的礼物等），应剔除；反之契合的优先。\n"
    "  2) 预算约束：用户预算为硬约束，价格明显偏离（远超上限或远低于下限）的商品降权或剔除；"
    "模糊预算（「元左右」）允许合理浮动，但不可大幅偏离。\n"
    "  3) 在都符合 1)、2) 的前提下，尽量兼顾【品类多样】，避免全部为同一细分类。\n"
    "  - 只输出一个 JSON 数组（0-based 候选序号，按匹配度从高到低排序，最多 top_n 个），不要解释、不要加 ``` 标记。\n"
    "示例：用户需求=「预算1000元送给亲友的实用礼物」，候选=[0:按摩仪,1:某私密用品,2:廉价贴纸,3:工具套装,4:护肤套装]"
    " -> [0,3,4]（剔除明显不适合作为礼物的私密类与廉价无关品，保留实用合适项）"
)


# 通用「私密/成人向」商品属性缓存：key=spu_id，value=bool（是否属私密/成人向）。
# 通用适用性判定缓存：key=(商品id, 场景上下文)，value=该商品是否不适合此场景/对象。
# 通过 LLM 通用分类得到（不枚举任何具体商品名/关键词/场景词），命中后稳定复用，避免重复调用。
_UNSUITABLE_CACHE: dict = {}


async def _is_unsuitable(card: dict, context: str = "") -> bool:
    """
    通用适用性判定：给定「场景/对象上下文」context（来自 scene_context，LLM 对任意需求场景的自由概括），
    判断该商品是否不适合此场景/对象。完全由 LLM 基于商品标题+简介+上下文理解，
    不依赖任何关键词/类目/对象枚举，也不对 context 的取值做分支判断。
    context 为空（纯自购/无场景）时，视为无适用约束，一律返回 False（放行）。
    失败按保守放行（False）处理，避免误杀正常商品（误杀比漏判体验更差，且用户可显式过滤）。
    """
    ctx = (context or "").strip()
    if not ctx:
        return False
    cache_key = (card.get("id"), ctx)
    if cache_key in _UNSUITABLE_CACHE:
        return _UNSUITABLE_CACHE[cache_key]
    title = card.get("title", "")
    brief = card.get("brief", "")
    system = (
        "你是购物场景适配助手。判断【商品】是否适合在【场景/对象】下被推荐给用户。\n"
        "判断依据是商品的实际用途与该场景/对象的匹配度，而不是某个固定品类清单。"
        "例如：私密/成人向商品在某些送礼对象（如长辈、亲友、同事）语境下不合适，"
        "但在其他语境（如送伴侣、自购）下可能合适——是否合适由你结合给出的场景/对象综合判断，不要套用固定规则。\n"
        "请基于商品标题、简介与给出的场景/对象综合判断，不要仅凭品牌名或个别字眼臆断。\n"
        "只输出一个 JSON 对象：{\"unsuitable\": true/false}。不要解释、不要加 ``` 标记。"
    )
    try:
        llm = get_llm(_SETTINGS.llm_provider)
        raw = (await _llm_text([
            {"role": "system", "content": system},
            {"role": "user", "content": f"场景/对象：{ctx}\n商品标题：{title}\n商品简介：{brief}"}
        ])).strip()
        raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        obj = _json.loads(raw)
        result = bool(obj.get("unsuitable", False))
    except Exception:
        result = False
    _UNSUITABLE_CACHE[cache_key] = result
    return result


async def _rerank_candidates(user_query: str, cards: list[dict], top_n: int,
                             unsuitable_context: str = "") -> list[dict]:
    """
    用 LLM 依据用户场景/人群/预算，从召回候选中精选 top_n 个最匹配商品，并尽量兼顾品类多样。
    失败或候选不足时回退原始顺序，保证检索链路不挂。
    - unsuitable_context：非空时（场景上下文来自 scene_context，LLM 对任意场景的自由概括），
      先用通用适配判定过滤掉「不适合该场景/对象」的候选；判定由 LLM 通用完成，
      不枚举关键词/场景/对象，也不对 context 取值做分支；是否排除完全由 LLM 按场景语义决定。
    """
    if not cards:
        return []
    # 场景适用性过滤：仅当存在场景/对象上下文时才触发，按上下文通用判定（确定性过滤）
    ctx = (unsuitable_context or "").strip()
    if ctx:
        kept = []
        for c in cards:
            if not await _is_unsuitable(c, ctx):
                kept.append(c)
        cards = kept
    if len(cards) <= top_n:
        return cards[:top_n]
    try:
        listing = "\n".join(
            f"{i}. {c.get('title', '')} | 价格:{c.get('price_yuan')}元 | {c.get('brief', '')[:40]}"
            for i, c in enumerate(cards)
        )
        messages = [
            {"role": "system", "content": _RERANK_SYSTEM},
            {"role": "user", "content": f"用户需求：{user_query}\n候选商品（共{len(cards)}个）：\n{listing}\n"
                                        f"请返回最匹配需求的 top_{top_n} 个候选序号（JSON 数组，0-based）。"},
        ]
        raw = (await _llm_text(messages)).strip()
        raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        idxs = _json.loads(raw)
        if isinstance(idxs, list):
            picked = [cards[i] for i in idxs if isinstance(i, int) and 0 <= i < len(cards)]
            # LLM 可能返回重复序号：按 id 去重保序；不足 top_n 时按候选原序补齐
            seen, ordered = set(), []
            for c in picked:
                if id(c) not in seen:
                    seen.add(id(c)); ordered.append(c)
            for c in cards:
                if len(ordered) >= top_n:
                    break
                if id(c) not in seen:
                    seen.add(id(c)); ordered.append(c)
            return ordered[:top_n]
    except Exception:
        pass
    return cards[:top_n]


def _build_intro(n: int) -> str:
    """开场白文案（固定文本，不调 LLM）。"""
    if n == 0:
        return "抱歉，没有找到与您需求相关的商品，您可以换个说法、补充预算或场景后重试。"
    return "已为您筛选出以下相关商品："


def _build_reason_prompt(query: str, cards: list[dict]) -> list[dict]:
    """构造 LLM 提示：为候选商品逐条撰写推荐理由（仅写理由，不做取舍）。"""
    goods = []
    for i, c in enumerate(cards):
        price = f"¥{c['price_yuan']}" if c["price_yuan"] is not None else "价格未知"
        goods.append(f"{i+1}. 标题:{c['title']} | 价格:{price} | 简介:{c.get('brief') or ''}")
    goods_text = "\n".join(goods)
    system = (
        "你是商城导购助手。下方【候选商品】是从商城数据库检索出的真实商品"
        "（已满足用户预算等条件），请为【每一个】商品写一段 2-3 句推荐理由，"
        "说明其适合的场景/人群与特点，必须基于商品真实信息，不得编造。\n"
        "只输出一个 JSON 数组，不要输出其他任何文字、不要加 ``` 标记。\n"
        "每个元素格式：{\"reason\": 推荐理由}，按商品顺序一一对应。\n"
        f"【候选商品】\n{goods_text}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"用户需求：{query}\n请返回每个商品的推荐理由（JSON 数组，按商品顺序）。"},
    ]


def _parse_reasons(text: str, cards: list[dict]) -> list[dict]:
    """解析 LLM 返回的 JSON 数组，按数组顺序对应候选；解析失败或数量不足时以真实字段兜底。"""
    t = text.strip()
    t = _re.sub(r"<think>.*?</think>", "", t, flags=_re.DOTALL).strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
        t = t.strip()
    try:
        arr = _json.loads(t)
    except Exception:
        arr = []
    if not isinstance(arr, list):
        arr = []

    result = []
    for i, item in enumerate(arr):
        if i >= len(cards):
            break
        result.append({"card": cards[i], "reason": str(item.get("reason", "")).strip()})

    # 不足部分以真实字段生成兜底理由，保证候选全部展示
    while len(result) < len(cards):
        c = cards[len(result)]
        price = f"¥{c['price_yuan']}" if c["price_yuan"] is not None else "价格未知"
        fallback = f"{c['title']}，价格 {price}，{c.get('brief') or '本商城在售商品'}。"
        result.append({"card": c, "reason": fallback})
    return result


async def _llm_text(messages: list[dict]) -> str:
    """调用 LLM（非流式，聚合为完整文本返回）。"""
    llm = get_llm(_SETTINGS.llm_provider)
    chunks = []
    async for c in llm.stream(messages):
        chunks.append(c)
    return "".join(chunks)


_REWRITE_SYSTEM = (
    "你是购物助手的「查询改写器」。用户可能有多轮对话，当前这句往往省略了前面提过的需求"
    "（如前面说「男士手表」，现在只说「预算2000内」）。\n"
    "请把【仅用户说过的话】综合改写成一句独立、可直达向量检索的购物查询，"
    "保留所有关键约束（品类/场景/人群/预算/数量/偏好等），去掉寒暄与无信息量的反馈词。\n"
    "【忠实度铁律（最重要）】：\n"
    "  - 只能迁移用户【在历史或当前句中明确说过的】约束（如预算、数量、已点名的品类/人群/场景）；\n"
    "  - 绝对禁止无中生有：不要把笼统词脑补成具体品类或子类。用户说「礼物」就保持「礼物」，"
    "不得自行添加用户没说过的品类、品牌或属性；\n"
    "  - 若当前句表达了【新的场景/对象/品类】（如历史聊某品类、当前转向其他对象或场景），必须以当前句为准，"
    "丢弃历史中被脑补的无关品类，不要被上一轮的推荐结果或历史里的笼统词带偏。\n"
    "同时判断用户当前这句话相对【上一轮推荐结果】的价格调整意图，注意以下区分：\n"
    "  - cheaper=true：用户希望比上一轮【更便宜】（如「太贵了/便宜点/再实惠些」），且未给出新的绝对预算；\n"
    "  - pricier=true：用户希望比上一轮【更贵/更好】（如「太便宜了/档次低点/换个好的」），且未给出新的绝对预算；\n"
    "  - 若用户给出新的绝对预算（如「50以内」「500以上」），或提出了【全新的需求/品类/场景】（如上一轮聊手表、现在说「送给父母的礼物」），"
    "则 cheaper 与 pricier 都应为 false，应按新需求全新检索，不要沿用上一轮价格区间。\n"
    "只输出一个 JSON 对象，不要解释、不要加 ``` 标记：\n"
    '{"query": "改写后的检索查询", "cheaper": true/false, "pricier": true/false, '
    '"price_min": 整数或null, "price_max": 整数或null}\n'
    "其中 price_min / price_max 为用户预算区间的【下限/上限】，单位人民币【元】（整数），未提及预算则为 null。\n"
    "请充分理解口语与模糊表述，给出合理的价格区间（不要只取单一数字当硬上限，否则会召回远低于预算的便宜货）：\n"
    "  - 「1千块左右的」「大概一千」→ min=800, max=1200（给 ±20% 容差区间）\n"
    "  - 「几百块」「两三百」→ min=200, max=900（按量级给区间）\n"
    "  - 「上千」「一千以上」→ min=1000, max=null\n"
    "  - 「300-600」「300到600」→ min=300, max=600\n"
    "  - 「500以上」→ min=500, max=null；「50以内」「预算2000内」→ min=null, max=50/2000\n"
    "  - 若用户给出了新的绝对预算，则 cheaper/pricier 均为 false（按新预算全新检索）。\n"
    '示例1：历史=[「推荐10款男士手表」]，当前=「太贵了，预算2000内」 -> '
    '{"query": "预算2000内的男士手表", "cheaper": false, "pricier": false, "price_min": null, "price_max": 2000}\n'
    '示例2：历史=[「推荐男士内裤」]，当前=「太贵了」 -> '
    '{"query": "男士内裤", "cheaper": true, "pricier": false, "price_min": null, "price_max": null}\n'
    '示例3：历史=[「推荐男士手表」]，当前=「太便宜了，要好的」 -> '
    '{"query": "男士手表", "cheaper": false, "pricier": true, "price_min": null, "price_max": null}\n'
    '示例4：历史=[「推荐男士手表」]，当前=「推荐5款送给父母的礼物」 -> '
    '{"query": "推荐5款送给父母的礼物", "cheaper": false, "pricier": false, "price_min": null, "price_max": null}\n'
    '示例5：历史=[[]]，当前=「要1千块左右的女士礼物」 -> '
    '{"query": "1千块左右的女士礼物", "cheaper": false, "pricier": false, "scene_context": "送女士的礼物", "price_min": 800, "price_max": 1200}\n'
    '示例6：历史=[「推荐1千左右的女士礼物」]，当前=「送长辈的生日礼物有什么推荐」 -> '
    '{"query": "送长辈的生日礼物", "cheaper": false, "pricier": false, "scene_context": "送长辈的生日礼物", "price_min": null, "price_max": null}'
    "（注意：当前句切换到新的对象/场景时，以当前句为准，不要沿用历史上被脑补出的具体品类）\n"
    "另输出 scene_context（字符串）：由你自由概括当前需求所处的「场景/对象」上下文"
    "（如「送长辈」「送伴侣」「送孩子」「送朋友」「自购」「囤货」等任意表述，不限于送礼）；"
    "纯自购、闲逛、问商品信息则为空字符串。该字段是 LLM 对任意场景的自由自然语言概括，"
    "代码不枚举、不分支任何取值；下游通用判定据此判断「某商品是否适合此场景/对象」，具体是否合适由 LLM 决定。\n"
    '完整 JSON 示例：{"query": "...", "cheaper": false, "pricier": false, "scene_context": "送长辈", "price_min": null, "price_max": null}'
)


async def _rewrite_query(query: str, history: list[dict]) -> dict:
    """
    通用多轮查询：只基于用户侧消息（排除 assistant 回复）凝练成独立可检索查询，
    并通用地识别「相对降价」意图（cheaper/pricier）与「场景上下文」（scene_context）。不枚举任何特定场景/品类词。
    返回 {"query": str, "cheaper": bool, "pricier": bool, "scene_context": str}。
    - cheaper：相对上一轮更便宜（且未给新绝对预算）
    - pricier：相对上一轮更贵/更好（且未给新绝对预算）
    - scene_context：LLM 对任意需求场景/对象的自由自然语言概括（如「送长辈」「送孩子」「自购」等），无则空串；
      作通用上下文供下游「某商品是否适合此场景/对象」的通用判定使用，代码不枚举、不分支其取值。
    全新需求（含新品类/场景）时二者皆 False，按新需求全新检索。
    任意失败（解析/调用异常）降级为「拼接所有 user 消息 + 当前 query」，cheaper/pricier 均为 False（安全侧：不下压，避免误出极端低价品）。
    """
    # 仅提取 history 中的 user 内容，绝不混入 assistant 回复
    user_turns = [m.get("content", "").strip() for m in (history or [])
                  if m.get("role") == "user" and (m.get("content") or "").strip()]
    current = (query or "").strip()
    if not user_turns and not current:
        return {"query": current, "cheaper": False, "pricier": False, "scene_context": ""}

    # 降级拼接：直接把所有用户话语 + 当前 query 合并（去重保序）
    fallback_query = " ".join(dict.fromkeys([*user_turns, current])).strip()

    try:
        llm = get_llm(_SETTINGS.llm_provider)
        messages = [{"role": "system", "content": _REWRITE_SYSTEM}]
        convo = "\n".join(f"用户：{t}" for t in user_turns)
        messages.append({"role": "user", "content": f"用户历史：\n{convo}\n当前：{current}\n请输出 JSON。"})
        raw = (await _llm_text(messages)).strip()
        raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        obj = _json.loads(raw)
        q = str(obj.get("query", "")).strip() or fallback_query
        cheaper = bool(obj.get("cheaper", False))
        pricier = bool(obj.get("pricier", False))
        scene_context = str(obj.get("scene_context", "") or "").strip()
        # 互斥保护：二者同时为真时视为无效，按全新需求处理
        if cheaper and pricier:
            cheaper, pricier = False, False
        # LLM 给出的预算区间（单位：元）→ 转「分」；解析异常则置 None（交由正则兜底）
        try:
            pmin = int(obj["price_min"]) * 100 if obj.get("price_min") is not None else None
            pmax = int(obj["price_max"]) * 100 if obj.get("price_max") is not None else None
        except (TypeError, ValueError):
            pmin = pmax = None
        return {"query": q, "cheaper": cheaper, "pricier": pricier, "scene_context": scene_context,
                "price_min": pmin, "price_max": pmax}
    except Exception:
        return {"query": fallback_query, "cheaper": False, "pricier": False, "scene_context": "",
                "price_min": None, "price_max": None}


_REFERENCE_SYSTEM = (
    "你是购物助手的「指代解析器」。用户可能基于上一轮已推荐的商品列表继续提问"
    "（如「详细介绍第一个」「刚才那个怎么样」「第二款和第四款对比」）。\n"
    "请判断用户当前这句话是否引用了【上一轮推荐列表中的具体商品】。\n"
    "若是，返回被引用的商品序号（1-based，取第一处引用即可）；若引用了多个或范围，取最小序号。\n"
    "若这句话是新的购物需求（重新检索类，如「再来点便宜的」「推荐耳机」）或不引用任何已列商品，返回 0。\n"
    "只输出一个 JSON 对象，不要解释、不要加 ``` 标记：\n"
    '{"ref_index": 0}  或  {"ref_index": 3}'
)


async def _resolve_reference(query: str, last_recs: list[dict]) -> int | None:
    """
    通用指代解析：判断 query 是否引用上一轮推荐列表中的某个商品，返回 1-based 索引；非引用返回 None。
    不枚举任何场景词/商品名，完全由 LLM 基于商品标题列表判断。
    LLM 失败或解析异常统一返回 None（交由常规检索流程处理）。
    """
    if not last_recs:
        return None
    llm = get_llm(_SETTINGS.llm_provider)
    listing = "\n".join(f"{i+1}. {c.get('title', '')}" for i, c in enumerate(last_recs))
    messages = [
        {"role": "system", "content": _REFERENCE_SYSTEM},
        {"role": "user", "content": f"上一轮推荐列表：\n{listing}\n\n用户当前说：{query}\n请判定引用的商品序号。"},
    ]
    try:
        raw = (await _llm_text(messages)).strip()
        raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        obj = _json.loads(raw)
        idx = int(obj.get("ref_index", 0))
        if idx <= 0 or idx > len(last_recs):
            return None
        return idx
    except Exception:
        return None


def _build_detail_prompt(query: str, card: dict) -> list[dict]:
    """构造 LLM 提示：针对单个已推荐商品，生成详细介绍/对比说明（基于真实字段，不编造）。"""
    price = f"¥{card['price_yuan']}" if card.get("price_yuan") is not None else "价格未知"
    info = f"标题:{card.get('title','')} | 价格:{price} | 简介:{card.get('brief') or ''} | 详情:{card.get('intro') or ''}"
    system = (
        "你是商城导购助手。用户针对【上一轮已推荐过的某个商品】进一步提问，"
        "请基于该商品的真实信息作答（详细介绍、对比、适用场景等），不得编造参数。\n"
        f"【商品信息】\n{info}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"用户说：{query}\n请基于上述商品信息作答。"},
    ]


def _resolve_budget(rewrite: dict, query: str, prev_constraints: dict,
                    eff_filters: dict) -> tuple:
    """
    解析并归一化价格约束（单位：分）。
    优先级：LLM 预算解析（理解模糊/量级/区间，给容差区间） > 正则兜底 > 继承上一轮预算。
    返回 (applied_min, applied_max, eff_filters)；模糊/区间预算会按配置比例外扩候选池。
    """
    rw_pmin, rw_pmax = rewrite.get("price_min"), rewrite.get("price_max")
    applied_min = applied_max = None
    if rw_pmin is not None or rw_pmax is not None:
        applied_min, applied_max = rw_pmin, rw_pmax
    else:
        budget_min, budget_max = parse_budget_range(query)
        if budget_min is not None or budget_max is not None:
            applied_min, applied_max = budget_min, budget_max
    # 本轮无新预算且非相对调价 → 继承上一轮约束，保证多轮预算持续生效
    if applied_min is None and applied_max is None and not (rewrite.get("cheaper") or rewrite.get("pricier")):
        applied_min = prev_constraints.get("price_min")
        applied_max = prev_constraints.get("price_max")

    if applied_min is not None and eff_filters.get("price_min") is None:
        eff_filters["price_min"] = applied_min
    if applied_max is not None and eff_filters.get("price_max") is None:
        eff_filters["price_max"] = applied_max

    # 模糊/区间预算：把硬过滤区间按配置比例外扩，避免候选被卡死在过窄区间（硬边界不扩）
    if eff_filters.get("price_min") is not None and eff_filters.get("price_max") is not None:
        r = _SETTINGS.budget_expand_ratio
        eff_filters["price_min"] = int(eff_filters["price_min"] * (1 - r))
        eff_filters["price_max"] = int(eff_filters["price_max"] * (1 + r))
    return applied_min, applied_max, eff_filters


def _build_retrieve_query(rewrite: dict, last_recs: list[dict]) -> str:
    """
    构造独立可检索 query。
    仅当用户在上一轮推荐上做相对调价（cheaper/pricier）时，才把旧标题并入查询锚定同批商品；
    全新需求/场景切换时不锚定，避免旧品类强加干扰。
    """
    q = rewrite["query"]
    if last_recs and (rewrite.get("cheaper") or rewrite.get("pricier")):
        anchor = " ".join(dict.fromkeys([c.get("title", "") for c in last_recs if c.get("title")]))
        if anchor:
            q = " ".join(dict.fromkeys([q, anchor])).strip()
    return q


def _apply_relative_price(rewrite: dict, last_recs: list[dict], eff_filters: dict) -> tuple:
    """
    相对价格约束推导（元→分）。cheaper 下压上限+下限保护；pricier 上提下限+放开上限。
    仅当用户未显式给绝对预算时应用（绝对预算优先）。返回 (cheaper_applied, pricier_applied)。
    """
    has_abs_budget = eff_filters.get("price_max") is not None or eff_filters.get("price_min") is not None
    cheaper_applied = pricier_applied = False
    if not (last_recs and not has_abs_budget):
        return cheaper_applied, pricier_applied
    prices = [c["price_yuan"] for c in last_recs if c.get("price_yuan") is not None]
    if not (prices and (rewrite.get("cheaper") or rewrite.get("pricier"))):
        return cheaper_applied, pricier_applied
    if rewrite.get("cheaper"):
        eff_filters["price_max"] = int(max(prices) * _SETTINGS.relative_price_factor * 100)
        eff_filters["price_min"] = max(1, int(min(prices) * _SETTINGS.relative_price_floor * 100))
        cheaper_applied = True
    elif rewrite.get("pricier"):
        eff_filters["price_min"] = int(min(prices) * _SETTINGS.relative_price_lift * 100)
        pricier_applied = True
    return cheaper_applied, pricier_applied


async def _search_candidates(retrieve_query: str, eff_filters: dict, want_n: int,
                             unsuitable_ctx: str, applied_min, applied_max) -> list[dict]:
    """检索 + LLM 精排；候选不足时按梯度重试以尽量满足用户数量诉求：
       1) 放大召回规模（recall_top_k 倍增）让重排有更多可挑候选；
       2) 若有预算约束，按配置渐进放宽预算（始终围绕原始预算，不放大到无约束）。
       任一步满足 want_n 即停止；均不满足则返回当前最优结果。"""
    async def _once(recall_top_k: int | None = None):
        d = await _run_sync(retrieve, retrieve_query, eff_filters, want_n, recall_top_k)
        return await _rerank_candidates(retrieve_query, _build_cards(d), want_n, unsuitable_ctx)

    cards = await _once()
    # 梯度1：放大召回规模，缓解「要N款却不足」的弱相关截断问题
    if len(cards) < want_n:
        cards = await _once(recall_top_k=max(_SETTINGS.retrieval_top_k, want_n) * 2)
    # 梯度2：有预算约束时渐进放宽（围绕原始预算）
    if len(cards) < want_n and applied_min is not None and applied_max is not None:
        ratios = _SETTINGS.budget_relax_ratios
        for i in range(0, len(ratios) - 1, 2):
            eff_filters["price_min"] = int(applied_min * ratios[i])
            eff_filters["price_max"] = int(applied_max * ratios[i + 1])
            cards = await _once()
            if len(cards) >= want_n:
                break
    return cards


def _need_clarify(rewrite: dict, last_recs: list[dict], intent: dict, has_cards: bool) -> bool:
    """
    澄清决策：在已知需求上继续（有上一轮推荐）仅在纯闲聊或无命中时引导；
    全新对话还需需求明确；无上下文的相对调价诉求无法落地 → 引导给预算。
    """
    if last_recs:
        need = (not intent["is_shoppable"]) or (not has_cards)
    else:
        need = (not intent["is_shoppable"]) or (not intent["is_clear"]) or (not has_cards)
    if (rewrite.get("cheaper") or rewrite.get("pricier")) and not last_recs:
        need = True
    return need


async def answer(session_id: str, query: str, filters: dict | None = None):
    """对话主流程（编排）：取上下文 → 改写/预算 → 指代 → 检索/精排 → 意图判定 → 流式推荐。"""
    eff_filters = dict(filters or {})

    # 1) 取会话上下文（已压缩+滑动窗口）与上一轮推荐列表（供指代引用）
    history = await _run_sync(get_context, session_id)
    last_recs = await _run_sync(load_last_recommendations, session_id)

    # 多轮改写（含 LLM 预算解析，单次调用同时完成）；price_min/price_max 已转「分」
    rewrite = await _rewrite_query(query, history)

    # 预算解析 + 外扩（优先级：LLM > 正则 > 继承）
    prev_constraints = await _run_sync(load_constraints, session_id)
    applied_min, applied_max, eff_filters = _resolve_budget(
        rewrite, query, prev_constraints, eff_filters)

    # 用户显式数量优先，否则用默认数量
    top_n = parse_count_max(query) or None

    # 2) 指代解析：追问「上一轮已推荐商品」（第一个/刚才那个/第二款…）直接复用卡片，跳过检索
    ref_index = await _resolve_reference(query, last_recs)
    if ref_index is not None:
        card = last_recs[ref_index - 1]
        detail = (await _llm_text(_build_detail_prompt(query, card))).strip()
        yield {"type": "intro", "data": _build_intro(1)}
        yield {"type": "item_text", "data": {
            "index": ref_index,                    # 沿用上一轮展示序号，保持连续
            "id": card.get("id"),
            "title": card.get("title"),
            "price_yuan": card.get("price_yuan"),
            "image": card.get("image"),
            "intro": card.get("intro") or "",
            "reason": detail,                      # 承载详细介绍/对比文本
        }}
        await _run_sync(save_turn, session_id, query, f"针对商品「{card.get('title')}」的追问：{detail}")
        await _run_sync(save_last_recommendations, session_id, last_recs)  # 维持列表，便于继续追问
        return

    # 3) 检索 query 构造 + 5) 相对价格推导 + 检索/精排
    retrieve_query = _build_retrieve_query(rewrite, last_recs)
    cheaper_applied, pricier_applied = _apply_relative_price(rewrite, last_recs, eff_filters)
    want_n = top_n or _SETTINGS.rerank_top_n
    unsuitable_ctx = str(rewrite.get("scene_context", "") or "").strip()
    cards = await _search_candidates(retrieve_query, eff_filters, want_n, unsuitable_ctx,
                                     applied_min, applied_max)

    # 相对调价兜底：下压/上提后无命中，按方向依次放开价格约束再检索，避免死循环
    if (cheaper_applied or pricier_applied) and len(cards) == 0:
        if eff_filters.get("price_max") is not None:
            eff_filters.pop("price_max", None)
            docs = await _run_sync(retrieve, retrieve_query, eff_filters, want_n)
            cards = await _rerank_candidates(retrieve_query, _build_cards(docs), want_n, unsuitable_ctx)
        if len(cards) == 0 and eff_filters.get("price_min") is not None:
            eff_filters.pop("price_min", None)
            docs = await _run_sync(retrieve, retrieve_query, eff_filters, want_n)
            cards = await _rerank_candidates(retrieve_query, _build_cards(docs), want_n, unsuitable_ctx)

    # 6) 意图判定 + 澄清决策
    intent = await judge_intent(query, history)
    if _need_clarify(rewrite, last_recs, intent, len(cards)):
        hint = "上一轮检索未匹配到商品" if len(cards) == 0 and intent["is_shoppable"] else ""
        clarify_msgs = build_clarify_prompt(query, history, hint)
        try:
            text = await _llm_text(clarify_msgs)
        except Exception:
            text = "您好，我是您的导购助手。请告诉我您想买什么类型的商品、用于什么场景或人群、以及预算范围，我会为您精准推荐。"
        await _run_sync(save_turn, session_id, query, text)
        yield {"type": "clarify", "data": text}
        return

    # 7) 需求明确且有命中：生成推荐理由并流式下发
    raw = await _llm_text(_build_reason_prompt(query, cards))
    reasoned = _parse_reasons(raw, cards)

    intro_text = _build_intro(len(cards))
    if cheaper_applied:
        intro_text = "明白，已为您下调预算并筛选更优惠的同类商品："
    elif pricier_applied:
        intro_text = "明白，已为您上调预算筛选更高品质的同类商品："
    yield {"type": "intro", "data": intro_text}

    for idx, item in enumerate(reasoned):
        card = item["card"]
        # 图文合并为单个 item_text 事件，附带序号、商品 id 与商品自身 intro
        yield {"type": "item_text", "data": {
            "index": idx + 1,                      # 展示序号（从 1 开始）
            "id": card["id"],                      # 商品唯一 id（spu_id）
            "title": card["title"],
            "price_yuan": card["price_yuan"],
            "image": card.get("image"),
            "intro": card.get("intro") or "",      # 商品自身的 intro（metadata.intro）
            "reason": item["reason"],
        }}

    summary = _build_intro(len(cards)) + " " + "、".join(c["title"] for c in cards)
    await _run_sync(save_turn, session_id, query, summary)
    await _run_sync(save_last_recommendations, session_id, cards)       # 供后续指代引用
    await _run_sync(save_constraints, session_id, applied_min, applied_max)  # 供后续轮次继承预算


async def _run_sync(func, *args):
    """唯一同步桥接入口：在线程池执行同步函数，避免阻塞 asyncio 事件循环。"""
    return await asyncio.to_thread(func, *args)
