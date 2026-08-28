"""
需求意图识别与引导。

策略（纯 LLM 判定）：
  1. judge_intent：每次请求都让 LLM 判断用户表述是否「购物意图」且「需求明确到可检索」，
     返回结构化 JSON：{is_shoppable, is_clear, reason}。
  2. build_clarify_prompt：当需求不明或无法命中商品时，构造引导提示，
     强制 LLM 仅围绕购物维度（品类/场景/人群/预算/数量）反问，禁止闲聊、禁止偏离导购目的。

判意结果配合检索命中数共同决定走向：
  - 非购物意图 / 需求不明 / 检索 0 命中  -> 走引导（clarify 事件）
  - 否则                              -> 走正常推荐流程
"""
import json as _json
import re as _re

from core.llm import get_llm
from config import get_settings

_SETTINGS = get_settings()


async def _aggregate(llm, messages: list[dict]) -> str:
    """非流式聚合：在主事件循环中 await LLM 流式输出并拼回完整文本。"""
    chunks = []
    async for c in llm.stream(messages):
        chunks.append(c)
    return "".join(chunks)


_JUDGE_SYSTEM = (
    "你是商城导购系统的「意图裁判」。请判断用户这句话：\n"
    "1) is_shoppable：是否表达「在商城购物的意图」（买/找/推荐/送/咨询某商品）。"
    "纯闲聊、问候、与购物无关的话题均为 false。\n"
    "2) is_clear：需求是否已「明确到可以检索商品」。"
    "缺少关键维度（不知道要什么品类/场景/人群，或只说「推荐点东西」这种空泛表述）则为 false；"
    "已给出品类/场景/人群/具体商品名/预算等任一可检索线索则为 true。\n"
    "只输出一个 JSON 对象，不要其他任何文字、不要加 ``` 标记。\n"
    '格式：{"is_shoppable": true/false, "is_clear": true/false, "reason": "简短说明"}'
)


async def judge_intent(query: str, history: list[dict] | None = None) -> dict:
    """
    纯 LLM 判定用户意图与需求明确性（async，需在事件循环内 await）。
    :param query: 当前用户问题
    :param history: 历史上下文（已压缩+滑动窗口后的 messages），可选
    :return: {is_shoppable: bool, is_clear: bool, reason: str}
    """
    llm = get_llm(_SETTINGS.llm_provider)
    messages = [{"role": "system", "content": _JUDGE_SYSTEM}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": f"用户说：{query}\n请判定意图与需求明确性。"})

    raw = (await _aggregate(llm, messages)).strip()
    raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        obj = _json.loads(raw)
    except Exception:
        # 解析失败：保守判定为「不明确」，交由引导流程
        return {"is_shoppable": False, "is_clear": False, "reason": "意图解析失败，转引导"}
    return {
        "is_shoppable": bool(obj.get("is_shoppable", False)),
        "is_clear": bool(obj.get("is_clear", False)),
        "reason": str(obj.get("reason", "")),
    }


_CLARIFY_SYSTEM = (
    "你是商城导购助手。当前用户的需求还不清楚，或未能匹配到合适商品。\n"
    "请引导用户说出明确的购物需求，严格遵守：\n"
    "1. 只围绕「购物」维度反问：想买什么品类/用于什么场景或人群/预算范围/期望数量/偏好（品牌、功效等）。\n"
    "2. 不得闲聊、不得回答与购物无关的问题、不得偏离导购目的。\n"
    "3. 语言简洁友好，最多提 2-3 个针对性问题；若已知部分信息，基于已知给出可选项（如「您更关注价格还是功效？」）。\n"
    "4. 不要编造商品，不要声称已找到商品。\n"
    "只输出引导文案本身，不要输出 JSON、不要加 ``` 标记。"
)


def build_clarify_prompt(query: str, history: list[dict] | None = None, hint: str = "") -> list[dict]:
    """
    构造引导 LLM 生成澄清文案的 messages。
    :param query: 当前用户问题
    :param history: 会话上下文（用于延续多轮）
    :param hint: 额外提示（如「上一轮未检索到商品」），帮助 LLM 聚焦
    """
    messages = [{"role": "system", "content": _CLARIFY_SYSTEM}]
    if history:
        messages.extend(history)
    user = f"用户说：{query}"
    if hint:
        user += f"\n（系统提示：{hint}）"
    user += "\n请引导用户明确购物需求。"
    messages.append({"role": "user", "content": user})
    return messages
