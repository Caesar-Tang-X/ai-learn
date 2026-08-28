"""
会话存储：基于 Redis 的 per-session 历史读写 + 上下文组合 + TTL 自动过期。
同步实现；service 层用 asyncio.to_thread 包裹。
key 设计：
  session:{sid}:history       -> JSON 列表 [{role, content}]（对话历史）
  session:{sid}:last_recs     -> 本轮推荐商品卡片（供指代引用）
  session:{sid}:constraints   -> 本轮生效价格约束（供多轮继承）
"""
import json
import redis

from config import get_settings
from core.memory.window import slide_window
from core.memory.compressor import compress

_SETTINGS = get_settings()
_r = redis.from_url(_SETTINGS.redis_url, decode_responses=True)


def _key(session_id: str) -> str:
    return f"session:{session_id}:history"


def load_history(session_id: str) -> list[dict]:
    """读取会话历史；不存在返回空列表。"""
    raw = _r.get(_key(session_id))
    if not raw:
        return []
    return json.loads(raw)


def save_history(session_id: str, history: list[dict]) -> None:
    """写入会话历史并刷新 TTL。"""
    _r.set(_key(session_id), json.dumps(history, ensure_ascii=False), ex=_SETTINGS.session_ttl_seconds)


def append_turn(session_id: str, user_msg: str, assistant_msg: str) -> None:
    """追加一轮对话（user + assistant）并刷新 TTL。"""
    history = load_history(session_id)
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})
    save_history(session_id, history)


def reset_session(session_id: str) -> None:
    """清空会话（新会话）。"""
    _r.delete(_key(session_id))
    _r.delete(f"session:{session_id}:last_recs")


def save_last_recommendations(session_id: str, cards: list[dict]) -> None:
    """保存本轮推荐的商品卡片（精简字段），供后续指代引用（「第一个」「刚才那个」）。"""
    payload = [
        {
            "id": c.get("id"),
            "title": c.get("title"),
            "price_yuan": c.get("price_yuan"),
            "image": c.get("image"),
            "intro": c.get("intro") or "",
        }
        for c in (cards or [])
    ]
    _r.set(
        f"session:{session_id}:last_recs",
        json.dumps(payload, ensure_ascii=False),
        ex=_SETTINGS.session_ttl_seconds,
    )


def load_last_recommendations(session_id: str) -> list[dict]:
    """读取上一轮推荐的商品卡片；不存在或解析失败返回空列表。"""
    raw = _r.get(f"session:{session_id}:last_recs")
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        return []


def save_constraints(session_id: str, price_min: int | None, price_max: int | None) -> None:
    """保存本轮生效的价格约束（分），供后续轮次继承（用户未改预算时延续）。"""
    payload = {"price_min": price_min, "price_max": price_max}
    _r.set(
        f"session:{session_id}:constraints",
        json.dumps(payload, ensure_ascii=False),
        ex=_SETTINGS.session_ttl_seconds,
    )


def load_constraints(session_id: str) -> dict:
    """读取上一轮生效的价格约束；不存在或解析失败返回空 dict。"""
    raw = _r.get(f"session:{session_id}:constraints")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def get_context(session_id: str) -> list[dict]:
    """
    取当前会话可用上下文（已压缩 + 滑动窗口截取）。
    顺序保证摘要在前、窗口在后，喂给 LLM 作为对话历史。
    """
    history = load_history(session_id)
    if not history:
        return []
    return slide_window(compress(history))


def save_turn(session_id: str, user_msg: str, assistant_msg: str) -> None:
    """持久化一轮对话（user + assistant）。"""
    append_turn(session_id, user_msg, assistant_msg)


def clear(session_id: str) -> None:
    """重置会话（清空历史与推荐/约束缓存）。"""
    reset_session(session_id)
