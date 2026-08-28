"""
记忆模块门面：对外暴露会话存储与上下文相关函数。
实现见 core.memory.store（组合 window/compressor 的上下文构建也在此完成）。
"""
from core.memory.store import (
    get_context,
    save_turn,
    clear,
    save_last_recommendations,
    load_last_recommendations,
    save_constraints,
    load_constraints,
)

__all__ = [
    "get_context",
    "save_turn",
    "clear",
    "save_last_recommendations",
    "load_last_recommendations",
    "save_constraints",
    "load_constraints",
]
