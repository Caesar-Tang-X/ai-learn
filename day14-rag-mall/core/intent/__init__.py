"""
意图识别模块门面：对外暴露 judge_intent / build_clarify_prompt。
"""
from core.intent.clarify import judge_intent, build_clarify_prompt

__all__ = ["judge_intent", "build_clarify_prompt"]
