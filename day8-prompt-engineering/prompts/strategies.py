"""Prompt 策略模板类。

封装 CoT / Few-Shot / JSON 三种策略为可复用类。
RouterPrompt 根据问题类型自动选择策略。
"""

import json
import re
from typing import Optional, List, Dict

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ══════════════════════════════════════════════
# 基类
# ══════════════════════════════════════════════

class BasePrompt:
    """Prompt 策略基类。"""

    def __init__(self, name: str):
        self.name = name

    def build(self, context: str, question: str) -> str:
        """子类实现：构建 prompt 字符串。"""
        raise NotImplementedError

    def run(self, llm, context: str, question: str) -> str:
        """构建 prompt → 调 LLM → 返回文本。"""
        prompt_text = self.build(context, question)
        return llm.invoke(prompt_text)


# ══════════════════════════════════════════════
# 普通 Prompt（Day7 原版，作为基线）
# ══════════════════════════════════════════════

class SimplePrompt(BasePrompt):
    """普通 Prompt，来自 Day7 main.py 第 26~34 行。"""

    def __init__(self):
        super().__init__("simple")
        self.template = PromptTemplate.from_template(
            """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。

【资料】
{context}

【问题】{question}

【回答】"""
        )

    def build(self, context: str, question: str) -> str:
        return self.template.format(context=context, question=question)


# ══════════════════════════════════════════════
# CoT Prompt（步骤 3）
# ══════════════════════════════════════════════

class CoTPrompt(BasePrompt):
    """CoT 思维链 Prompt——先回答再简要分析。"""

    def __init__(self):
        super().__init__("cot")
        self.template = PromptTemplate.from_template(
            """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。

【资料】
{context}

【问题】{question}

请按以下步骤回答：
1. 【回答】先给出完整答案
2. 【分析】简要说明答案的依据（不超过 80 字）

【回答】
【分析】"""
        )

    def build(self, context: str, question: str) -> str:
        return self.template.format(context=context, question=question)


# ══════════════════════════════════════════════
# Few-Shot Prompt（步骤 4）
# ══════════════════════════════════════════════

class FewShotPrompt(BasePrompt):
    """Few-Shot Prompt——用范例教 LLM 输出格式。"""

    EXAMPLES = [
        {
            "question": "如何启动服务",
            "answer": (
                "步骤1：确保虚拟环境已激活\n"
                "步骤2：在项目根目录执行 uvicorn main:app --reload --host 0.0.0.0 --port 8000\n"
                "步骤3：打开浏览器访问 http://localhost:8000/docs 验证"
            ),
        },
    ]

    def __init__(self):
        super().__init__("few_shot")
        self.template = PromptTemplate.from_template(
            """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。

回答格式要求：按步骤编号列出，每步格式为「步骤X：操作内容（涉及的命令或工具）」

示例——
问题：{example_question}
回答：
{example_answer}

现在回答以下问题：
【资料】
{context}

【问题】{question}

【回答】"""
        )

    def build(self, context: str, question: str) -> str:
        ex = self.EXAMPLES[0]
        return self.template.format(
            example_question=ex["question"],
            example_answer=ex["answer"],
            context=context,
            question=question,
        )


# ══════════════════════════════════════════════
# JSON Prompt（步骤 5）
# ══════════════════════════════════════════════

class JsonPrompt(BasePrompt):
    """JSON 结构化输出——约束 schema + 容错解析 + 重试。"""

    def __init__(self, max_retries: int = 2):
        super().__init__("json")
        self.max_retries = max_retries
        self.template = PromptTemplate.from_template(
            """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。

【资料】
{context}

【问题】{question}

请按以下 JSON 格式回答（仅输出 JSON，不要其他文字）：
{{
    "steps": ["步骤1描述", "步骤2描述", ...]
}}

JSON："""
        )
        self.fix_template = PromptTemplate.from_template(
            """你之前输出了以下内容，但它不是合法的 JSON：

{raw_output}

解析错误：{error}

请修正后重新输出合法的 JSON（仅输出 JSON，不要其他文字）：

JSON："""
        )

    def build(self, context: str, question: str) -> str:
        return self.template.format(context=context, question=question)

    def run(self, llm, context: str, question: str) -> Optional[dict]:
        """重写 run：返回 dict 而非 str。"""
        raw = super().run(llm, context, question)
        result = self._parse(raw)
        if result is not None:
            return result

        # 重试
        for _ in range(self.max_retries):
            try:
                json.loads(raw)
            except json.JSONDecodeError as e:
                fix_text = self.fix_template.format(raw_output=raw, error=str(e))
                raw = llm.invoke(fix_text)
                result = self._parse(raw)
                if result is not None:
                    return result
        return None

    @staticmethod
    def _parse(text: str) -> Optional[dict]:
        """三层容错解析（与 Day7 self_query.py 一致）。"""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None


# ══════════════════════════════════════════════
# Router：根据问题类型自动选择策略
# ══════════════════════════════════════════════

class RouterPrompt:
    """根据问题 type 自动选择 Prompt 策略。

    multi_step → CoT
    extraction → Simple
    """

    def __init__(self):
        self.strategies: Dict[str, BasePrompt] = {
            "multi_step": CoTPrompt(),
            "extraction": SimplePrompt(),
        }

    def run(self, llm, context: str, question: str, qtype: str) -> str:
        strategy = self.strategies.get(qtype)
        if strategy is None:
            # 未知类型，回退到简单 Prompt
            strategy = SimplePrompt()
        return strategy.run(llm, context, question)

    def get_strategy_name(self, qtype: str) -> str:
        strategy = self.strategies.get(qtype)
        return strategy.name if strategy else "simple"
