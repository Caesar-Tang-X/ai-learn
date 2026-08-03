"""结构化 JSON 输出演示。

对比：同一个问题 + 同一段资料 + 同一个 LLM，
对比自由文本 vs JSON 约束输出的差异。
"""

import json
import re
from typing import Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from adapters.ollama_llm import OllamaLLM
from test_data import TEST_CASES


# ── 自由文本 Prompt（对照）──
FREE_TEXT_PROMPT = PromptTemplate.from_template(
    """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。

【资料】
{context}

【问题】{question}

【回答】"""
)

# ── JSON Prompt（约束 schema）──
JSON_PROMPT = PromptTemplate.from_template(
    """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。

【资料】
{context}

【问题】{question}

请按以下 JSON 格式回答（仅输出 JSON，不要其他文字）：
{{
    "steps": ["步骤1描述", "步骤2描述", ...],
    "tools": ["用到的工具或命令", ...],
    "summary": "一句话总结"
}}

JSON："""
)

# ── 修正 Prompt（解析失败时用）──
FIX_PROMPT = PromptTemplate.from_template(
    """你之前输出了以下内容，但它不是合法的 JSON：

{raw_output}

解析错误：{error}

请修正后重新输出合法的 JSON（仅输出 JSON，不要其他文字）：

JSON："""
)


def parse_json(text: str) -> Optional[dict]:
    """三层容错解析 JSON（与 Day7 self_query.py 第 64~83 行逻辑一致）。"""
    # 第 1 层：直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 第 2 层：提取 ```json ... ``` 块
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # 第 3 层：提取第一个 {...}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def ask_json(llm, context: str, question: str, max_retries: int = 2) -> Optional[dict]:
    """让 LLM 输出 JSON，解析失败时重试。

    Args:
        llm: LLM 实例。
        context: 资料文本。
        question: 问题。
        max_retries: 最大重试次数。

    Returns:
        dict 或 None（全部重试失败）。
    """
    raw = (JSON_PROMPT | llm | StrOutputParser()).invoke({
        "context": context, "question": question,
    })

    result = parse_json(raw)
    if result is not None:
        return result

    # 重试：把错误信息反馈给 LLM
    for attempt in range(max_retries):
        try:
            json.loads(raw)  # 触发具体异常信息
        except json.JSONDecodeError as e:
            fix_prompt_text = FIX_PROMPT.format(raw_output=raw, error=str(e))
            raw = llm.invoke(fix_prompt_text)
            result = parse_json(raw)
            if result is not None:
                return result

    return None


def run_comparison():
    llm = OllamaLLM()

    for case in TEST_CASES:
        question = case["question"]
        context = case["context"]

        print(f"\n{'='*60}")
        print(f"问题：{question}")
        print(f"{'='*60}")

        # 自由文本
        chain_free = FREE_TEXT_PROMPT | llm | StrOutputParser()
        answer_free = chain_free.invoke({
            "context": context, "question": question
        })
        print(f"\n── 自由文本 ──")
        print(answer_free[:200])

        # JSON
        result_json = ask_json(llm, context, question)
        print(f"\n── JSON 输出 ──")
        if result_json:
            print(json.dumps(result_json, ensure_ascii=False, indent=2))
            # 验证 schema 字段
            for field in ["steps", "tools", "summary"]:
                print(f"  {field}: {'✓ 存在' if field in result_json else '✗ 缺失'}")
        else:
            print("（解析失败，所有重试均未产出合法 JSON）")


if __name__ == "__main__":
    run_comparison()
