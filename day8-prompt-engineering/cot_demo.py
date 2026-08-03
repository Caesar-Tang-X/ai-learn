"""CoT（Chain of Thought）思维链演示。

对比：同一个问题 + 同一段资料 + 同一个 LLM，
唯一变量是 Prompt 是否包含「逐步分析」指令。
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from adapters.ollama_llm import OllamaLLM
from test_data import TEST_CASES


# ── 普通 Prompt（对照，来自 Day7 main.py 第 26~34 行）──
BASELINE_PROMPT = PromptTemplate.from_template(
    """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。

【资料】
{context}

【问题】{question}

【回答】"""
)

# ── CoT Prompt（新增「分析」步骤）──
COT_PROMPT = PromptTemplate.from_template(
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


def run_comparison():
    llm = OllamaLLM()

    for case in TEST_CASES:
        question = case["question"]
        context = case["context"]

        print(f"\n{'='*60}")
        print(f"问题：{question}")
        print(f"{'='*60}")

        # 普通 Prompt
        chain_baseline = BASELINE_PROMPT | llm | StrOutputParser()
        answer_baseline = chain_baseline.invoke({
            "context": context, "question": question
        })
        print(f"\n── 普通 Prompt ──")
        print(answer_baseline[:300])

        # CoT Prompt
        chain_cot = COT_PROMPT | llm | StrOutputParser()
        answer_cot = chain_cot.invoke({
            "context": context, "question": question
        })
        print(f"\n── CoT Prompt ──")
        print(answer_cot[:500])

        # 检查是否命中期望关键词
        keywords = case.get("expect_keywords", [])
        qtype = case.get("type", "")
        if keywords:
            print(f"\n期望关键词: {keywords}")
            baseline_hit = [kw for kw in keywords if kw.lower() in answer_baseline.lower()]
            cot_hit = [kw for kw in keywords if kw.lower() in answer_cot.lower()]
            print(f"普通 Prompt 命中: {baseline_hit} ({len(baseline_hit)}/{len(keywords)})")
            print(f"CoT Prompt  命中: {cot_hit} ({len(cot_hit)}/{len(keywords)})")
            if qtype == "extraction":
                print("（注：此问题类型为「信息提取」，CoT 不一定优于普通 Prompt）")


if __name__ == "__main__":
    run_comparison()
