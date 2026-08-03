"""Few-Shot Prompting 演示。

对比：同一个问题 + 同一段资料 + 同一个 LLM，
唯一变量是 Prompt 中是否包含范例。
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from adapters.ollama_llm import OllamaLLM
from test_data import TEST_CASES


# ── Zero-Shot（无范例，来自 cot_demo.py 的普通 Prompt）──
ZERO_SHOT_PROMPT = PromptTemplate.from_template(
    """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。

【资料】
{context}

【问题】{question}

【回答】"""
)

# ── Few-Shot（带范例）──
FEW_SHOT_PROMPT = PromptTemplate.from_template(
    """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。

回答格式要求：按步骤编号列出，每步格式为「步骤X：操作内容（涉及的命令或工具）」

示例——
问题：如何启动服务
资料：...在项目根目录执行 uvicorn main:app --reload --host 0.0.0.0 --port 8000...
回答：
步骤1：确保虚拟环境已激活
步骤2：在项目根目录执行 uvicorn main:app --reload --host 0.0.0.0 --port 8000
步骤3：打开浏览器访问 http://localhost:8000/docs 验证

现在回答以下问题：
【资料】
{context}

【问题】{question}

【回答】"""
)


def run_comparison():
    llm = OllamaLLM()

    for case in TEST_CASES:
        question = case["question"]
        context = case["context"]
        qtype = case.get("type", "")

        print(f"\n{'='*60}")
        print(f"问题：{question}（类型：{qtype}）")
        print(f"{'='*60}")

        # Zero-Shot
        chain_zero = ZERO_SHOT_PROMPT | llm | StrOutputParser()
        answer_zero = chain_zero.invoke({
            "context": context, "question": question
        })
        print(f"\n── Zero-Shot（无范例）──")
        print(answer_zero[:300])

        # Few-Shot
        chain_few = FEW_SHOT_PROMPT | llm | StrOutputParser()
        answer_few = chain_few.invoke({
            "context": context, "question": question
        })
        print(f"\n── Few-Shot（有范例）──")
        print(answer_few[:500])

        # 关键词命中对比
        keywords = case.get("expect_keywords", [])
        if keywords:
            zero_hit = [kw for kw in keywords if kw.lower() in answer_zero.lower()]
            few_hit = [kw for kw in keywords if kw.lower() in answer_few.lower()]
            print(f"\n期望关键词: {keywords}")
            print(f"Zero-Shot 命中: {zero_hit} ({len(zero_hit)}/{len(keywords)})")
            print(f"Few-Shot  命中: {few_hit} ({len(few_hit)}/{len(keywords)})")


if __name__ == "__main__":
    run_comparison()
