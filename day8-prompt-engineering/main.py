"""Day8 完整对比实验。

设计原则：同一 LLM(qwen2.5:3b)、同一问题、同一 context，
唯一变量是 Prompt 策略。差异 100% 归因 Prompt 优化。

实验组：
  A. SimplePrompt    → Day7 基线
  B. CoTPrompt       → 思维链
  C. FewShotPrompt   → 范例引导
  D. JsonPrompt      → 结构化 JSON
  E. RouterPrompt    → 自动选择策略
"""

import json

from adapters.ollama_llm import OllamaLLM
from test_data import TEST_CASES
from prompts.strategies import (
    SimplePrompt, CoTPrompt, FewShotPrompt, JsonPrompt, RouterPrompt,
)


def main():
    llm = OllamaLLM()

    # 策略注册表
    strategies = {
        "A. Simple": SimplePrompt(),
        "B. CoT": CoTPrompt(),
        "C. Few-Shot": FewShotPrompt(),
    }
    router = RouterPrompt()

    for case in TEST_CASES:
        question = case["question"]
        context = case["context"]
        qtype = case.get("type", "")
        keywords = case.get("expect_keywords", [])

        print(f"\n{'#'*60}")
        print(f"问题：{question}")
        print(f"类型：{qtype}  |  期望关键词：{keywords}")
        print(f"{'#'*60}")

        # A/B/C
        for label, strategy in strategies.items():
            answer = strategy.run(llm, context, question)
            hit = [kw for kw in keywords if kw.lower() in answer.lower()]
            print(f"\n── {label} ──")
            print(answer[:250])
            print(f"关键词命中: {hit} ({len(hit)}/{len(keywords)})")

        # D. JSON（仅 multi_step）
        if qtype == "multi_step":
            json_prompt = JsonPrompt()
            result = json_prompt.run(llm, context, question)
            print(f"\n── D. JSON ──")
            if result:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print("解析失败（模型未输出合法 JSON）")

        # E. Router
        router_answer = router.run(llm, context, question, qtype)
        strategy_name = router.get_strategy_name(qtype)
        hit = [kw for kw in keywords if kw.lower() in router_answer.lower()]
        print(f"\n── E. Router（自动选择：{strategy_name}）──")
        print(router_answer[:250])
        print(f"关键词命中: {hit} ({len(hit)}/{len(keywords)})")


if __name__ == "__main__":
    main()
