"""Prompt 策略类统一验证。"""

import json
from adapters.ollama_llm import OllamaLLM
from test_data import TEST_CASES
from prompts.strategies import (
    SimplePrompt, CoTPrompt, FewShotPrompt, JsonPrompt, RouterPrompt,
)


def main():
    llm = OllamaLLM()
    router = RouterPrompt()

    for case in TEST_CASES:
        question = case["question"]
        context = case["context"]
        qtype = case.get("type", "")
        keywords = case.get("expect_keywords", [])

        print(f"\n{'='*60}")
        print(f"问题：{question}（类型：{qtype}）")
        print(f"Router 自动选择策略：{router.get_strategy_name(qtype)}")
        print(f"{'='*60}")

        # Router 自动选择
        answer = router.run(llm, context, question, qtype)
        print(f"\n── Router（自动选择）──")
        print(answer[:300])

        # 关键词命中
        if keywords:
            hit = [kw for kw in keywords if kw.lower() in answer.lower()]
            print(f"\n关键词命中: {hit} ({len(hit)}/{len(keywords)})")

        # JSON 策略单独演示（仅 multi_step 类型）
        if qtype == "multi_step":
            json_prompt = JsonPrompt()
            result = json_prompt.run(llm, context, question)
            if result:
                print(f"\n── JSON 输出 ──")
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(f"\n── JSON 输出 ── 解析失败")


if __name__ == "__main__":
    main()
