"""步骤六（方案3）：本地 Ollama 原生 tool calling（与 tongyi 版同构，仅换后端）。

验证结论：本地模型「不是不能调工具」。之前 Day9 初期的失败，根因是「高层框架工厂
（create_agent / create_react_agent）未把 tools 绑进请求」，而非模型能力问题。
只要用 llm.bind_tools([...]) 手工绑定 + 标准消息循环，qwen2.5:3b / 7b 都能原生
tool calling（实测各跑 5 次，调工具率与成功率均 100%）。

实测速度：3b ≈ 5.5s/次，7b ≈ 34s/次；简单任务 3b 完全够用，本文件默认即 3b。
"""
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

from core.adapters.ollama_chat import build_chat_model
from tools.kb_tool import search_kb
from tools.api_tool import call_api
from tools.local_api_tool import local_status

TOOLS = {
    "search_kb": search_kb,
    "call_api": call_api,
    "local_status": local_status,
}


def run_agent(question: str, model: str = "qwen2.5:3b") -> str:
    llm = build_chat_model(model)
    llm_with_tools = llm.bind_tools(list(TOOLS.values()))

    messages = [
        SystemMessage(content=(
            "你是一个智能助手，可以使用工具回答用户问题。"
            "需要信息时请调用工具，拿到结果后再综合回答。最后用中文给出明确答案。")),
        HumanMessage(content=question),
    ]

    MAX_TURNS = 5
    for turn in range(MAX_TURNS):
        resp = llm_with_tools.invoke(messages)
        messages.append(resp)

        if resp.tool_calls:
            for tc in resp.tool_calls:
                name = tc["name"]
                args = tc.get("args", {})
                print(f"> 调用工具 {name}({args})")
                try:
                    observation = TOOLS[name].invoke(args)
                except Exception as e:
                    observation = f"（工具出错：{e}）"
                print(f"  -> {str(observation)[:120]}")
                messages.append(ToolMessage(content=str(observation),
                                            tool_call_id=tc["id"]))
        else:
            return resp.content

    return "（达到最大轮次，未得出最终答案）"


if __name__ == "__main__":
    q = "我们项目 Day7 学了什么？另外本地服务现在在线吗？"
    print("\n=== 最终答案 ===")
    answer = run_agent(q)
    try:
        print(answer)
    except UnicodeEncodeError:
        print(answer.encode("gbk", errors="replace").decode("gbk"))
