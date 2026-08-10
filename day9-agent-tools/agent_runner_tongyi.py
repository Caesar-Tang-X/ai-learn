"""步骤六（方案2）：通义 qwen3.7-plus 原生 tool calling（手工 bind_tools + 自循环）。
与方案1（本地Ollama文本ReAct）共用同一套 @tool 定义，体现"工具写一次、换后端/换调度方式复用"。
"""
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

from core.adapters.tongyi_chat import build_tongyi_chat
from tools.kb_tool import search_kb
from tools.api_tool import call_api
from tools.local_api_tool import local_status

TOOLS = {
    "search_kb": search_kb,
    "call_api": call_api,
    "local_status": local_status,
}


def run_agent(question: str) -> str:
    llm = build_tongyi_chat("qwen3.7-plus")
    tools = list(TOOLS.values())

    # 关键：手工 bind_tools，确保工具定义进入请求（高层 create_agent 在本环境不生效）
    llm_with_tools = llm.bind_tools(tools)

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

        # 若模型返回工具调用，则逐个执行并回填 ToolMessage
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
            # 没有工具调用，视为最终答案
            return resp.content

    return "（达到最大轮次，未得出最终答案）"


if __name__ == "__main__":
    q = "我们项目 Day7 学了什么？另外本地服务现在在线吗？"
    print("\n=== 最终答案 ===")
    # Windows 控制台为 GBK，模型答案可能含 emoji 等非 GBK 字符导致 UnicodeEncodeError，
    # 用 errors="replace" 兜底打印，不影响 Agent 逻辑本身。
    answer = run_agent(q)
    try:
        print(answer)
    except UnicodeEncodeError:
        print(answer.encode("gbk", errors="replace").decode("gbk"))
