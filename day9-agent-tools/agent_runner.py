"""步骤六（方案1）：手写轻量 ReAct Agent，纯文本驱动工具调用。"""
import re
from core.adapters.ollama_chat import build_chat_model
from tools.kb_tool import search_kb
from tools.api_tool import call_api
from tools.local_api_tool import local_status

TOOLS = {
    "search_kb": (search_kb, "查询项目内部知识库（Day1~Day8 学习笔记）。输入：自然语言查询字符串"),
    "call_api": (call_api, "调用外部公开 HTTP 接口。输入：完整 URL 字符串"),
    "local_status": (local_status, "查询本地 FastAPI 服务状态。输入：空字符串即可"),
}

SYSTEM_PROMPT = """你是一个智能助手，可以使用以下工具：
{tool_desc}

当需要获取信息时，请严格按以下格式输出（不要有多余内容，一次只输出一个 Action）：
Thought: 你的思考
Action: 工具名
Action Input: 工具所需的参数

工具执行后你会收到 Observation，然后你再输出下一步（继续 Action 或 Final Answer）。
当你已拿到所有需要的信息后，输出：
Thought: 我已有足够信息
Final Answer: 给用户的完整中文答案

重要规则：
1. 每次回复只能包含一个 Action，严禁在一次回复里写多个 Action。
2. 用户问题涉及多个信息源时，必须逐个调用对应工具，不得凭空猜测未调用工具得到的结果。
3. 只有所有需要的 Observation 都拿到后，才能输出 Final Answer。"""

MAX_STEPS = 6


def _build_tool_desc() -> str:
    return "\n".join(f"- {name}: {desc}" for name, (_, desc) in TOOLS.items())


def run_agent(question: str) -> str:
    llm = build_chat_model("qwen2.5:3b")
    system = SYSTEM_PROMPT.format(tool_desc=_build_tool_desc())
    history = f"用户问题：{question}\n"

    for step in range(1, MAX_STEPS + 1):
        prompt = system + "\n" + history + f"\n请继续（第{step}步）：\n"
        reply = llm.invoke(prompt).content.strip()
        print(f"\n--- 第{step}步 模型输出 ---\n{reply}")

        m_final = re.search(r"Final Answer:\s*(.*)", reply, re.DOTALL)
        if m_final:
            return m_final.group(1).strip()

        m_act = re.search(r"Action:\s*(\w+)", reply)
        # 关键修复：Action Input 在遇到下一个 Thought:/Action:/Final Answer: 时截断
        m_in = re.search(
            r"Action Input:\s*(.*?)(?=\n(Thought|Action|Final Answer):|$)",
            reply, re.DOTALL)
        if m_act:
            name = m_act.group(1).strip()
            arg = m_in.group(1).strip() if m_in else ""
            if name in TOOLS:
                try:
                    observation = TOOLS[name][0].invoke(arg)
                except Exception as e:
                    observation = f"（工具执行出错：{e}）"
                print(f"> 执行工具 {name}({arg!r}) -> {str(observation)[:120]}")
                history += reply + f"\nObservation: {observation}\n"
            else:
                history += reply + f"\nObservation: 未知工具 {name}\n"
        else:
            history += reply + "\nObservation: 请严格按格式只输出一个 Action 或 Final Answer。\n"

    return "（已达到最大步数，未能得出最终答案）"


if __name__ == "__main__":
    q = "我们项目 Day7 学了什么？另外本地服务现在在线吗？"
    answer = run_agent(q)
    print("\n=== 最终答案 ===")
    try:
        print(answer)
    except UnicodeEncodeError:
        print(answer.encode("gbk", errors="replace").decode("gbk"))
