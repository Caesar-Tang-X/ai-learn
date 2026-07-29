"""自查询检索——LLM 从问题中自动提取结构化过滤条件。

核心思想：用户说「前几章讲了什么」→ LLM 翻译成
  filter={"chunk_id": {"$lte": 4}} + query="讲了什么"

然后：PGVector.similarity_search(query, filter=filter)

限制：qwen2.5:3b 是 3B 小模型，结构化提取不稳定——这是本地小模型的天然局限。
目标不是完美运行，而是理解 Self-Query 的设计模式。
"""

from typing import List, Optional
import json
import re

from langchain_core.documents import Document

from rag_chain import build_vectorstore
from adapters.ollama_llm import OllamaLLM


# 元数据字段描述——LLM 用来理解可过滤的字段及其含义
METADATA_FIELDS = [
    {"name": "chunk_id", "type": "integer",
     "description": "文档分段编号，从 0 开始递增，对应文档的前后顺序"},
    {"name": "source", "type": "string",
     "description": "文档来源路径，如 'samples/README.md'"},
]


def _build_filter_prompt(question: str) -> str:
    """构建 prompt，引导 LLM 从问题中提取过滤条件。"""
    fields_desc = "\n".join(
        f"- {f['name']} ({f['type']}): {f['description']}"
        for f in METADATA_FIELDS
    )
    return f"""你是一个查询分析器。根据用户问题，提取结构化的过滤条件。

可用的元数据字段：
{fields_desc}

支持的运算符：$eq（等于）、$gte（大于等于）、$lte（小于等于）、$and（且）

规则：
1. 从问题中提取过滤意图，翻译成 filter 字典
2. 同时提取纯搜索查询词（去掉过滤条件的剩余部分）
3. 返回 JSON：{{"query": "搜索词", "filter": filter字典 或 null}}

示例：
- 问题："前 5 个步骤" → {{"query": "步骤内容", "filter": {{"chunk_id": {{"$lte": 4}}}}}}
- 问题："如何安装依赖" → {{"query": "如何安装依赖", "filter": null}}
- 问题："README 的配置部分" → {{"query": "配置", "filter": {{"source": "samples/README.md"}}}}

用户问题：{question}

仅返回 JSON，不要其他文字。"""


def _parse_llm_output(text: str) -> dict:
    """从 LLM 输出中提取 JSON。

    qwen2.5:3b 小模型可能输出多余的 Markdown 包裹或解释，需要容错解析。
    """
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 尝试提取 ```json ... ``` 块
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # 尝试提取第一个 {...}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"query": text.strip(), "filter": None}


def self_query_search(
    question: str,
    file_path: str,
    k: int = 3,
    llm=None,
) -> List[Document]:
    """自查询检索：LLM 提取 filter → PGVector 带 filter 检索。

    Args:
        question: 用户问题（自然语言，如「前几个步骤讲了什么」）。
        file_path: 源文档。
        k: 返回条数。
        llm: LLM 实例（默认 OllamaLLM）。

    Returns:
        List[Document]。
    """
    if llm is None:
        llm = OllamaLLM()

    # 1. LLM 提取结构化查询
    prompt = _build_filter_prompt(question)
    raw_output = llm.invoke(prompt)
    print(f"[LLM 输出] {raw_output.strip()}")

    parsed = _parse_llm_output(raw_output)
    search_query = parsed.get("query", question)
    filters = parsed.get("filter") or parsed.get("filters")

    # 2. 用提取的参数做检索
    vs = build_vectorstore(file_path)
    kwargs = {"k": k}
    if filters and isinstance(filters, dict) and len(filters) > 0:
        kwargs["filter"] = filters
        print(f"[应用 filter] {filters}")
    else:
        print("[无 filter] 纯语义检索")

    return vs.similarity_search(search_query, **kwargs)


class SelfQueryApp:
    """自查询演示器——对比有/无 filter 的检索差异。"""

    def __init__(self, file_path: str = "samples/README.md"):
        self.file_path = file_path
        self.llm = OllamaLLM()

    def compare(self, question: str, k: int = 3):
        """并排对比：纯语义 vs 自查询。"""
        print(f"\n{'='*60}")
        print(f"问题：{question}")
        print(f"{'='*60}")

        # 无 filter
        vs = build_vectorstore(self.file_path)
        baseline = vs.similarity_search(question, k=k)
        print(f"\n--- 无 filter（纯语义）---")
        for i, doc in enumerate(baseline, 1):
            print(f"  {i} | chunk_id {doc.metadata['chunk_id']}")

        # 自查询
        print(f"\n--- 自查询（LLM 提取 filter）---")
        results = self_query_search(question, self.file_path, k=k, llm=self.llm)
        if not results:
            print("  (无结果)")
        for i, doc in enumerate(results, 1):
            print(
                f"  {i} | chunk_id {doc.metadata['chunk_id']} "
                f"| {doc.page_content[:60]}..."
            )


# ========== 单测 ==========
if __name__ == "__main__":
    app = SelfQueryApp()

    # 测试用例：不同复杂度的过滤意图
    for q in [
        "前几个步骤讲了什么",
        "如何安装依赖",
    ]:
        app.compare(q, k=3)
