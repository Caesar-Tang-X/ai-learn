"""完整优化 RAG 链(LCEL) + 与 Day5 基线并排对比（步骤5）。

实验设计原则：同一 prompt、同一 LLM(qwen2.5:3b)，唯一变量是"召回的文档"，
对比才公平——两者答案差异即可 100% 归因到"Day6 检索优化"。

运行：在 day6-hybrid-retrieval/ 根目录执行 `python main.py`
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag_chain import build_vectorstore
from adapters.ollama_llm import OllamaLLM
from retrieval import hybrid_search_reranked

# 与 Day5 手写版口径一致的中文 prompt，保证对比公平
PROMPT = PromptTemplate.from_template(
    """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。

【资料】
{context}

【问题】{question}

【回答】"""
)


def run_with_docs(docs, question: str, llm):
    """用给定文档拼 context，经 LCEL 链 PROMPT | llm | StrOutputParser 生成答案。

    Args:
        docs: List[Document] 召回文档。
        question: 问题串。
        llm: LangChain LLM 实例（qwen2.5:3b）。

    Returns:
        str：生成的答案。
    """
    context = "\n\n".join(d.page_content for d in docs)
    return (PROMPT | llm | StrOutputParser()).invoke({"context": context, "question": question})


def baseline_docs(question: str, file_path: str, k: int = 3):
    """Day5 风格基线：纯语义 similarity 检索 Top-K。

    Args:
        question: 问题串。
        file_path: 源文档路径。
        k: 返回条数。

    Returns:
        List[Document]。
    """
    vs = build_vectorstore(file_path)
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": k}).invoke(question)


def optimized_docs(question: str, file_path: str, threshold: float = 0.3):
    """Day6 优化：混合检索 + Rerank + 阈值过滤弱相关。

    Args:
        question: 问题串。
        file_path: 源文档路径。
        threshold: 重排分数阈值，过滤弱相关。

    Returns:
        List[Document]。
    """
    results = hybrid_search_reranked(question, file_path, fetch_k=10, final_k=3, threshold=threshold)
    return [d for d, _ in results]


def main():
    file_path = "samples/README.md"
    question = "如何安装依赖"
    llm = OllamaLLM()                       # 默认 qwen2.5:3b

    print("########## 基线（Day5 风格：纯语义 similarity k=3）##########")
    b_docs = baseline_docs(question, file_path)
    print("召回 chunk_id:", [d.metadata["chunk_id"] for d in b_docs])
    print(run_with_docs(b_docs, question, llm))

    print("\n\n########## 优化（Day6：混合检索 + Rerank, threshold=0.3）##########")
    o_docs = optimized_docs(question, file_path)
    print("召回 chunk_id:", [d.metadata["chunk_id"] for d in o_docs])
    print(run_with_docs(o_docs, question, llm))


if __name__ == "__main__":
    main()
