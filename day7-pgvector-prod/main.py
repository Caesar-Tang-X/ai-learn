"""Day7 完整对比实验。

设计原则：同一 prompt、同一 LLM(qwen2.5:3b)、同一问题，
唯一变量是召回文档。差异 100% 归因检索策略。

实验组：
  A. PGVector 纯语义        → 验证 PGVector ≈ Day5 Chroma
  B. PGVector MMR            → 验证 PGVector ≈ Day6 MMR
  C. 混合检索 + Rerank       → Day6 完整链路在 PGVector 上的复现
  D. 元数据过滤              → PGVector 独有能力
  E. HyDE 假想文档检索        → 通用策略
  F. 混合+Rerank+chunk_id重排 → 检索负责准 + 排序负责顺
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag_chain import build_vectorstore
from adapters.ollama_llm import OllamaLLM
from retrieval.pg_retriever import pg_similarity_search, pg_mmr_search
from retrieval.reranker import hybrid_search_reranked
from retrieval.metadata_filter import metadata_filter_search
from retrieval.hyde import hyde_search

# 与 Day6 一致的 prompt
PROMPT = PromptTemplate.from_template(
    """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。

【资料】
{context}

【问题】{question}

【回答】"""
)


def answer_with_docs(docs, question: str, llm, label: str = ""):
    """用给定文档拼 context，生成答案并打印。"""
    if not docs:
        print(f"  (无召回文档)")
        return

    context = "\n\n".join(d.page_content for d in docs)
    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    print(f"\n{'='*60}")
    print(f"【{label}】")
    cids = [d.metadata.get("chunk_id", "?") for d in docs]
    print(f"召回 chunk_id: {cids}")
    print(f"召回数量: {len(docs)}")
    print(f"-" * 40)
    print(answer)
    print(f"{'='*60}")


def main():
    file_path = "samples/README.md"
    question = "如何安装依赖"
    llm = OllamaLLM()

    # A. PGVector 纯语义
    docs_a = pg_similarity_search(question, file_path, k=3)
    answer_with_docs(docs_a, question, llm, "A. PGVector 纯语义")

    # B. PGVector MMR
    docs_b = pg_mmr_search(question, file_path, k=3)
    answer_with_docs(docs_b, question, llm, "B. PGVector MMR")

    # C. 混合检索 + Rerank（Day6 完整链路复现）
    results_c = hybrid_search_reranked(
        question, file_path, fetch_k=10, final_k=3, threshold=0.3
    )
    docs_c = [d for d, _ in results_c]
    answer_with_docs(docs_c, question, llm, "C. 混合 + Rerank (threshold=0.3)")

    # D. 元数据过滤（chunk_id 0~5）
    docs_d = metadata_filter_search(
        question,
        file_path,
        filters={
            "$and": [
                {"chunk_id": {"$gte": 0}},
                {"chunk_id": {"$lte": 5}},
            ]
        },
        k=3,
    )
    answer_with_docs(docs_d, question, llm, "D. 语义 + 元数据过滤 (chunk_id 0~5)")

    # E. HyDE
    docs_e = hyde_search(question, file_path, k=3)
    answer_with_docs(docs_e, question, llm, "E. HyDE 假想文档检索")

    # F. 混合检索 + Rerank + 按 chunk_id 重排顺序
    results_f = hybrid_search_reranked(
        question, file_path, fetch_k=10, final_k=3, threshold=0.3
    )
    docs_f_raw = [d for d, _ in results_f]
    # 语义检索只看「内容有多相关」，不看「步骤谁先谁后」——
    # 加一行 sorted 把逻辑顺序还回去
    docs_f = sorted(docs_f_raw, key=lambda d: d.metadata["chunk_id"])
    answer_with_docs(
        docs_f, question, llm,
        "F. 混合 + Rerank + chunk_id 升序（还原步骤顺序）"
    )



if __name__ == "__main__":
    main()
