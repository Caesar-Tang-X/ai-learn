"""HyDE（Hypothetical Document Embedding）假想文档检索。

核心洞察：LLM 生成的「假想答案」的向量比用户「短问题」的向量更接近知识库文档。
原因：假想答案与知识库文档都是「叙述性长文本」，在嵌入空间中天然更近。

流程：
  用户问题 → LLM 生成假想答案 → embed → similarity_search → 返回真实文档

限制：qwen2.5:3b 是小模型，生成的假想答案可能较短或不够准确。
"""

from typing import List, Optional
from langchain_core.documents import Document

from rag_chain import build_vectorstore
from adapters.ollama_llm import OllamaLLM


HYDE_PROMPT = """你是一个技术助手。请根据以下问题，写一段假设性的回答。
要求：
1. 用技术文档的口吻，包含具体命令、步骤或术语
2. 50-100 字
3. 不要说"我不知道"或"资料未提及"——请尽力推测

问题：{question}

假设回答："""


class HyDERetriever:
    """HyDE 检索器。"""

    def __init__(self, file_path: str = "samples/README.md"):
        self.file_path = file_path
        self.llm = OllamaLLM()

    def generate_hypothetical(self, question: str) -> str:
        """让 LLM 根据问题生成假想答案。"""
        prompt = HYDE_PROMPT.format(question=question)
        return self.llm.invoke(prompt)

    def search(
        self, question: str, k: int = 3
    ) -> List[tuple]:
        """HyDE 检索：假想答案嵌入 → 搜索。

        Returns:
            List[tuple]: (Document, hypothetical_text)
                        返回的仍是知识库的真实文档，附带假想答案供参考。
        """
        # 1. 生成假想答案
        hypothetical = self.generate_hypothetical(question)
        print(f"[假想答案] {hypothetical.strip()[:100]}...")

        # 2. 用假想答案嵌入去检索
        vs = build_vectorstore(self.file_path)
        docs = vs.similarity_search(hypothetical, k=k)

        return [(doc, hypothetical) for doc in docs]


def hyde_search(
    question: str,
    file_path: str,
    k: int = 3,
) -> List[Document]:
    """便捷函数：HyDE 检索，只返回 Document。"""
    retriever = HyDERetriever(file_path)
    results = retriever.search(question, k=k)
    return [doc for doc, _ in results]


# ========== 单测 ==========
if __name__ == "__main__":
    retriever = HyDERetriever("samples/README.md")
    Q = "如何安装依赖"
    print(f"===== HyDE 检索：{Q} =====\n")
    results = retriever.search(Q, k=3)
    for i, (doc, hypo) in enumerate(results, 1):
        print(
            f"{i} | chunk_id {doc.metadata['chunk_id']} "
            f"| {doc.page_content[:60]}..."
        )

    # 对比：普通语义检索
    print(f"\n===== 对比：普通语义检索 =====")
    vs = build_vectorstore("samples/README.md")
    baseline = vs.similarity_search(Q, k=3)
    for i, doc in enumerate(baseline, 1):
        print(
            f"{i} | chunk_id {doc.metadata['chunk_id']} "
            f"| {doc.page_content[:60]}..."
        )
