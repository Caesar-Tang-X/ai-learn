"""
author: tang
description: Day5 步骤4 —— 手写最简 RAG 链，剖开 retriever | prompt | llm 的 LCEL 写法
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from rag_chain import build_retriever
from adapters.ollama_llm import OllamaLLM


def handwritten_rag(question: str, retriever, llm):
    """手写 RAG：召回 → 拼 context → 格式化 prompt → 调 LLM，每步打印，看清数据流"""
    # 1) 检索：retriever.invoke 返回相关 Document 列表（新版 LangChain 统一用 invoke）
    docs = retriever.invoke(question)
    print(f"\n===== [步骤1 检索] 召回 {len(docs)} 条 =====")
    for i, d in enumerate(docs):
        print(f"--- 相关文档 {i+1} (来源 {d.metadata['source']} | chunk_id {d.metadata['chunk_id']}) ---")
        print(d.page_content[:100], "...\n")
    # 2) 拼上下文：把召回的 chunk 拼成一段 context
    context = "\n\n".join(d.page_content for d in docs)
    # 3) 构造 Prompt：用 LangChain 的 PromptTemplate 格式化
    prompt = PromptTemplate.from_template(
        """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。
        【资料】
        {context}
        【问题】{question}
        【回答】"""
    )
    final_prompt = prompt.format(context=context, question=question)
    print("===== [步骤3 拼好的 Prompt]（前 600 字）=====")
    print(final_prompt[:600], "...\n")
    # 4) 生成：把拼好的 prompt 丢给 LLM（同样用 invoke）
    answer = llm.invoke(final_prompt)
    print("===== [步骤4 生成答案] =====")
    print(answer)
    return answer


def format_docs(docs):
    """把召回的 Document 列表拼成一段 context（对应手写版第2步'拼 context'）"""
    return "\n\n".join(d.page_content for d in docs)

def lcel_rag(question: str, retriever, llm):
    prompt = PromptTemplate.from_template(
        """你是一个严谨的助手。仅根据下面【资料】回答问题，资料中没有相关信息就明确说"资料中未提及"。
        【资料】
        {context}
        【问题】{question}
        【回答】"""
    )
    # LCEL 链：retriever 召回 → format_docs 拼 context → prompt 格式化 → llm 生成 → 解析字符串
    # 这正是 RetrievalQA(chain_type="stuff") 的现代等价实现，且每一步可见
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = rag_chain.invoke(question)
    # 单独再调一次 retriever，打印命中来源，便于和手写版 [步骤1] 核对
    docs = retriever.invoke(question)
    print("\n===== [LCEL 命中来源] =====")
    for d in docs:
        print(f"  {d.metadata['source']} | chunk_id {d.metadata['chunk_id']} | {d.page_content[:60]}...")
    print("===== [LCEL RAG 答案] =====")
    print(answer)


def main():
    file_path = "samples/README.md"
    retriever = build_retriever(file_path, k=3)
    llm = OllamaLLM()   # 默认 model=qwen2.5:latest，改成你实际 pull 的对话模型名
    question = "如何安装依赖"
    print("\n\n########## 下面是 RAG 链手写版 ##########\n")
    handwritten_rag(question, retriever, llm)
    print("\n\n########## 下面是 RetrievalQA 官方版 ##########\n")
    lcel_rag(question, retriever, llm)


if __name__ == "__main__":
    main()
