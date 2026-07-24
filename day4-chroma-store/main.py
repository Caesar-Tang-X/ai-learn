"""
author: tang
description: Day4 入口 —— Day2 分块 → 存入 Chroma → 检索 → 简易问答 Demo（增 + 查 + 生成）
"""

import os
import requests
from dotenv import load_dotenv

from pipelines.loaders.document_loader import DocumentLoader
from pipelines.cleaners.text_cleaner import TextCleaner
from pipelines.splitters.recursive_splitter import RecursiveTextSplitter

from store.chroma_store import ChromaStore

load_dotenv()


def build_chunks(file_path: str, chunk_size: int = 300, chunk_overlap: int = 50) -> list[str]:
    """复用 Day2 流水线：加载 → 清洗 → 分块，返回 chunk 列表"""
    raw_text = DocumentLoader().load(file_path)
    clean_text = TextCleaner().clean(raw_text)
    chunks = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split(clean_text)
    print(f"[分块] 共切出 {len(chunks)} 块")
    return chunks

def call_ollama_generate(prompt: str, model: str = "qwen2.5:3b") -> str:
    """轻量封装 Day1 的 /api/generate：把 prompt 发给 Ollama，返回模型回复"""
    url = f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}  # stream=False 拿完整回复
    try:
        resp = requests.post(url, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()["response"]
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Ollama 服务未启动，请先执行 ollama serve")
    except Exception as e:
        raise RuntimeError(f"生成失败: {str(e)}")

def build_qa_prompt(question: str, contexts: list[str]) -> str:
    """把检索到的资料拼成上下文，构造 RAG 提示词"""
    context_text = "\n\n".join(f"【资料 {i+1}】\n{c}" for i, c in enumerate(contexts))
    return (
        "你是一个严谨的助手。请仅根据下面提供的【资料】回答问题。"
        "如果资料中没有相关信息，请明确回答“资料中未提及”。\n\n"
        f"【资料】\n{context_text}\n\n"
        f"【问题】{question}\n\n"
        "【回答】"
    )

def answer_question(store: ChromaStore, question: str, n_results: int = 3):
    """RAG 问答：检索 → 拼上下文 → 调 Ollama 生成 → 标注来源"""
    result = store.query(question, n_results=n_results)
    contexts = result["documents"][0]
    metas = result["metadatas"][0]

    prompt = build_qa_prompt(question, contexts)
    answer = call_ollama_generate(prompt)

    print(f"\n===== 问答 Demo =====")
    print(f"问题: {question}")
    print(f"回答: {answer}")
    print(f"\n引用来源 (Top {n_results}):")
    for i, meta in enumerate(metas):
        print(f"  [{i+1}] {meta['source']} | chunk_id: {meta['chunk_id']}")


def main():
    # 1. 用 Day2 流程产出 chunks
    file_path = "samples/README.md"
    chunks = build_chunks(file_path, chunk_size=300, chunk_overlap=50)

    # 2. 给每条 chunk 造元数据（解决 Day2/Day3 遗留的"无 source/chunk_id"问题）
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": file_path, "chunk_id": i, "length": len(c)}
        for i, c in enumerate(chunks)
    ]

    # 3. 入库（增）：ChromaStore 内部会用 Day3 的 EmbeddingClient 把 chunks 变向量
    store = ChromaStore(collection_name="rag_docs", persist_dir="./chroma_db")
    store.upsert(chunks, metadatas=metadatas, ids=ids)

    # 4. 检索验证（查）：问一个和文档相关的问题
    question = "如何安装依赖"
    print(f"\n===== 提问: {question} =====")
    result = store.query(question, n_results=3)
    # result 是字典，含 ids / documents / metadatas / distances
    for i, (doc, meta, dist) in enumerate(zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    )):
        print(f"\n--- Top {i+1} (距离 {dist:.4f}, 相似度 {1-dist:.4f}) ---")
        print(f"来源: {meta['source']} | chunk_id: {meta['chunk_id']}")
        print(f"内容: {doc[:120]}...")

    # 5. 问答 Demo（RAG 闭环）
    answer_question(store, question, n_results=3)

if __name__ == "__main__":
    main()
