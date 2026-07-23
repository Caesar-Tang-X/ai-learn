"""
author: tang
description: Day3 进阶 —— 把 Day2 的 chunk 文字向量化，并用余弦相似度做语义检索
             流水线：加载 → 清洗 → 分块(Day2) → 向量化(Day3) → 余弦相似度检索
"""

from loaders.document_loader import DocumentLoader
from cleaners.text_cleaner import TextCleaner
from splitters.recursive_splitter import RecursiveTextSplitter

from embeddings.embedding_client import EmbeddingClient
from utils.cosine import cosine_similarity


def build_chunks(file_path: str, chunk_size: int = 300, chunk_overlap: int = 50) -> list[str]:
    """Day2 流水线：加载 → 清洗 → 分块，返回 chunk 列表"""
    raw_text = DocumentLoader().load(file_path)
    clean_text = TextCleaner().clean(raw_text)
    chunks = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split(clean_text)
    print(f"[分块] 共切出 {len(chunks)} 块")
    return chunks


def main():
    # ---------- 1. 用 Day2 的流程产出 chunks ----------
    # day3 目录下需放一个测试文档（可从 day2 的 samples/ 复制一份 README.md）
    file_path = "samples/README.md"
    chunks = build_chunks(file_path, chunk_size=300, chunk_overlap=50)

    # ---------- 2. 用 Day3 的客户端把每个 chunk 变成向量 ----------
    client = EmbeddingClient(model="bge-m3")
    print("开始向量化 chunks ...")
    chunk_vectors = [client.embed(c) for c in chunks]
    print(f"向量化完成，向量维度: {len(chunk_vectors[0])}")

    # ---------- 3. 用余弦相似度衡量"语义接近程度" ----------
    # 场景：用户提一个问题，我们找最相关的 chunk（这就是 RAG 检索的核心）
    query = "如何安装依赖"
    query_vec = client.embed(query)

    # 计算 query 和每个 chunk 的余弦相似度
    scored = [(cosine_similarity(query_vec, cv), i) for i, cv in enumerate(chunk_vectors)]

    # 按相似度从高到低排序，取最相关的
    scored.sort(reverse=True, key=lambda x: x[0])
    print(f"\n===== 查询: {query} =====")
    print("与各个 chunk 的余弦相似度（取前 3 高）:")
    for sim, i in scored[:3]:
        print(f"  相似度 {sim:.4f}  <- Chunk {i+1}: {chunks[i][:60]}...")

    # ---------- 4. 验证：内容连续的 chunk 相似度更高 ----------
    sim_adjacent = cosine_similarity(chunk_vectors[0], chunk_vectors[1])
    print(f"\n相邻两个 chunk 的余弦相似度: {sim_adjacent:.4f}（通常较高，因为来自同一文档相邻处）")


if __name__ == "__main__":
    main()
