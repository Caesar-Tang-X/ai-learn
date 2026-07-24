from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from loaders.document_loader import DocumentLoader
from cleaners.text_cleaner import TextCleaner
from adapters.ollama_embeddings import OllamaEmbeddings


def build_retriever(file_path: str, collection_name: str = "rag_docs_lc",
                    persist_dir: str = "./chroma_db", chunk_size: int = 300,
                    chunk_overlap: int = 50, k: int = 3):
    embedding = OllamaEmbeddings(model="bge-m3")

    # 先加载（或创建）持久化 collection，避免每次 from_documents 都重复累积
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=persist_dir,
    )

    # 库为空才构建并写入，否则直接复用已有数据（根治"重复运行累积"）
    if len(vectorstore.get()["ids"]) == 0:
        raw_text = DocumentLoader().load(file_path)
        clean_text = TextCleaner().clean(raw_text)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(clean_text)
        docs = [
            Document(page_content=c, metadata={"source": file_path, "chunk_id": i})
            for i, c in enumerate(chunks)
        ]
        vectorstore.add_documents(docs)
        print(f"[入库] Chroma 已写入 {len(docs)} 条")
    else:
        print(f"[复用] 库已有 {len(vectorstore.get()['ids'])} 条，跳过写入")

    return vectorstore.as_retriever(search_kwargs={"k": k})
