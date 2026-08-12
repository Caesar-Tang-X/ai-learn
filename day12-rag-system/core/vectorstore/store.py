"""
PGVector 向量存储。

职责：管理 documents 表的创建与向量化文本的增、查。
不负责检索策略（混合检索/Rerank 在 retrieval 层）。
"""
import psycopg
from psycopg.rows import dict_row

from config import get_settings
from core.embeddings import get_embedding_client


class VectorStore:
    """
    PGVector 文档存储（同步）。
    """

    def __init__(self) -> None:
        self._dsn = get_settings().database_url
        self._dim = get_embedding_client().dimension

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self._dsn, row_factory=dict_row)

    def init(self) -> None:
        """
        重置初始化：删表重建（清空全部文档）。可重复调用。
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute("DROP TABLE IF EXISTS documents;")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS documents (
                        id        BIGSERIAL PRIMARY KEY,
                        content   TEXT NOT NULL,
                        source    TEXT,
                        embedding vector({self._dim})
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS documents_embedding_idx
                    ON documents USING hnsw (embedding vector_cosine_ops);
                    """
                )
            conn.commit()


    def add(self, texts: list[str], source: str | None = None) -> int:
        """
        批量插入文本及其向量。返回插入条数。
        """
        if not texts:
            return 0
        client = get_embedding_client()
        vectors = client.embed(texts)
        rows = [(t, source, v) for t, v in zip(texts, vectors)]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO documents (content, source, embedding) "
                    "VALUES (%s, %s, %s);",
                    rows,
                )
            conn.commit()
        return len(rows)

    def search(self, query_vector: list[float], top_k: int = 8) -> list[dict]:
        """
        按向量余弦相似度召回 top_k 条最相近文档。
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, content, source,
                           1 - (embedding <=> %s::vector) AS score
                    FROM documents
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (query_vector, query_vector, top_k),
                )
                return cur.fetchall()

    def count(self) -> int:
        """
        返回文档总数。
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) AS n FROM documents;")
                return cur.fetchone()["n"]

