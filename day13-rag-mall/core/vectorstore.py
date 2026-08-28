"""
day13 PGVector 商品向量存储。
表 products：content(商品文本) + metadata(JSONB 结构化字段) + embedding。
"""
import psycopg
from psycopg.rows import dict_row

from config import get_settings
from core.embeddings import get_embedding_client
from psycopg.types.json import Json


class ProductVectorStore:
    def __init__(self) -> None:
        self._dsn = get_settings().database_url
        self._dim = get_embedding_client().dimension

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self._dsn, row_factory=dict_row)

    def init(self) -> None:
        """删表重建，清空商品向量。"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute("DROP TABLE IF EXISTS products;")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS products (
                        id        BIGSERIAL PRIMARY KEY,
                        content   TEXT NOT NULL,
                        metadata  JSONB NOT NULL DEFAULT '{{}}',
                        embedding vector({self._dim})
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS products_embedding_idx
                    ON products USING hnsw (embedding vector_cosine_ops);
                    """
                )
            conn.commit()

    def add(self, texts: list[str], metadatas: list[dict],
            batch_size: int = 50) -> int:
        """批量插入商品文本+向量+metadata。内部按 batch_size 分批，避免大批量超时。"""
        if not texts:
            return 0
        client = get_embedding_client()
        total = 0
        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            batch_texts = texts[start:end]
            batch_meta = metadatas[start:end]
            vectors = client.embed(batch_texts)
            rows = [(t, Json(m), v) for t, m, v in zip(batch_texts, batch_meta, vectors)]
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO products (content, metadata, embedding) "
                        "VALUES (%s, %s, %s);",
                        rows,
                    )
                conn.commit()
            total += len(rows)
            print(f"[入库进度] {end}/{len(texts)}")
        return total


    def upsert(self, texts: list[str], metadatas: list[dict],
               batch_size: int = 50) -> int:
        """增量插入：按 spu_id 去重，已存在则先删后插，不存在则新增。内部分批。"""
        if not texts:
            return 0
        client = get_embedding_client()
        total = 0
        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            batch_texts = texts[start:end]
            batch_meta = metadatas[start:end]
            vectors = client.embed(batch_texts)
            rows = [(t, Json(m), v) for t, m, v in zip(batch_texts, batch_meta, vectors)]
            spu_ids = [str(m.get("spu_id")) for m in batch_meta]
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM products WHERE metadata->>'spu_id' = ANY(%s);",
                        (spu_ids,),
                    )
                    cur.executemany(
                        "INSERT INTO products (content, metadata, embedding) "
                        "VALUES (%s, %s, %s);",
                        rows,
                    )
                conn.commit()
            total += len(rows)
            print(f"[upsert进度] {end}/{len(texts)}")
        return total


    def search(self, query_vector: list[float], top_k: int = 20,
               filters: dict | None = None) -> list[dict]:
        """
        向量召回 + metadata 硬过滤。
        filters 形如:
          {"doctor_id": 123, "channel_type": 0,
           "exclude_catalog_ids": [5,8], "price_min": 10, "price_max": 300}
        """
        where = []
        params: list = []
        if filters:
            if filters.get("doctor_id") is not None:
                where.append("metadata->>'doctor_id' = %s")
                params.append(str(filters["doctor_id"]))
            if filters.get("channel_type") is not None:
                where.append("metadata->>'channel_type' = %s")
                params.append(str(filters["channel_type"]))
            if filters.get("exclude_catalog_ids"):
                ex = filters["exclude_catalog_ids"]
                placeholders = ", ".join("%s" for _ in ex)
                where.append(f"metadata->>'catalog_id' NOT IN ({placeholders})")
                params.extend(str(x) for x in ex)
            if filters.get("price_min") is not None:
                where.append("(metadata->>'price_yuan')::float >= %s")
                params.append(filters["price_min"])
            if filters.get("price_max") is not None:
                where.append("(metadata->>'price_yuan')::float <= %s")
                params.append(filters["price_max"])

        sql = """
            SELECT id, content, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM products
        """
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY embedding <=> %s::vector LIMIT %s;"
        params = [query_vector] + params + [query_vector, top_k]

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()

    def count(self) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) AS n FROM products;")
                return cur.fetchone()["n"]
