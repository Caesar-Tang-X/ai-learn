"""
初始化商品向量库（PGVector）。

建表 products：
- id          主键
- content     商品文本（用于展示/全文检索源）
- metadata    JSONB 结构化字段（doctor_id/catalog_id/price_yuan/channel_type 等，供硬过滤）
- embedding   vector(维度) 语义向量（供向量召回）
- content_tsv tsvector 全文检索列（供全文召回，由 content 生成）

索引：
- embedding 向量索引（HNSW, cosine）
- content_tsv Gin 索引（全文）

维度由 embedding provider 动态决定。
"""
import sys

import psycopg
from psycopg.rows import dict_row

from config import get_settings
from core.embeddings import get_embedding_client

_SETTINGS = get_settings()

def _dim() -> int:
    """从选中的 embedding provider 取维度，避免硬编码。"""
    provider = getattr(_SETTINGS, "embedding_provider", "ollama")
    return get_embedding_client(provider).dimension


def init() -> None:
    dim = _dim()
    print(f"[init_db] 使用 embedding 维度: {dim}")
    dsn = _SETTINGS.postgres_database_url
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            # 1. 启用 vector 扩展
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # 2. 删表重建
            cur.execute("DROP TABLE IF EXISTS products;")

            # 3. 建表
            cur.execute(
                f"""
                CREATE TABLE products (
                    id          BIGSERIAL PRIMARY KEY,
                    content     TEXT NOT NULL,
                    metadata    JSONB NOT NULL DEFAULT '{{}}',
                    embedding   vector({dim}),
                    content_tsv TSVECTOR
                );
                """
            )

            # 4. 向量索引（HNSW cosine）
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS products_embedding_idx
                ON products USING hnsw (embedding vector_cosine_ops);
                """
            )

            # 5. 全文索引（Gin）
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS products_tsv_idx
                ON products USING gin (content_tsv);
                """
            )

            # 6. 触发器：content 写入/更新时自动维护 content_tsv
            cur.execute(
                """
                CREATE OR REPLACE FUNCTION products_tsv_trigger() RETURNS trigger AS $$
                BEGIN
                    NEW.content_tsv := to_tsvector('simple', COALESCE(NEW.content, ''));
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
                """
            )
            cur.execute(
                """
                DROP TRIGGER IF EXISTS products_tsv_update ON products;
                CREATE TRIGGER products_tsv_update
                    BEFORE INSERT OR UPDATE OF content ON products
                    FOR EACH ROW EXECUTE FUNCTION products_tsv_trigger();
                """
            )
        conn.commit()
    print("[init_db] 表 products 初始化完成（含向量+全文索引+自动维护触发器）")


if __name__ == "__main__":
    # 可通过环境变量 EMBEDDING_PROVIDER 选择维度来源
    provider = _SETTINGS.embedding_provider
    print(f"[init_db] embedding provider = {provider}")
    try:
        init()
    except Exception as e:
        print(f"[init_db] 失败: {e}", file=sys.stderr)
        sys.exit(1)
