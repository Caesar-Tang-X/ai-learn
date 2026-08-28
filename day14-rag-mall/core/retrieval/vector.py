"""
向量召回：用 embedding 将查询转向量，从 PGVector 按余弦距离取 top_k。
同步实现；service 层用 asyncio.to_thread 包裹。
依赖：core.embeddings（取查询向量）、psycopg（查库）、core.retrieval.filters。
"""
import psycopg
from psycopg.rows import dict_row

from config import get_settings
from core.embeddings import get_embedding_client
from core.retrieval.filters import build_filter_clauses

_SETTINGS = get_settings()


def vector_search(
    query: str, top_k: int | None = None, filters: dict | None = None
) -> list[dict]:
    """
    向量召回。
    :param query: 用户查询
    :param top_k: 返回条数，默认 settings.retrieval_top_k
    :param filters: 语义化硬过滤（可选，透传给 filters.build_filter_clauses）
    :return: [{id, content, metadata, score}, ...] score 为余弦相似度(0~1)，降序
    """
    top_k = top_k or _SETTINGS.retrieval_top_k
    provider = _SETTINGS.embedding_provider
    vec = get_embedding_client(provider).embed_query(query)
    # PG vector 文本字面量格式："[v1,v2,...,vN]"
    vec_literal = "[" + ",".join(str(x) for x in vec) + "]"

    sql = """
        SELECT id, content, metadata,
               1 - (embedding <=> %s::vector) AS score
        FROM products
    """
    params: list = [vec_literal]
    clauses, fparams = build_filter_clauses(filters or {})
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
        params.extend(fparams)
    sql += " ORDER BY embedding <=> %s::vector LIMIT %s;"
    params.append(vec_literal)
    params.append(top_k)

    dsn = _SETTINGS.postgres_database_url
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
