"""
全文召回：用 tsvector 倒排索引做关键词匹配，补充向量召回的漏召回。
    中文查询（无空格长句）无法走 tsquery 拆词时，按常见连接词把 query 切碎，
    取各段中的连续中文片段（≥2 字）作为 LIKE 关键词，保证核心词（如「礼物」）
    能从「元以内的礼物」这类长串中独立出来命中商品标题/正文。
同步实现；service 层用 asyncio.to_thread 包裹。
依赖：psycopg、core.retrieval.filters。
"""
import re

import psycopg
from psycopg.rows import dict_row

from config import get_settings
from core.retrieval.filters import build_filter_clauses

_SETTINGS = get_settings()

# 中文连接/量词片段：用于将长句切碎，使核心检索词独立出来（不含业务语义词）。
_STOP_CN = {
    "的", "以", "内", "元", "款", "个", "一", "这", "那", "和", "与", "及",
    "或", "中", "下", "上", "请", "帮", "想", "要", "有", "看", "买", "送",
    "需要", "适合", "左右", "预算", "推荐", "以内", "一款", "一个",
}


def fulltext_search(
    query: str, top_k: int | None = None, filters: dict | None = None
) -> list[dict]:
    """
    全文召回。
    :param query: 用户查询
    :param top_k: 返回条数，默认 settings.retrieval_top_k
    :param filters: 语义化硬过滤（等值 / include / exclude）
    :return: [{id, content, metadata, score}, ...]，score 降序
    """
    top_k = top_k or _SETTINGS.retrieval_top_k

    # 含中文：按连接词切碎后取各段连续中文（≥2 字）作为 LIKE 关键词
    has_cn = bool(re.search(r"[一-鿿]", query))
    stop_pat = "|".join(re.escape(w) for w in _STOP_CN)
    cn_words = [
        w
        for seg in re.split(stop_pat, query)
        for w in re.findall(r"[一-鿿]{2,}", seg)
    ]
    # 不含中文：走英文 tsquery 倒排匹配
    terms = (
        [t for t in re.split(r"\s+", query.replace("（", " ").replace("）", " ")) if t]
        if not has_cn
        else []
    )

    if not has_cn and not terms:
        return []
    if has_cn and not cn_words:
        return []

    dsn = _SETTINGS.postgres_database_url
    params: list = []

    if terms:
        # 英文/数字查询：tsquery 倒排匹配
        tsquery = " & ".join(terms)
        sql = """
            SELECT id, content, metadata,
                   ts_rank(content_tsv, to_tsquery('simple', %s)) AS score
            FROM products
            WHERE content_tsv @@ to_tsquery('simple', %s)
        """
        params = [tsquery, tsquery]
    else:
        # 中文查询：任一关键词出现在 content 即召回（OR）
        like_clauses = []
        for w in cn_words:
            like_clauses.append("content LIKE %s")
            params.append(f"%{w}%")
        sql = f"""
            SELECT id, content, metadata,
                   1.0 AS score
            FROM products
            WHERE ({' OR '.join(like_clauses)})
        """

    clauses, fparams = build_filter_clauses(filters or {})
    if clauses:
        sql += " AND " + " AND ".join(clauses)
        params.extend(fparams)
    sql += " ORDER BY score DESC LIMIT %s;"
    params.append(top_k)

    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
