"""
商品入库脚本（PGVector + 全文）。

支持三种模式（命令行参数 --mode）：
- rebuild : 清空 products 表后全量写入（依赖 init_products_db.py 已建表）
- append  : 增量插入，不删已有数据（spu_id 重复会报错，慎用）
- upsert  : 按 metadata.spu_id 覆盖——已存在则先删后插，不存在则新增

文本拼接：title + brief + intro + 价格
向量：使用 settings.embedding_provider 指定的 embedding 后端
全文：content_tsv 由数据库触发器自动维护，本脚本不手动处理

用法：
    python -m scripts.ingest_products_into_db --mode rebuild
    python -m scripts.ingest_products_into_db --mode append
    python -m scripts.ingest_products_into_db --mode upsert
"""
import argparse
import json
import sys

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from config import get_settings
from core.embeddings import get_embedding_client

_SETTINGS = get_settings()
_DATA_PATH = "data/products.json"
_BATCH = 50


def _load_products() -> list[dict]:
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_text(p: dict) -> str:
    """拼接商品文本，空值兜底为空串。"""
    title = p.get("title") or ""
    brief = p.get("brief") or ""
    intro = p.get("intro") or ""
    price = p.get("price")                 

    price_text = f"价格{price/100}元" if price is not None else ""
    return f"{title} {brief} {intro} {price_text}".strip()


def _build_metadata(p: dict) -> dict:
    """结构化字段全量存入 metadata，供检索时硬过滤。"""
    return {
        "spu_id": p.get("spu_id"),
        "doctor_id": p.get("doctor_id"),
        "catalog_id": p.get("catalog_id"),
        "title": p.get("title"),
        "brief": p.get("brief"),
        "intro": p.get("intro"),
        "thumbnail_img": p.get("thumbnail_img"),
        "price": p.get("price"),           
        "channel_type": p.get("channel_type"),
        "is_enable": p.get("is_enable"),
        "is_delete": p.get("is_delete"),
    }


def _insert_rows(cur, rows: list[tuple]) -> None:
    cur.executemany(
        "INSERT INTO products (content, metadata, embedding) "
        "VALUES (%s, %s, %s);",
        rows,
    )


def _process_batch(
    cur, batch_products: list[dict], client, *, pre_delete_spu: bool
) -> int:
    """
    单批处理：本批内做 embedding -> 组行 -> (可选删旧) -> 插入 -> 提交。
    每批独立完成，避免数据量大时一次性预计算 embedding 导致长时间阻塞/超时。
    :param pre_delete_spu: upsert 模式需先按 spu_id 删除旧记录；rebuild/append 为 False
    :return: 本批成功写入条数
    """
    texts = [_build_text(p) for p in batch_products]
    metas = [_build_metadata(p) for p in batch_products]
    vectors = client.embed(texts)  # 仅对本批做 embedding，耗时可控
    rows = [(t, Json(m), v) for t, m, v in zip(texts, metas, vectors)]

    if pre_delete_spu:
        spu_ids = [str(m.get("spu_id")) for m in metas]
        cur.execute(
            "DELETE FROM products WHERE metadata->>'spu_id' = ANY(%s);",
            (spu_ids,),
        )
    _insert_rows(cur, rows)
    return len(rows)


def rebuild(products: list[dict], client) -> int:
    """清空后全量写入。"""
    dsn = _SETTINGS.postgres_database_url
    total, failed = 0, 0
    with psycopg.connect(dsn, row_factory=dict_row, connect_timeout=30) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE products;")
            conn.commit()
            for start in range(0, len(products), _BATCH):
                end = start + _BATCH
                try:
                    total += _process_batch(
                        cur, products[start:end], client, pre_delete_spu=False
                    )
                    conn.commit()
                except Exception as e:  # 单批失败不中断整体，跳过并继续
                    conn.rollback()
                    failed += (end - start)
                    print(f"[rebuild] 批次 {start}-{end} 失败，已跳过: {e}")
                print(f"[rebuild] 进度 {min(end, len(products))}/{len(products)}")
    if failed:
        print(f"[rebuild] 警告：{failed} 条写入失败")
    return total


def append(products: list[dict], client) -> int:
    """增量插入（不删历史；spu_id 重复会触发唯一约束错误）。"""
    dsn = _SETTINGS.postgres_database_url
    total, failed = 0, 0
    with psycopg.connect(dsn, row_factory=dict_row, connect_timeout=30) as conn:
        with conn.cursor() as cur:
            for start in range(0, len(products), _BATCH):
                end = start + _BATCH
                try:
                    total += _process_batch(
                        cur, products[start:end], client, pre_delete_spu=False
                    )
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    failed += (end - start)
                    print(f"[append] 批次 {start}-{end} 失败，已跳过: {e}")
                print(f"[append] 进度 {min(end, len(products))}/{len(products)}")
    if failed:
        print(f"[append] 警告：{failed} 条写入失败")
    return total


def upsert(products: list[dict], client) -> int:
    """按 spu_id 覆盖：已存在先删后插，不存在新增。"""
    dsn = _SETTINGS.postgres_database_url
    total, failed = 0, 0
    with psycopg.connect(dsn, row_factory=dict_row, connect_timeout=30) as conn:
        with conn.cursor() as cur:
            for start in range(0, len(products), _BATCH):
                end = start + _BATCH
                try:
                    total += _process_batch(
                        cur, products[start:end], client, pre_delete_spu=True
                    )
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    failed += (end - start)
                    print(f"[upsert] 批次 {start}-{end} 失败，已跳过: {e}")
                print(f"[upsert] 进度 {min(end, len(products))}/{len(products)}")
    if failed:
        print(f"[upsert] 警告：{failed} 条写入失败")
    return total


_MODES = {"rebuild": rebuild, "append": append, "upsert": upsert}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="商品入库")
    parser.add_argument(
        "--mode", choices=list(_MODES.keys()), default="upsert",
        help="rebuild=清空重建, append=增量, upsert=按spu_id覆盖(默认)",
    )
    parser.add_argument(
        "--batch", type=int, default=_BATCH, metavar="N",
        help=f"每批处理条数（默认 {_BATCH}），数据量大/embedding 慢时可调小",
    )
    args = parser.parse_args()
    _BATCH = max(1, args.batch) 

    provider = _SETTINGS.embedding_provider
    print(f"[ingest] embedding provider = {provider}, mode = {args.mode}, batch = {_BATCH}")
    try:
        products = _load_products()
        client = get_embedding_client(provider)
        n = _MODES[args.mode](products, client)
        print(f"[ingest] 完成，写入 {n} 条")
    except FileNotFoundError:
        print(f"[ingest] 找不到数据文件: {_DATA_PATH}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ingest] 失败: {e}", file=sys.stderr)
        sys.exit(1)


