"""
入库脚本：读取 data/products.json，
将每个商品拼成 (title+brief+intro) 文本，带 metadata 写入 PGVector。
"""
import argparse
import json

from core.vectorstore import ProductVectorStore


def build_text(p: dict) -> str:
    """拼接商品文本，空值兜底为空串。价格补充进去，便于语义匹配与 LLM 判断。"""
    title = p.get("title") or ""
    brief = p.get("brief") or ""
    intro = p.get("intro") or ""
    price = p.get("price_yuan")
    price_text = f" 价格：{price}元" if price is not None else ""
    return f"{title} {brief} {intro}{price_text}".strip()


def build_metadata(p: dict) -> dict:
    """结构化字段存入 metadata，供检索时硬过滤。"""
    return {
        "spu_id": p.get("spu_id"),
        "doctor_id": p.get("doctor_id"),
        "catalog_id": p.get("catalog_id"),
        "channel_type": p.get("channel_type"),
        "price_yuan": p.get("price_yuan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="商品向量入库")
    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="full",
        help="full=清空重建; incremental=增量 upsert(按 spu_id 覆盖)",
    )
    args = parser.parse_args()

    with open("data/products.json", "r", encoding="utf-8") as f:
        products = json.load(f)

    texts = [build_text(p) for p in products]
    metadatas = [build_metadata(p) for p in products]

    store = ProductVectorStore()
    if args.mode == "full":
        store.init()  # 删表重建，清空旧数据
        n = store.add(texts, metadatas)
        print(f"[全量入库] 已写入 {n} 条（总计 {store.count()} 条）")
    else:
        n = store.upsert(texts, metadatas)
        print(f"[增量入库] 已 upsert {n} 条（总计 {store.count()} 条）")


if __name__ == "__main__":
    main()
