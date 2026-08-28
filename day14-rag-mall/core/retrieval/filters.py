"""
filters 解析：将语义化 filters 转为 SQL WHERE 子句 + 参数。
所有用户传入的过滤键均为可选，未提供或值为 None 的键不参与过滤。
"""
from typing import Any


# 基础硬过滤：始终生效，过滤未上架与异常价格的脏数据。
# is_enable=1（已上架）；price 为「分」整数且 > 0（排除 -1 / 0 等脏价）。
BASE_HARD_CLAUSES: list[str] = [
    "metadata->>'is_enable' = '1'",
    "(metadata->>'price' IS NULL OR (metadata->>'price')::numeric > 0)",
]


def _field_of(key: str) -> str:
    """从语义键提取真实 metadata 字段名。"""
    if key.startswith("exclude_catalog_ids"):
        return "catalog_id"
    if key.startswith("include_catalog_ids"):
        return "catalog_id"
    if key.startswith("price_min") or key.startswith("price_max"):
        return "price"
    return key


def build_filter_clauses(filters: dict[str, Any]) -> tuple[list[str], list[str]]:
    """
    将语义化 filters 转为 (SQL 子句列表, 参数列表)。

    键语义约定：
    - exclude_catalog_ids  →  metadata->>'catalog_id' NOT IN (...)
    - include_catalog_ids  →  metadata->>'catalog_id' IN (...)
    - price_min / price_max→  metadata->>'price' 数值比较（单位：分）
    - 其他普通键            →  metadata->>'key' = %s（等值）
    metadata 为 jsonb；普通键取出为 text，参数统一转 str 比较，
    price 字段存「分」，需用 ::numeric 做数值范围比较。
    返回结果始终以 BASE_HARD_CLAUSES 开头，保证基础过滤不受用户 filters 影响。
    """
    clauses: list[str] = []
    params: list[str] = []
    for key, value in filters.items():
        if value is None:
            continue
        field = _field_of(key)
        col = f"metadata->>'{field}'"
        if key.startswith("exclude_catalog_ids"):
            items = [str(v) for v in (value if isinstance(value, list) else [value])]
            placeholders = ",".join(["%s"] * len(items))
            clauses.append(f"{col} NOT IN ({placeholders})")
            params.extend(items)
        elif key.startswith("include_catalog_ids"):
            items = [str(v) for v in (value if isinstance(value, list) else [value])]
            placeholders = ",".join(["%s"] * len(items))
            clauses.append(f"{col} IN ({placeholders})")
            params.extend(items)
        elif key.startswith("price_min"):
            # 下限：价格未知(NULL)的商品不满足「≥X」，必须 IS NOT NULL
            clauses.append(f"({col} IS NOT NULL AND ({col})::numeric >= %s)")
            params.append(float(value))
        elif key.startswith("price_max"):
            # 上限：价格未知(NULL)的商品不应被上限拦截，保持 IS NULL 放行
            clauses.append(f"({col} IS NULL OR ({col})::numeric <= %s)")
            params.append(float(value))
        else:
            clauses.append(f"{col} = %s")
            params.append(str(value))
    return BASE_HARD_CLAUSES + clauses, params
