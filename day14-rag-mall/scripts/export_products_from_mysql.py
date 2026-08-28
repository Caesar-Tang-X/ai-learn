"""
一次性数据准备脚本：从源商品库（MySQL）导出商品到本地 data/products.json，
供 init_products_db.py 导入 PostgreSQL 向量库。

凭据通过环境变量注入（不硬编码）：
  export MYSQL_HOST=... MYSQL_USER=... MYSQL_PASSWORD=... MYSQL_DB=...
"""
import json
import os

import pymysql

MYSQL_HOST = os.getenv("MYSQL_HOST", "")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", ""))
MYSQL_USER = os.getenv("MYSQL_USER", "")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "")

conn = pymysql.connect(
    host=MYSQL_HOST,
    port=MYSQL_PORT,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    database=MYSQL_DB,
    charset="utf8mb4",
)
try:
    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute(
            """
            SELECT
                id AS spu_id,
                doctor_id,
                catalog_id,
                title,
                brief,
                intro,
                thumbnail_img,
                price,
                channel_type,
                is_enable,
                is_delete
            FROM djk_doctor_goods_spu
            WHERE is_enable = 1
              AND is_delete = 0
            LIMIT 3000
            """
        )
        rows = cur.fetchall()
finally:
    conn.close()

os.makedirs("data", exist_ok=True)
with open("data/products.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print(f"导出 {len(rows)} 条")
