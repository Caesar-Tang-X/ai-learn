"""
读取商品数据库并将数据保存在本地。
"""
import pymysql, json, os

conn = pymysql.connect(
    host="", 
    port=3306, 
    user="", 
    password="",
    database="", 
    charset="utf8mb4"
)
try:
    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute(
            """
                SELECT 
                    id AS spu_id, doctor_id, catalog_id,
                    channel_type, title, brief, intro, price
                FROM 
                    djk_doctor_goods_spu
                WHERE 
                    is_enable=1 
                    AND is_delete=0
                LIMIT 3000
            """
        )
        rows = cur.fetchall()
finally:
    conn.close()

# price 分 → 元
for r in rows:
    r["price_yuan"] = round(r["price"] / 100, 2)
    del r["price"]

os.makedirs("data", exist_ok=True)
with open("data/products.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print(f"导出 {len(rows)} 条")
