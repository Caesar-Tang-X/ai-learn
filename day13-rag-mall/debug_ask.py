from config import get_settings
from core.embeddings import get_embedding_client
from core.vectorstore import ProductVectorStore
from query import ask
import asyncio

embed = get_embedding_client()
store = ProductVectorStore()

prompt = "价格在10000以内的男士手表" + "给出性价比最高的5款商品"
filters = {"doctor_id": 1, "channel_type": 1, "exclude_catalog_ids": [2]}

qvec = embed.embed_query(prompt)
hits = store.search(qvec, top_k=20, filters=filters)

print("命中条数:", len(hits))
for h in hits:
    print("score=", round(h["score"], 3),
          "| spu_id=", h["metadata"]["spu_id"],
          "| content=", h["content"])

answer = asyncio.run(ask(prompt, filters, top_k=20))
print("\n===== LLM 回答 =====")
print(answer)
