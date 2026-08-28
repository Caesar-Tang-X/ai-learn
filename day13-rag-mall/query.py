"""
day13 查询脚本：接收应用端 prompt + filters，
向量检索(含 metadata 硬过滤) → 拼 context → LLM 生成答案。
LLM 调用复用 day12 已验证的 AssistantAgent 封装（autogen 0.7.5）。
"""
import asyncio

from config import get_settings
from core.embeddings import get_embedding_client
from core.vectorstore import ProductVectorStore
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import json
import re


def _build_llm_client():
    """构造指向本地 Ollama 的 OpenAI 兼容客户端（与 day12 一致）。"""
    cfg = get_settings()
    return OpenAIChatCompletionClient(
        model=cfg.llm_model,
        base_url=f"{cfg.ollama_base_url}/v1",
        api_key="ollama",
        timeout=300,
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "structured_output": False,
            "family": "unknown",
            "context_window": 8192,
        },
    )


def _format_context(hits: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        meta = h["metadata"]
        lines.append(
            f"[{i}] spu_id={meta.get('spu_id')} | "
            f"doctor_id={meta.get('doctor_id')} | "
            f"catalog_id={meta.get('catalog_id')} | "
            f"channel_type={meta.get('channel_type')} | "
            f"price_yuan={meta.get('price_yuan')} | "
            f"文本={h['content']}"
        )
    return "\n".join(lines)


def _extract_json_array(text: str) -> list:
    """从 LLM 输出提取商品数组并统一映射成带 key 的 dict 列表。

    兼容两种 LLM 输出：
      - 标准对象数组: [{"spu_id":1,...}, ...]
      - 位置数组:     [[1,2,3,4,5.0,"标题"], ...]
    最终都转成 [{"spu_id":1,...}, ...]。
    兼容 ```json 包裹或前后多余文字。
    """
    # 1) 直接解析
    raw = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            raw = parsed
    except json.JSONDecodeError:
        pass
    # 2) 抠第一个 [ ... ]（处理 markdown 包裹/前后废话）
    if raw is None:
        match = re.search(r"$$.*$$", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    raw = parsed
            except json.JSONDecodeError:
                pass
    # 3) 兜底空数组
    if raw is None:
        return []
    # 4) 把每个元素映射成带 key 的 dict
    _PRODUCT_FIELDS = ["spu_id", "doctor_id", "catalog_id", "channel_type", "price_yuan", "title"]
    result = []
    for item in raw:
        if isinstance(item, dict):
            result.append(item)
        elif isinstance(item, (list, tuple)):
            obj = {}
            for i, field in enumerate(_PRODUCT_FIELDS):
                if i < len(item):
                    obj[field] = item[i]
            result.append(obj)
    return result


async def ask(prompt: str, filters: dict | None = None, top_k: int = 20) -> list:
    # 1. 向量检索 + 硬过滤（阻塞调用包进 to_thread）
    embed_client = get_embedding_client()
    qvec = await asyncio.to_thread(embed_client.embed_query, prompt)
    store = ProductVectorStore()
    hits = await asyncio.to_thread(store.search, qvec, top_k, filters)

    # 2. 拼 context
    context = _format_context(hits)

    # 3. 调用 LLM（用 AssistantAgent，与 day12 一致）
    client = _build_llm_client()
    agent = AssistantAgent(
        name="shopping_assistant",
        model_client=client,
        system_message=(
            "你是严谨的商品导购助手。仅根据下面【商品资料】筛选并输出结果。"
            "必须只输出一个 JSON 数组，不要输出任何解释性文字、不要使用 markdown 代码块。"
            "数组每个元素是一个商品对象，字段必须包含："
            "spu_id(整数)、doctor_id(整数)、catalog_id(整数)、"
            "channel_type(整数)、price_yuan(数字)、title(字符串,即商品文本)。"
            "只输出资料中符合条件(用户问题)的商品；若无符合的，输出空数组 []。"
        ),
    )
    user_text = f"【商品资料】\n{context}\n\n【用户问题】{prompt}\n【输出】请只输出JSON数组："
    result = await agent.run(task=user_text)
    return _extract_json_array(result.messages[-1].content)


if __name__ == "__main__":
    # 示例：模拟应用端传入的 prompt + filters
    demo_prompt = (
        "已知：doctor_id=1, channel_type=0, 需排除 catalog_id=[2]。"
        "用户想找：当归相关商品"
    )
    demo_filters = {
        "doctor_id": 1,
        "channel_type": 0,
        "exclude_catalog_ids": [2],
    }
    print(asyncio.run(ask(demo_prompt, demo_filters)))
