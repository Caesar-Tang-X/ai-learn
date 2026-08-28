"""
检索编排门面：多路召回（向量 + 全文）-> 去重融合 -> 重排精排 -> 相关性截断。
rerank 未配置或调用失败时，降级为融合分相对阈值截断，保证检索链路不挂。
同步实现；调用方在异步环境中用线程池包裹。
依赖：core.retrieval.vector / fulltext、core.rerank。
"""
from config import get_settings
from core.retrieval.vector import vector_search
from core.retrieval.fulltext import fulltext_search
from core.rerank import get_reranker

_SETTINGS = get_settings()


def _merge_dedup(vector_hits: list[dict], fulltext_hits: list[dict]) -> list[dict]:
    """合并两路召回，按 id 去重，保留较高 score。"""
    merged: dict[int, dict] = {}
    for h in list(vector_hits) + list(fulltext_hits):
        hid = h["id"]
        if hid not in merged or h["score"] > merged[hid]["score"]:
            merged[hid] = h
    return list(merged.values())


def _relative_truncate(candidates: list[dict], top_score) -> list[dict]:
    """
    按相对阈值截断：保留 score >= max(top_score*relative, abs_floor) 的项。
    用于无 rerank 或 rerank 失败时的相关性兜底，避免弱相关项（如「墨镜」下的中药）混入。
    score 可能为 Decimal（pgvector 返回），统一转 float 比较。
    """
    top = float(top_score)
    floor = max(top * _SETTINGS.rerank_relative_threshold, _SETTINGS.rerank_threshold)
    return [c for c in candidates if float(c["score"]) >= floor]


def _fallback(candidates: list[dict], top_n: int) -> list[dict]:
    """无 rerank 时的降级：按融合分相对阈值截断无关项，保留候选池（>=top_n）供下游精选。"""
    if not candidates:
        return []
    truncated = _relative_truncate(candidates, candidates[0]["score"])
    pool_size = max(_SETTINGS.retrieval_final_top_k, top_n)
    if len(truncated) < pool_size:
        seen = {id(t) for t in truncated}
        for c in candidates:
            if id(c) not in seen:
                truncated.append(c)
            if len(truncated) >= pool_size:
                break
    return truncated[:pool_size]


def retrieve(query: str, filters: dict | None = None, top_n: int | None = None,
             recall_top_k: int | None = None) -> list[dict]:
    """
    检索主入口。
    流程：向量召回（按 vector_threshold 过滤） + 全文召回 -> 融合 ->
          重排（按 rerank_threshold 截断无关项，未配置 rerank 时按融合分相对截断）-> 按 top_n 限制条数。
    :param query: 用户查询
    :param filters: metadata 硬过滤（如价格/类目），透传给两路召回
    :param top_n: 最终返回商品数量上限；为 None 时取 settings.rerank_top_n（默认值）
    :param recall_top_k: 每路召回规模上限；为 None 时取 settings.retrieval_top_k。
        当用户显式要求更多数量（如「10款」）时由调用方放大，确保召回池 >= top_n。
    :return: 通过相关性过滤的商品 [{id, content, metadata, score}, ...] 降序，最多 top_n 条
    """
    if top_n is None:
        top_n = _SETTINGS.rerank_top_n
    if recall_top_k is None:
        recall_top_k = _SETTINGS.retrieval_top_k
    # 召回规模至少覆盖最终需求数量，避免「要10款却只有4款」
    recall_top_k = max(recall_top_k, top_n)

    vector_hits = vector_search(query, top_k=recall_top_k, filters=filters)
    fulltext_hits = fulltext_search(query, top_k=recall_top_k, filters=filters)

    # 向量召回按相似度阈值过滤明显无关项（如「墨镜」查询下的中药）
    vector_hits = [h for h in vector_hits if h["score"] >= _SETTINGS.vector_threshold]

    merged = _merge_dedup(vector_hits, fulltext_hits)
    merged.sort(key=lambda x: x["score"], reverse=True)
    # 融合候选池至少覆盖 top_n，供重排充分挑选
    candidates = merged[: max(_SETTINGS.retrieval_final_top_k, top_n)]
    if not candidates:
        return []

    # 最高分低于相关性下限：知识库中无相关商品，直接返回空（避免展示无关品）
    if float(candidates[0]["score"]) < _SETTINGS.min_relevance_score:
        return []

    # rerank_provider 为空/未配置：不使用重排，按融合分相对阈值截断后限制条数
    if not _SETTINGS.rerank_provider:
        return _fallback(candidates, top_n)

    # 启用 rerank：调用重排模型；任意异常降级为融合分相对截断（避免检索链路单点故障）
    try:
        documents = [c["content"] for c in candidates]
        reranker = get_reranker(_SETTINGS.rerank_provider)
        # 重排召回的候选数需 >= 最终需求数量，否则下游 LLM 精排无从挑选满 top_n；
        # fusion 只做「去噪」，不在此处把候选截断到 top_n。
        rerank_k = max(_SETTINGS.rerank_top_n, top_n, len(candidates))
        ranked = reranker.rerank(query, documents, top_n=rerank_k)

        if not ranked:
            return []
        # 按 rerank 结果重排 candidates 并附带 rerank 分数，再按相对阈值截断明显无关项
        top_score = ranked[0][1]
        floor = max(top_score * _SETTINGS.rerank_relative_threshold, _SETTINGS.rerank_threshold)
        result = []
        for idx, score in ranked:
            if idx < len(candidates) and score >= floor:
                item = dict(candidates[idx])
                item["score"] = score
                result.append(item)
        # 候选池至少保留 top_n 供下游 LLM 精排精选；不足时回退原始 candidates 补齐（去噪后仍尽量保量）
        pool_size = max(_SETTINGS.retrieval_final_top_k, top_n)
        if len(result) < pool_size:
            seen = {id(r) for r in result}
            for c in candidates:
                if id(c) not in seen:
                    result.append(c)
                if len(result) >= pool_size:
                    break
        return result[:pool_size]
    except Exception:
        # 网络不通 / 401 / provider 非法 等，降级为融合分相对截断
        return _fallback(candidates, top_n)
