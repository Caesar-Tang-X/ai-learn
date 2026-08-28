"""
FastAPI 入口：C 端 RAG 导购流式接口。
端点：
  GET  /            —— 静态聊天页（C 端用户界面）
  POST /chat        —— 接收 {session_id, query, filters?}，SSE 流式返回回复文本
  GET  /health      —— 健康检查
C 端多用户隔离由 session_id 保证（memory 层按 session 分桶）。
"""
import os
import json
import contextlib

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from service import answer
from core.llm import close_all as close_llm
from core.embeddings import close_all as close_embeddings
from core.rerank import close_all as close_rerank


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动：无额外初始化（各客户端懒加载为单例）
    yield
    # 关闭：释放所有缓存的客户端的连接资源
    await close_llm()
    close_embeddings()
    close_rerank()


app = FastAPI(title="day14 RAG Mall", version="1.0", lifespan=lifespan)

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")


class ChatRequest(BaseModel):
    session_id: str
    query: str
    filters: dict | None = None


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    流式对话端点。
    :param req.session_id: 会话 ID（前端每次会话生成唯一 UUID）
    :param req.query: 用户问题
    :param req.filters: 语义化硬过滤（可选）。支持的键：
        - include_catalog_ids / exclude_catalog_ids : list[int] | int —— 类目白/黑名单
        - price_min / price_max                      : int          —— 价格上下限（单位：分）
        - spu_id / doctor_id / catalog_id / title / brief / intro /
          thumbnail_img / price / channel_type / is_enable / is_delete
                                                       : any          —— 按 metadata 等值匹配
        所有键均可选；未提供或值为 null 的键不参与过滤。
        后端会始终附加 is_enable=1 / is_delete=0 及预算上限（从 query 解析），
        因此调用方通常无需手动传入这些基础条件。
    :return: text/event-stream，逐事件推送 JSON；结束发送 [DONE]
             事件类型：
               - clarify   : 需求不明/非购物意图/0 命中时，下发引导文案（引导用户说出明确需求）
               - intro     : 正常推荐流程开场白
               - item_text : 单个商品（标题/价格/理由/图片/intro）图文合一
    """
    if not req.query or not req.query.strip():
        return JSONResponse(status_code=400, content={"error": "query 不能为空"})

    # 合并默认过滤条件：始终只召回已生效、未删除的商品
    merged_filters: dict = dict(req.filters or {})
    merged_filters["is_enable"] = 1
    merged_filters["is_delete"] = 0

    async def event_stream():
        async for event in answer(req.session_id, req.query.strip(), merged_filters):
            # 每个事件是一行 JSON：{"type":"cards"|"text","data":...}
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


# 挂载静态资源（聊天页）。放最后，避免覆盖 /chat、/health 等 API 路由。
if os.path.isdir(_STATIC_DIR):
    _static_app = StaticFiles(directory=_STATIC_DIR, html=True)

    async def _static_no_cache(scope, receive, send):
        # 包装 send，给静态响应注入 no-cache，避免浏览器缓存旧版页面/JS
        async def _send_no_cache(message):
            if message["type"] == "http.response.start":
                headers = message.get("headers") or []
                headers.append((b"cache-control", b"no-cache"))
                message["headers"] = headers
            await send(message)

        await _static_app(scope, receive, _send_no_cache)

    app.mount("/", _static_no_cache, name="static")
