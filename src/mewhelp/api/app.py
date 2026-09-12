"""FastAPI 服务层 —— 把图接上 HTTP 和 SSE。

这个模块只做三件事：

1. **装配**（lifespan 里）：init_db + 把 `data/knowledge` 灌进检索器。
   用 lifespan 而不是 `@app.on_event("startup")` —— 后者在 FastAPI 里已经废弃，
   而且没法保证"关服时把观测缓冲刷干净"。
2. **路由**：每条路由只负责"解析入参 → 调服务 → 整形返回"，
   业务逻辑都在 `service.py`，这里读起来就是一张路由表。
3. **异常兜底**：任何异常都要变成 `event: error` 或一句降级答复，
   **不允许让 SSE 连接裸崩** —— 连接一断，前端连"出了什么事"都看不到。

关于同步/异步：图是同步的（LangGraph 的 invoke，内部还有几十毫秒到几秒的
模型调用），所以一律用 `run_in_threadpool` 丢到线程池，绝不阻塞事件循环。
在 mock 模式下这点代价看不出来，但一旦换成真实模型，阻塞事件循环会让
**所有**并发请求一起卡住。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from ..config import Settings
from ..domain import FlywheelSource
from ..flywheel.feedback import triage_hint, triage_label
from ..observability.tracing import build_tracer
from ..schemas import FlywheelRecord
from ..storage import FeedbackRow, init_db, session_scope
from ..workflow.graph import GraphDeps, build_deps, build_graph
from .service import (
    EVENT_ERROR,
    ChatService,
    CachedRun,
    ResultCache,
    SessionStore,
    sse_event,
    stream_events,
)

logger = logging.getLogger("mewhelp.api")

#: 演示页所在目录。用 `__file__` 定位而不是 cwd —— 从任意目录启动服务都要能找到它。
STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

#: 启动时灌库的默认目录（settings 里没配或不存在时的退路）。
DEFAULT_KB_DIR = Path("data/knowledge")


# ============================================================
# 请求体
# ============================================================
class ChatRequest(BaseModel):
    """对话入参。

    三个字段都给了默认值：少传一个字段应该是"用默认值继续"，而不是 422。
    演示页、curl、CLI 都受益于这个宽松度，而真正的输入校验在 service 层。
    """

    user_id: str = "u-anonymous"
    session_id: str = "s-default"
    message: str = ""


class FeedbackRequest(BaseModel):
    session_id: str = ""
    request_id: str = ""
    thumbs_down: bool = False
    comment: str | None = None


class ReviewRequest(BaseModel):
    record_id: str
    approved: bool = True
    reviewer: str = "anonymous"
    answer: str | None = None


# ============================================================
# 装配辅助
# ============================================================
def _init_db_safely(settings: Settings) -> bool:
    """建表。失败不阻断启动 —— 演示环境里数据库不可用，也不该让整个服务起不来。"""
    try:
        init_db(settings=settings)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("[api] 数据库初始化失败（会话/反馈将不可用）：%s", exc)
        return False


def _index_kb(deps: GraphDeps) -> int:
    """把知识库灌进检索器，返回 chunk 数。

    优先用 settings 里的绝对路径，取不到再退回相对路径 —— 后者依赖 cwd，
    只在"从项目根启动"时才对。
    """
    retriever = getattr(deps, "retriever", None)
    if retriever is None or not hasattr(retriever, "load_from_dir"):
        return 0
    kb_dir = Path(getattr(deps.settings, "knowledge_dir", "") or DEFAULT_KB_DIR)
    if not kb_dir.is_dir():
        kb_dir = DEFAULT_KB_DIR
    try:
        return int(retriever.load_from_dir(kb_dir) or 0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[api] 知识库加载失败：%s", exc)
        return 0


def _db_backend(settings: Settings) -> str:
    return "sqlite" if str(settings.database_url).startswith("sqlite") else "mysql"


def _retriever_stats(deps: GraphDeps) -> dict[str, Any]:
    fn = getattr(getattr(deps, "retriever", None), "stats", None)
    if not callable(fn):
        return {}
    try:
        return dict(fn())
    except Exception as exc:  # noqa: BLE001 — 自检不能把健康检查搞挂
        return {"error": f"{type(exc).__name__}: {exc}"}


def _flywheel_stats(deps: GraphDeps) -> dict[str, Any]:
    fn = getattr(getattr(deps, "flywheel", None), "stats", None)
    if not callable(fn):
        return {}
    try:
        return dict(fn())
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def _flywheel_item(rec: FlywheelRecord, settings: Settings) -> dict[str, Any]:
    """一条待审记录 → 审核台要看到的全部信息。

    快照摘要（几条证据、最高分多少）是这里最要紧的字段：审核人靠它一眼判断
    "该补文档"还是"该调检索"，而不用去翻原始日志。
    """
    snap = rec.retrieval_snapshot
    hint = triage_hint(rec, settings)
    return {
        "record_id": rec.record_id,
        "session_id": rec.session_id,
        "question": rec.question,
        "resolved_question": rec.resolved_question,
        "intent": rec.intent.value if rec.intent else None,
        "source": rec.source.value if rec.source else None,
        "refusal_reason": rec.refusal_reason.value if rec.refusal_reason else None,
        "ask_count": rec.ask_count,
        "status": rec.status,
        "reviewer": rec.reviewer,
        "answer": rec.answer,
        "created_at": rec.created_at.isoformat(),
        "reviewed_at": rec.reviewed_at.isoformat() if rec.reviewed_at else None,
        "triage": hint,
        "triage_label": triage_label(hint),
        "snapshot": (
            {
                "strategy": snap.strategy,
                "evidence_count": len(snap.evidences),
                "top_score": round(snap.top_score, 4),
                "chunks": [
                    {"chunk_id": e.chunk_id, "title": e.chunk.title, "score": round(e.score, 4)}
                    for e in snap.evidences
                ],
            }
            if snap is not None
            else None
        ),
    }


def _record_thumbs_down(
    deps: GraphDeps, run: CachedRun | None, payload: FeedbackRequest
) -> str | None:
    """点踩 → 落问题池。

    点踩是三个飞轮入口里**信号最强**的一个（用户亲自告诉你答案不行），
    所以它不该只变成 `feedback` 表里的一行计数。带上当时的检索快照落池，
    这条反馈才有资格进审核队列。

    拿不到当时的问题（进程重启、缓存已淘汰）时**不硬造**一条记录：
    没有问题和快照的飞轮条目只会污染审核队列。
    """
    if run is None:
        return None
    result = run.result
    try:
        rec = FlywheelRecord(
            # 这个 id 会被 FlywheelStore 按问题指纹重算，这里只是占位
            record_id=f"{result.request_id}-td",
            session_id=payload.session_id or result.session_id,
            question=result.question,
            resolved_question=result.resolved_question,
            intent=result.intent,
            source=FlywheelSource.THUMBS_DOWN,
            refusal_reason=result.refusal_reason,
            retrieval_snapshot=run.retrieval,
        )
        saved = deps.flywheel.record(rec)
        return saved.record_id
    except Exception as exc:  # noqa: BLE001 — 落池失败不该让反馈接口失败
        logger.warning("[api] 点踩落池失败：%s", exc)
        return None


def _save_feedback(
    deps: GraphDeps, cache: ResultCache, payload: FeedbackRequest
) -> dict[str, Any]:
    feedback_id: int | None = None
    try:
        with session_scope(settings=deps.settings) as s:
            row = FeedbackRow(
                session_id=payload.session_id or "",
                request_id=payload.request_id or "",
                thumbs_down=bool(payload.thumbs_down),
                comment=payload.comment,
            )
            s.add(row)
            s.flush()
            feedback_id = int(row.id or 0)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "flywheel": None}

    flywheel_id = None
    if payload.thumbs_down:
        flywheel_id = _record_thumbs_down(deps, cache.get(payload.request_id), payload)
    return {
        "ok": True,
        "feedback_id": feedback_id,
        "thumbs_down": bool(payload.thumbs_down),
        "flywheel": flywheel_id,
    }


# ============================================================
# 应用
# ============================================================
def create_app(deps: GraphDeps | None = None) -> FastAPI:
    """装配 FastAPI 应用。

    `deps` 为 None 时按默认配置装配一套（全 mock，离线可跑）。
    传入 deps 的口子是为测试和演示留的：换掉 retriever 就能演消融，
    换掉 settings 就能把数据写到临时库、不污染仓库里的 mewhelp.db。
    """
    deps = deps or build_deps()
    graph = build_graph(deps)
    tracer = build_tracer(deps.settings)
    cache = ResultCache()
    service = ChatService(
        graph=graph,
        sessions=SessionStore(deps.settings),
        tracer=tracer,
        cache=cache,
        settings=deps.settings,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 建表 + 灌知识库都放在启动时做一次：前者是后续所有会话/审计的前提，
        # 后者的索引成本（几十毫秒到几秒）不能摊到每个请求上。
        app.state.db_ready = _init_db_safely(deps.settings)
        app.state.kb_chunks = _index_kb(deps)
        logger.info(
            "[api] 启动完成：kb_chunks=%s db_ready=%s fully_mocked=%s",
            app.state.kb_chunks,
            app.state.db_ready,
            deps.settings.is_fully_mocked,
        )
        try:
            yield
        finally:
            tracer.close()

    app = FastAPI(
        title="MewHelp API",
        version="0.1.0",
        description="电商 AI 客服：Workflow + Agent 混合架构，SSE 逐字推送",
        lifespan=lifespan,
    )
    # 演示页可能被从 file:// 或另一个端口打开，放开跨域省掉一类"打不开"的求助。
    # 离线演示用不上鉴权，这里没有额外风险面。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.deps = deps
    app.state.tracer = tracer
    app.state.service = service
    app.state.cache = cache
    app.state.db_ready = False
    app.state.kb_chunks = 0

    # ------------------------------------------------------------
    # 对话
    # ------------------------------------------------------------
    @app.post("/api/chat")
    async def chat(payload: ChatRequest) -> EventSourceResponse:
        """SSE 流式问答：meta → evidence → tool* → token* → done。"""

        async def gen():
            # 图是同步的，丢线程池跑；否则一个请求会把事件循环卡住，
            # 其它请求（包括前端拉健康检查）全部排队。
            try:
                result = await run_in_threadpool(
                    service.ask,
                    user_id=payload.user_id,
                    session_id=payload.session_id,
                    message=payload.message,
                )
            except Exception as exc:  # noqa: BLE001 — 第一道兜底
                logger.exception("[api] /api/chat 执行失败")
                yield sse_event(EVENT_ERROR, {"message": f"{type(exc).__name__}: {exc}"})
                return
            try:
                async for event in stream_events(result):
                    yield event
            except Exception as exc:  # noqa: BLE001 — 推流中途出错也不能裸崩
                logger.exception("[api] SSE 推送中断")
                yield sse_event(EVENT_ERROR, {"message": f"{type(exc).__name__}: {exc}"})

        return EventSourceResponse(gen())

    @app.post("/api/chat/sync")
    async def chat_sync(payload: ChatRequest) -> JSONResponse:
        """同一套逻辑的非流式版本，给脚本/评估/调试用。"""
        try:
            result = await run_in_threadpool(
                service.ask,
                user_id=payload.user_id,
                session_id=payload.session_id,
                message=payload.message,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[api] /api/chat/sync 执行失败")
            return JSONResponse(
                status_code=200,
                content={
                    "error": f"{type(exc).__name__}: {exc}",
                    "answer": "抱歉，服务暂时不可用，帮你转人工确认。",
                    "handoff": True,
                    "path": [],
                },
            )
        return JSONResponse(content=result.model_dump(mode="json"))

    # ------------------------------------------------------------
    # 反馈 / 飞轮
    # ------------------------------------------------------------
    @app.post("/api/feedback")
    async def feedback(payload: FeedbackRequest) -> dict[str, Any]:
        return await run_in_threadpool(_save_feedback, deps, cache, payload)

    @app.get("/api/flywheel/pending")
    async def flywheel_pending(limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
        """待审问题池。按被问频次降序 —— 排队顺序就是运营优先级。"""
        try:
            rows = deps.flywheel.pending(limit=limit)
        except Exception as exc:  # noqa: BLE001
            return {"count": 0, "items": [], "error": f"{type(exc).__name__}: {exc}"}
        items = [_flywheel_item(r, deps.settings) for r in rows]
        return {"count": len(items), "items": items, "stats": _flywheel_stats(deps)}

    @app.post("/api/flywheel/review")
    async def flywheel_review(payload: ReviewRequest) -> dict[str, Any]:
        """人工审核。

        审核通过后**不自动写向量库**：知识库的回灌是批量动作（要重建索引、
        要能整体回滚），把它塞进一个点击接口里会让"审核"和"发布"两件事
        纠缠在一起。这里只把回灌所需的 chunk 准备好并返回它的 id。
        """
        try:
            rec = deps.flywheel.review(
                payload.record_id, payload.approved, payload.reviewer, payload.answer
            )
            chunk = deps.flywheel.to_kb_chunk(rec) if rec.status == "approved" else None
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "record": None}
        return {
            "ok": True,
            "record": rec.model_dump(mode="json"),
            "backfill_chunk_id": chunk.chunk_id if chunk else None,
        }

    # ------------------------------------------------------------
    # 运维 / 可观测
    # ------------------------------------------------------------
    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        s: Settings = deps.settings
        return {
            "status": "ok",
            # 每个组件的模式都摊开写：看到这一行就知道当前是不是纯离线跑，
            # 不用去猜 .env 里配了什么。
            "components": {
                "llm": s.llm_mode,
                "embedding": s.embedding_mode,
                "vector_store": s.vector_store_mode,
                "reranker": s.reranker_mode,
                "database": _db_backend(s),
                "observability": "langfuse" if s.langfuse_enabled else "local",
            },
            "fully_mocked": bool(s.is_fully_mocked),
            "kb_chunks": int(app.state.kb_chunks),
            "db_ready": bool(app.state.db_ready),
            "retriever": _retriever_stats(deps),
            "flywheel": _flywheel_stats(deps),
        }

    @app.get("/api/trace/{request_id}")
    async def trace(request_id: str) -> dict[str, Any]:
        steps = tracer.trace(request_id)
        return {
            "request_id": request_id,
            "count": len(steps),
            "steps": [step.model_dump(mode="json") for step in steps],
        }

    # ------------------------------------------------------------
    # 演示页
    # ------------------------------------------------------------
    @app.get("/")
    async def index() -> Any:
        if not INDEX_HTML.is_file():
            return JSONResponse(status_code=404, content={"error": "演示页缺失", "path": str(INDEX_HTML)})
        return FileResponse(INDEX_HTML, media_type="text/html; charset=utf-8")

    return app


__all__ = ["create_app", "ChatRequest", "FeedbackRequest", "ReviewRequest"]
