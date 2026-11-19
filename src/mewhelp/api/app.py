"""FastAPI 服务层 —— 把图接上 HTTP 和 SSE。

这个模块只做三件事：

1. **装配**（lifespan 里）：init_db + 把 `data/knowledge` 灌进检索器。
   用 lifespan 而不是 `@app.on_event("startup")` —— 后者在 FastAPI 里已经废弃，
   而且没法保证"关服时把观测缓冲刷干净"。
2. **路由**：每条路由只负责"解析入参 → 调服务 → 整形返回"，
   业务逻辑都在 `service.py`，这里读起来就是一张路由表。
3. **异常兜底**：任何异常都要变成 `event: error` 或一句降级答复，
   **不允许让 SSE 连接裸崩** —— 连接一断，前端连"出了什么事"都看不到。

图仍在线程中同步执行；最终生成经请求级异步模型连接实时推送，断连时取消。
"""

from __future__ import annotations

import logging
import secrets
from contextlib import aclosing, asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from ..config import Settings
from ..domain import FlywheelSource
from ..memory.extract import KIND_MANUAL, MemoryCandidate
from ..memory.store import MemoryStore
from ..observability.tracing import build_tracer
from ..schemas import FlywheelRecord
from ..storage import FeedbackRow, ensure_utc, init_db, session_scope
from ..tools.business import list_orders
from ..workflow.graph import GraphDeps, build_deps, build_graph
from .auth import Account, AccountError, AccountStore, SESSION_COOKIE, SESSION_TTL
from .service import (
    EVENT_ERROR,
    ChatService,
    CachedRun,
    ResultCache,
    SessionOwnershipError,
    SessionStore,
    memory_to_public,
    sse_event,
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

    四个字段都给了默认值：少传一个字段应该是"用默认值继续"，而不是 422。
    演示页、curl、CLI 都受益于这个宽松度，而真正的输入校验在 service 层。
    """

    session_id: str = Field(
        default_factory=lambda: f"s_{secrets.token_hex(12)}", max_length=64
    )
    message: str = ""
    #: 引用提问：用户在界面上明确指向的那一句原话。
    #: 它进的是**当轮上下文**，不落库 —— 引用的消息本身已经在会话历史里了，
    #: 再存一遍会让它在下一次提问时出现两次。
    quote: str = ""


class FeedbackRequest(BaseModel):
    session_id: str = ""
    request_id: str = ""
    thumbs_down: bool = False
    comment: str | None = None


class ReviewRequest(BaseModel):
    """飞轮人工审核 —— 已从 HTTP 关闭，留在契约里仅供 CLI/内部脚本复用。"""
    record_id: str
    approved: bool = True
    reviewer: str = "anonymous"
    answer: str | None = None


class MemoryRequest(BaseModel):
    """用户在记忆面板里自己写一条备忘。"""

    text: str = ""
    #: 默认 manual —— 从面板写进来的都是"我自己的备忘"，和系统自动抽的分开记
    kind: str = KIND_MANUAL


class MemoryPatchRequest(BaseModel):
    pinned: bool = True


class SessionPatchRequest(BaseModel):
    title: str = ""


class AuthRequest(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(min_length=1, max_length=128)


#: 时间序列化。真正的时区归一化发生在存储层（`storage.ensure_utc`）——
#: 这里只是把契约模型里的 datetime 转成字符串。保留这一层而不是直接
#: `model_dump(mode="json")`，是因为这些路由返回的是普通字典，
#: 没有 Pydantic 帮忙做那一步。
def _iso_utc(value: datetime | None) -> str:
    stamped = ensure_utc(value)
    return stamped.isoformat() if stamped else ""


def _public_account(account: Account) -> dict[str, str]:
    return {"id": account.user_id, "email": account.email}


def _graph_leaf_label(chunk: Any) -> str:
    """图谱叶子的标签：取**这一条**的内容摘要，而不是它所在小节的标题。

    切分后的 `chunk.title` 就是最近的标题，同一个章节下的几条规则会得到
    完全一样的标签（实测「生鲜类」下面三条都叫「生鲜类」）—— 树看起来
    像标题被复制了几遍，用户点哪一条都分不出来。正文第一句才是这一条
    真正在讲什么；表格没有"第一句"，退回列头，至少说明这张表覆盖哪些字段。
    """
    from .service import _user_snippet

    if chunk.is_table and chunk.table:
        head = [c.strip() for c in chunk.table[0] if c.strip()]
        if head:
            # 表格的列头往往一模一样（三张退款表都叫「商品状态、申请时间窗、
            # 退款比例」），只取前三列的话三张叶子标签完全相同，用户还是
            # 分不出点哪一张。最后一列才是各表自己的差异（是否需要寄回 /
            # 检测要求 / 备注），所以中间用省略号跳过、把尾列带上。
            if len(head) > 3:
                return "速查表：" + "、".join(head[:2]) + "…" + head[-1]
            return "速查表：" + "、".join(head)

    snippet = _user_snippet(chunk, limit=80)
    # 在第一个句号/分号处收住：标签是给人一眼扫的，不是给人读完的。
    # 但只在句号落在合理位置时才收 —— 一句 80 字的规则截在 28 字处
    # 也比截在逗号处更像一句完整的话。
    for sep in ("。", "；"):
        cut = snippet.find(sep)
        if 0 < cut <= 28:
            return snippet[:cut]
    return snippet[:28] + ("…" if len(snippet) > 28 else "")


def _fold_sections(
    doc_id: str, tree: dict[str, Any], path: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    """把面包屑折叠出来的 dict 树递归转成接口约定的 list 树。

    必须递归：前端按 `Array.isArray` 判断"有没有子章节"，中间某一层
    还是 dict 的话，那一层以下的所有章节都会被静默丢掉 —— 数据现在是
    两层，看起来没坏，但三层的知识文档一加进来就会整枝消失。

    顺序用 dict 的**插入顺序**，不排序：这棵树是按 chunk_id 升序遍历
    chunk 时逐条建出来的，而 chunk_id 的序号就是原文顺序，所以插入顺序
    恰好就是文档里的章节顺序。以前这里写的是 `sorted(tree)` —— 按标题
    的 Unicode 码点排，结果是「退款总则」被推到「退款到账时效」后面、
    每篇文档的章节都和原文对不上。同一份知识库两次请求仍然返回同一棵树
    （chunk 输入顺序本身是确定的），稳定性不受影响。
    """
    out: list[dict[str, Any]] = []
    for key, value in tree.items():
        here = (*path, key)
        out.append(
            {
                "id": f"{doc_id}#{'/'.join(here)}",
                "title": key,
                "sections": _fold_sections(doc_id, value["sections"], here),
                "leafs": value["leafs"],
            }
        )
    return out


def _graph_from_chunks(chunks: list[Any]) -> list[dict[str, Any]]:
    """把切分时留下的面包屑折叠成一棵「文档 → 章节 → 小节」的树。

    这是 `retrieval/chunking.py` 那条工程主张的免费红利：切分时严格按
    标题层级、条件句永不断开，于是每个 chunk 都带着完整的祖先链。
    图谱不需要任何新的解析步骤 —— 反过来，如果切分是定长滑窗，
    这里就没有面包屑可折，图谱也就无从谈起。

    折叠时做三件收纳：
      * 跳级标题会产生空层（breadcrumb 里的 ""），跳过 —— 它只是切分
        时补的占位，不是真实存在的章节；
      * 小节下挂着该小节的原文片段，作为点开时的预览（和「依据」卡片
        同一份口径：`_user_snippet` 在 service.py）；
      * doc_id / seq 从 chunk_id 里拆出来排序，保证同一份知识库
        两次请求返回同一棵树。
    """
    from .service import _user_snippet

    docs: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        doc = docs.setdefault(
            chunk.doc_id,
            # 文档节点自己也带 leafs：H1 底下没有小节、直接是正文的片段
            # 会挂在文档这一层 —— 没有这个口子，这类片段就从图上消失了
            {"id": chunk.doc_id, "title": chunk.title, "sections": {}, "leafs": []},
        )
        # 文档层标题取文档名而非最后一个面包屑：面包屑的最后一段是"当前
        # 小节"，文档节点的标题应该一直是它自己（chunking 把 H1 当根）
        if not doc["sections"] and chunk.breadcrumb:
            doc["title"] = chunk.breadcrumb[0] or doc["title"]

        node = doc
        # 面包屑[1:]：第 0 段是文档根，已经消费掉了
        for name in chunk.breadcrumb[1:]:
            if not name:
                continue  # 跳级标题补出来的空层
            node = node["sections"].setdefault(name, {"sections": {}, "leafs": []})
        node["leafs"].append(
            {
                "id": chunk.chunk_id,
                "label": _graph_leaf_label(chunk),
                "text": _user_snippet(chunk),
            }
        )

    out: list[dict[str, Any]] = []
    for doc_id, doc in docs.items():
        out.append(
            {
                "id": doc_id,
                "title": doc["title"],
                "sections": _fold_sections(doc_id, doc["sections"]),
                "leafs": doc["leafs"],
            }
        )
    return out


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
    deps: GraphDeps, cache: ResultCache, payload: FeedbackRequest, user_id: str
) -> dict[str, Any]:
    feedback_id: int | None = None
    try:
        with session_scope(settings=deps.settings) as s:
            row = FeedbackRow(
                user_id=user_id,
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
        run = cache.get(payload.request_id)
        if run and run.result.user_id == user_id and run.result.session_id == payload.session_id:
            flywheel_id = _record_thumbs_down(deps, run, payload)
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
    deps.enforce_order_ownership = True
    graph = build_graph(deps)
    tracer = build_tracer(deps.settings)
    cache = ResultCache()
    memory_store = MemoryStore(deps.settings)
    service = ChatService(
        graph=graph,
        sessions=SessionStore(deps.settings),
        tracer=tracer,
        cache=cache,
        settings=deps.settings,
        # 记忆接在这里而不是在节点内部 new 一个：agent 的每一步都可能被单独
        # 测试，注入进去的依赖才是可替换的（测试里塞个内存实现就能跑）。
        memory=memory_store,
        llm=deps.llm,
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
        # 前端是给**普通用户**看的。这几样开发者通道全都关掉：
        #   docs_url      —— /docs 的 Swagger UI
        #   redoc_url     —— /redoc 的 ReDoc
        #   openapi_url   —— /openapi.json 的 schema（关了上面两个还能直接拿到）
        # 开发者排查另有 CLI / logs / 数据库，不需要给面向用户的界面留后门。
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    account_store = AccountStore(deps.settings)

    def require_account(request: Request) -> Account:
        if not request.app.state.db_ready:
            raise HTTPException(status_code=503, detail="服务暂时不可用")
        try:
            account = account_store.from_token(request.cookies.get(SESSION_COOKIE))
        except Exception as exc:  # noqa: BLE001
            logger.exception("[api] 登录状态读取失败")
            raise HTTPException(status_code=503, detail="服务暂时不可用") from exc
        if account is None:
            raise HTTPException(status_code=401, detail="请先登录")
        return account

    app.state.deps = deps
    app.state.tracer = tracer
    app.state.service = service
    app.state.cache = cache
    app.state.memory = memory_store
    app.state.accounts = account_store
    app.state.db_ready = False
    app.state.kb_chunks = 0

    def set_session_cookie(response: Response, token: str, request: Request) -> None:
        response.set_cookie(
            SESSION_COOKIE,
            token,
            max_age=int(SESSION_TTL.total_seconds()),
            httponly=True,
            secure=request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https",
            samesite="lax",
            path="/",
        )

    def account_error(exc: AccountError) -> HTTPException:
        if exc.code == "EMAIL_EXISTS":
            return HTTPException(status_code=409, detail="该邮箱已注册")
        if exc.code == "INVALID_CREDENTIALS":
            return HTTPException(status_code=401, detail="邮箱或密码不正确")
        if exc.code == "INVALID_EMAIL":
            return HTTPException(status_code=422, detail="请输入有效邮箱地址")
        if exc.code == "PASSWORD_TOO_SHORT":
            return HTTPException(status_code=422, detail="密码至少需要 10 个字符")
        if exc.code == "PASSWORD_TOO_LONG":
            return HTTPException(status_code=422, detail="密码不能超过 128 个字符")
        return HTTPException(status_code=400, detail="无法完成此操作")

    @app.get("/api/auth/me")
    async def current_account(request: Request) -> dict[str, Any]:
        if not request.app.state.db_ready:
            raise HTTPException(status_code=503, detail="服务暂时不可用")
        try:
            account = await run_in_threadpool(
                account_store.from_token, request.cookies.get(SESSION_COOKIE)
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[api] 登录状态读取失败")
            raise HTTPException(status_code=503, detail="服务暂时不可用") from exc
        return {"authenticated": account is not None, "user": _public_account(account) if account else None}

    @app.post("/api/auth/register")
    async def register(payload: AuthRequest, request: Request, response: Response) -> dict[str, Any]:
        if not app.state.db_ready:
            raise HTTPException(status_code=503, detail="服务暂时不可用")
        try:
            account, token = await run_in_threadpool(
                account_store.register, payload.email, payload.password
            )
        except AccountError as exc:
            raise account_error(exc) from exc
        set_session_cookie(response, token, request)
        return {"user": _public_account(account)}

    @app.post("/api/auth/login")
    async def login(payload: AuthRequest, request: Request, response: Response) -> dict[str, Any]:
        if not app.state.db_ready:
            raise HTTPException(status_code=503, detail="服务暂时不可用")
        try:
            account, token = await run_in_threadpool(
                account_store.login, payload.email, payload.password
            )
        except AccountError as exc:
            raise account_error(exc) from exc
        set_session_cookie(response, token, request)
        return {"user": _public_account(account)}

    @app.post("/api/auth/logout")
    async def logout(request: Request, response: Response) -> dict[str, bool]:
        if not app.state.db_ready:
            raise HTTPException(status_code=503, detail="服务暂时不可用")
        try:
            await run_in_threadpool(
                account_store.logout, request.cookies.get(SESSION_COOKIE)
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[api] 退出登录失败")
            raise HTTPException(status_code=503, detail="暂时无法退出，请稍后重试") from exc
        response.delete_cookie(SESSION_COOKIE, path="/", samesite="lax")
        return {"ok": True}

    # ------------------------------------------------------------
    # 对话
    # ------------------------------------------------------------
    @app.post("/api/chat")
    async def chat(
        payload: ChatRequest, account: Account = Depends(require_account)
    ) -> EventSourceResponse:
        """SSE 流式问答：meta → evidence → tool* → token* → done。"""

        async def gen():
            try:
                async with aclosing(service.stream(
                    user_id=account.user_id,
                    session_id=payload.session_id,
                    message=payload.message,
                    quote=payload.quote,
                )) as events:
                    async for event in events:
                        yield event
            except SessionOwnershipError:
                yield sse_event(EVENT_ERROR, {"message": "这次对话不可用，请新建对话后重试。"})
                return
            except Exception as exc:  # noqa: BLE001 — 第一道兜底
                logger.exception("[api] /api/chat 执行失败")
                yield sse_event(EVENT_ERROR, {"message": "服务暂时不可用，请稍后再试。"})
                return
        return EventSourceResponse(gen())

    # ------------------------------------------------------------
    # 反馈
    # ------------------------------------------------------------
    @app.post("/api/feedback")
    async def feedback(
        payload: FeedbackRequest, account: Account = Depends(require_account)
    ) -> dict[str, Any]:
        """用户点赞/点踩。

        只回「收没收到」。点踩之后内部发生了什么（落问题池、带检索快照、
        分诊）是**运营侧**的事 —— 那些字段原本就站在这里往外发，
        等于把一个审核台接口开给了普通用户。审核台走 `python -m mewhelp.cli flywheel`。
        """
        if service.sessions.get_session(payload.session_id, user_id=account.user_id) is None:
            return {"ok": False}
        saved = await run_in_threadpool(_save_feedback, deps, cache, payload, account.user_id)
        # 收敛成用户视角的应答：详情、flywheel id、快照一律不对外
        return {"ok": saved.get("ok", False)}
    # ------------------------------------------------------------
    @app.get("/api/sessions")
    async def sessions(
        account: Account = Depends(require_account),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, Any]:
        """历史会话列表，最近更新的在前。

        列表里刻意没有摘要和路径 —— 这一栏的作用是"让用户认出是哪次对话"，
        多给的信息只会挤掉真正有用的那行标题。
        """
        rows = service.sessions.list_sessions(account.user_id, limit=limit)
        return {
            "count": len(rows),
            "sessions": [
                {
                    "session_id": r.session_id,
                    "title": r.title,
                    "turns": r.turns,
                    "preview": r.preview,
                    "updated_at": _iso_utc(r.updated_at),
                }
                for r in rows
            ],
        }

    @app.get("/api/sessions/{session_id}")
    async def session_detail(
        session_id: str, account: Account = Depends(require_account)
    ) -> dict[str, Any]:
        """恢复一次历史对话。

        返回的消息**已经按用户视角过滤过**（见 `to_session_messages`）：
        意图、节点路径、出口这些存在库里但不该回到界面上。过滤写在服务层
        而不是这里，是为了让 CLI 复用同一份口径。
        """
        detail = service.sessions.get_session(session_id, user_id=account.user_id)
        if detail is None:
            return {"ok": False, "error": "NOT_FOUND", "session_id": session_id}
        return {
            "ok": True,
            "session_id": detail.session_id,
            "title": detail.title,
            "updated_at": _iso_utc(detail.updated_at),
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "handoff": m.handoff,
                    "request_id": m.request_id,
                }
                for m in detail.messages
            ],
        }

    @app.patch("/api/sessions/{session_id}")
    async def session_rename(
        session_id: str,
        payload: SessionPatchRequest,
        account: Account = Depends(require_account),
    ) -> dict[str, Any]:
        """改名。传空标题表示"恢复成自动生成的标题"（下一次追加时会重算）。"""
        ok = service.sessions.rename(
            session_id, payload.title, user_id=account.user_id
        )
        return {"ok": ok, "session_id": session_id, "title": payload.title}

    @app.delete("/api/sessions/{session_id}")
    async def session_delete(
        session_id: str, account: Account = Depends(require_account)
    ) -> dict[str, Any]:
        """删掉一次会话。

        注意它**不动长期记忆**：用户清掉的是聊天记录，不是"你认识我"。
        这两件事被混在一起的那天，用户会发现自己写下的备忘也跟着没了。
        """
        ok = service.sessions.delete(session_id, user_id=account.user_id)
        return {"ok": ok, "session_id": session_id}

    # ------------------------------------------------------------
    # 长期记忆
    # ------------------------------------------------------------
    @app.get("/api/memories")
    async def memories(account: Account = Depends(require_account)) -> dict[str, Any]:
        """这个用户的全部记忆（钉住的排前面）。"""
        items = memory_store.list(account.user_id)
        return {"count": len(items), "items": [memory_to_public(i) for i in items]}

    @app.post("/api/memories")
    async def add_memory(
        payload: MemoryRequest, account: Account = Depends(require_account)
    ) -> dict[str, Any]:
        """用户自己写一条备忘。

        面板里写的默认**钉住**：用户亲手敲进去的东西，不该被"记忆条数上限"
        的自动清理顺手淘汰掉 —— 那是他明确表达过"这条重要"的信号。
        """
        text = (payload.text or "").strip()
        if not text:
            return {"ok": False, "error": "EMPTY", "item": None}
        item = memory_store.remember(
            account.user_id,
            MemoryCandidate(kind=payload.kind or KIND_MANUAL, key="", text=text),
            source="manual",
            pinned=True,
        )
        return {"ok": item is not None, "item": memory_to_public(item) if item else None}

    @app.patch("/api/memories/{memory_id}")
    async def patch_memory(
        memory_id: int,
        payload: MemoryPatchRequest,
        account: Account = Depends(require_account),
    ) -> dict[str, Any]:
        ok = memory_store.set_pinned(account.user_id, memory_id, payload.pinned)
        return {"ok": ok, "id": memory_id, "pinned": payload.pinned}

    @app.delete("/api/memories/{memory_id}")
    async def delete_memory(
        memory_id: int, account: Account = Depends(require_account)
    ) -> dict[str, Any]:
        ok = memory_store.delete(account.user_id, memory_id)
        return {"ok": ok, "id": memory_id}

    # ------------------------------------------------------------
    # 订单快捷选择
    # ------------------------------------------------------------
    @app.get("/api/orders")
    async def orders(account: Account = Depends(require_account)) -> dict[str, Any]:
        """当前用户的订单摘要 —— 给界面的"订单快捷选择"用。

        订单号是用户最不可能记住的东西，让他手打 ORD-1007 是把系统的记忆
        成本转嫁给用户。这里让他点一下就行。

        用户身份来自登录态。没有接入真实订单来源时，返回空列表，避免把共享
        样例数据误当成当前用户的订单。
        """
        try:
            rows = list_orders(account.user_id)
        except Exception as exc:  # noqa: BLE001 — 夹具坏了不该让界面打不开
            logger.warning("[api] /api/orders 查询失败: %s", exc)
            return {"count": 0, "orders": []}
        return {"count": len(rows), "orders": rows}

    # ------------------------------------------------------------
    # 运维 / 可观测
    # ------------------------------------------------------------
    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        """存活探针。

        只回「在不在」。组件模式、模式开关、检索引擎、飞轮状态这些内部信息
        一律不下发 —— 它们是部署和排查用的话，就该待在对方自己的监控系统里，
        而不是暴露给任何能访问这个域名的前端。
        """
        return {"status": "ok"}

    # ------------------------------------------------------------
    # 知识图谱
    # ------------------------------------------------------------
    @app.get("/api/knowledge/graph")
    async def knowledge_graph(account: Account = Depends(require_account)) -> dict[str, Any]:
        """知识库的知识节点图谱。

        检索层的 `_chunks` 里有每一个片段的**面包屑**（按标题层级切分时
        天然留下的痕迹）—— 不用再解析一遍 Markdown，直接把面包屑折叠成
        一棵树：文档 → 章节 → 小节。原文片段作为小节的摘要附在叶子上，
        让图谱是"知识结构的地图"，而不是一张只有标题的空壳。

        拿不到检索器或里面没东西（比如启动时知识库目录没配）就回空树，
        界面按空状态处理 —— 这和其它读接口的纪律一致：缺一块比白屏好。
        """
        retriever = getattr(deps, "retriever", None)
        list_chunks = getattr(retriever, "list_chunks", None)
        if not callable(list_chunks):
            return {"count": 0, "docs": []}
        try:
            chunks = list_chunks()
        except Exception as exc:  # noqa: BLE001 — 图谱缺一块不该让服务报错
            logger.warning("[api] /api/knowledge/graph 读取失败：%s", exc)
            return {"count": 0, "docs": []}
        return {"count": len(chunks), "docs": _graph_from_chunks(chunks)}

    # ------------------------------------------------------------
    # 前端（React 构建产物）
    # ------------------------------------------------------------
    @app.get("/")
    async def index() -> Any:
        """用户界面入口。

        页面只呈现对话本身，任何开发者/调试者视角的内容都不出现。
        """
        if not INDEX_HTML.is_file():
            # 前端资源缺失是部署问题，不是用户问题 —— 不该把构建命令、
            # 目录路径这些事讲给普通用户听。它们属于启动日志，不属于页面。
            return JSONResponse(status_code=404, content={"error": "页面暂时不可用"})
        return FileResponse(INDEX_HTML, media_type="text/html; charset=utf-8")

    # 构建产物带 hash 的静态资源（/assets/*.js、/assets/*.css）
    assets_dir = STATIC_DIR / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    # 界面底层的风景照片。单独挂一条，因为它**不是**构建产物（没有 hash 文件名），
    # 而是从 web/public/ 原样拷过来的资源 —— 和 /assets 混在一个挂载点里，
    # 下次有人按"带 hash 的产物"去清理目录时会把它一起删掉。
    scene_dir = STATIC_DIR / "scene"
    if scene_dir.is_dir():
        app.mount("/scene", StaticFiles(directory=scene_dir), name="scene")

    return app


__all__ = ["create_app", "ChatRequest", "FeedbackRequest", "ReviewRequest"]
