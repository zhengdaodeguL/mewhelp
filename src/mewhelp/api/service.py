"""服务层的内部件：会话历史落库、图调用、SSE 事件构造。

单独拆出来而不是全塞进 `app.py`，有三个具体理由：

1. 这些逻辑（历史取回、结果整形、事件顺序）**跟 HTTP 无关** —— CLI 的
   `chat` / `demo` 子命令要复用同一套，否则就会长出第二份"怎么调图"的实现，
   两边的口径迟早分叉；
2. 它们可以被单测直接调用，不必每次都起 TestClient；
3. `app.py` 只剩路由和装配，读起来就是一张路由表。

这里所有的对外方法都遵守同一条纪律：**异常不外泄**。历史读不出来就当没有
历史，写不进去就记一条 warning —— 会话存储是"锦上添花"，它挂了不该让用户
连一句答复都拿不到。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from ..config import Settings, get_settings
from ..schemas import ConversationResult, RetrievalResult, TokenUsage, Turn
from ..storage import SessionRow, session_scope
from ..workflow.state import AgentState, new_state

logger = logging.getLogger("mewhelp.api")

# ============================================================
# 常量
# ============================================================
#: 单条消息的字符上限。超长输入几乎从来不是"用户想问得更细"，而是误粘贴
#: （聊天记录、日志、整篇文档）。直接丢给模型既慢又挤爆上下文，所以显式挡掉。
MAX_MESSAGE_CHARS = 2000

#: 逐字推送的节奏。mock 下 answer 是瞬间生成的，靠这个间隔造出"模型在打字"
#: 的观感；将来接了真实流式输出，把常量调成 0 即可，调用方一行不用改。
STREAM_CHUNK_SLEEP = 0.01

#: 每次推送的字数，循环取值 —— 2~4 字一块接近人眼的阅读节奏，
#: 定长切块在中文里会出现"恰好切在词中间"的机械感。
CHUNK_SIZES: tuple[int, ...] = (3, 2, 4)

#: SSE 事件名。前端按这些名字解析，**不要改**。
EVENT_META = "meta"
EVENT_EVIDENCE = "evidence"
EVENT_TOOL = "tool"
EVENT_TOKEN = "token"
EVENT_DONE = "done"
EVENT_ERROR = "error"

EMPTY_ANSWER_FALLBACK = "抱歉，这个问题我暂时没能给出答复，帮你转人工确认。"
EMPTY_INPUT_ANSWER = (
    "你想咨询什么问题？直接说就行，比如「帮我查下订单 ORD-1002 到哪了」。"
)
TOO_LONG_ANSWER = f"你的问题太长了（超过 {MAX_MESSAGE_CHARS} 字），麻烦精简一下再发我。"

#: 会话里保留的最大轮数（含 user/assistant）。滑窗/摘要只关心最近若干轮，
#: 但多留一个远大于窗口的余量，好让指代消解在长会话里仍有东西可查。
DEFAULT_MAX_TURNS = 60


# ============================================================
# 小工具
# ============================================================
def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 3)


def _value(x: Any) -> Any:
    """枚举取 value，其它原样返回 —— 事件载荷里只放能 JSON 化的值。"""
    return x.value if hasattr(x, "value") else x


def _request_id(session_id: str, user_id: str, message: str, history_len: int) -> str:
    """由输入内容推导的 request_id，而不是 uuid4。

    这是刻意的：整条链路承诺"同输入同输出"，一个随机 id 会让重放调试、审计
    对账、飞轮记录回溯全部失效 —— 同一句话问两次会得到两个对不上的 id，
    而它们的处理过程本该完全一样。

    碰撞面足够小：同一会话里，只要历史长度变了（上一轮已经落库），
    同样的问句就会拿到不同的 id；不同用户/会话天然隔离。
    """
    raw = f"{session_id}\x1f{user_id}\x1f{message}\x1f{history_len}"
    return "req_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def chunk_text(text: str, sizes: Sequence[int] = CHUNK_SIZES) -> Iterator[str]:
    """把答复切成 2~4 字的小块，供 SSE 逐块推送。

    切块规则是**确定**的（不像随机长度那样看起来更"自然"）：同一句话每次
    推出来的分块完全一致，前端快照和回放才对得上。
    """
    if not text:
        return
    i, k = 0, 0
    while i < len(text):
        n = max(1, int(sizes[k % len(sizes)]))
        yield text[i : i + n]
        i += n
        k += 1


def sse_event(event: str, payload: dict[str, Any]) -> dict[str, str]:
    """构造一个 sse-starlette 能直接消费的事件字典。

    `ensure_ascii=False` 让中文在流里保持可读（调试时肉眼能看懂）；
    换行由 JSON 自己转义成 `\\n`，不会把 SSE 的"一行一个字段"拆坏。
    """
    return {"event": event, "data": json.dumps(payload, ensure_ascii=False)}


# ============================================================
# 会话存储
# ============================================================
def decode_turns(raw: str | None) -> list[Turn]:
    """把 `SessionRow.turns_json` 解回 list[Turn]。

    坏行逐条跳过：一条脏数据不该让整段历史不可用 —— 那会让指代消解突然
    "失明"，而症状是"多轮问答偶尔答非所问"，极难定位。
    """
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    turns: list[Turn] = []
    for item in data:
        try:
            turns.append(Turn.model_validate(item))
        except Exception:  # noqa: BLE001 — 单条坏数据只丢这一条
            continue
    return turns


def encode_turns(turns: Sequence[Turn]) -> str:
    return json.dumps([t.model_dump(mode="json") for t in turns], ensure_ascii=False)


class SessionStore:
    """会话历史的读写。

    只读写 `SessionRow`，不碰 Checkpointer：LangGraph 的 checkpointer 存的是
    "图的状态"（中断恢复用），这里存的是"人看得懂的那份对话"。两者职责不同，
    混在一起会让中断恢复和多轮上下文互相污染。
    """

    def __init__(self, settings: Settings | None = None, *, max_turns: int = DEFAULT_MAX_TURNS):
        self.settings = settings or get_settings()
        self.max_turns = max(2, int(max_turns))
        #: 最近一次存储失败的原因。不抛异常，但必须能被诊断（否则"历史丢了"
        #: 会表现为"多轮突然失忆"，而没有任何线索）。
        self.last_error: str | None = None

    def load(self, session_id: str) -> list[Turn]:
        if not session_id:
            return []
        try:
            with session_scope(settings=self.settings) as s:
                row = s.get(SessionRow, session_id)
                if row is None:
                    return []
                return decode_turns(row.turns_json)
        except Exception as exc:  # noqa: BLE001 — 读不到历史 = 没有历史
            self.last_error = f"load failed: {exc!r}"
            logger.warning("[api] 会话历史读取失败（按无历史继续）：%s", exc)
            return []

    def append(self, session_id: str, user_id: str, turns: Sequence[Turn]) -> list[Turn]:
        """把新的轮次追加进会话，返回追加后的完整历史。

        截断保留**最近的**轮次：滑窗和摘要都只看近处，丢最早的那些对回答
        影响最小；反过来（丢最近的）会让多轮指代立刻失效。
        """
        if not session_id or not turns:
            return []
        try:
            with session_scope(settings=self.settings) as s:
                row = s.get(SessionRow, session_id)
                if row is None:
                    row = SessionRow(
                        session_id=session_id,
                        user_id=user_id or "",
                        summary="",
                        turns_json="[]",
                        summarized_turns=0,
                        token_estimate=0,
                    )
                    s.add(row)
                history = decode_turns(row.turns_json)
                history.extend(turns)
                if len(history) > self.max_turns:
                    history = history[-self.max_turns :]
                row.turns_json = encode_turns(history)
                if user_id:
                    row.user_id = user_id
            return history
        except Exception as exc:  # noqa: BLE001 — 写不进去不影响这次答复
            self.last_error = f"append failed: {exc!r}"
            logger.warning("[api] 会话历史写入失败（本次答复照常返回）：%s", exc)
            return []


# ============================================================
# 结果缓存
# ============================================================
@dataclass
class CachedRun:
    """一次请求的可复用现场。

    只缓存 `ConversationResult` 是不够的：用户点踩落飞轮时，审核台最需要的
    是**当时的检索快照**（判断"真缺知识"还是"没检索到"的唯一依据），而它不在
    ConversationResult 里。缓存住原始 state 才能把这份现场原样带进问题池。
    """

    result: ConversationResult
    retrieval: RetrievalResult | None = None


class ResultCache:
    """request_id → 最近一次请求现场的有界缓存。

    有界是必须的：无界字典在长跑的服务里就是内存泄漏，而症状是"跑几天之后
    内存爆掉"，跟请求量没有直观联系。
    """

    def __init__(self, maxsize: int = 256):
        self.maxsize = max(1, int(maxsize))
        self._rows: OrderedDict[str, CachedRun] = OrderedDict()

    def put(self, run: CachedRun) -> None:
        rid = run.result.request_id
        if not rid:
            return
        self._rows[rid] = run
        self._rows.move_to_end(rid)
        while len(self._rows) > self.maxsize:
            self._rows.popitem(last=False)

    def get(self, request_id: str) -> CachedRun | None:
        return self._rows.get(request_id or "")

    def __len__(self) -> int:
        return len(self._rows)


# ============================================================
# 状态 → 契约模型
# ============================================================
def result_from_state(
    state: AgentState,
    *,
    latency_ms: float,
    session_id: str = "",
    user_id: str = "",
    question: str = "",
) -> ConversationResult:
    """把图跑完后的 state 整形成 `ConversationResult`。

    这一层存在的意义是**防腐**：state 是图的内部结构，会随节点增删而变形；
    API / CLI / 评估脚本只认这一个契约模型。转换写在一处，改动就只在一处。
    """
    retrieval = state.get("retrieval")
    answer = (state.get("answer") or "").strip()
    handoff = bool(state.get("handoff"))
    if not answer:
        # 空答复绝不能发出去：前端会看到一片空白，用户不知道发生了什么。
        # 兜一句"转人工"并同步把 handoff 置真，语义上才是自洽的。
        answer = EMPTY_ANSWER_FALLBACK
        handoff = True

    return ConversationResult(
        request_id=state.get("request_id", "") or "",
        session_id=state.get("session_id", "") or session_id,
        user_id=state.get("user_id", "") or user_id,
        question=state.get("question", "") or question,
        resolved_question=state.get("resolved_question", "") or "",
        intent=state.get("intent"),
        intent_confidence=float(state.get("intent_confidence") or 0.0),
        exit=state.get("exit"),
        answer=answer,
        path=list(state.get("path") or []),
        refusal_reason=state.get("refusal_reason"),
        handoff=handoff,
        evidences=list(retrieval.evidences) if retrieval else [],
        tool_calls=list(state.get("tool_calls") or []),
        interrupted=bool(state.get("interrupt_prompt")),
        interrupt_prompt=state.get("interrupt_prompt"),
        trace=list(state.get("trace") or []),
        usage=state.get("usage") or TokenUsage(),
        latency_ms=latency_ms,
    )


def _guarded_result(
    *,
    request_id: str,
    session_id: str,
    user_id: str,
    question: str,
    answer: str,
    history_len: int = 0,
    handoff: bool = False,
) -> ConversationResult:
    """不跑图、直接给一句答复的结果（空输入 / 超长输入 / 图崩了）。

    这条路径下 `path` 是空的 —— 前端会如实显示"没走流水线"，
    而不是伪造一条看起来正常的分流路径。

    `handoff` 默认 False：让用户"再说一遍"或"说得短点"并不是转人工，
    把它标成转人工会让转人工率这个指标凭空变高，也就失去了意义。
    """
    return ConversationResult(
        request_id=request_id or _request_id(session_id, user_id, question, history_len),
        session_id=session_id,
        user_id=user_id,
        question=question,
        resolved_question=question,
        answer=answer,
        path=[],
        handoff=handoff,
    )


# ============================================================
# 服务
# ============================================================
class ChatService:
    """一次问答的完整动作：取历史 → 跑图 → 整形 → 落库 → 留痕。

    图是**注入**进来的（而不是在这里 build）：这样测试可以塞一张假图进来，
    评估脚本可以塞挂了 checkpointer 的图，而 `ask()` 的语义不变。
    """

    def __init__(
        self,
        *,
        graph: Any,
        sessions: SessionStore | None = None,
        tracer: Any | None = None,
        cache: ResultCache | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.graph = graph
        self.settings = settings or get_settings()
        # 注意这里是显式的 `is None` 判断，不是 `or`：ResultCache 定义了
        # `__len__`，一个**空**缓存是 falsy 的 —— 用 `cache or ResultCache()`
        # 会把调用方注入的缓存悄悄换成一个新对象，症状是"点踩永远落不进问题池"
        # （缓存里明明有，但查的是另一个）。这类 bug 靠读代码几乎发现不了。
        self.sessions = sessions if sessions is not None else SessionStore(self.settings)
        self.tracer = tracer
        self.cache = cache if cache is not None else ResultCache()

    # ------------------------------------------------------------------
    def ask(self, *, user_id: str, session_id: str, message: str) -> ConversationResult:
        text = (message or "").strip()
        history = self.sessions.load(session_id)

        # 两道输入护栏。它们不跑图，但照样落库、照样缓存 ——
        # 否则"用户说了句废话"这条记录在会话里会凭空消失，多轮上下文会断。
        if not text:
            result = _guarded_result(
                request_id="",
                session_id=session_id,
                user_id=user_id,
                question="",
                answer=EMPTY_INPUT_ANSWER,
                history_len=len(history),
            )
            return self._finish(result, state=None)

        if len(text) > MAX_MESSAGE_CHARS:
            result = _guarded_result(
                request_id="",
                session_id=session_id,
                user_id=user_id,
                question=text[:MAX_MESSAGE_CHARS],
                answer=TOO_LONG_ANSWER,
                history_len=len(history),
            )
            return self._finish(result, state=None)

        request_id = _request_id(session_id, user_id, text, len(history))
        t0 = time.perf_counter()
        try:
            state = self.graph.invoke(
                new_state(
                    request_id=request_id,
                    session_id=session_id,
                    user_id=user_id,
                    question=text,
                    history=list(history),
                )
            )
        except Exception as exc:  # noqa: BLE001
            # 图自己已经层层兜底了，走到这里说明是编排层的意外（比如状态结构
            # 不兼容）。仍然要给用户一句话，而不是一个 500。
            logger.exception("[api] 图执行异常")
            return self._finish(
                _guarded_result(
                    request_id=request_id,
                    session_id=session_id,
                    user_id=user_id,
                    question=text,
                    answer=EMPTY_ANSWER_FALLBACK,
                    history_len=len(history),
                    handoff=True,
                ),
                state=None,
                error=f"{type(exc).__name__}: {exc}",
            )

        result = result_from_state(
            state,
            latency_ms=_ms(t0),
            session_id=session_id,
            user_id=user_id,
            question=text,
        )
        return self._finish(result, state=state)

    # ------------------------------------------------------------------
    def _finish(
        self,
        result: ConversationResult,
        *,
        state: AgentState | None,
        error: str | None = None,
    ) -> ConversationResult:
        """收尾：缓存现场、登记 trace、把这一轮写回会话。"""
        retrieval = state.get("retrieval") if state else None
        if error and result.trace == []:
            # 崩在编排层时留一条自己的痕迹，否则 /api/trace 会空得让人以为
            # "这条请求根本没进来过"。
            from ..schemas import TraceStep

            result.trace = [
                TraceStep(node="api", error=error, details={"request_id": result.request_id})
            ]
        self.cache.put(CachedRun(result=result, retrieval=retrieval))
        self._record_trace(result)
        self._persist(result)
        return result

    def _record_trace(self, result: ConversationResult) -> None:
        """把 state 里攒的 TraceStep 交给 Tracer，喂给 `GET /api/trace/{id}`。

        观测层是零侵入挂的：节点只往 `state["trace"]` 追加，谁需要谁去取。
        这里补上 request_id —— step 是在节点里造的，那时还没有"归到哪条请求"
        这回事。
        """
        if self.tracer is None:
            return
        for step in result.trace:
            step.details.setdefault("request_id", result.request_id)
            try:
                self.tracer.record(step)
            except Exception as exc:  # noqa: BLE001 — 留痕失败不能影响答复
                logger.warning("[api] trace 登记失败：%s", exc)

    def _persist(self, result: ConversationResult) -> None:
        turns = [
            Turn(
                role="user",
                content=result.question,
                metadata={"request_id": result.request_id},
            ),
            Turn(
                role="assistant",
                content=result.answer,
                intent=result.intent,
                metadata={
                    "request_id": result.request_id,
                    "intent": _value(result.intent),
                    "exit": _value(result.exit),
                    "path": result.path,
                    "handoff": result.handoff,
                },
            ),
        ]
        self.sessions.append(result.session_id, result.user_id, turns)


# ============================================================
# SSE 事件流
# ============================================================
def meta_payload(result: ConversationResult) -> dict[str, Any]:
    return {
        "request_id": result.request_id,
        "intent": _value(result.intent),
        "exit": _value(result.exit),
        "path": list(result.path),
        "resolved_question": result.resolved_question,
    }


def evidence_payload(result: ConversationResult) -> dict[str, Any]:
    return {
        "chunks": [
            {
                "chunk_id": e.chunk_id,
                "title": e.chunk.title,
                "score": round(float(e.score), 4),
            }
            for e in result.evidences
        ]
    }


def tool_payload(result: ConversationResult) -> list[dict[str, Any]]:
    return [
        {"name": tc.name, "ok": bool(tc.ok), "error": tc.error} for tc in result.tool_calls
    ]


def done_payload(result: ConversationResult) -> dict[str, Any]:
    return {
        "answer": result.answer,
        "usage": result.usage.model_dump(mode="json"),
        "latency_ms": result.latency_ms,
        "handoff": result.handoff,
        "refusal_reason": _value(result.refusal_reason),
    }


async def stream_events(
    result: ConversationResult, *, sleep: float = STREAM_CHUNK_SLEEP
) -> AsyncIterator[dict[str, str]]:
    """按 meta → evidence → tool* → token* → done 的顺序推事件。

    顺序是硬约束，不只是为了好看：前端靠"先拿到 path 和证据、再逐字收答复"
    来布局（左侧对话框 / 右侧检视面板）。顺序一乱，检视面板就会漏掉内容。

    逐字推送是**事后切块**的：图本身是同步跑完的（mock 下几十毫秒），
    想让它看起来像流式，只能在这里人为放慢节奏。真接了流式模型之后，
    这个函数改成边收边推即可，事件名和顺序都不用动。
    """
    yield sse_event(EVENT_META, meta_payload(result))
    yield sse_event(EVENT_EVIDENCE, evidence_payload(result))
    for payload in tool_payload(result):
        yield sse_event(EVENT_TOOL, payload)
    for piece in chunk_text(result.answer):
        yield sse_event(EVENT_TOKEN, {"text": piece})
        if sleep:
            await asyncio.sleep(sleep)
    yield sse_event(EVENT_DONE, done_payload(result))


__all__ = [
    "MAX_MESSAGE_CHARS",
    "STREAM_CHUNK_SLEEP",
    "CHUNK_SIZES",
    "EVENT_META",
    "EVENT_EVIDENCE",
    "EVENT_TOOL",
    "EVENT_TOKEN",
    "EVENT_DONE",
    "EVENT_ERROR",
    "ChatService",
    "CachedRun",
    "ResultCache",
    "SessionStore",
    "chunk_text",
    "decode_turns",
    "done_payload",
    "encode_turns",
    "evidence_payload",
    "meta_payload",
    "result_from_state",
    "sse_event",
    "stream_events",
    "tool_payload",
]
