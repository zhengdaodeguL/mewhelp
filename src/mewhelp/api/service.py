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
import re
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterator, Sequence
from concurrent.futures import Future
from contextvars import copy_context
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from ..config import Settings, get_settings
from ..generation import GenerationContext, current_generation, check_cancelled
from ..memory.extract import extract_candidates, prompt_lines, used_memory_ids
from ..memory.store import MemoryStore
from ..schemas import (
    MEMORY_KIND_LABELS,
    ConversationResult,
    Evidence,
    KbChunk,
    RetrievalResult,
    SessionDetail,
    SessionMessage,
    SessionSummary,
    TokenUsage,
    Turn,
)
from ..storage import SessionRow, ensure_utc, session_scope
from ..workflow.state import AgentState, new_state

logger = logging.getLogger("mewhelp.api")

# ============================================================
# 常量
# ============================================================
#: 单条消息的字符上限。超长输入几乎从来不是"用户想问得更细"，而是误粘贴
#: （聊天记录、日志、整篇文档）。直接丢给模型既慢又挤爆上下文，所以显式挡掉。
MAX_MESSAGE_CHARS = 2000

#: 每次推送的字数，循环取值 —— 2~4 字一块接近人眼的阅读节奏，
#: 定长切块在中文里会出现"恰好切在词中间"的机械感。
CHUNK_SIZES: tuple[int, ...] = (3, 2, 4)

#: SSE 事件名。前端按这些名字解析，**不要改**。
EVENT_META = "meta"
EVENT_EVIDENCE = "evidence"
EVENT_MEMORY = "memory"
EVENT_TOOL = "tool"
EVENT_TOKEN = "token"
EVENT_DONE = "done"
EVENT_ERROR = "error"

EMPTY_ANSWER_FALLBACK = "抱歉，这个问题我暂时没能给出答复，帮你转人工确认。"
EMPTY_INPUT_ANSWER = (
    "你想咨询什么问题？直接说就行，比如「帮我查下订单 ORD-1002 到哪了」。"
)
TOO_LONG_ANSWER = f"你的问题太长了（超过 {MAX_MESSAGE_CHARS} 字），麻烦精简一下再发我。"
_STOP_STREAM = object()

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


#: 会话标题长度上限。历史列表一行放不下更多；截断比换行好 ——
#: 列表要能一眼扫过去，两行的条目会毁掉这个节奏。
TITLE_CHARS = 24

#: 列表里那行灰色预览的长度
PREVIEW_CHARS = 48


def _flatten(text: str) -> str:
    """把一段可能带换行/Markdown 记号的文本压成一行。

    标题和预览都会出现在**列表**里，而列表里的换行会把一条撑成三行高；
    `**` 之类的记号则是写给渲染器的，直接显示出来就是噪声。
    """
    line = " ".join((text or "").split())
    return line.replace("**", "").replace("__", "").lstrip("#").strip()


def make_title(text: str, limit: int = TITLE_CHARS) -> str:
    """由第一句用户消息推出会话标题。

    用用户原话而不是让模型总结：历史列表要的是"认得出这是哪次对话"，
    用户自己那句话永远比一句概括更好认，而且这是零成本、零延迟的。
    """
    line = _flatten(text)
    if not line:
        return ""
    return line[:limit] + ("…" if len(line) > limit else "")


def _title_from(turns: Sequence[Turn]) -> str:
    """从一串轮次里推标题：优先第一句用户消息，没有就退回第一条内容。"""
    for t in turns:
        if t.role == "user" and t.content.strip():
            return make_title(t.content)
    for t in turns:
        if t.content.strip():
            return make_title(t.content)
    return ""


def _preview_from(turns: Sequence[Turn]) -> str:
    """列表里的预览行：最后一条助手答复的开头。"""
    for t in reversed(list(turns)):
        if t.role == "assistant" and t.content.strip():
            line = _flatten(t.content)
            return line[:PREVIEW_CHARS] + ("…" if len(line) > PREVIEW_CHARS else "")
    return ""


def to_session_messages(turns: Sequence[Turn]) -> list[SessionMessage]:
    """库里的轮次 → 恢复对话要用的消息列表。

    两道过滤，缺一不可：

      1. **只留 user / assistant**。tool、system 轮是给模型看的，界面里
         没有它们的容身之处（也没有对应的气泡样式）。
      2. **只回用户视角的字段**。`intent` / `exit` / `path` 就存在
         `Turn.metadata` 里，是审计用的；顺手把它们带出去，历史一恢复
         就等于把这些内部信息又摆回了界面上。字段的白名单在这里显式写死，
         而不是"刨掉几个已知的 key" —— 后者在下次有人加字段时会漏。
    """
    out: list[SessionMessage] = []
    for t in turns:
        if t.role not in ("user", "assistant"):
            continue
        meta = t.metadata or {}
        out.append(
            SessionMessage(
                role=t.role,
                content=t.content,
                handoff=bool(meta.get("handoff")) if t.role == "assistant" else False,
                request_id=str(meta.get("request_id") or ""),
                status=meta.get("status", "complete"),
                evidence=meta.get("evidence", []) if t.role == "assistant" else [],
                tools=meta.get("tools", []) if t.role == "assistant" else [],
            )
        )
    return out


class SessionOwnershipError(Exception):
    """Raised when a session identifier belongs to another account."""


class SessionStore:
    """会话历史的读写。

    只读写 `SessionRow`，不碰 Checkpointer：LangGraph 的 checkpointer 存的是
    "图的状态"（中断恢复用），这里存的是"人看得懂的那份对话"。两者职责不同，
    混在一起会让中断恢复和多轮上下文互相污染。
    """

    def __init__(self, settings: Settings | None = None, *, max_turns: int = DEFAULT_MAX_TURNS):
        self.settings = settings or get_settings()
        self.max_turns = max(2, int(max_turns))
        self._write_lock = threading.RLock()
        #: 最近一次存储失败的原因。不抛异常，但必须能被诊断（否则"历史丢了"
        #: 会表现为"多轮突然失忆"，而没有任何线索）。
        self.last_error: str | None = None

    def load(self, session_id: str, user_id: str | None = None) -> list[Turn]:
        if not session_id:
            return []
        try:
            with session_scope(settings=self.settings) as s:
                row = s.get(SessionRow, session_id)
                if row is None:
                    return []
                if user_id is not None and row.user_id != user_id:
                    raise SessionOwnershipError
                return decode_turns(row.turns_json)
        except SessionOwnershipError:
            raise
        except Exception as exc:  # noqa: BLE001 — 读不到历史 = 没有历史
            self.last_error = f"load failed: {exc!r}"
            logger.warning("[api] 会话历史读取失败（按无历史继续）：%s", exc)
            return []

    def append(
        self, session_id: str, user_id: str, turns: Sequence[Turn], *, replace: bool = False,
        existing_only: bool = False,
    ) -> list[Turn]:
        """把新的轮次追加进会话，返回追加后的完整历史。

        截断保留**最近的**轮次：滑窗和摘要都只看近处，丢最早的那些对回答
        影响最小；反过来（丢最近的）会让多轮指代立刻失效。

        顺带维护标题：只在标题为空时写一次（第一次有内容的时候）。
        每次都重算的话，用户聊到第三轮会发现历史列表里那条的名字变了 ——
        列表跳动比名字取得不完美更烦人。
        """
        if not session_id or not turns:
            return []
        check_cancelled()
        try:
            with self._write_lock, session_scope(settings=self.settings) as s:
                row = s.get(SessionRow, session_id)
                if row is None:
                    if existing_only:
                        return []
                    row = SessionRow(
                        session_id=session_id,
                        user_id=user_id or "",
                        title="",
                        summary="",
                        turns_json="[]",
                        summarized_turns=0,
                        token_estimate=0,
                    )
                    s.add(row)
                elif row.user_id != user_id:
                    raise SessionOwnershipError
                history = decode_turns(row.turns_json)
                if replace:
                    # SSE 的待核验快照、最终核验结果与停止结果共用同一个问答 ID。
                    # 原位更新保留轮次顺序，不为异步收尾再追加一轮。
                    positions = {
                        (t.metadata.get("request_id"), t.role): i
                        for i, t in enumerate(history) if t.metadata.get("request_id")
                    }
                    if existing_only and any((t.metadata.get("request_id"), t.role) not in positions for t in turns):
                        return []
                    for turn in turns:
                        key = (turn.metadata.get("request_id"), turn.role)
                        if key in positions:
                            history[positions[key]] = turn
                        else:
                            positions[key] = len(history)
                            history.append(turn)
                else:
                    history.extend(turns)
                if len(history) > self.max_turns:
                    history = history[-self.max_turns :]
                row.turns_json = encode_turns(history)
                if not (row.title or "").strip():
                    row.title = _title_from(history)
                check_cancelled()
            return history
        except SessionOwnershipError:
            raise
        except Exception as exc:  # noqa: BLE001 — 写不进去不影响这次答复
            self.last_error = f"append failed: {exc!r}"
            logger.warning("[api] 会话历史写入失败（本次答复照常返回）：%s", exc)
            return []

    # ------------------------------------------------------------------
    # 历史列表 / 恢复 / 改名 / 删除
    # ------------------------------------------------------------------
    def list_sessions(self, user_id: str, *, limit: int = 50) -> list[SessionSummary]:
        """一个用户的历史会话，最近更新的排前面。

        按服务端解析的 `user_id` 过滤，避免不同账号之间互相看到历史标题。
        """
        if not user_id:
            return []
        try:
            with session_scope(settings=self.settings) as s:
                rows = list(
                    s.scalars(
                        select(SessionRow)
                        .where(SessionRow.user_id == user_id)
                        .order_by(SessionRow.updated_at.desc(), SessionRow.session_id.desc())
                        .limit(max(1, int(limit)))
                    )
                )
                out: list[SessionSummary] = []
                for row in rows:
                    turns = decode_turns(row.turns_json)
                    if not turns:
                        # 空会话不进列表：它是"点了新会话又没说啥"留下的壳子，
                        # 列出来只会让历史栏里堆一串"未命名"
                        continue
                    out.append(
                        SessionSummary(
                            session_id=row.session_id,
                            title=(row.title or "").strip() or _title_from(turns) or "未命名会话",
                            turns=len(turns),
                            preview=_preview_from(turns),
                            updated_at=ensure_utc(row.updated_at),
                        )
                    )
                return out
        except Exception as exc:  # noqa: BLE001 — 列表拉不出来，界面就当没有历史
            self.last_error = f"list failed: {exc!r}"
            logger.warning("[api] 会话列表读取失败：%s", exc)
            return []

    def get_session(self, session_id: str, *, user_id: str = "") -> SessionDetail | None:
        """取一次会话的完整对话（已按用户视角过滤）。

        HTTP 路径总是传入登录账号；归属不符时与不存在使用同一种结果。
        """
        if not session_id:
            return None
        try:
            with session_scope(settings=self.settings) as s:
                row = s.get(SessionRow, session_id)
                if row is None:
                    return None
                if user_id and (row.user_id or "") != user_id:
                    return None
                turns = decode_turns(row.turns_json)
                return SessionDetail(
                    session_id=row.session_id,
                    title=(row.title or "").strip() or _title_from(turns) or "未命名会话",
                    updated_at=ensure_utc(row.updated_at),
                    messages=to_session_messages(turns),
                )
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"get failed: {exc!r}"
            logger.warning("[api] 会话读取失败：%s", exc)
            return None

    def rename(self, session_id: str, title: str, *, user_id: str = "") -> bool:
        """改名。空标题是合法输入（表示"恢复成自动生成的标题"）。"""
        if not session_id:
            return False
        try:
            with session_scope(settings=self.settings) as s:
                row = s.get(SessionRow, session_id)
                if row is None or (user_id and (row.user_id or "") != user_id):
                    return False
                row.title = _flatten(title)[:TITLE_CHARS + 8]
                return True
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"rename failed: {exc!r}"
            logger.warning("[api] 会话改名失败：%s", exc)
            return False

    def delete(self, session_id: str, *, user_id: str = "") -> bool:
        """删掉一次会话。

        只删这一行 —— 记忆在另一张表（`user_memory`），**不会**被连带删掉。
        这正是两张表分开的理由：用户清空的是"聊天记录"，不是"你认识我"。
        """
        if not session_id:
            return False
        try:
            with session_scope(settings=self.settings) as s:
                row = s.get(SessionRow, session_id)
                if row is None or (user_id and (row.user_id or "") != user_id):
                    return False
                s.delete(row)
                return True
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"delete failed: {exc!r}"
            logger.warning("[api] 会话删除失败：%s", exc)
            return False


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


def _extract_in_background(llm: Any, question: str) -> Any:
    """在后台线程里跑记忆抽取，返回一个已经开跑的 Future。

    只用来把"问模型要候选"这一步从关键路径上挪走，**不改变任何顺序**：
    真正决定先后的是 `_memorize` 里的三步（先记命中、再落库、最后回读），
    这里的线程只负责算，落库仍然由 `_memorize` 在主线程上按原顺序做。

    用裸线程而不是线程池：这个 Future 的存活期就是一次请求，不需要池化；
    而要管好一个长生命周期线程池的关闭时机，比省下的这点开销麻烦得多。
    """
    future: Future = Future()

    def run() -> None:
        try:
            future.set_result(extract_candidates(llm, question))
        except BaseException as exc:  # noqa: BLE001 —— 抽取失败不能杀死线程
            future.set_exception(exc)

    context = copy_context()
    threading.Thread(target=context.run, args=(run,), daemon=True, name="mewhelp-memory").start()
    return future


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
        memory: MemoryStore | None = None,
        llm: Any | None = None,
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
        #: 长期记忆（跨会话的用户档案）。默认**关闭** —— 不注入就当没有这回事，
        #: 这样单测里塞一张假图进来跑流程时，不会悄悄去连真实数据库。
        self.memory = memory
        #: 抽取记忆用的模型。没给就只读不写（读是纯查询，写要花一次模型调用）。
        self.llm = llm
        # 持有收尾任务直到完成；HTTP 断连不应让落库任务被垃圾回收或取消。
        self._background: set[asyncio.Task] = set()
        self._active: dict[tuple[str, str, str], GenerationContext] = {}
        self._session_tail: dict[tuple[str, str], asyncio.Future] = {}
        self._stopping: dict[tuple[str, str, str], asyncio.Task] = {}
        self._deleted_sessions: OrderedDict[tuple[str, str], None] = OrderedDict()

    def _track(self, coroutine) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self._background.add(task)

        def finished(task):
            self._background.discard(task)
            if not task.cancelled() and task.exception() is not None:
                logger.error("[api] 异步收尾失败", exc_info=task.exception())

        task.add_done_callback(finished)
        return task

    async def drain(self) -> None:
        """关服/离线调用等待已登记的收尾；不放在 HTTP 正文流的关闭路径上。"""
        while self._background:
            await asyncio.gather(*list(self._background), return_exceptions=True)

    async def wait_for_session(self, user_id: str, session_id: str = "") -> None:
        """正文结束后的读操作等收尾，保证刷新/反馈能读到刚完成的那轮。"""
        pending = [
            g.settled for (uid, sid, _), g in self._active.items()
            if uid == user_id and (not session_id or sid == session_id)
            and (g.done_sent or g.cancelled.is_set())
        ]
        pending.extend(
            task for (uid, sid, _), task in self._stopping.items()
            if uid == user_id and (not session_id or sid == session_id)
        )
        if pending:
            await asyncio.gather(*(asyncio.shield(task) for task in pending))

    def _stored_result(self, user_id: str, session_id: str, request_id: str) -> ConversationResult | None:
        turns = self.sessions.load(session_id, user_id=user_id)
        reply = next((t for t in reversed(turns) if t.role == "assistant" and t.metadata.get("request_id") == request_id), None)
        if reply is None:
            return None
        question = next((t.content for t in reversed(turns) if t.role == "user" and t.metadata.get("request_id") == request_id), "")
        return ConversationResult(
            user_id=user_id, session_id=session_id, request_id=request_id, question=question,
            answer=reply.content, status=reply.metadata.get("status", "complete"),
            handoff=bool(reply.metadata.get("handoff")),
            evidences=[Evidence.model_validate(e) for e in reply.metadata.get("retrieval_evidence", [])],
        )

    async def stop(
        self, *, user_id: str, session_id: str, request_id: str, message: str, answer: str | None = None,
    ) -> ConversationResult | None:
        """明确停止 + 保存确认；也覆盖原 POST 尚未到达的窗口。"""
        key = (user_id, session_id, request_id)
        if key[:2] in self._deleted_sessions:
            return None
        if any(uid != user_id and sid == session_id for uid, sid, _ in self._active):
            raise SessionOwnershipError
        # 先检查所有权，不能用停止接口给别人的会话写入一轮。
        stored = self._stored_result(user_id, session_id, request_id)
        if generation := self._active.get(key):
            if not generation.done_sent:
                source = generation.text or (generation.state or {}).get("answer", "")
                # 只接受确实生成过的前缀，不能把客户端提供的任意文本当作助手答复。
                generation.stop_answer = answer if answer is not None and source.startswith(answer) else generation.delivered
                generation.cancel()
                if generation.producer is not None:
                    generation.producer.cancel()
            await asyncio.shield(generation.settled)
            return await self._confirm_stopped_prefix(self._stored_result(user_id, session_id, request_id), answer)
        if stored is not None:
            return await self._confirm_stopped_prefix(stored, answer)
        if key not in self._stopping:
            previous = self._session_tail.get((user_id, session_id))

            async def save():
                if previous is not None:
                    await asyncio.shield(previous)
                result = ConversationResult(
                    user_id=user_id, session_id=session_id, request_id=request_id,
                    question=message.strip()[:MAX_MESSAGE_CHARS], answer="", status="stopped",
                )
                await asyncio.to_thread(self._persist, result)

            self._stopping[key] = task = self._track(save())
            self._session_tail[key[:2]] = task

            def stopped(task):
                if self._stopping.get(key) is task:
                    self._stopping.pop(key, None)
                if self._session_tail.get(key[:2]) is task:
                    self._session_tail.pop(key[:2], None)

            task.add_done_callback(stopped)
        task = self._stopping[key]
        try:
            await asyncio.shield(task)
        finally:
            if task.done():
                self._stopping.pop(key, None)
        return self._stored_result(user_id, session_id, request_id)

    async def _confirm_stopped_prefix(self, result, received: str | None):
        # 浏览器可能先断连再送达停止确认。此时后端已保存的发送队列可能比
        # 浏览器实际收到的多几片，只能收缩为已生成过的前缀，不能接受任意替换。
        if result is None or result.status != "stopped" or received is None or result.answer == received:
            return result
        if not result.answer.startswith(received):
            return result
        turns = await asyncio.to_thread(self.sessions.load, result.session_id, user_id=result.user_id)
        reply = next((t for t in turns if t.role == "assistant" and t.metadata.get("request_id") == result.request_id), None)
        if reply is None:
            return None
        saved = await asyncio.to_thread(
            self.sessions.append, result.session_id, result.user_id,
            [reply.model_copy(update={"content": received})], replace=True, existing_only=True,
        )
        if not saved:
            return None
        result.answer = received
        return result

    async def delete_session(self, user_id: str, session_id: str) -> bool:
        """先结束在途写入再删除，晚到的停止确认不能重建已删除的会话。"""
        self.sessions.load(session_id, user_id=user_id)
        key = (user_id, session_id)
        self._deleted_sessions[key] = None
        self._deleted_sessions.move_to_end(key)
        while len(self._deleted_sessions) > 256:
            self._deleted_sessions.popitem(last=False)
        for (uid, sid, _), generation in list(self._active.items()):
            if (uid, sid) == key and not generation.done_sent:
                generation.cancel()
                if generation.producer is not None:
                    generation.producer.cancel()
        await self.wait_for_session(user_id, session_id)
        return await asyncio.to_thread(self.sessions.delete, session_id, user_id=user_id)

    # ------------------------------------------------------------------
    def ask(
        self,
        *,
        user_id: str,
        session_id: str,
        message: str,
        quote: str = "",
        request_id: str = "",
    ) -> ConversationResult:
        check_cancelled()
        text = (message or "").strip()
        history = self.sessions.load(session_id, user_id=user_id)

        # 引用提问：用户在界面上明确指向的那一句，要**显式**带进这一轮的上下文。
        #
        # 这不是"多给点信息"。被引用的往往是较早的消息 —— 轮到它的时候很可能
        # 已经滑出 `window_turns` 了，只靠会话历史根本看不见。挂成一条 user 轮次
        # 放在**最后**：指代消解找"最近实体"时会因此指向被引用的那一句，
        # 这正是"关于这句再问一句"该有的行为。
        #
        # 它只活在这一次请求里，不落库（`_persist` 只写 question 和 answer）——
        # 被引用的消息本身就在会话历史里，再存一遍会让它下次出现两次。
        quoted = (quote or "").strip()
        if quoted:
            history = list(history) + [
                Turn(
                    role="user",
                    content=f"（我引用了自己先前说过的一句）{quoted}",
                    metadata={"quoted": True},
                )
            ]

        # 用户档案在跑图之前取一次：指代消解要靠它认出"我的订单"是哪一单。
        # 失败时返回空列表（读不到记忆 = 没有记忆），不影响这一轮的任何流程。
        memories = self.memory.for_prompt(user_id) if self.memory else []
        memory_lines = prompt_lines(memories)

        # 两道输入护栏。它们不跑图，但照样落库、照样缓存 ——
        # 否则"用户说了句废话"这条记录在会话里会凭空消失，多轮上下文会断。
        if not text:
            result = _guarded_result(
                request_id=request_id,
                session_id=session_id,
                user_id=user_id,
                question="",
                answer=EMPTY_INPUT_ANSWER,
                history_len=len(history),
            )
            return self._finish(result, state=None)

        if len(text) > MAX_MESSAGE_CHARS:
            result = _guarded_result(
                request_id=request_id,
                session_id=session_id,
                user_id=user_id,
                question=text[:MAX_MESSAGE_CHARS],
                answer=TOO_LONG_ANSWER,
                history_len=len(history),
            )
            return self._finish(result, state=None)

        request_id = request_id or _request_id(session_id, user_id, text, len(history))
        t0 = time.perf_counter()

        # 同步调用的记忆抽取与图并行；SSE 则在 done 之后启动，避免档案
        # 模型的等待挡住首字。两条路径仍由 _memorize 按相同顺序保存候选。
        generation = current_generation.get()
        learner = _extract_in_background(self.llm, text) if (self.memory and self.llm and generation is None) else None
        if generation is not None:
            generation.learner = learner
            generation.state = new_state(
                request_id=request_id, session_id=session_id, user_id=user_id,
                question=text, history=list(history), memories=memory_lines,
            )

        try:
            state = self.graph.invoke(
                new_state(
                    request_id=request_id,
                    session_id=session_id,
                    user_id=user_id,
                    question=text,
                    history=list(history),
                    memories=memory_lines,
                )
            )
        except Exception as exc:  # noqa: BLE001
            check_cancelled()
            if generation is not None and generation.text:
                raise
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
        return self._finish(result, state=state, learner=learner)

    # ------------------------------------------------------------------
    def _finish(
        self,
        result: ConversationResult,
        *,
        state: AgentState | None,
        error: str | None = None,
        learner: Any = None,
    ) -> ConversationResult:
        """收尾：缓存现场、登记 trace、把这一轮写回会话。"""
        check_cancelled()
        generation = current_generation.get()
        retrieval = state.get("retrieval") if state else None
        if error and result.trace == []:
            # 崩在编排层时留一条自己的痕迹，否则 /api/trace 会空得让人以为
            # "这条请求根本没进来过"。
            from ..schemas import TraceStep

            result.trace = [
                TraceStep(node="api", error=error, details={"request_id": result.request_id})
            ]
        if generation is not None:
            # SSE 消费完所有正文后才提交，不能在排队发送前就保存整轮。
            generation.finish = state
            if state is not None:
                generation.state = state
            return result
        self.cache.put(CachedRun(result=result, retrieval=retrieval))
        self._record_trace(result)
        self._persist(result)
        self._memorize(result, learner=learner)
        return result

    def _memorize(self, result: ConversationResult, *, learner: Any = None) -> None:
        """把这一轮里"值得长期记住的东西"写进用户档案，并把最新档案挂回结果。

        顺序是有讲究的，三步不能换：

          1. **先记命中**。判断"这条记忆有没有被用上"要拿**注入时**那份档案
             和 `resolved_question` 比（补进去的那个单号就是记忆的功劳）。
             放到抽取之后做的话，抽取刚写进去的新值会被算成"被用上了" ——
             自己给自己记功，这个计数就再也没有意义了。
          2. **再抽新记忆**。抽取永远只从**用户这句话**里抽：档案是"关于这个人
             的事实"，助手说过什么不算。
          3. **最后回读全量**。推给界面的是这一轮之后该用户的完整档案，
             前端拿它直接替换面板，不需要做增量合并 —— 增量合并漏掉一次事件，
             面板就会和真实记忆永久对不上，而且没人会发现。

        整个方法包在 try 里：记忆是锦上添花，它出任何问题都不该让这一轮答复
        （已经算完了）变成错误。
        """
        if self.memory is None or not result.user_id:
            return
        try:
            check_cancelled()
            injected = self.memory.for_prompt(result.user_id)
            self.memory.mark_used(
                result.user_id,
                used_memory_ids(injected, result.question, result.resolved_question),
            )
            if self.llm is not None:
                # 候选多半已经在跑图的那段时间里算好了（见 `ask` 里的
                # `_extract_in_background`）；`learner` 是 None 时才是同步抽。
                candidates = learner.result() if learner is not None else extract_candidates(
                    self.llm, result.question
                )
                check_cancelled()
                self.memory.remember_many(result.user_id, candidates)
            result.memories = self.memory.list(result.user_id)
        except Exception as exc:  # noqa: BLE001 —— 记忆失败不影响答复
            logger.warning("[api] 记忆更新失败（本轮答复不受影响）：%s", exc)

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
        check_cancelled()
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
                    "status": result.status,
                    "evidence": evidence_payload(result)["chunks"],
                    # 摘要只用于展示；多轮追问需要完整正文、表格与来源。
                    "retrieval_evidence": [e.model_dump(mode="json") for e in result.evidences],
                    "tools": tool_payload(result),
                },
            ),
        ]
        self.sessions.append(
            result.session_id, result.user_id, turns,
            replace=current_generation.get() is not None or result.status in {"stopped", "interrupted"},
        )

    async def stream(self, **kwargs: Any) -> AsyncIterator[dict[str, str]]:
        """正文以 done 结束；核验、记忆与持久化由保留引用的任务完成。

        HTTP 消费者收到 done 就关闭迭代器。离线消费者若遍历到耗尽，则同时
        等待收尾，便于测试/CLI 读取最终结果。aclose 不会取消已交付的答案。
        """
        generation = GenerationContext()
        kwargs = dict(kwargs)
        kwargs["request_id"] = kwargs.get("request_id") or f"req_{secrets.token_hex(12)}"
        generation.request_id = kwargs["request_id"]
        key = (kwargs["user_id"], kwargs["session_id"], generation.request_id)
        session_key = key[:2]
        if session_key in self._deleted_sessions:
            raise SessionOwnershipError
        if pending_stop := self._stopping.get(key):
            await asyncio.shield(pending_stop)
        if existing := self._active.get(key):
            await asyncio.shield(existing.settled)
        if stored := self._stored_result(*key):
            async for event in stream_events(stored):
                yield event
            return
        if any(uid != key[0] and sid == key[1] for uid, sid, _ in self._active):
            raise SessionOwnershipError
        # 同会话的前一轮，以及同用户已交付的档案更新，须先可读再开始下一轮。
        # 其它用户/仍在独立会话生成的请求可以并行。
        predecessors = {
            g.settled for (uid, _, _), g in self._active.items() if uid == key[0] and g.done_sent
        }
        if previous := self._session_tail.get(session_key):
            predecessors.add(previous)
        self._session_tail[session_key] = generation.settled
        self._active[key] = generation
        generation.state = new_state(
            request_id=generation.request_id, session_id=kwargs["session_id"],
            user_id=kwargs["user_id"], question=(kwargs.get("message") or "").strip(),
        )
        queue: asyncio.Queue = asyncio.Queue()
        generation.on_cancel = lambda: queue.put_nowait(_STOP_STREAM)
        prepared: ConversationResult | None = None
        delivered_done = False
        failed = False
        result: ConversationResult | None = None
        started = time.perf_counter()
        first_token_at: float | None = None
        last_token_at: float | None = None

        def settle():
            if not generation.settled.done():
                generation.settled.set_result(None)
            if self._active.get(key) is generation:
                self._active.pop(key, None)
            if self._session_tail.get(session_key) is generation.settled:
                self._session_tail.pop(session_key, None)

        def emit(event):
            generation.check()
            generation.loop.call_soon_threadsafe(queue.put_nowait, event)

        def prepare(state):
            nonlocal prepared
            generation.check()
            generation.state = state
            prepared = result_from_state(state, latency_ms=0)
            # 只读已有档案；抽取/更新绝不挡住首字。新档案由收尾后 REST 同步。
            prepared.memories = self.memory.list(prepared.user_id) if self.memory else []
            for event in _opening_events(prepared):
                emit(event)

        def token(piece):
            generation.check()
            generation.text += piece
            emit(sse_event(EVENT_TOKEN, {"text": piece}))

        def complete(state, *, verifying=False):
            if generation.response_ready:
                return
            generation.check()
            generation.state = state
            answer = result_from_state(state, latency_ms=_ms(started))
            answer.status = "verifying" if verifying else "complete"
            generation.response_ready = True
            emit(answer)
            # done 发出、初始快照持久化之后再开始模型自评/飞轮。
            # 等待本身也可取消；停止时不会继续跑自评或发更正。
            generation.run(generation.released.wait())

        generation.prepare = prepare
        generation.token = token
        generation.complete = complete

        async def produce():
            context_token = current_generation.set(generation)
            try:
                if predecessors:
                    await asyncio.gather(*(asyncio.shield(p) for p in predecessors))
                generation.check()
                result = await asyncio.to_thread(self.ask, **kwargs)
                if not generation.response_ready:
                    queue.put_nowait(result)
                await generation.closed.wait()
                await generation.released.wait()
                generation.check()
                await asyncio.to_thread(self._memorize, result, learner=generation.learner)
                await asyncio.to_thread(self._commit_stream, generation, result)
            except asyncio.CancelledError:
                # 也处理宿主任务取消，确保同步图里等待的异步 HTTP 一并结束。
                generation.cancel()
                queue.put_nowait(_STOP_STREAM)
            except Exception as exc:
                if generation.done_sent:
                    logger.exception("[api] 答复收尾失败 request_id=%s", generation.request_id)
                    failed_result = result_from_state(generation.state, latency_ms=_ms(started))
                    failed_result.answer = generation.delivered + "\n\n答复核验未完成，请重新提问或联系人工客服。"
                    failed_result.status = "interrupted"
                    await asyncio.to_thread(self._persist, failed_result)
                else:
                    queue.put_nowait(exc)
            finally:
                if generation.done_sent:
                    settle()
                current_generation.reset(context_token)

        producer = self._track(produce())
        generation.producer = producer

        async def release():
            context_token = current_generation.set(generation)
            try:
                if self.memory and self.llm and result is not None:
                    # 只依赖用户原话，和后续自评/飞轮并行；仍在 done 之后启动。
                    generation.learner = _extract_in_background(self.llm, result.question)
                if result is not None and generation.response_ready:
                    # 核验期间刷新页面仍能恢复正文与“核对中”状态。
                    await asyncio.to_thread(self._persist, result)
            finally:
                generation.released.set()
                current_generation.reset(context_token)

        def sent(event):
            nonlocal first_token_at, last_token_at
            if event["event"] == EVENT_TOKEN:
                now = time.perf_counter()
                first_token_at = first_token_at or now
                last_token_at = now
                generation.delivered += json.loads(event["data"])["text"]
            return event

        try:
            while True:
                item = await queue.get()
                if item is _STOP_STREAM:
                    return
                if isinstance(item, Exception):
                    raise item
                if isinstance(item, ConversationResult):
                    result = item
                    break
                yield sent(item)
            generation.check()
            if prepared is None:
                result.memories = self.memory.list(result.user_id) if self.memory else []
                for event in _opening_events(result):
                    yield event
            else:
                result.memories = prepared.memories
            if not result.answer.startswith(generation.text):
                raise RuntimeError("流式正文与最终答复不一致")
            suffix = result.answer[len(generation.text):]
            # 规则/工具/拒答本来已是完整结果，直接分片排空，不伪造打字等待。
            for piece in chunk_text(suffix):
                generation.check()
                yield sent(sse_event(EVENT_TOKEN, {"text": piece}))
            generation.check()
            delivered_done = True
            generation.done_sent = True
            now = time.perf_counter()
            logger.info(
                "[api] stream done request_id=%s first_token_ms=%.1f tail_ms=%.1f elapsed_ms=%.1f status=%s",
                generation.request_id, ((first_token_at or now) - started) * 1000,
                (now - (last_token_at or now)) * 1000, (now - started) * 1000, result.status,
            )
            yield sse_event(EVENT_DONE, done_payload(result))
        except Exception:
            failed = True
            raise
        finally:
            if delivered_done:
                generation.closed.set()
                self._track(release())
            else:
                generation.cancel()
                producer.cancel()
                if not failed or generation.delivered:
                    partial = result_from_state(generation.state, latency_ms=_ms(started))
                    partial.answer = generation.stop_answer if generation.stop_answer is not None else generation.delivered
                    partial.status = "interrupted" if failed else "stopped"
                    partial.handoff = False
                    partial.refusal_reason = None

                    async def save_partial():
                        # 取消模型请求，仍须保存已交付的部分；写入脱离取消上下文。
                        context_token = current_generation.set(None)
                        try:
                            if predecessors:
                                await asyncio.gather(*(asyncio.shield(p) for p in predecessors))
                            await asyncio.to_thread(self._persist, partial)
                        finally:
                            settle()
                            current_generation.reset(context_token)

                    saving = self._track(save_partial())
                    try:
                        await asyncio.shield(saving)
                    except asyncio.CancelledError:
                        pass  # ASGI 断连可能再次取消；任务仍由服务持有并在关服时 drain。
                else:
                    settle()
        # 完整耗尽迭代器的离线调用等待最终状态；HTTP 在 done 处 aclose，跳过此处。
        await asyncio.shield(producer)

    def _commit_stream(self, generation, result):
        generation.check()
        state = generation.finish
        retrieval = state.get("retrieval") if state else None
        self._persist(result)
        generation.check()
        self.cache.put(CachedRun(result=result, retrieval=retrieval))
        self._record_trace(result)


# ============================================================
# SSE 事件流
# ============================================================
#: Markdown 表格行（含 `| - | - |` 这种分隔行）
_MD_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
#: `**加粗**` / `__加粗__`
_MD_EMPHASIS = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")


def _user_snippet(chunk: KbChunk, limit: int = 140) -> str:
    """把知识片段收拾成**用户读得懂**的一小段。

    知识库是 Markdown 写的，整段搬上界面会把 `**加粗**` 的星号、表格分隔行
    （`| - | - |`）原样露出来 —— 那些是写给渲染器看的记号，不是给人看的内容。
    表格片段则只取表头当摘要，比把整张表拍在卡片上清楚得多。
    """
    if chunk.is_table and chunk.table:
        head = [c.strip() for c in chunk.table[0] if c.strip()]
        if head:
            return "表格：" + "、".join(head)

    lines: list[str] = []
    for raw in (chunk.text or "").splitlines():
        line = raw.strip()
        if not line or _MD_TABLE_ROW.match(line):
            continue
        line = _MD_EMPHASIS.sub(lambda m: m.group(1) or m.group(2), line.lstrip("#").strip())
        if line:
            lines.append(line)

    text = " ".join(lines)
    if not text:
        # 整段都被过滤掉了（比如内容本身就是一张表）—— 退回标题，至少不是空白
        return chunk.title
    return text[:limit] + ("…" if len(text) > limit else "")


def meta_payload(result: ConversationResult) -> dict[str, Any]:
    return {"request_id": result.request_id}


def evidence_payload(result: ConversationResult) -> dict[str, Any]:
    """检索依据。

    `snippet` 是给**用户看**的原文摘录，右侧工作区据此渲染「依据」卡片。
    只返回界面需要呈现的标题和摘录。
    """
    chunks: list[dict[str, Any]] = []
    for e in result.evidences:
        chunks.append(
            {
                "title": e.chunk.title,
                "snippet": _user_snippet(e.chunk),
            }
        )
    return {"chunks": chunks}


def tool_payload(result: ConversationResult) -> list[dict[str, Any]]:
    """把可展示的业务结果投影成工作区卡片数据。"""
    payload: list[dict[str, Any]] = []
    for tc in result.tool_calls:
        if not isinstance(tc.result, (dict, list)):
            continue
        data = tc.result
        # 操作是否获准与订单是否查到是两件事。快照是资料，拒绝不是操作成功。
        snapshot = data.get("data") if isinstance(data, dict) else None
        if isinstance(snapshot, dict) and snapshot.get("order_id") and isinstance(snapshot.get("items"), list):
            payload.append({"kind": "order", "ok": True, "result": snapshot})
        if not tc.ok:
            continue
        kind = "logistics" if tc.name == "track_logistics" else "order"
        payload.append({"kind": kind, "ok": True, "result": data})
    return payload


def memory_payload(result: ConversationResult) -> dict[str, Any]:
    """用户档案。

    推的是**全量**而不是"本轮新增/变化"：前端拿它直接替换面板状态即可。
    增量同步看着更省，但它要求前后端对"什么算一次变化"达成一致 ——
    一旦某次事件丢了，面板和真实记忆就会永久对不上，而且没有任何报错。

    `mem_key` / `user_id` 这类内部口径不在这里：面板要展示的就是
    id、类型、内容、是否钉住、被用上过几次。
    """
    return {"items": [memory_to_public(m) for m in result.memories]}


def memory_to_public(item: Any) -> dict[str, Any]:
    """一条记忆 → 界面能直接渲染的字典。

    `kind_label` 由服务端算好一起下发，而不是让前端自己维护一张
    "order → 关注的订单"的映射表：那张表是**产品文案**，两处各写一份，
    迟早会出现"服务端日志写'关注的订单'、界面显示'订单'"的不一致，
    而且没有任何测试会因此失败。

    刻意用**白名单**而不是 `model_dump`：`MemoryItem` 里有 `hits`（被用上
    几次）这种内部口径 —— 它回答的是"这条记忆有没有价值"，是给运营/调试
    看的，不该出现在面对用户的 payload 里。`model_dump` 会把字段一股脑带出去，
    下次加一个内部字段就又会漏到界面上；白名单则是一道显式的闸。
    """
    return {
        "id": item.id,
        "kind": item.kind,
        "kind_label": MEMORY_KIND_LABELS.get(item.kind, "记忆"),
        "text": item.text,
        "source": item.source,
        "pinned": item.pinned,
        "created_at": item.created_at.isoformat() if item.created_at else "",
        "updated_at": item.updated_at.isoformat() if item.updated_at else "",
    }


def done_payload(result: ConversationResult) -> dict[str, Any]:
    payload = {
        "answer": result.answer,
        "handoff": result.handoff,
    }
    if result.status != "complete":
        payload["status"] = result.status
    return payload


def _opening_events(result: ConversationResult) -> Iterator[dict[str, str]]:
    yield sse_event(EVENT_META, meta_payload(result))
    yield sse_event(EVENT_EVIDENCE, evidence_payload(result))
    if result.memories:
        yield sse_event(EVENT_MEMORY, memory_payload(result))
    for payload in tool_payload(result):
        yield sse_event(EVENT_TOOL, payload)


async def stream_events(result: ConversationResult) -> AsyncIterator[dict[str, str]]:
    """按 meta → evidence → memory? → tool* → token* → done 的顺序推事件。

    顺序是硬约束，不只是为了好看：前端靠"先拿到路径和证据、再逐字收答复"
    来布局（左侧对话框 / 右侧检视面板）。顺序一乱，检视面板就会漏掉内容。

    `memory` 事件只在**该用户确实有档案**时才推（新用户一个事件都不多推）：
    前端因此可以把"收到 memory 事件"直接当成"面板该更新了"的信号，
    不必自己判断"这次是不是空列表"。也正因如此，`test_stream_events_on_short_answer`
    里那个没有任何记忆的结果，事件序列仍然是老的四件套。

    已完成结果立即分片排空，不添加打字等待；在线生成即时转发模型分片。
    """
    for event in _opening_events(result):
        yield event
    for piece in chunk_text(result.answer):
        yield sse_event(EVENT_TOKEN, {"text": piece})
    yield sse_event(EVENT_DONE, done_payload(result))


__all__ = [
    "MAX_MESSAGE_CHARS",
    "CHUNK_SIZES",
    "EVENT_META",
    "EVENT_EVIDENCE",
    "EVENT_MEMORY",
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
    "make_title",
    "memory_payload",
    "memory_to_public",
    "meta_payload",
    "result_from_state",
    "sse_event",
    "stream_events",
    "to_session_messages",
    "tool_payload",
]
