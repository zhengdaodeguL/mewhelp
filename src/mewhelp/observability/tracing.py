"""可观测层。

对外只有两个东西：`Tracer`（记录）和 `build_tracer()`（装配）。设计上有三条
不可让步的约束，写在这里以免后来者改坏：

1. **离线必须零成本**。`langfuse_enabled=False` 时本模块不 import langfuse、
   不建 HTTP 连接、不碰 OpenTelemetry。做法是把 langfuse 的 import 全部关在
   `_LangfuseBridge` 里（延迟导入），Tracer 关闭时连这个类都不会被实例化。
   没做这件事的项目，离线用户会被一个观测 SDK 拖慢甚至拖挂。

2. **trace 是产品功能，不是调试开关**。`GET /api/trace/{request_id}` 要能返回
   调用树，所以就算不开 Langfuse，`TraceStep` 也照收。`langfuse_enabled` 只决定
   "要不要往远端导出"，不决定"要不要留痕"。实测空 step 约 12µs（pydantic 构造
   占大头），一条请求十几步 ≈ 0.2ms，相对于一次 LLM 调用可以忽略。

3. **观测层绝不能反过来影响业务**。Langfuse 连不上、SDK 版本不对、key 没配，
   一律降级成"只留本地 trace + 打一条 warning"，业务代码无感。这也是为什么
   `_StepScope.__exit__` 永远返回 False —— 异常必须原样冒出去，不能被
   观测代码吞掉。
"""

from __future__ import annotations

import functools
import json
import logging
import time
from collections import OrderedDict, deque
from contextlib import AbstractContextManager
from datetime import date, datetime
from enum import Enum
from types import TracebackType
from typing import Any, Callable, Mapping, TypeVar

from pydantic import BaseModel

from ..config import Settings, get_settings
from ..domain import NodeName
from ..schemas import (
    Evidence,
    GateDecision,
    RetrievalResult,
    TokenUsage,
    ToolCall,
    TraceStep,
)

logger = logging.getLogger("mewhelp.observability")

# ---------------------------------------------------------------- 常量
#: 摘要字符上限 —— 只存"够用来定位问题"的长度。日志爆炸的代价是没人再看日志。
_MAX_DIGEST_CHARS = 280
#: 单个 detail 字符串的上限
_MAX_DETAIL_STR = 200
#: 列表类 detail（证据、工具调用）保留条数
_MAX_LIST_ITEMS = 8
#: 单步 details 序列化后的总长度上限，超了按 value 体积从大到小丢
_MAX_DETAILS_BYTES = 4000
#: 内存里保留的 step 总数 / trace 条数上限。服务长跑时这是防内存泄漏的兜底。
_MAX_STEPS = 4000
_MAX_TRACES = 500

F = TypeVar("F", bound=Callable[..., Any])


# ============================================================
# 摘要工具：任何对象 → 短、可 JSON 序列化、确定性的表示
# ============================================================
def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"…(+{len(text) - limit})"


def _jsonable(value: Any, depth: int = 0) -> Any:
    """把任意对象压成可 JSON 序列化的浅结构。

    递归深度和列表长度都设了上限：真出问题的时候，日志里有一条"被截断了"
    就足够定位，把整个 RetrievalResult 打进去只会让人不想看。
    """
    if depth > 4:
        return _truncate(str(value), _MAX_DETAIL_STR)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate(value, _MAX_DETAIL_STR)
    if isinstance(value, Enum):  # 含 StrEnum / Intent / Exit 等
        return _jsonable(value.value, depth + 1)
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json"), depth + 1)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        items = list(value.items())[: _MAX_LIST_ITEMS * 4]
        return {str(k): _jsonable(v, depth + 1) for k, v in items}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v, depth + 1) for v in list(value)[: _MAX_LIST_ITEMS * 4]]
    return _truncate(str(value), _MAX_DETAIL_STR)


def digest(value: Any, limit: int = _MAX_DIGEST_CHARS) -> str:
    """生成输入/输出摘要。

    刻意不保留原文结构：调用树要回答的是"这一步吃了什么、吐了什么"，
    而不是"把整个 state 再存一遍"。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(_jsonable(value), ensure_ascii=False, default=str)
        except (TypeError, ValueError):  # pragma: no cover - 兜底
            text = str(value)
    # 收起换行：一条 trace 一行，前端不用处理多行
    text = " ".join(text.split())
    return _truncate(text, limit)


def _evidence_digest(ev: Any) -> dict[str, Any]:
    """一条证据只留"是谁、多相关" —— 正文不进 trace，否则接口体被撑爆。"""
    chunk = getattr(ev, "chunk", None)
    return {
        "chunk_id": getattr(chunk, "chunk_id", "") or getattr(ev, "chunk_id", ""),
        "title": getattr(chunk, "title", ""),
        "score": round(float(getattr(ev, "score", 0.0) or 0.0), 4),
    }


def _retrieval_digest(r: RetrievalResult) -> dict[str, Any]:
    """检索结果的摘要 —— "检索回了哪些 chunk" 就在 chunks 里。"""
    return {
        "query": _truncate(r.query, _MAX_DETAIL_STR),
        "strategy": r.strategy,
        "dense_hits": r.dense_hits,
        "sparse_hits": r.sparse_hits,
        "fused_hits": r.fused_hits,
        "reranked_hits": r.reranked_hits,
        "top_score": round(r.top_score, 4),
        "latency_ms": round(r.latency_ms, 3),
        "chunks": [_evidence_digest(e) for e in r.evidences[:_MAX_LIST_ITEMS]],
    }


def _tool_digest(t: ToolCall) -> dict[str, Any]:
    """工具摘要 —— "调了哪个工具、成没成" 一眼可见，成败都要留。"""
    return {
        "name": t.name,
        "args": _jsonable(t.args),
        "ok": t.ok,
        "error": t.error,
        "elapsed_ms": round(t.elapsed_ms, 3),
        "attempts": t.attempts,
    }


def _summarize(value: Any) -> Any:
    """details 里每个值的统一收口：结构化对象走专用摘要，其余走通用裁剪。"""
    if isinstance(value, RetrievalResult):
        return _retrieval_digest(value)
    if isinstance(value, Evidence):
        return _evidence_digest(value)
    if isinstance(value, ToolCall):
        return _tool_digest(value)
    if isinstance(value, TokenUsage):
        return value.model_dump(mode="json")
    if isinstance(value, GateDecision):
        return {
            "passed": value.passed,
            "confidence": value.confidence,
            "reason": value.reason.value if value.reason else None,
            "checks": _jsonable(value.checks),
        }
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(v, Evidence) for v in value):
            return [_evidence_digest(v) for v in value[:_MAX_LIST_ITEMS]]
        if value and all(isinstance(v, ToolCall) for v in value):
            return [_tool_digest(v) for v in value[:_MAX_LIST_ITEMS]]
    return _jsonable(value)


def _cap_details(details: dict[str, Any], limit: int = _MAX_DETAILS_BYTES) -> dict[str, Any]:
    """details 总量兜底。

    逐个丢"体积最大的键"而不是整体丢弃：整体丢会让人完全看不出这一步干了什么，
    丢几个大键至少保留了骨架，且顺序确定（先按体积、体积相同按 key 名）。

    先走一个廉价的必要性判断再序列化：大多数 step 的 details 只有三五个短标量，
    为了它们做一次 json.dumps 纯属白烧 CPU，而 step() 在热路径上。
    """
    if not details:
        return details
    if len(details) * (_MAX_DETAIL_STR + 32) + 2 <= limit:
        # 全是短标量时体积有上界，不用真序列化就能断定不会超
        if all(
            v is None
            or isinstance(v, (bool, int, float))
            or (isinstance(v, str) and len(v) <= _MAX_DETAIL_STR)
            for v in details.values()
        ):
            return details

    try:
        size = len(json.dumps(details, ensure_ascii=False, default=str))
    except (TypeError, ValueError):  # pragma: no cover - _jsonable 之后不该发生
        return {"_unserializable": True}
    if size <= limit:
        return details

    dropped: list[str] = []
    kept = dict(details)
    while kept:
        biggest = max(
            sorted(kept, key=str),
            key=lambda k: len(json.dumps(kept[k], ensure_ascii=False, default=str)),
        )
        kept.pop(biggest)
        dropped.append(biggest)
        try:
            size = len(json.dumps(dict(kept, _dropped=dropped), ensure_ascii=False, default=str))
        except (TypeError, ValueError):  # pragma: no cover
            size = 0
        if size <= limit:
            break
    kept["_dropped"] = sorted(dropped)
    return kept


# ============================================================
# Langfuse 桥接 —— 只有 enabled=True 时才会被 import 到
# ============================================================
class _LangfuseBridge:
    """TraceStep ↔ Langfuse span 的映射。

    这个类被单独隔出来，是为了让"没开观测"的进程永远不加载 langfuse 及其
    背后的 OpenTelemetry。所有远端调用都包在 try 里，失败只降级不中断。
    """

    def __init__(self, settings: Settings) -> None:
        # 延迟导入（硬要求）：这行只在 langfuse 开启时才会被执行
        from langfuse import Langfuse

        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            raise RuntimeError(
                "langfuse_enabled=True 但缺少 public_key/secret_key，"
                "已降级为仅本地 trace"
            )
        self._client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )

    def open_span(self, name: str, input_digest: str, metadata: dict[str, Any]) -> Any:
        """返回一个 Langfuse 的上下文管理器；调用方负责 __enter__/__exit__。"""
        return self._client.start_as_current_observation(
            name=name,
            as_type="span",
            input=input_digest or None,
            metadata=metadata or None,
        )

    def flush(self) -> None:
        self._client.flush()

    def shutdown(self) -> None:
        self._client.shutdown()


# ============================================================
# 一步的上下文
# ============================================================
class TracedStep(TraceStep):
    """`with tracer.step(...) as st:` 里拿到的那个对象。

    做成 TraceStep 的子类，而不是包一层代理：编排层会把 step 直接塞进
    `state["trace"]` / `ConversationResult.trace`，那两处的类型就是 TraceStep，
    任何"长得像但不是"的对象都会在别处炸掉。子类同时满足类型契约和便利方法。
    """

    def note(self, **details: Any) -> "TracedStep":
        """补记细节（统一裁剪后并入 details）。返回自身，方便链式调用。"""
        self.details.update({k: _summarize(v) for k, v in details.items()})
        return self

    def set_output(self, value: Any) -> "TracedStep":
        """事后补输出摘要（比如拿不到返回值、只能从 state 里回读时）。"""
        self.output_digest = digest(value)
        return self

    def set_input(self, value: Any) -> "TracedStep":
        self.input_digest = digest(value)
        return self


class _StepScope(AbstractContextManager[TraceStep]):
    """`tracer.step(...)` 返回的上下文管理器。

    退出时补齐 elapsed_ms，并（开启时）收掉对应的 Langfuse span。
    注意 `__exit__` 永远返回 False：观测代码的职责是记录异常，不是处理异常。
    """

    __slots__ = ("_tracer", "step", "_t0", "_span", "_span_cm")

    def __init__(self, tracer: "Tracer", step: TracedStep) -> None:
        self._tracer = tracer
        self.step = step
        self._t0 = time.perf_counter()
        self._span: Any = None
        self._span_cm: Any = None

    def __enter__(self) -> TraceStep:
        # span 的开启做完了再返回：span 建不起来不该影响 with 体内的业务
        self._span_cm, self._span = self._tracer._open_span(self.step)
        return self.step

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        self.step.elapsed_ms = round((time.perf_counter() - self._t0) * 1000, 3)
        if exc is not None and not self.step.error:
            # 只记类型 + 消息，不记 traceback：调用树上看一眼就够
            self.step.error = f"{exc_type.__name__ if exc_type else 'Error'}: {exc}"
            self.step.details.setdefault("error_type", exc_type.__name__ if exc_type else "Error")
        self.step.details = _cap_details(self.step.details)
        self._tracer._close_span(self.step, self._span, self._span_cm)
        self._tracer._emit(self.step)
        return False  # 绝不放行任何异常


# ============================================================
# Tracer
# ============================================================
class Tracer:
    """一条请求 → 一棵调用树。

    `enabled` 只控制"要不要往 Langfuse 导出"，本地 TraceStep 始终收集 ——
    因为前端 `GET /api/trace/{request_id}` 依赖它，而这跟观测后端无关。
    """

    def __init__(self, settings: Settings | None = None, enabled: bool | None = None) -> None:
        self.settings = settings or get_settings()
        self.enabled = bool(self.settings.langfuse_enabled if enabled is None else enabled)
        # 用 deque(maxlen=) 而不是 list：服务长跑时 step 会几十万条，
        # 靠 `del list[:n]` 裁剪是 O(n) 的列表搬移，会把热路径拖慢。
        self._steps: deque[TraceStep] = deque(maxlen=_MAX_STEPS)
        self._by_request: OrderedDict[str, list[TraceStep]] = OrderedDict()
        self._current_request_id: str | None = None
        self._bridge: _LangfuseBridge | None = None
        #: 去重后的告警集合 —— 每步刷一条 warning 会把日志淹掉
        self._warned: set[str] = set()
        if self.enabled:
            self._bridge = self._build_bridge()

    # ------------------------------------------------ 构造期
    def _build_bridge(self) -> _LangfuseBridge | None:
        try:
            return _LangfuseBridge(self.settings)
        except Exception as exc:  # noqa: BLE001 — 缺包/缺 key/版本不兼容都走这条
            self._warn_once(
                "bridge_init",
                f"Langfuse 初始化失败，已降级为仅本地 trace："
                f"{type(exc).__name__}: {exc}",
            )
            return None

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning("[observability] %s", message)

    # ------------------------------------------------ 请求上下文
    def bind(self, request_id: str) -> AbstractContextManager[None]:
        """把接下来的 step 归到这条 request 下（with 用法，退出后复原）。

        用 bind 而不是让每个 step 都传 request_id：业务代码里 request_id 的
        作用域本来就是一次请求，重复传参只会漏。
        """
        return _RequestBinding(self, request_id)

    def set_request(self, request_id: str | None) -> None:
        self._current_request_id = request_id or None

    @property
    def current_request_id(self) -> str | None:
        return self._current_request_id

    # ------------------------------------------------ 记录
    def step(self, node: str | NodeName, **details: Any) -> AbstractContextManager[TraceStep]:
        """开一步。

        `node` 直接收 NodeName 常量（StrEnum，天然是 str），这样节点名
        全局只有一处拼写来源，不会出现 "retreive"/"retrieve" 并存。
        """
        request_id = details.pop("request_id", None) or self._current_request_id
        input_value = details.pop("input", None)
        output_value = details.pop("output", None)
        tokens_in = details.pop("tokens_in", 0)
        tokens_out = details.pop("tokens_out", 0)

        summary = {k: _summarize(v) for k, v in details.items()}
        if request_id:
            summary["request_id"] = request_id

        step = TracedStep(
            node=node.value if isinstance(node, Enum) else str(node),
            input_digest=digest(input_value) if input_value is not None else "",
            output_digest=digest(output_value) if output_value is not None else "",
            details=_cap_details(summary),
            tokens_in=int(tokens_in or 0),
            tokens_out=int(tokens_out or 0),
        )
        return _StepScope(self, step)

    def record(self, step: TraceStep) -> TraceStep:
        """登记一个外部构造好的 TraceStep（零侵入挂载用）。

        编排层自己往 state["trace"] 里追加 step，最后一次性交给 Tracer 即可，
        不需要在业务代码里到处 with。
        """
        self._emit(step)
        return step

    @property
    def steps(self) -> list[TraceStep]:
        return list(self._steps)

    def trace(self, request_id: str) -> list[TraceStep]:
        """`GET /api/trace/{request_id}` 的数据源。查不到返回空列表，不抛异常。"""
        return list(self._by_request.get(request_id, ()))

    def request_ids(self) -> list[str]:
        return list(self._by_request.keys())

    def clear(self) -> None:
        self._steps.clear()
        self._by_request.clear()

    def wrap(self, fn: F, node: str | NodeName | None = None) -> F:
        """把一个节点函数包成"自动记一步"的版本。

        这是"零侵入"里"侵入最小"的那一档：改的是装配处一行代码，
        节点函数本身一行不动。
        """

        @functools.wraps(fn)
        def _inner(*args: Any, **kwargs: Any) -> Any:
            name = node or getattr(fn, "__name__", "node")
            with self.step(name) as st:
                if args or kwargs:
                    # 只取入参的浅摘要，避免把整个 LangGraph state 复制一遍。
                    # args[0] 是 self，对还原调用树没有价值，丢掉。
                    st.set_input({"args": args[1:] if args else (), "kwargs": kwargs})
                out = fn(*args, **kwargs)
                st.set_output(out)
                return out

        return _inner  # type: ignore[return-value]

    # ------------------------------------------------ 内部：span 生命周期
    def _open_span(self, step: TraceStep) -> tuple[Any, Any]:
        """开 span。任何失败 → 关掉桥并降级，返回值永远安全可解包。"""
        if self._bridge is None:
            return None, None
        try:
            cm = self._bridge.open_span(step.node, step.input_digest, step.details)
            span = cm.__enter__()
            return cm, span
        except Exception as exc:  # noqa: BLE001
            self._warn_once(
                "span_open",
                f"Langfuse span 开启失败，后续步骤只留本地 trace："
                f"{type(exc).__name__}: {exc}",
            )
            return None, None

    def _close_span(self, step: TraceStep, span: Any, cm: Any) -> None:
        if span is not None:
            try:
                span.update(
                    output=step.output_digest or None,
                    metadata=dict(step.details, elapsed_ms=step.elapsed_ms),
                    level="ERROR" if step.error else "DEFAULT",
                    status_message=step.error or None,
                )
            except Exception as exc:  # noqa: BLE001
                self._warn_once("span_update", f"Langfuse span 更新失败：{exc}")
        if cm is not None:
            try:
                # 传 (None, None, None)：不让 SDK 决定异常要不要被吞；
                # 异常由 _StepScope.__exit__ 原样往上抛。
                cm.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                self._warn_once("span_close", f"Langfuse span 关闭失败：{exc}")

    # ------------------------------------------------ 内部：落盘（内存）
    def _emit(self, step: TraceStep) -> None:
        # deque(maxlen=) 自动淘汰最老的，不需要手工裁剪
        self._steps.append(step)

        rid = step.details.get("request_id")
        if not rid:
            return
        bucket = self._by_request.get(str(rid))
        if bucket is None:
            bucket = []
            self._by_request[str(rid)] = bucket
            while len(self._by_request) > _MAX_TRACES:
                self._by_request.popitem(last=False)
        bucket.append(step)

    # ------------------------------------------------ 收尾
    def flush(self) -> None:
        """把缓冲的 span 推给 Langfuse。永远不抛异常。"""
        if self._bridge is None:
            return  # 关闭态：真正的 no-op，连函数调用都不必进
        try:
            self._bridge.flush()
        except Exception as exc:  # noqa: BLE001 — 连不上后端绝不能影响收尾
            self._warn_once("flush", f"Langfuse flush 失败（已忽略）：{exc}")

    def close(self) -> None:
        self.flush()
        if self._bridge is None:
            return
        try:
            self._bridge.shutdown()
        except Exception as exc:  # noqa: BLE001
            self._warn_once("shutdown", f"Langfuse shutdown 失败（已忽略）：{exc}")

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.close()
        return False


class _RequestBinding(AbstractContextManager[None]):
    """`with tracer.bind(rid):` —— 退出时还原上一个 request_id，支持嵌套。"""

    __slots__ = ("_tracer", "_request_id", "_prev")

    def __init__(self, tracer: Tracer, request_id: str) -> None:
        self._tracer = tracer
        self._request_id = request_id
        self._prev: str | None = None

    def __enter__(self) -> None:
        self._prev = self._tracer.current_request_id
        self._tracer.set_request(self._request_id)
        return None

    def __exit__(self, *exc_info: object) -> bool:
        self._tracer.set_request(self._prev)
        return False


# ============================================================
# 工厂
# ============================================================
def build_tracer(settings: Settings | None = None) -> Tracer:
    """按配置装配 Tracer。开启与否只在 Tracer 内部判断一次。"""
    settings = settings or get_settings()
    return Tracer(settings=settings, enabled=settings.langfuse_enabled)


_DEFAULT_TRACER: Tracer | None = None


def get_tracer(settings: Settings | None = None) -> Tracer:
    """进程级默认 Tracer。

    服务层需要"写 trace 的地方"和"读 trace 的接口"拿到同一个实例，
    所以在模块级缓存一份；这跟模块内部是否缓存配置无关，是两回事。
    """
    global _DEFAULT_TRACER
    if _DEFAULT_TRACER is None:
        _DEFAULT_TRACER = build_tracer(settings)
    return _DEFAULT_TRACER


def reset_tracer() -> None:
    """测试用：丢掉进程级实例，让它按新配置重建。"""
    global _DEFAULT_TRACER
    if _DEFAULT_TRACER is not None:
        _DEFAULT_TRACER.close()
    _DEFAULT_TRACER = None


__all__ = [
    "Tracer",
    "TraceStep",
    "build_tracer",
    "get_tracer",
    "reset_tracer",
    "digest",
]
