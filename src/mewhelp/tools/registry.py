"""工具注册表：把"调用一个工具"变成一件可审计、可限时、可重试的事。

为什么值得单独一层？因为工具是整条链路里**唯一会碰外部世界**的地方：
网络会抖、下游会超时、参数会写错。如果这些失败直接以异常形式冒到图上，
一次抽风就会把整请求打挂，而用户只是想查个订单。

所以这一层定了三条规矩：
  1. **异常不外泄**：任何失败都变成 `ToolCall(ok=False, error=...)`。
  2. **每次调用都留痕**：成功失败都产出 ToolCall，`attempts` / `elapsed_ms` 如实填。
     审计里最怕"看起来成功但其实重试了 3 次"，所以重试次数必须落进记录。
  3. **schema 由 ToolSpec 单向生成**：模型看到的工具描述和真正执行的是同一份定义，
     避免"schema 说参数叫 order_id、实现却叫 oid"这种只能靠线上发现的错。

超时实现用线程池而不是信号量：信号量只在主线程可用，而工具可能被
FastAPI 的线程池调用；线程池方案在两种场景下行为一致。
"""

from __future__ import annotations

import inspect
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable

from ..config import Settings, get_settings
from ..domain import ToolError
from ..schemas import ToolCall, ToolSpec

#: 工具函数返回这个值时表示"业务规则拒绝"。
#: 为什么不让业务工具直接返回 ToolCall 或抛异常？
#:   - 直接返回 ToolCall 会让业务代码知道审计字段（elapsed_ms/attempts），耦合太紧；
#:   - 抛异常在语义上等于"意外失败"，而业务拒绝是**预期内**的结果。
#: 用一个纯 dict 当契约，两边都能各自演化。
_KEY_OK = "ok"
_KEY_ERROR = "error"
_KEY_RESULT = "result"
_KEY_DATA = "data"
_ACCOUNT_ORDER_TOOLS = frozenset(
    {"get_order", "track_logistics", "request_refund", "cancel_order", "change_address"}
)


class ToolRegistry:
    """工具注册与调用。

    `today` 是注入点：退款时效这类规则依赖"今天是几号"，但直接用
    `date.today()` 会让"正好第 30 天可退"这种边界用例明天就失效。
    把当前日期显式传进来，测试才能钉死在一个日期上复现。
    （业务工具本身也支持逐次覆盖，见 business.py 的 `today` 参数。）
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        today: Any = None,
    ) -> None:
        self.settings = settings or get_settings()
        # 默认 None 而不是 date.today()：让"没注入"这件事保持可见，
        # 真正的兜底发生在 business 层，那里才知道该怎么算。
        self.today = today
        self._specs: dict[str, ToolSpec] = {}
        self._fns: dict[str, Callable[..., Any]] = {}
        #: 审计流水 —— 由调用方 drain。不设上限是有意的：
        #: 工具调用次数本来就少（一次请求个位数），截断反而会丢关键现场。
        self.audit: list[ToolCall] = []
        #: 单次调用覆盖项，走显式开关而不是 monkeypatch，
        #: 这样"测试强制超时"这件事在测试代码里是自解释的。
        self._timeout_override: float | None = None
        self._max_attempts_override: int | None = None
        #: 注册期自检留下的问题清单（schema 与实现不一致）。
        #: 存下来而不是打印：测试可以直接断言它为空，把一致性变成可回归的。
        self.binding_problems: list[str] = []

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------
    def register(self, spec: ToolSpec, fn: Callable[..., Any]) -> None:
        """登记一个工具。同名重复注册以最后一次为准（热更新场景需要）。"""
        if not spec.name:
            raise ValueError("ToolSpec.name 不能为空")
        if not callable(fn):
            raise TypeError(f"工具 {spec.name} 的实现不可调用: {type(fn)!r}")
        self._specs[spec.name] = spec
        self._fns[spec.name] = fn
        # 注册时就把参数校验一遍，把"schema 和实现不一致"从运行时错误
        # 提前成启动期错误。这里不抛给调用方（契约要求异常不外泄），
        # 只记进 binding_problems，由测试断言。
        self.binding_problems.extend(_validate_binding(spec, fn))

    def names(self) -> list[str]:
        """已注册工具名，保持注册顺序 —— 顺序即"给模型的推荐优先级"。"""
        return list(self._specs)

    def spec_of(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    # ------------------------------------------------------------------
    # 对外描述
    # ------------------------------------------------------------------
    def specs(self) -> list[dict]:
        """OpenAI function-calling 格式的工具声明。"""
        out: list[dict] = []
        for spec in self._specs.values():
            params = dict(spec.parameters or {})
            # 补齐 OpenAI 要求的最小合法结构：类型 + 属性表。
            # 缺 properties 会被服务端直接拒掉整个请求，属于"一个工具写错、
            # 全场工具用不了"，所以在这里兜住。
            params.setdefault("type", "object")
            params.setdefault("properties", {})
            fn: dict[str, Any] = {
                "name": spec.name,
                "description": spec.description,
                "parameters": params,
            }
            # high_risk 不是 OpenAI 标准字段，放进 function 里透传。
            # 这么做的理由：它描述的是**这个动作本身**的性质（能不能被 Agent
            # 直接执行），属于工具定义的一部分；塞在 parameters 里会被模型
            # 当成一个可以自己填的参数，属于语义错误。
            # 图上的高风险判定读的是 ToolSpec.high_risk，不依赖这里的字段。
            fn["high_risk"] = bool(spec.high_risk)
            out.append({"type": "function", "function": fn})
        return out

    def high_risk_actions(self) -> frozenset[str]:
        """Agent 不允许直接执行的动作名 —— 高风险子流程用它做白名单。"""
        return frozenset(n for n, s in self._specs.items() if s.high_risk)

    def describe(self) -> str:
        """给 prompt 用的工具清单（人类可读，不是 JSON）。"""
        if not self._specs:
            return "当前没有可用工具。"
        lines = ["可用工具："]
        for spec in self._specs.values():
            required = set(spec.parameters.get("required", []) or [])
            sig = ", ".join(
                name if name in required else f"{name}?"
                for name in (spec.parameters.get("properties") or {})
            )
            flag = "  [高风险·需人工确认]" if spec.high_risk else ""
            lines.append(f"- {spec.name}({sig})：{spec.description}{flag}")
        high = self.high_risk_actions()
        if high:
            lines.append(
                "注意：" + "、".join(sorted(high)) + " 属于高风险动作，"
                "你只能向用户提议，不能直接执行；是否生效由高风险审核流程决定。"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 调用
    # ------------------------------------------------------------------
    def call(
        self, name: str, args: dict | None = None, *, principal_id: str | None = None
    ) -> ToolCall:
        """执行工具。永远返回 ToolCall，永远不抛异常。"""
        # 模型偶尔会把 action_input 生造成数组或字符串（function-calling 的
        # 经典翻车点）。在这里收敛成空 dict，让它走"缺必填参数"的正常报错路径，
        # 而不是在 dict() 上抛 TypeError 把整条链路带下去。
        args = _coerce_args(args)
        t0 = time.perf_counter()

        if principal_id is not None and name in _ACCOUNT_ORDER_TOOLS:
            return self._record(
                ToolCall(
                    name=name,
                    args=args,
                    ok=False,
                    error="ORDER_ACCOUNT_UNLINKED",
                    result={
                        "message": "您的账号暂未关联订单信息，暂时无法查询或办理订单业务。",
                        "code": "ORDER_ACCOUNT_UNLINKED",
                    },
                    elapsed_ms=_ms(t0),
                    attempts=0,
                )
            )

        fn = self._fns.get(name)
        if fn is None:
            # 未知工具名是最常见的模型幻觉形态，必须返回可解释的错误码，
            # 而不是 KeyError —— 上层要拿它去回灌给模型让它换个动作。
            return self._record(
                ToolCall(
                    name=name,
                    args=args,
                    ok=False,
                    error=ToolError.UPSTREAM_ERROR.value,
                    result={"message": f"未注册的工具：{name}", "available": self.names()},
                    elapsed_ms=_ms(t0),
                    # 0 而不是 1：一次都没真正执行过。审计里"尝试了几次"
                    # 必须反映对下游的真实压力，不能凑数。
                    attempts=0,
                )
            )

        # 参数对不上时早退 —— 重试再多次也不会让 schema 变对，
        # 白白等满超时窗口对用户更糟（体验上表现为"卡住"）。
        problem = _arg_problem(self._specs[name], fn, args)
        if problem:
            code, message = problem
            return self._record(
                ToolCall(
                    name=name,
                    args=args,
                    ok=False,
                    error=ToolError.UPSTREAM_ERROR.value,
                    result={
                        "code": code,
                        "message": message,
                        "expected": _accepted_params(self._specs[name]),
                    },
                    elapsed_ms=_ms(t0),
                    attempts=1,
                )
            )

        timeout_s = self._timeout_override or float(self.settings.tool_timeout_s)
        call_args = self._with_ambient_date(name, args)
        ok, payload, attempts = self._invoke(fn, name, call_args, timeout_s, t0)
        elapsed = _ms(t0)

        if ok:
            if isinstance(payload, dict) and payload.get(_KEY_OK) is False:
                # 业务拒绝：不改判成异常，只把错误码原样带出去。
                return self._record(
                    ToolCall(
                        name=name,
                        args=args,
                        ok=False,
                        error=str(payload.get(_KEY_ERROR) or ToolError.UPSTREAM_ERROR.value),
                        result={k: v for k, v in payload.items() if k != _KEY_OK},
                        elapsed_ms=elapsed,
                        attempts=attempts,
                    )
                )
            if isinstance(payload, dict) and _KEY_DATA in payload:
                # 业务工具的通用返回形状 {"ok": True, "data": ...} ——
                # 剥掉信封，让上层拿到的 result 就是数据本身。
                result: Any = payload[_KEY_DATA]
            else:
                result = payload
            return self._record(
                ToolCall(name=name, args=args, ok=True, result=result, elapsed_ms=elapsed, attempts=attempts)
            )

        error, exc = payload
        return self._record(
            ToolCall(
                name=name,
                args=args,
                ok=False,
                error=error,
                result={
                    "message": _friendly(name, error),
                    "code": error,
                    # 异常细节保留在结果里而不是丢掉 —— 排障时这一行往往就是全部线索。
                    "exception": f"{type(exc).__name__}: {exc}"[:300] if exc else None,
                },
                elapsed_ms=elapsed,
                attempts=attempts,
            )
        )

    def _with_ambient_date(self, name: str, args: dict) -> dict:
        """把注册表上的 `today` 透明地注入到支持它的工具里。

        为什么需要这一层：高风险子流程调用工具时不该关心"今天是几号"
        （那是环境事实，不是业务参数），但测试又必须能钉死日期。
        让 registry 持有这个环境值，调用方就只需要在**一个地方**设置它（测试夹具），
        业务调用点保持干净。已显式传了 today 的调用以调用方为准。
        """
        if self.today is None or "today" in args:
            return args
        fn = self._fns.get(name)
        if fn is None:
            return args
        try:
            accepted = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return args
        if "today" not in accepted:
            return args
        return {**args, "today": self.today}

    # ------------------------------------------------------------------
    def _invoke(
        self, fn: Callable[..., Any], name: str, args: dict, timeout_s: float, t0: float
    ) -> tuple[bool, Any, int]:
        """执行 + 超时 + 重试。

        超时用**剩余总预算**而不是每个 attempt 都给满：3 次 × 5 秒 = 15 秒
        是用户无法接受的等待，而"总预算 5 秒、用不完就少试几次"才是对的。
        最后那个分支（预算耗尽但一次都没跑完）保证 attempts 永远 >= 1，
        审计里不会出现"调了 0 次"这种解释不通的记录。
        """
        deadline = t0 + timeout_s
        max_attempts = self._max_attempts_override or max(1, int(self.settings.tool_max_retries))
        attempts = 0
        last: BaseException | None = None

        # 不用 `with ThreadPoolExecutor(...)`：with 退出时会 wait=True，
        # 而超时的任务可能一直阻塞——那反而把"超时保护"变成了"无限等"。
        # 这里自己 shutdown(wait=False)，让它成为游离线程，调用方立刻拿到结果。
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tool-{name}")
        try:
            for _ in range(max_attempts):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                attempts += 1
                future = pool.submit(_call_safely, fn, args)
                try:
                    # 每个 attempt 的等待上限也受总预算约束。
                    value, exc = future.result(timeout=remaining)
                except FutureTimeoutError:
                    return False, (ToolError.TIMEOUT.value, None), attempts
                if exc is None:
                    return True, value, attempts
                last = exc
        finally:
            # Python 3.9+ 起 cancel_futures 只取消未开始的任务，不影响已跑的。
            pool.shutdown(wait=False, cancel_futures=True)

        if attempts == 0:
                # 预算在首次尝试前就已用尽（或配置成 0 秒）。
                # 把它标成 TIMEOUT 是唯一诚实的解释：确实是等超时了。
                return False, (ToolError.TIMEOUT.value, None), 1

        return False, (ToolError.UPSTREAM_ERROR.value, last), attempts

    def _record(self, call: ToolCall) -> ToolCall:
        self.audit.append(call)
        return call

    def drain_audit(self) -> list[ToolCall]:
        """取走并清空审计流水 —— 交给 TraceStep / 落库。"""
        out = list(self.audit)
        self.audit.clear()
        return out


# ======================================================================
# 模块级小工具
# ======================================================================


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 3)


def _coerce_args(args: Any) -> dict:
    """把模型可能给歪的 action_input 收敛成 dict。

    宁可把它变成空参数、走"缺必填参数"的清晰报错，也不要在这里抛异常 ——
    工具层的整个存在意义就是不让任何东西冒到调用方。
    """
    if args is None:
        return {}
    if isinstance(args, dict):
        return dict(args)
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _call_safely(fn: Callable[..., Any], args: dict) -> tuple[Any, BaseException | None]:
    """在工作线程里把异常收成值。

    必须在这里就 catch：异常如果从 future 里抛出，会被 `future.result()`
    重新抛到调用方，那就破坏了"异常不外泄"这条底线。
    """
    try:
        return fn(**args), None
    except BaseException as exc:  # noqa: BLE001 — 有意收窄所有失败形态
        return None, exc


def _accepted_params(spec: ToolSpec) -> list[str]:
    return list((spec.parameters or {}).get("properties") or {})


def _arg_problem(
    spec: ToolSpec, fn: Callable[..., Any], args: dict
) -> tuple[str, str] | None:
    """参数检查。返回 (场景码, 说明)，None 表示可以执行。

    先查必填再查参数名：调用方最需要知道的是"你少给了一个必需的参数"，
    而不是"你给了一个我不认识的参数"。
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    accepted = set(sig.parameters)

    declared_required = set((spec.parameters or {}).get("required") or [])
    missing = sorted(
        n
        for n in declared_required
        # 有默认值的实现参数不算缺 —— schema 的 required 描述的是"模型必须给"，
        # 而实现给了默认值就意味着它可以被省略。
        if n not in args and sig.parameters[n].default is inspect.Parameter.empty
    )
    if missing:
        return (
            "MISSING_ARGUMENTS",
            "缺少必需参数：" + "、".join(missing) + "。",
        )

    unknown = sorted(k for k in args if k not in accepted)
    if unknown:
        return (
            "UNKNOWN_ARGUMENTS",
            "参数不匹配：" + "、".join(unknown) + "；可用参数：" + "、".join(sorted(accepted)),
        )
    return None


def _validate_binding(spec: ToolSpec, fn: Callable[..., Any]) -> list[str]:
    """注册期自检：schema 声明的参数必须真能被实现接住。

    只返回问题清单、不抛异常 —— 一个工具写错不该让整个进程起不来。
    全量校验留在测试里做（test_tools.py 会断言这个清单为空）。
    """
    problems: list[str] = []
    declared = set(_accepted_params(spec))
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return problems
    accepted = set(sig.parameters)
    for name in sorted(declared - accepted):
        problems.append(f"{spec.name}: schema 声明了参数 {name}，实现未接受")
    return problems


def _friendly(name: str, error: str) -> str:
    if error == ToolError.TIMEOUT.value:
        return f"工具 {name} 响应超时，请稍后再试。"
    return f"工具 {name} 执行失败，请稍后再试。"
