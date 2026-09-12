"""工作流的共享状态。

LangGraph 的每个节点都读写同一个 `AgentState`。把状态定义单独拿出来，
是因为它是"图"和"各模块"之间唯一的耦合面：节点只做两件事 —— 读状态、
往状态里追加自己的产出。这样每个节点都能被单独测试。

约定：节点**只追加，不覆盖**。`trace` 和 `path` 是审计的骨架，一旦丢步，
"一条请求还原成一棵调用树"就不成立了。
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from ..domain import Exit, FlywheelSource, Intent, RefusalReason
from ..schemas import (
    ClassifyResult,
    Evidence,
    GateDecision,
    ResolveResult,
    RetrievalResult,
    SlotExtraction,
    TokenUsage,
    ToolCall,
    TraceStep,
    Turn,
)


def _append(left: list, right: list) -> list:
    """list 字段的 reducer —— 让并行分支各自追加后能合起来。

    没有它，LangGraph 在两条边汇合时会把其中一条的结果整个丢掉。
    这是个静默 bug：图跑得通，日志却少了一半节点。
    """
    return (left or []) + (right or [])


def _merge_usage(left: TokenUsage | None, right: TokenUsage | None) -> TokenUsage:
    """TokenUsage 的 reducer —— 并行分支的成本要累加，不能互相覆盖。"""
    if left is None:
        return right or TokenUsage()
    if right is None:
        return left
    return left.add(right)


class AgentState(TypedDict, total=False):
    """一条用户消息在流水线里流转时携带的全部信息。"""

    # ---------------- 输入 ----------------
    request_id: str
    session_id: str
    user_id: str
    question: str          # 用户原话（半截话也算）
    history: list[Turn]    # 本次会话的历史轮次

    # ---------------- ③ 指代消解 + 意图识别 ----------------
    resolved_question: str  # 补全后的完整问题
    resolve: ResolveResult
    intent: Intent
    intent_confidence: float
    classify: ClassifyResult
    slots: SlotExtraction

    # ---------------- ④ 分流 ----------------
    exit: Exit

    # ---------------- 检索 ----------------
    retrieval: RetrievalResult

    # ---------------- 闸门 ----------------
    gate: GateDecision

    # ---------------- 工具 / Agent ----------------
    tool_calls: Annotated[list[ToolCall], _append]

    # ---------------- ⑤ 输出 ----------------
    answer: str
    refusal_reason: RefusalReason | None
    handoff: bool

    # ---------------- 高风险子流程的中断 ----------------
    #: 缺槽位时给用户的追问话术；非空即代表流程暂停在这里
    interrupt_prompt: str | None
    missing_slots: list[str]
    #: 用户补充的信息。挂 checkpointer 时由 Command(resume=...) 注入，
    #: 高风险节点把它当作"本轮问题"继续执行。
    interrupt_resume: str | None
    #: 用户补充信息后，从哪个节点继续
    resume_from: str | None

    # ---------------- 飞轮 ----------------
    flywheel_source: FlywheelSource | None

    # ---------------- 记分牌 ----------------
    #: 经过的节点序列 —— 面试演示时一眼看出分流走对没有
    path: Annotated[list[str], _append]
    trace: Annotated[list[TraceStep], _append]
    usage: Annotated[TokenUsage, _merge_usage]
    #: 出错时记一笔，但流程继续（异常不外泄）
    errors: Annotated[list[str], _append]


def new_state(
    *,
    request_id: str,
    session_id: str,
    user_id: str,
    question: str,
    history: list[Turn] | None = None,
) -> AgentState:
    """构造一个干净的初始状态。

    显式把所有 list/dict 字段初始化成空容器：LangGraph 的 reducer 在字段
    缺失和字段为空时的行为不一样，靠默认值容易踩坑。
    """
    return AgentState(
        request_id=request_id,
        session_id=session_id,
        user_id=user_id,
        question=question,
        history=list(history or []),
        resolved_question="",
        tool_calls=[],
        path=[],
        trace=[],
        usage=TokenUsage(),
        errors=[],
        missing_slots=[],
        answer="",
        handoff=False,
        refusal_reason=None,
        interrupt_prompt=None,
        interrupt_resume=None,
        resume_from=None,
        flywheel_source=None,
    )


#: 恢复执行时，每个中断点对应的落点节点。
#: 抽成一张表，是为了让"补完信息从哪继续"这件事显式可见，
#: 而不是散落在节点的 if/else 里。
RESUME_TARGETS: dict[str, str] = {
    "missing_order_id": "high_risk_flow",
    "missing_address": "high_risk_flow",
}


def state_summary(state: AgentState) -> dict[str, Any]:
    """给 Langfuse / 审计日志用的轻量摘要 —— 不打印全量证据，避免日志爆炸。"""
    retrieval = state.get("retrieval")
    return {
        "request_id": state.get("request_id"),
        "question": state.get("question"),
        "resolved_question": state.get("resolved_question"),
        "intent": _v(state.get("intent")),
        "exit": _v(state.get("exit")),
        "path": state.get("path", []),
        "handoff": state.get("handoff", False),
        "refusal_reason": _v(state.get("refusal_reason")),
        "evidence_count": len(retrieval.evidences) if retrieval else 0,
        "tool_calls": [tc.name for tc in state.get("tool_calls", [])],
        "errors": state.get("errors", []),
    }


def _v(x: Any) -> Any:
    """枚举取 value，其它原样返回 —— 方便直接序列化成 JSON。"""
    return x.value if hasattr(x, "value") else x
