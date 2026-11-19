"""图节点。

每个节点只做三件事：读状态 → 调一个模块 → 往状态里追加自己的产出。

三条贯穿所有节点的纪律：

1. **节点不抛异常。** 流水线的价值在于"部分失败也能给出有用结果"：
   精排挂了还有召回结果，模型挂了还有兜底话术。异常冒到图外面，
   用户拿到的就是一个 500，而不是一句"我暂时查不到，帮你转人工"。

2. **每步都写 trace 和 path。** 一条请求要能还原成一棵调用树，
   靠的就是节点自己记账 —— 事后从返回值反推是还原不出来的。

3. **节点之间不共享可变状态。** 只通过 AgentState 传值，
   这样任意节点都能被单独替换或测试。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ..domain import Exit, Intent, NodeName, RefusalReason, ToolError
from ..generation import current_generation
from ..llm import LLMResponse, extract_json
from ..schemas import (
    Evidence,
    GateDecision,
    RetrievalResult,
    SelfEvalResult,
    SlotExtraction,
    TokenUsage,
    ToolCall,
    TraceStep,
    Turn,
)
from .state import AgentState


def _usage_from(llm_response: Any) -> TokenUsage:
    u = getattr(llm_response, "usage", None)
    return u if isinstance(u, TokenUsage) else TokenUsage()


def _step(node: str, **details: Any) -> TraceStep:
    return TraceStep(node=node, details=details)


# ============================================================
# ③ 指代消解 + 意图识别
# ============================================================


def make_resolve_node(deps) -> Any:
    def resolve_node(state: AgentState) -> dict[str, Any]:
        q = state.get("question", "")
        history = state.get("history", [])
        memories = state.get("memories", [])
        try:
            result = deps.resolver.resolve(q, history, memory=memories)
        except Exception as exc:  # noqa: BLE001 — 消解失败不该阻断问答
            from ..schemas import ResolveResult

            result = ResolveResult(original=q, resolved=q, rewritten=False)
            return {
                "resolved_question": q,
                "resolve": result,
                "path": [NodeName.RESOLVE.value],
                "trace": [
                    _step(NodeName.RESOLVE.value, error=str(exc), fallback=True)
                ],
                "errors": [f"resolve: {exc}"],
            }

        return {
            "resolved_question": result.resolved or q,
            "resolve": result,
            "path": [NodeName.RESOLVE.value],
            "trace": [
                _step(
                    NodeName.RESOLVE.value,
                    original=result.original,
                    resolved=result.resolved,
                    rewritten=result.rewritten,
                    substitutions=result.substitutions,
                    # 记下"这一轮有几条用户档案参与"—— 排查"它怎么知道我的单号"
                    # 时，这是唯一的线索
                    memory_items=len(memories),
                )
            ],
        }

    return resolve_node


def make_classify_node(deps) -> Any:
    def classify_node(state: AgentState) -> dict[str, Any]:
        text = state.get("resolved_question") or state.get("question", "")
        history = state.get("history", [])
        try:
            classify = deps.classifier.classify(text)
            slots = deps.classifier.extract_slots(state.get("question", ""), history)
        except Exception as exc:  # noqa: BLE001
            from ..domain import INTENT_LABELS
            from ..schemas import ClassifyResult

            classify = ClassifyResult(
                intent=Intent.POLICY_FAQ,
                confidence=0.0,
                label=INTENT_LABELS.get(Intent.POLICY_FAQ, ""),
                method="fallback",
            )
            slots = SlotExtraction()
            return {
                "intent": classify.intent,
                "intent_confidence": 0.0,
                "classify": classify,
                "slots": slots,
                "exit": INTENT_TO_EXIT_FALLBACK,
                "path": [NodeName.CLASSIFY.value],
                "trace": [_step(NodeName.CLASSIFY.value, error=str(exc), fallback=True)],
                "errors": [f"classify: {exc}"],
            }

        from ..domain import INTENT_TO_EXIT

        exit_ = INTENT_TO_EXIT.get(classify.intent, Exit.FALLBACK)
        return {
            "intent": classify.intent,
            "intent_confidence": classify.confidence,
            "classify": classify,
            "slots": slots,
            "exit": exit_,
            "path": [NodeName.CLASSIFY.value],
            "trace": [
                _step(
                    NodeName.CLASSIFY.value,
                    intent=classify.intent.value,
                    confidence=classify.confidence,
                    method=classify.method,
                    matched=classify.matched,
                    exit=exit_.value,
                )
            ],
        }

    return classify_node


#: 分类降级时的出口。走知识检索而不是兜底，因为"答不上来"比"直接放弃"更有用；
#: 真正的兜底由闸门在检索之后决定。
INTENT_TO_EXIT_FALLBACK = Exit.KNOWLEDGE


# ============================================================
# 条件边的路由函数
# ============================================================


def route_by_exit(state: AgentState) -> str:
    """按出口分流 —— 图上最重要的那道岔口。

    九类意图收敛到五个出口，收敛的意义在这里体现：图只有五条边，
    而不是九条。新增一类意图时，只要它落进已有出口，图一行都不用改。
    """
    exit_ = state.get("exit") or Exit.FALLBACK
    return {
        Exit.KNOWLEDGE: "retrieve",
        Exit.BUSINESS_TOOL: "agent",
        Exit.HIGH_RISK: "high_risk_flow",
        Exit.HANDOFF: "handoff",
        Exit.FALLBACK: "retrieve",
    }.get(exit_, "retrieve")


# ============================================================
# 检索
# ============================================================


def make_retrieve_node(deps) -> Any:
    def retrieve_node(state: AgentState) -> dict[str, Any]:
        query = state.get("resolved_question") or state.get("question", "")
        try:
            result = deps.retriever.retrieve(query)
        except Exception as exc:  # noqa: BLE001 — 检索挂了要能退化成"无证据"，而不是崩
            result = RetrievalResult(query=query, strategy="failed")
            # 检索失败 = 一条证据都没有 = 闸门必然拒答，但拒答话术得是人话。
            # 不在这里给 answer 的话，handoff 节点会输出一句通用的转人工，
            # 用户完全不知道发生了什么。
            return {
                "retrieval": result,
                "answer": "抱歉，知识库暂时查询不通，我帮你转人工核实一下。",
                "path": [NodeName.RETRIEVE.value],
                "trace": [_step(NodeName.RETRIEVE.value, error=str(exc), fallback=True)],
                "errors": [f"retrieve: {exc}"],
            }

        return {
            "retrieval": result,
            "path": [NodeName.RETRIEVE.value],
            "trace": [
                _step(
                    NodeName.RETRIEVE.value,
                    strategy=result.strategy,
                    dense_hits=result.dense_hits,
                    sparse_hits=result.sparse_hits,
                    fused_hits=result.fused_hits,
                    reranked_hits=result.reranked_hits,
                    top_score=round(result.top_score, 4),
                    # 只记 chunk_id 和标题，不记全文 —— 日志不该被证据正文撑爆
                    chunks=[
                        {"chunk_id": e.chunk_id, "title": e.chunk.title, "score": round(e.score, 4)}
                        for e in result.evidences
                    ],
                )
            ],
        }

    return retrieve_node


def make_gate_node(deps) -> Any:
    def gate_node(state: AgentState) -> dict[str, Any]:
        try:
            decision = deps.gate.check(
                state.get("resolved_question") or state.get("question", ""),
                state.get("retrieval"),
                state.get("intent"),
                state.get("slots"),
            )
        except Exception as exc:  # noqa: BLE001 — 闸门自己坏了，宁可不放行
            decision = GateDecision.reject(
                0.0, RefusalReason.NO_EVIDENCE, {}, "系统校验异常，帮你转人工确认。"
            )
            return {
                "gate": decision,
                "refusal_reason": decision.reason,
                "path": [NodeName.GATE.value],
                "trace": [_step(NodeName.GATE.value, error=str(exc), fallback=True)],
                "errors": [f"gate: {exc}"],
            }

        return {
            "gate": decision,
            "refusal_reason": None if decision.passed else decision.reason,
            "path": [NodeName.GATE.value],
            "trace": [
                _step(
                    NodeName.GATE.value,
                    passed=decision.passed,
                    confidence=round(decision.confidence, 4),
                    reason=decision.reason.value if decision.reason else None,
                    # 四个判据逐条记下来 —— "为什么判它不过"要能一眼看到
                    checks={k: round(v, 4) for k, v in decision.checks.items()},
                    checked_before_generation=True,
                )
            ],
        }

    return gate_node


def gate_route(state: AgentState) -> str:
    """闸门之后走生成还是转人工。"""
    gate = state.get("gate")
    return "generate" if (gate and gate.passed) else "handoff"


# ============================================================
# 主力 Agent（ReAct）
# ============================================================


def make_agent_node(deps) -> Any:
    """主力 Agent：ReAct 循环。

    **循环上限是必需的。** 模型可以一直在"再查一个工具"里打转，
    没有上限的话一次请求能烧掉几十次调用，而用户只是问了个物流。
    """

    def agent_node(state: AgentState) -> dict[str, Any]:
        question = state.get("resolved_question") or state.get("question", "")

        # 缺必填槽位时先追问，而不是让 Agent 拿 None 去调工具。
        # 不拦这一道，用户说"我的订单到哪了"会收到"没有查到这个订单号"——
        # 他没报错任何东西，只是我们没问。这条判断和高风险子流程共用
        # 同一张槽位表（`SlotExtraction.missing_for`），"哪些意图必须有订单号"
        # 只能有一份定义。
        missing = _missing_slots(state)
        if missing:
            from .high_risk import interrupt_prompt

            prompt = interrupt_prompt(state.get("intent") or Intent.POLICY_FAQ, missing)
            return {
                "answer": prompt,
                "interrupt_prompt": prompt,
                "missing_slots": missing,
                "handoff": False,
                "path": [NodeName.AGENT.value],
                "trace": [
                    _step(NodeName.AGENT.value, skipped=True, missing=missing, reason="missing_slot")
                ],
            }

        specs = deps.tools.specs()
        calls: list[ToolCall] = []
        usage = TokenUsage()
        steps_detail: list[dict[str, Any]] = []
        errors: list[str] = []

        scratch: list[dict[str, Any]] = [{"role": "user", "content": question}]
        for step in range(deps.max_agent_steps):
            try:
                resp = deps.llm.chat(scratch, tools=specs, task="plan_tool")
                usage = usage.add(_usage_from(resp))
                actions = _read_plan(resp)
            except Exception as exc:  # noqa: BLE001 — 规划失败就收手，不用模型兜
                errors.append(f"agent: {exc}")
                break

            steps_detail.append(
                {"step": step, "actions": [{"action": a, "args": d} for a, d in actions]}
            )
            if not actions:
                break

            # 高风险工具不允许 Agent 直接调用 —— 只能由 high_risk_flow 决定。
            # 这是"业务铁律写死在图上"最具体的一处体现。
            #
            # 先整轮扫一遍再决定执行与否，而不是"能跑的先跑、危险的丢一边"：
            # 模型越过这条线的时候，这一轮的规划整体已经不可信了，只执行其中
            # "看起来安全"的那几个，等于让越界的模型拿到一半的执行权。
            risky = [(a, d) for a, d in actions if a in deps.tools.high_risk_actions()]
            if risky:
                calls.extend(
                    ToolCall(name=a, args=dict(d), ok=False, error="HIGH_RISK_REQUIRES_FLOW")
                    for a, d in risky
                )
                steps_detail[-1]["blocked"] = "high_risk"
                break

            executed = [
                (
                    deps.tools.call(
                        action,
                        dict(action_input),
                        principal_id=state.get("user_id") or None,
                    )
                    if deps.enforce_order_ownership
                    else deps.tools.call(action, dict(action_input))
                )
                for action, action_input in actions
            ]
            calls.extend(executed)
            # 把这一步的动作与结果写回对话，供下一轮继续推理。
            # 正文为空（模型只用 function calling 说话）时要补一句动作本身 ——
            # 空着一轮，下一步就失去了"我刚要干什么"的线索，只能对着工具结果猜。
            scratch.append({"role": "assistant", "content": resp.text or _plan_text(actions)})
            scratch.append(
                {
                    "role": "user",
                    "content": "\n".join(
                        f"工具 {c.name} 返回：{_brief(c.result if c.ok else c.error)}"
                        for c in executed
                    ),
                }
            )
        else:
            errors.append("agent: 达到最大步数上限")

        return {
            "tool_calls": calls,
            "usage": usage,
            "path": [NodeName.AGENT.value],
            "trace": [
                _step(
                    NodeName.AGENT.value,
                    steps=steps_detail,
                    tool_calls=[{"name": c.name, "ok": c.ok, "error": c.error} for c in calls],
                    max_steps=deps.max_agent_steps,
                )
            ],
            "errors": errors,
        }

    return agent_node


def _read_plan(resp: LLMResponse) -> list[tuple[str, dict[str, Any]]]:
    """从一次规划响应里读出这一步要调哪些工具、参数是什么。

    两种约定都要认，而且**必须**都认：

      * **function calling** —— 真模型按 OpenAI 协议把动作放在 `tool_calls` 里，
        正文往往是空的、或者是一句"我先查一下订单"的旁白；
      * **正文 JSON** —— 离线 mock 把 `{"action": ..., "action_input": ...}`
        写在正文里。

    只读正文的话，接上真模型会**安静地一个工具都不调**：意图、出口、置信度
    全都正常，就是没有数据，最后生成节点只能答"没有可用的证据"。而这条路径
    离线跑永远不复现 —— mock 永远走的是第二种约定。

    function calling 优先：正文此时本来就可能没有可解析的内容。
    返回空列表表示"模型认为可以收尾了"。
    """
    if resp.tool_calls:
        return [
            (str(call.get("name") or ""), _as_args(call.get("arguments")))
            for call in resp.tool_calls
            if call.get("name")
        ]
    try:
        plan = extract_json(resp.text)
    except Exception:  # noqa: BLE001 — 解析不出计划就是"没有下一步"，不是错误
        return []
    if not isinstance(plan, dict):
        return []
    action = plan.get("action")
    if not action or action == "finish":
        return []
    return [(str(action), _as_args(plan.get("action_input")))]


def _as_args(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _plan_text(actions: Sequence[tuple[str, dict[str, Any]]]) -> str:
    """把这一步的动作写成一句模型看得懂的记录，回灌进对话。"""
    return json.dumps(
        [{"action": a, "action_input": d} for a, d in actions], ensure_ascii=False
    )


def _brief(value: Any, limit: int = 400) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


def _missing_slots(state: AgentState) -> list[str]:
    """这次请求走到 Agent 之前，还缺哪些必填槽位。

    用的是**补全后**的问题再抽一次单号，这一点是必须的：指代消解（包括长期记忆
    那一层）存在的意义就是把"我的订单"补成"订单 ORD-1002"。拿用户原话判，
    记忆刚补上来的单号会被当成"缺失"，于是系统明明知道是哪一单，
    却回一句"请把订单号发我"—— 记忆功能在这里就白做了。
    """
    intent = state.get("intent")
    slots = state.get("slots")
    if intent is None or slots is None:
        return []
    from .high_risk import extract_order_id

    resolved = state.get("resolved_question") or state.get("question", "")
    order_id = slots.order_id or extract_order_id(resolved)
    return slots.model_copy(update={"order_id": order_id}).missing_for(intent)


def _render_tool_answer(calls: list[ToolCall]) -> str:
    """把工具结果翻成给用户的人话。

    和 high_risk 子流程同一原则：这是"查到了什么"的事实陈述，不交给模型写。
    模型写只会引入"工具说待付款、模型说已发货"这类不一致；这里直接引用字段值。

    每个工具的结果都是 dict，带自己的 `status_label` / `note` 字段，优先用它们。
    这样新增一个业务工具时，只要它返回规范的 dict，这里就不用改。

    **有成功的就只渲染成功的。** 这一轮里失败的那些，多半已经被模型自己纠正
    （它换个工具名或补上参数再调一次），把"系统暂时不可用"拼进答复等于让用户
    读一段他不需要知道的故障前言 —— 数据是对的、前言是假的，整条读起来像系统坏了。
    全部失败才算真的没办成，那时候才渲染错误。
    """
    if not calls:
        return "没有可用的查询结果。"

    successful = [c for c in calls if c.ok]
    renderable = successful or calls

    parts: list[str] = []
    for call in renderable:
        if not call.ok:
            parts.append(_tool_error_text(call.error))
            continue
        data = call.result if isinstance(call.result, dict) else {}
        note = data.get("note")
        status_label = data.get("status_label")
        if note:
            parts.append(note)
        elif status_label:
            # 有状态标签但没有现成话术的，拼一句最小的状态陈述
            oid = data.get("order_id", "")
            parts.append(f"订单 {oid} 当前状态：{status_label}。")

    if parts:
        return "\n".join(parts)
    return "已经查到相关信息，具体的我帮你转人工核对一下。"


#: 服务侧故障 —— 重试也不会好，这一轮是真的没办成，该转人工。
#: 与业务侧拒绝（订单不存在、不可取消、不在退款条件内）区分开：那是"查到了，
#: 结果是不同意"，用户知道下一步该干什么，再转人工只会多一道没人接的工序。
#:
#: 注意 `ToolError.UPSTREAM_ERROR` 同时承载"下游真的挂了"和"模型把工具名
#: 写错了"两种成因（见 tools/registry.py 的 `call`）—— 后者会在多步循环里
#: 被模型自己纠正，所以判升级时还要看**有没有任何一条成功**，不能只看错误码。
_SERVICE_ERRORS = frozenset({ToolError.TIMEOUT.value, ToolError.UPSTREAM_ERROR.value})


def _should_escalate(calls: list[ToolCall]) -> bool:
    """这一轮工具全都没办成，且至少一条是服务侧故障 → 升级转人工。"""
    if any(c.ok for c in calls):
        return False
    return any((c.error or "") in _SERVICE_ERRORS for c in calls)


#: 工具错误码 → 人话。工具名对用户是内部实现细节，不该泄漏到答复里。
#:
#: **这里的每一句都不许承诺"已帮你转人工"。** 这句话只出现在答复正文里，
#: 而 agent 出口的 handoff 标志是由 `_should_escalate` 单独决定的 ——
#: 正文说转了、标志是 false，用户就会读到一个不存在的转接（前端只在
#: handoff=true 时才补那条"已为你转接人工客服"提示条）。真要转人工时，
#: 由节点把 handoff 置真、再由 handoff 节点补那句提示，正文不抢这个话。
_TOOL_ERROR_TEXT: dict[str, str] = {
    "ORDER_ACCOUNT_UNLINKED": "您的账号暂未关联订单信息，暂时无法查询或办理订单业务。",
    "ORDER_NOT_FOUND": "没有查到这个订单号，麻烦核对一下再发我（形如 ORD-1007）。",
    "ORDER_NOT_CANCELLABLE": "这笔订单已经发货了，没法直接取消。",
    "ORDER_ADDRESS_LOCKED": "订单已发出，收货地址改不了了。",
    "REFUND_NOT_ELIGIBLE": "这笔订单不在退款条件内，需要的话我可以帮你转人工复核。",
    "TIMEOUT": "系统查询超时了，请稍后再试。",
    "UPSTREAM_ERROR": "系统暂时没查出来，请稍后再试。",
}


def _tool_error_text(error: str | None) -> str:
    if error and error in _TOOL_ERROR_TEXT:
        return _TOOL_ERROR_TEXT[error]
    return "暂时没查到，麻烦核对一下信息再发我。"


# ============================================================
# 高风险子流程（含中断）
# ============================================================


def make_high_risk_node(deps) -> Any:
    from langgraph.types import interrupt

    from .high_risk import run_high_risk_flow

    def high_risk_node(state: AgentState) -> dict[str, Any]:
        intent = state.get("intent") or Intent.POLICY_FAQ
        question = state.get("question", "")

        principal_id = (state.get("user_id") or None) if deps.enforce_order_ownership else None
        outcome = run_high_risk_flow(intent, question, deps.tools, principal_id=principal_id)

        if outcome.interrupted and not deps.checkpointer:
            # 没挂 checkpointer 就没有"挂起"这回事（interrupt() 会直接抛异常，
            # 把整条请求打成 500）。这时正确的行为是**把追问当作答复返回**：
            # 用户的诉求没被满足，但他明确知道下一步该给什么。
            return {
                "interrupt_prompt": outcome.interrupt_prompt,
                "missing_slots": outcome.missing,
                "answer": outcome.interrupt_prompt,
                "handoff": False,
                "path": [NodeName.HIGH_RISK_FLOW.value],
                "trace": [
                    _step(
                        NodeName.HIGH_RISK_FLOW.value,
                        interrupted=True,
                        missing=outcome.missing,
                        prompt=outcome.interrupt_prompt,
                    )
                ],
            }

        if outcome.interrupted:
            # 挂了 checkpointer：真的把图挂在这一行。状态被原样存下，
            # 用户补充信息后从本节点开头继续 —— 不是重跑整条流水线。
            # 恢复时 LangGraph 会把"补充内容 + 原状态"一起回放给本节点，
            # 而补充内容本身就是新的 question，所以直接重跑子流程即可。
            payload = interrupt({"prompt": outcome.interrupt_prompt, "missing": outcome.missing})
            resumed_question = payload if isinstance(payload, str) else question
            outcome = run_high_risk_flow(
                intent, resumed_question, deps.tools, principal_id=principal_id
            )

        escalate = bool(outcome.error) and outcome.error in _ESCALATE
        return {
            "interrupt_prompt": None,
            "missing_slots": [],
            "tool_calls": outcome.tool_calls,
            "answer": outcome.answer or "",
            "handoff": escalate,
            "refusal_reason": RefusalReason.HIGH_RISK_UNAPPROVED if outcome.error else None,
            "path": [NodeName.HIGH_RISK_FLOW.value],
            "trace": [
                _step(
                    NodeName.HIGH_RISK_FLOW.value,
                    interrupted=False,
                    tool_calls=[
                        {"name": c.name, "ok": c.ok, "error": c.error} for c in outcome.tool_calls
                    ],
                    error=outcome.error,
                    escalate=escalate,
                )
            ],
        }

    return high_risk_node


def _esc() -> frozenset[str]:
    from .high_risk import ESCALATE_TO_HUMAN

    return ESCALATE_TO_HUMAN


_ESCALATE = _esc()

#: 高风险流程跑完后：已经给出明确答复的走 END，需要人工的转 handoff
def high_risk_route(state: AgentState) -> str:
    return "handoff" if state.get("handoff") else "__end__"


# ============================================================
# 生成 / 转人工 / 飞轮
# ============================================================


def make_generate_node(deps) -> Any:
    """生成回答。

    两条来路：
      * 从闸门过来 —— 有检索证据，生成前先自评（自评不过就转人工）
      * 从 Agent / 高风险子流程过来 —— 已经有确定性的答案，直接沿用
    """

    def generate_node(state: AgentState) -> dict[str, Any]:
        existing = state.get("answer") or ""
        if existing:
            return {
                "answer": existing,
                "path": [NodeName.GENERATE.value],
                "trace": [_step(NodeName.GENERATE.value, source="deterministic", chars=len(existing))],
            }

        # 工具路径（business_tool 出口）：Agent 已经调过工具，工具结果本身就是答案。
        # 这里不"生成"，只把结构化结果翻成人话 —— 和 high_risk 子流程同一个原则：
        # 事实陈述不交给模型写，避免引入与工具结果不一致的措辞。
        tool_calls = state.get("tool_calls") or []
        if tool_calls:
            answer = _render_tool_answer(tool_calls)
            escalate = _should_escalate(tool_calls)
            return {
                "answer": answer,
                "handoff": escalate,
                # 转人工时给一个拒答原因，飞轮才知道该把这条归到哪个入口 ——
                # 不写的话它落进问题池却标不出"为什么答不上来"，人工审核时
                # 分不清"缺知识"还是"下游挂了"。BUSINESS_TOOL 出口没有闸门，
                # 没有别的来源能补这个字段。
                "refusal_reason": RefusalReason.NO_EVIDENCE if escalate else None,
                "path": [NodeName.GENERATE.value],
                "trace": [
                    _step(
                        NodeName.GENERATE.value,
                        source="tool_result",
                        tool_calls=[c.name for c in tool_calls],
                        chars=len(answer),
                        escalate=escalate,
                    )
                ],
            }

        # 业务工具出口却一个工具都没调成 —— 这不是"答不上来"，是"没查到"。
        #
        # 不能让它掉进下面那条知识库通道：那条通道会拿着**空的证据**去生成
        # （business_tool 出口不经过 retrieve，`retrieval` 是 None），模型只能
        # 编一句"没有依据"，而置信度闸门压根没参与过 —— 它只服务知识出口。
        # 于是一次工具规划的失败，被静默粉饰成一次正常的知识问答。
        if state.get("exit") is Exit.BUSINESS_TOOL:
            return {
                "answer": "抱歉，这笔信息我这边暂时没查到，帮你转人工确认一下。",
                "handoff": True,
                "refusal_reason": RefusalReason.NO_EVIDENCE,
                "path": [NodeName.GENERATE.value],
                "trace": [
                    _step(NodeName.GENERATE.value, source="no_tool_call", fallback=True)
                ],
            }

        retrieval = state.get("retrieval")
        context = retrieval.to_context() if retrieval else ""
        usage = TokenUsage()

        # 用户档案作为**背景**注入，和知识库证据分开成两块。
        # 分开是有意的：证据是"这次回答的依据"（自评要逐条核对它），
        # 档案是"关于这个人的前提"（自评不该要求它出现在证据里）。
        # 混在一块会让自评把"尺码：L"当成"证据里没有的数字"而误判不可信。
        memories = state.get("memories") or []
        memory_block = ""
        if memories:
            memory_block = "已知的用户信息：\n" + "\n".join(
                f"- {m}" for m in memories
            ) + "\n\n"

        generation = current_generation.get()
        if generation is not None:
            generation.prepare(state)
        generate = deps.llm.chat
        if generation is not None:
            generate = lambda messages, **kw: deps.llm.chat_stream(
                messages, on_token=generation.token, **kw
            )
        try:
            resp = generate(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是电商客服。只根据下面给出的知识库证据回答，"
                            "不要引入证据之外的信息。证据不足时直接说无法确认。"
                            "已知的用户信息可以用来理解用户在说哪一单、什么偏好，"
                            "但它同样不是回答依据。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"{memory_block}"
                            f"知识库证据：\n{context}\n\n"
                            f"用户问题：{state.get('resolved_question') or state.get('question', '')}"
                        ),
                    },
                ],
                task="generate",
            )
            answer = resp.text
            usage = usage.add(_usage_from(resp))
        except Exception as exc:  # noqa: BLE001 — 生成失败走兜底话术，不是 500
            if generation is not None and generation.text:
                # 已发出的正文无法撤回，流失败只能报 error，不能把半句存成完整回答。
                raise
            return {
                "answer": "抱歉，我这边暂时没能组织出可靠的答复，帮你转人工确认。",
                "handoff": True,
                "path": [NodeName.GENERATE.value],
                "trace": [_step(NodeName.GENERATE.value, error=str(exc), fallback=True)],
                "errors": [f"generate: {exc}"],
            }

        # 自评仍在生成完成后执行，决定 done 的转接标志；流式正文已逐片发出。
        selfeval = SelfEvalResult(grounded=True, coverage=1.0)
        trace_err: str | None = None
        try:
            selfeval = deps.self_eval.evaluate(answer, context)
        except Exception as exc:  # noqa: BLE001 —— 自评挂了就当"没评过"，不能因此降级答复
            trace_err = str(exc)

        if not selfeval.grounded:
            return {
                "answer": answer,
                "handoff": True,
                "refusal_reason": RefusalReason.SELF_EVAL_FAILED,
                "usage": usage,
                "path": [NodeName.GENERATE.value],
                "trace": [
                    _step(
                        NodeName.GENERATE.value,
                        self_eval=False,
                        coverage=selfeval.coverage,
                        unsupported=selfeval.unsupported_claims,
                        error=trace_err,
                    )
                ],
            }

        return {
            "answer": answer,
            "usage": usage,
            "path": [NodeName.GENERATE.value],
            "trace": [
                _step(
                    NodeName.GENERATE.value,
                    self_eval=True,
                    coverage=selfeval.coverage,
                    chars=len(answer),
                    error=trace_err,
                )
            ],
        }

    return generate_node


def make_handoff_node(deps) -> Any:
    def handoff_node(state: AgentState) -> dict[str, Any]:
        gate = state.get("gate")
        # 已经有答复的（高风险流程业务拒绝等）沿用答复，只补一句转人工
        existing = state.get("answer") or ""
        if existing and state.get("handoff"):
            answer = existing + "\n\n已经帮你转接人工客服进一步处理。"
        elif gate and not gate.passed and gate.message:
            answer = gate.message
        else:
            answer = "这个问题我帮你转人工处理，稍后会有客服同事联系你。"

        return {
            "answer": answer,
            "handoff": True,
            "refusal_reason": state.get("refusal_reason") or (gate.reason if gate and not gate.passed else None),
            "path": [NodeName.HANDOFF.value],
            "trace": [
                _step(
                    NodeName.HANDOFF.value,
                    reason=(gate.reason.value if gate and not gate.passed and gate.reason else None),
                )
            ],
        }

    return handoff_node


def make_flywheel_node(deps) -> Any:
    """把答不上的问题落进问题池 —— 数据飞轮的入口。

    落池时**连同当时的检索快照一起存**。没有快照，人工审核时就没法判断
    "真缺知识"还是"知识有但没检索到"，而这两种情况的处理方式完全不同。
    """

    def flywheel_node(state: AgentState) -> dict[str, Any]:
        if not state.get("handoff"):
            return {"path": [NodeName.FLYWHEEL.value], "trace": [_step(NodeName.FLYWHEEL.value, skipped=True)]}

        from ..flywheel.feedback import source_for
        from ..schemas import FlywheelRecord

        reason = state.get("refusal_reason")
        source = source_for(reason, thumbs_down=False)

        try:
            record = FlywheelRecord(
                record_id=f"{state.get('request_id', 'req')}-fw",
                session_id=state.get("session_id", ""),
                question=state.get("question", ""),
                resolved_question=state.get("resolved_question", ""),
                intent=state.get("intent"),
                source=source,
                refusal_reason=reason,
                retrieval_snapshot=state.get("retrieval"),
            )
            saved = deps.flywheel.record(record)
            detail: dict[str, Any] = {
                "source": source.value,
                "reason": reason.value if reason else None,
                "ask_count": saved.ask_count,
                "has_snapshot": saved.retrieval_snapshot is not None,
            }
            errors: list[str] = []
        except Exception as exc:  # noqa: BLE001 — 落池失败不该影响这次答复
            detail = {"error": str(exc)}
            errors = [f"flywheel: {exc}"]

        return {
            "flywheel_source": source,
            "path": [NodeName.FLYWHEEL.value],
            "trace": [_step(NodeName.FLYWHEEL.value, **detail)],
            "errors": errors,
        }

    return flywheel_node
