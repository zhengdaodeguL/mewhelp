"""Issue #13：固定日期、模型响应和账号，复现退款路由、提示丢失及串单。"""

from __future__ import annotations

import json

import pytest

from mewhelp.config import PROJECT_ROOT, Settings
from mewhelp.domain import Exit, Intent
from mewhelp.llm import LLMResponse, MockLLMClient
from mewhelp.schemas import ClassifyResult, SlotExtraction, ToolCall, Turn
from mewhelp.tools import ToolRegistry, register_business_tools
from mewhelp.tools import business
from mewhelp.workflow.graph import build_deps, build_graph
from mewhelp.workflow.high_risk import run_high_risk_flow
from mewhelp.workflow.nodes import _render_tool_answer, make_agent_node
from mewhelp.workflow.state import new_state


BASE_DATE = "2026-08-20"
REPORT_DATE = "2026-11-20"
ORDER_CASES = [
    ("ORD-1007", "U1003", "OK"),
    ("ORD-1004", "U1002", "OK"),
    ("ORD-1005", "U1003", "OK"),
    ("ORD-1006", "U1003", "REFUND_WINDOW_EXPIRED"),
    ("ORD-1010", "U1005", "REFUND_ALREADY_APPLIED"),
    ("ORD-1011", "U1005", "REFUND_ALREADY_APPLIED"),
    ("ORD-1001", "U1001", "REFUND_STATUS_NOT_DELIVERED"),
    ("ORD-1003", "U1002", "REFUND_STATUS_NOT_DELIVERED"),
    ("ORD-1009", "U1004", "REFUND_STATUS_NOT_DELIVERED"),
    ("ORD-1008", "U1004", "OK"),
]
REFUND_PHRASES = [
    ("我要退款 {order_id}", Intent.REFUND_REQUEST),
    ("订单 {order_id} 可以退款吗？", Intent.REFUND_REQUEST),
    ("{order_id} 能退吗？", Intent.RETURN_REQUEST),
    ("这单 {order_id} 能退款吗？", Intent.REFUND_REQUEST),
    ("订单{order_id}能否退款？", Intent.REFUND_REQUEST),
    ("我想退货 {order_id}", Intent.RETURN_REQUEST),
]


class PlannedLLM(MockLLMClient):
    """只固定工具规划；分类、槽位和指代仍走真实 NLU + 离线模型。"""

    def __init__(self, settings, plans, protocol):
        super().__init__(settings)
        self.plans = list(plans)
        self.protocol = protocol

    def chat(self, messages, **kwargs):
        if kwargs.get("task") != "plan_tool":
            return super().chat(messages, **kwargs)
        self.calls.append({"task": "plan_tool", "messages": list(messages)})
        plan = self.plans.pop(0) if self.plans else []
        if self.protocol == "json":
            if not plan:
                return LLMResponse(text='{"action": "finish"}')
            name, args = plan[0]
            return LLMResponse(text=json.dumps({"action": name, "action_input": args}))
        return LLMResponse(text="", tool_calls=[
            {"id": f"call_{i}", "name": name, "arguments": args}
            for i, (name, args) in enumerate(plan)
        ])


@pytest.fixture
def deps_factory(tmp_path, monkeypatch):
    settings = Settings(
        _env_file=None,
        llm_mode="mock",
        embedding_mode="mock",
        reranker_mode="mock",
        vector_store_mode="memory",
        langfuse_enabled=False,
        MEWHELP_DATABASE_URL=f"sqlite:///{tmp_path / 'issue13.db'}",
        data_dir=PROJECT_ROOT / "data",
        knowledge_dir=PROJECT_ROOT / "data" / "knowledge",
    )
    # 工具加载同一份仓库夹具，避免开发者 .env 指向其他数据集。
    monkeypatch.setattr(business, "get_settings", lambda: settings)

    def factory(*, today=BASE_DATE, plans=None, protocol="function"):
        registry = register_business_tools(ToolRegistry(settings, today=today))
        llm = (
            MockLLMClient(settings)
            if plans is None
            else PlannedLLM(settings, plans, protocol)
        )
        deps = build_deps(settings=settings, llm=llm, tools=registry)
        deps.enforce_order_ownership = True
        return deps

    return factory


def ask(graph, question, user_id, *, history=None):
    return graph.invoke(new_state(
        request_id="issue13-turn",
        session_id="issue13-session",
        user_id=user_id,
        question=question,
        history=history,
    ))


@pytest.mark.parametrize("phrase,intent", REFUND_PHRASES)
@pytest.mark.parametrize("order_id,owner,code", ORDER_CASES)
def test_refund_matrix_routes_to_real_order(deps_factory, phrase, intent, order_id, owner, code):
    deps = deps_factory()
    out = ask(build_graph(deps), phrase.format(order_id=order_id), owner)

    assert out["intent"] is intent, out["classify"]
    assert out["classify"].method == "rule", "明确的订单退款问法不应依赖模型采样"
    assert out["exit"] is Exit.HIGH_RISK
    assert "high_risk_flow" in out["path"]
    call = out["tool_calls"][-1]
    assert call.name == "request_refund"
    assert all(c.args["order_id"] == order_id for c in out["tool_calls"])
    assert call.result["order_id"] == order_id
    assert call.ok is (code == "OK")
    if not call.ok:
        assert call.result["code"] == code
    assert call.result["data"]["items"], "拒绝退款也应保留已找到的订单快照"
    assert out["answer"].startswith(f"订单 {order_id}：")
    assert "暂时没查到" not in out["answer"]


@pytest.mark.parametrize("intent", [Intent.REFUND_REQUEST, Intent.RETURN_REQUEST])
@pytest.mark.parametrize("order_id,owner", [
    ("ORD-1004", "U1002"), ("ORD-1005", "U1003"),
    ("ORD-1007", "U1003"), ("ORD-1008", "U1004"),
])
def test_refund_answer_preserves_eligibility_and_review_note(deps_factory, intent, order_id, owner):
    deps = deps_factory()
    outcome = run_high_risk_flow(intent, f"订单 {order_id}", deps.tools, principal_id=owner)
    call = outcome.tool_calls[-1]
    assert call.ok
    assert outcome.answer.startswith(f"订单 {order_id}：")
    assert "30 天" in outcome.answer
    assert "可以申请退款" in outcome.answer
    assert "人工审核" in outcome.answer
    assert call.result["note"] in outcome.answer
    if order_id in {"ORD-1007", "ORD-1008"}:
        assert "生鲜" in outcome.answer
        assert "24 小时内" in outcome.answer
        assert "照片" in outcome.answer
    for promise in ("已提交", "已为你登记", "1 到 3 个工作日"):
        assert promise not in outcome.answer


@pytest.mark.parametrize("order_id,owner,phrase", [
    ("ORD-1007", "U1003", "订单 {order_id} 可以退款吗？"),
    ("ORD-1008", "U1004", "{order_id} 能退吗？"),
])
def test_expired_fresh_order_at_report_date_keeps_short_window(deps_factory, order_id, owner, phrase):
    deps = deps_factory(today=REPORT_DATE)
    out = ask(build_graph(deps), phrase.format(order_id=order_id), owner)
    call = out["tool_calls"][-1]
    assert call.result["code"] == "REFUND_WINDOW_EXPIRED"
    assert call.result["days_since_delivery"] > 90
    assert out["answer"].startswith(f"订单 {order_id}：")
    for text in ("超过 30 天", "生鲜", "24 小时内", "人工"):
        assert text in out["answer"]
    assert "已提交" not in out["answer"]


@pytest.mark.parametrize("question", ["生鲜类可以退款吗？", "退款要多久到账？", "退货运费谁出？"])
def test_generic_policy_questions_do_not_become_refund_requests(deps_factory, question):
    deps = deps_factory()
    out = ask(build_graph(deps), question, "U1003")
    assert out["exit"] is Exit.KNOWLEDGE
    assert out["tool_calls"] == []
    assert out["missing_slots"] == []


@pytest.mark.parametrize("protocol", ["function", "json"])
def test_blocked_refund_plan_is_not_reported_as_missing_order(deps_factory, monkeypatch, protocol):
    deps = deps_factory(plans=[[("request_refund", {"order_id": "ORD-1007"})]], protocol=protocol)
    # 固定一次误分类，确保路由修好后仍能独立验证 Agent 的拦截话术。
    monkeypatch.setattr(deps.classifier, "classify", lambda _: ClassifyResult(
        intent=Intent.ORDER_QUERY, confidence=0.9, method="llm",
    ))
    out = ask(build_graph(deps), "订单 ORD-1007 可以退款吗？", "U1003")
    assert out["tool_calls"][0].error == "HIGH_RISK_REQUIRES_FLOW"
    assert deps.tools.drain_audit() == [], "拦截发生在查单之前"
    assert "审核" in out["answer"]
    assert "HIGH_RISK_REQUIRES_FLOW" not in out["answer"]
    for text in ("没查到", "没有查到", "没有找到", "核对一下信息", "已提交"):
        assert text not in out["answer"]


@pytest.mark.parametrize("error", ["UNKNOWN_FAILURE", None])
def test_unclassified_tool_failure_does_not_claim_order_is_missing(error):
    answer = _render_tool_answer([ToolCall(
        name="get_order", args={"order_id": "ORD-1007"}, ok=False, error=error,
    )])
    assert "没查到" not in answer
    assert "没有查到" not in answer
    assert "没有找到" not in answer
    assert "未完成" in answer or "未能完成" in answer or "暂时无法" in answer


def test_no_tool_plan_does_not_claim_order_is_missing(deps_factory):
    deps = deps_factory(plans=[])
    out = ask(build_graph(deps), "查订单 ORD-1007 的状态", "U1003")
    assert out["tool_calls"] == []
    assert out["handoff"] is True
    assert "没查到" not in out["answer"]
    assert "未完成" in out["answer"] or "未能完成" in out["answer"]


@pytest.mark.parametrize("question", [
    "这单 ORD-1009 能退款吗？",
    "它能退款吗，订单 ORD-1009",
    "这单ord1009能退吗？",
])
def test_explicit_order_after_pronoun_does_not_borrow_old_order(deps_factory, question):
    deps = deps_factory()
    history = [
        Turn(role="user", content="查订单 ORD-1007"),
        Turn(role="assistant", content="订单 ORD-1007 已签收。"),
    ]
    resolved = deps.resolver.resolve(question, history, context="订单 ORD-1007", memory=["最近关注的订单：ORD-1007"])
    assert "ORD-1007" not in resolved.resolved
    assert "1009" in resolved.resolved
    assert deps.classifier.extract_slots(resolved.resolved, history).order_id == "ORD-1009"


@pytest.mark.parametrize("protocol", ["function", "json"])
@pytest.mark.parametrize("first_id,owner,error", [
    ("ORD-1008", "U1004", None),
    ("ORD-1007", "U1003", "ORDER_ACCOUNT_UNLINKED"),
])
def test_consecutive_queries_constrain_stale_model_order(deps_factory, protocol, first_id, owner, error):
    # 同账号 1008 -> 1009 应成功；1007 -> 1009 属于跨账号，必须明确拒绝。
    plan = [("get_order", {"order_id": first_id})]
    deps = deps_factory(plans=[plan, [], plan, []], protocol=protocol)
    graph = build_graph(deps)
    first_question = f"查订单 {first_id} 的状态"
    first = ask(graph, first_question, owner)
    assert first["tool_calls"][0].ok
    history = [Turn(role="user", content=first_question), Turn(role="assistant", content=first["answer"])]
    second = ask(graph, "查这单 ORD-1009 的状态", owner, history=history)

    assert first_id not in second["resolved_question"]
    assert second["slots"].order_id == "ORD-1009"
    assert second["tool_calls"][0].args["order_id"] == "ORD-1009"
    assert second["tool_calls"][0].error == error
    assert first_id not in second["answer"]
    if error is None:
        assert second["tool_calls"][0].result["order_id"] == "ORD-1009"
        assert "已取消" in second["answer"]
    else:
        assert "未关联这笔订单" in second["answer"]
        assert "items" not in second["tool_calls"][0].result


@pytest.mark.parametrize("tool", ["get_order", "track_logistics"])
@pytest.mark.parametrize("protocol", ["function", "json"])
def test_raw_current_order_overrides_stale_slots_and_model_args(deps_factory, tool, protocol):
    deps = deps_factory(plans=[[(tool, {"order_id": "ORD-1007"})]], protocol=protocol)
    state = new_state(request_id="current-order", session_id="s", user_id="U1004", question="查订单ord1009")
    state.update(intent=Intent.ORDER_QUERY, resolved_question="查订单 ORD-1007", slots=SlotExtraction(order_id="ORD-1007"))
    out = make_agent_node(deps)(state)
    call = out["tool_calls"][0]
    assert call.args["order_id"] == "ORD-1009"
    assert call.ok
    assert call.result["order_id"] == "ORD-1009"


def test_multiple_explicit_current_orders_remain_queryable(deps_factory):
    deps = deps_factory(plans=[[
        ("get_order", {"order_id": "ORD-1008"}),
        ("get_order", {"order_id": "ORD-1009"}),
    ]])
    out = ask(build_graph(deps), "查订单 ORD-1008 和 ORD-1009 的状态", "U1004")
    assert [c.result["order_id"] for c in out["tool_calls"]] == ["ORD-1008", "ORD-1009"]
    assert all(c.ok for c in out["tool_calls"])


def test_multiple_current_orders_do_not_allow_an_unmentioned_order(deps_factory):
    deps = deps_factory(plans=[[("get_order", {"order_id": "ORD-1004"})]])
    out = ask(build_graph(deps), "查订单 ORD-1008 和 ORD-1009 的状态", "U1002")
    assert deps.tools.drain_audit() == [], "不能查当轮未指定的订单，即使它属于当前账号"
    assert not any(c.ok for c in out["tool_calls"])
    assert "ORD-1004" not in out["answer"]
    assert "没查到" not in out["answer"]


def test_consecutive_refunds_use_current_order(deps_factory):
    deps = deps_factory()
    graph = build_graph(deps)
    first_question = "我要退款 ORD-1008"
    first = ask(graph, first_question, "U1004")
    second = ask(graph, "这单 ORD-1009 能退款吗？", "U1004", history=[
        Turn(role="user", content=first_question), Turn(role="assistant", content=first["answer"]),
    ])
    assert "ORD-1008" not in second["resolved_question"]
    assert second["tool_calls"][-1].result["order_id"] == "ORD-1009"
    assert second["answer"].startswith("订单 ORD-1009：")
    assert "已取消" in second["answer"]
    assert "ORD-1008" not in second["answer"]


@pytest.mark.parametrize("phrase,intent", REFUND_PHRASES[:3])
def test_refunds_preserve_cross_account_denial(deps_factory, phrase, intent):
    deps = deps_factory()
    out = ask(build_graph(deps), phrase.format(order_id="ORD-1007"), "U1001")
    assert out["tool_calls"][-1].error == "ORDER_ACCOUNT_UNLINKED"
    assert "未关联这笔订单" in out["answer"]
    assert not out["answer"].startswith("订单 ORD-1007：")
    assert all("data" not in c.result and "items" not in c.result for c in out["tool_calls"])


@pytest.mark.parametrize("phrase,intent", REFUND_PHRASES[:3])
def test_unknown_order_has_genuine_not_found_error(deps_factory, phrase, intent):
    deps = deps_factory()
    out = ask(build_graph(deps), phrase.format(order_id="ORD-9999"), "U1001")
    assert out["tool_calls"][-1].error == "ORDER_NOT_FOUND"
    assert "没有找到订单" in out["answer"] or "没有查到这个订单号" in out["answer"]


@pytest.mark.parametrize("order_id,owner", [("ORD-1001", "U1001"), ("ORD-1003", "U1002")])
def test_query_and_cancel_controls_use_same_order_store(deps_factory, order_id, owner):
    deps = deps_factory(plans=[[("get_order", {"order_id": order_id})]])
    graph = build_graph(deps)
    query = ask(graph, f"查订单 {order_id} 的状态", owner)
    cancel = ask(graph, f"帮我取消订单 {order_id}", owner)
    assert query["tool_calls"][0].ok
    assert cancel["tool_calls"][0].result["order_id"] == order_id
    assert cancel["tool_calls"][0].error != "ORDER_NOT_FOUND"
    if order_id == "ORD-1001":
        assert "人工审核" in cancel["answer"] and "未付款" in cancel["answer"]
    else:
        assert "已发货" in cancel["answer"] and "无法直接取消" in cancel["answer"]
