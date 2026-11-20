"""issue #4 回归：离线复现依据恢复、相关性与订单展示。"""

import pytest
from fastapi.testclient import TestClient

from mewhelp.api import create_app
from mewhelp.api.service import ChatService, SessionStore, evidence_payload, tool_payload
from mewhelp.config import Settings
from mewhelp.domain import Intent
from mewhelp.retrieval import build_pipeline
from mewhelp.schemas import ConversationResult, Evidence, KbChunk, Turn
from mewhelp.storage import get_engine, init_db
from mewhelp.tools.business import build_business_registry, get_order
from mewhelp.workflow.graph import build_deps
from mewhelp.workflow.high_risk import run_high_risk_flow


@pytest.fixture
def settings(tmp_path):
    config = Settings(database_url=f"sqlite:///{tmp_path / 'issue4.db'}")
    init_db(get_engine(config), config)
    return config


def test_evidence_survives_save_reopen_and_http_restore(settings):
    result = ConversationResult(
        request_id="依据轮次", session_id="依据会话", user_id="用户", question="苹果怎么保存",
        answer="冷藏保存。", evidences=[Evidence(chunk=KbChunk(
            chunk_id="苹果#1", doc_id="苹果", title="苹果", text="苹果冷藏保存。",
        ))], tool_calls=[build_business_registry().call("get_order", {"order_id": "ORD-1010"})],
    )
    ChatService(graph=None, settings=settings)._persist(result)
    # 新建存储实例模拟刷新/重启，不使用内存里的结果缓存。
    reopened = SessionStore(settings).get_session("依据会话", user_id="用户")
    assert reopened is not None
    saved = reopened.messages[-1].model_dump()
    assert saved.get("evidence") == evidence_payload(result)["chunks"]
    assert saved.get("tools") == tool_payload(result)
    assert SessionStore(settings).get_session("依据会话", user_id="其他用户") is None
    # 切到别的会话，再把原会话追加一轮无依据答复：每轮快照各自保存。
    service = ChatService(graph=None, settings=settings)
    service._persist(result.model_copy(update={"session_id": "另一个会话", "evidences": []}))
    service._persist(result.model_copy(update={"request_id": "库外轮次", "evidences": [], "tool_calls": []}))
    messages = SessionStore(settings).get_session("依据会话", user_id="用户").messages
    assert messages[1].evidence == saved["evidence"]
    assert messages[-1].evidence == []

    app = create_app(build_deps(settings=settings))
    with TestClient(app) as client:
        account = client.post("/api/auth/register", json={
            "email": "issue4@example.org", "password": "issue4-test-password",
        }).json()["user"]["id"]
        result.session_id = "恢复接口"
        result.user_id = account
        app.state.service._persist(result)
        restored = client.get("/api/sessions/恢复接口").json()
        assert restored["messages"][-1]["evidence"] == evidence_payload(result)["chunks"]
        assert restored["messages"][-1]["tools"] == tool_payload(result)
        assert "intent" not in restored["messages"][-1]


def test_old_sessions_without_workspace_still_load(settings):
    store = SessionStore(settings)
    store.append("旧会话", "用户", [Turn(role="assistant", content="旧答复")])
    restored = store.get_session("旧会话", user_id="用户")
    assert restored.messages[0].content == "旧答复"


@pytest.fixture
def product_pipeline(settings):
    pipeline = build_pipeline(settings)
    for product in ["苹果", "番茄", "猪肉", "大虾", "牛奶", "羊角蜜", "红薯", "金鲳鱼"]:
        pipeline.index([KbChunk(
            chunk_id=f"{product}#1", doc_id=product, title=f"{product}商品说明",
            breadcrumb=[product, "保存和规格"],
            text=f"{product}请冷藏保存。规格为500g。保质期3天。质量问题可联系客服。",
        )])
    return pipeline


@pytest.mark.parametrize("product", ["苹果", "番茄", "猪肉", "大虾", "牛奶"])
def test_evidence_only_contains_the_requested_product(product_pipeline, product):
    found = product_pipeline.retrieve(f"{product}怎么保存，有什么规格？")
    assert found.evidences
    assert all(ev.chunk.doc_id == product for ev in found.evidences), [
        (ev.chunk.doc_id, ev.score) for ev in found.evidences
    ]


@pytest.mark.parametrize("question", ["你们卖火箭吗？", "火箭有什么规格？"])
@pytest.mark.parametrize("strategy", ["dense", "hybrid_rerank", "hybrid"])
def test_out_of_library_question_has_no_unrelated_evidence(product_pipeline, question, strategy):
    assert product_pipeline.retrieve(question, strategy=strategy).evidences == []


def test_generic_section_heading_does_not_match_an_unknown_product(product_pipeline):
    product_pipeline.index([KbChunk(
        chunk_id="饼干#1", doc_id="饼干", title="规格", breadcrumb=["饼干", "规格"],
        text="饼干规格100g。",
    )])
    assert product_pipeline.retrieve("火箭有什么规格？").evidences == []


def test_each_low_scoring_evidence_is_removed(product_pipeline):
    class FixedReranker:
        def rerank(self, query, evidences, top_k):
            return [ev.model_copy(update={"score": score, "rerank_score": score})
                    for ev, score in zip(evidences, [0.9, 0.1])][:top_k]

    product_pipeline.reranker = FixedReranker()
    result = product_pipeline.retrieve("冷藏保存")
    assert len(result.evidences) == 1
    assert result.evidences[0].score == 0.9


def test_reranker_failure_still_filters_out_of_library_evidence(product_pipeline):
    class BrokenReranker:
        def rerank(self, *args, **kwargs):
            raise RuntimeError("模拟精排不可用")

    product_pipeline.reranker = BrokenReranker()
    assert product_pipeline.retrieve("你们卖火箭吗？").evidences == []


@pytest.mark.parametrize("oid,label", [("ORD-1010", "已退款"), ("ORD-1011", "退款中"), ("ORD-1009", "已取消")])
def test_order_snapshot_uses_current_refund_status(oid, label):
    assert get_order(oid)["data"]["status_label"] == label


@pytest.mark.parametrize("intent,oid", [
    (Intent.CANCEL_ORDER, "ORD-1003"), (Intent.REFUND_REQUEST, "ORD-1010"),
    (Intent.REFUND_REQUEST, "ORD-1011"),
])
def test_rejected_operation_still_exposes_the_found_order(intent, oid):
    outcome = run_high_risk_flow(intent, f"订单 {oid}", build_business_registry())
    result = ConversationResult(request_id="操作", session_id="会话", user_id="用户",
                                question="操作", tool_calls=outcome.tool_calls)
    payload = tool_payload(result)
    assert payload and payload[0]["result"]["items"]
    assert payload[0]["result"]["order_id"] == oid


def test_unpaid_cancellation_has_no_paid_refund_promise():
    registry = build_business_registry()
    call = registry.call("cancel_order", {"order_id": "ORD-1001"})
    assert "未付款" in call.result["refund_note"]
    assert "已支付金额将" not in call.result["refund_note"]
    answer = run_high_risk_flow(Intent.CANCEL_ORDER, "取消 ORD-1001", registry).answer
    assert "未付款" in answer
    assert "退款将按原支付方式退回" not in answer


def test_cancellation_without_payment_record_uses_conditional_refund_wording():
    call = build_business_registry().call("cancel_order", {"order_id": "ORD-1002"})
    assert "如已付款" in call.result["refund_note"]


def test_unlinked_account_has_no_order_snapshot_or_order_prefix():
    outcome = run_high_risk_flow(Intent.CANCEL_ORDER, "取消 ORD-1001",
                                build_business_registry(), principal_id="未关联账号")
    assert "ORD-1001" not in outcome.answer
    result = ConversationResult(request_id="未关联", session_id="会话", user_id="用户",
                                question="取消", tool_calls=outcome.tool_calls)
    assert tool_payload(result) == []


def test_order_query_answer_contains_dates_and_items():
    data = get_order("ORD-1007")["data"]
    answer = data["note"]
    assert data["created_at"][:10] in answer
    assert data["delivered_on"] in answer
    assert all(item["name"] in answer and f"×{item['qty']}" in answer for item in data["items"])


def test_shipped_order_answer_includes_available_logistics_date():
    assert "2026-08-19" in get_order("ORD-1003")["data"]["note"]


@pytest.mark.parametrize("intent,oid", [
    *[(Intent.REFUND_REQUEST, f"ORD-{number}") for number in range(1001, 1013)],
    (Intent.CANCEL_ORDER, "ORD-1001"), (Intent.CANCEL_ORDER, "ORD-1003"),
])
def test_refund_and_cancel_answers_always_identify_the_order(intent, oid):
    answer = run_high_risk_flow(intent, f"订单 {oid}", build_business_registry()).answer
    assert answer.startswith(f"订单 {oid}："), answer


def test_nonexistent_orders_do_not_create_workspace_cards():
    outcome = run_high_risk_flow(Intent.CANCEL_ORDER, "取消 ORD-9999", build_business_registry())
    result = ConversationResult(request_id="无单", session_id="会话", user_id="用户",
                                question="取消", tool_calls=outcome.tool_calls)
    assert tool_payload(result) == []
