"""Issue #12：固定模型重放报告中的拒答、误转接、答非所问和依据丢失。

报告没有附完整生鲜数据集；下面只转录报告明确给出的事实。价格/规格的
跨商品对照使用仓库 products.json，政策对照使用原有 Markdown 知识库。
"""

from __future__ import annotations

import json

import pytest

from mewhelp.api.service import ChatService, SessionStore, done_payload, result_from_state
from mewhelp.config import PROJECT_ROOT, Settings
from mewhelp.domain import Exit, Intent, RefusalReason
from mewhelp.llm import LLMResponse, MockLLMClient
from mewhelp.quality import ConfidenceGate, SelfEvaluator
from mewhelp.retrieval import chunk_document, chunk_document_file
from mewhelp.schemas import Evidence, KbChunk, RetrievalResult, SlotExtraction, Turn
from mewhelp.storage import init_db
from mewhelp.workflow.graph import build_deps, build_graph
from mewhelp.workflow.state import new_state


def card(name, text, *, doc=None, table=None):
    return KbChunk(
        chunk_id=name, doc_id=doc or name, title=name,
        breadcrumb=[doc or name, name], text=text,
        is_table=bool(table), table=table,
    )


SWEET_POTATO = card("红薯规格", "红薯重量规格有9斤、5斤、3斤、1斤；单果分大果、中果、小果。", doc="红薯")
POTATO = card("黑土豆规格与重量", "黑土豆1斤档净重0.8–1.2斤。", doc="黑土豆")
POTATO_POLICY = card("黑土豆赔付说明", "假一赔四针对非正品，不等于腐坏、化冻、少重都赔四倍。", doc="黑土豆")
MELON = card("羊角蜜规格", "羊角蜜单果250–500g，规格有带箱1斤、3斤、5斤，分基础普货、优选果、出口级严选。", doc="羊角蜜")
PACKING = card("羊角蜜包装与重量", "带箱1/3/5斤含包装，不能说是果肉净重或商品净重。", doc="羊角蜜")
PENALTY = card("腐坏变质与15%违约赔付", "15%违约赔付针对未履行服务，不是坏损比例，也不是只退15%。", doc="售后规则")
PORK = card("猪肉梅花肉价格", "淘严选原切梅花肉¥23.8起；猪梅花肉¥35.38起。", doc="猪肉")


class FixedModel(MockLLMClient):
    def __init__(self, settings, *, answer=None, verdict=None, classify=None, product=None):
        super().__init__(settings)
        self.answer = answer
        self.verdict = verdict
        self.classify = classify
        self.product = product
        self.tasks = []

    def chat(self, messages, **kwargs):
        task = kwargs.get("task")
        self.tasks.append(task)
        if task == "generate" and self.answer is not None:
            if isinstance(self.answer, Exception):
                raise self.answer
            return LLMResponse(text=self.answer)
        if task == "self_eval" and self.verdict is not None:
            return LLMResponse(text=json.dumps(self.verdict))
        if task == "classify" and self.classify:
            return LLMResponse(text=json.dumps({"intent": self.classify, "confidence": 0.99}))
        if task == "slots" and self.product:
            return LLMResponse(text=json.dumps({"product": self.product}))
        return super().chat(messages, **kwargs)


class FixedRetriever:
    def __init__(self, chunks, score=0.9):
        self.chunks = chunks
        self.score = score

    def retrieve(self, query):
        return RetrievalResult(query=query, evidences=[
            Evidence(chunk=c, score=self.score) for c in self.chunks
        ])


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None, llm_mode="mock", embedding_mode="mock",
        reranker_mode="mock", vector_store_mode="memory", langfuse_enabled=False,
        MEWHELP_DATABASE_URL=f"sqlite:///{tmp_path / 'issue12.db'}",
        data_dir=PROJECT_ROOT / "data", knowledge_dir=PROJECT_ROOT / "data/knowledge",
    )


def ask(deps, question, history=None):
    return build_graph(deps).invoke(new_state(
        request_id="issue12", session_id="issue12", user_id="u",
        question=question, history=history,
    ))


def supported(out):
    assert out["gate"].passed, out["gate"]
    assert out["handoff"] is False, out["answer"]
    assert out["refusal_reason"] is None
    assert "handoff" not in out["path"]
    assert "转人工" not in out["answer"] and "转接" not in out["answer"]
    assert done_payload(result_from_state(out, latency_ms=0))["handoff"] is False


@pytest.mark.parametrize("answer", [
    "红薯可选9斤、5斤、3斤和1斤装，单果有大、中、小果。",
    "红薯规格有9斤、5斤、3斤、1斤。已经帮你转接人工客服。",
])
def test_correct_specs_do_not_handoff_on_model_self_eval_disagreement(settings, answer):
    llm = FixedModel(settings, answer=answer, verdict={"grounded": False, "coverage": 0})
    out = ask(build_deps(settings=settings, llm=llm, retriever=FixedRetriever([SWEET_POTATO])), "红薯有哪些重量规格？")
    supported(out)
    for weight in ("9斤", "5斤", "3斤", "1斤"):
        assert weight in out["answer"].replace(" ", "")


@pytest.mark.parametrize("answer", [
    "无法确认。证据没有提供少重的具体赔偿规则。",
    "暂时无法确定，需要更多信息。",
    "抱歉，目前没有找到能够回答这个问题的依据。",
    "",
])
def test_non_answer_cannot_pass_as_a_successful_answer(settings, answer):
    llm = FixedModel(settings, answer=answer, verdict={"grounded": True, "coverage": 1})
    deps = build_deps(settings=settings, llm=llm, retriever=FixedRetriever([POTATO_POLICY]))
    out = ask(deps, "黑土豆少重如何处理？")
    assert out["handoff"] or out["interrupt_prompt"], out["answer"]
    if out["handoff"]:
        assert out["refusal_reason"] is not None
        assert out["path"][-2:] == ["handoff", "flywheel"]


def test_self_eval_checks_whether_the_question_was_answered(settings):
    llm = FixedModel(settings, verdict={
        "grounded": True, "coverage": 1, "answered": False,
        "note": "回答了如何赔偿，没有回答重量是否在规格范围内",
    })
    evaluator = SelfEvaluator(llm, settings)
    result = evaluator.evaluate(
        POTATO_POLICY.text, POTATO.text + POTATO_POLICY.text,
        question="黑土豆1斤档收到0.9斤算少重吗？",
    )
    assert result.grounded is False


@pytest.mark.parametrize("question,chunk,product", [
    ("黑土豆 1 斤档收到 0.9 斤，一定算少重吗？", card(
        "黑土豆规格与重量", "", doc="黑土豆",
        table=[["规格", "净重范围"], ["1斤档", "0.8–1.2斤"]],
    ), "黑土豆 1 斤档"),
    ("坏单包退写15%，就是只退15%吗？", PENALTY, "坏单包"),
])
def test_coverage_uses_the_rule_in_evidence_not_literal_slot_text(settings, question, chunk, product):
    decision = ConfidenceGate(settings).check(
        question, FixedRetriever([chunk]).retrieve(question),
        Intent.POLICY_FAQ, SlotExtraction(product=product),
    )
    assert decision.passed, decision


@pytest.mark.parametrize("weight,expected", [
    ("0.8", "范围内"), ("0.9", "范围内"), ("1.2", "范围内"), ("0.7", "低于"),
])
@pytest.mark.parametrize("as_table", [False, True])
def test_weight_comparison_uses_the_quoted_range(settings, weight, expected, as_table):
    chunk = POTATO
    if as_table:
        chunk = card("黑土豆规格与重量", "", doc="黑土豆", table=[
            ["规格", "净重范围"], ["1斤档", "0.8–1.2斤"],
        ])
    llm = FixedModel(settings, answer="无法确认，没有提供少重赔偿规则。", product="黑土豆 1 斤档")
    deps = build_deps(settings=settings, llm=llm, retriever=FixedRetriever([POTATO_POLICY, chunk]))
    out = ask(deps, f"黑土豆 1 斤档收到 {weight} 斤，一定算少重吗？")
    supported(out)
    assert expected in out["answer"]
    assert "0.8" in out["answer"] and "1.2" in out["answer"]
    assert "赔四倍" not in out["answer"]
    assert "generate" not in llm.tasks, "可核验的区间比较不应依赖模型采样"


@pytest.mark.parametrize("question,chunks,expected", [
    ("羊角蜜有哪些规格？", [MELON, PACKING], ["250", "500", "基础普货", "优选果", "出口级严选"]),
    ("坏单包退写15%，就是只退15%吗？", [PENALTY], ["未履行服务", "不是坏损比例"]),
    ("猪肉梅花肉有哪些款，多少钱？", [PORK], ["淘严选原切梅花肉", "23.8", "猪梅花肉", "35.38"]),
])
def test_covered_facts_survive_generation_failure(settings, question, chunks, expected):
    llm = FixedModel(settings, answer=RuntimeError("固定模拟生成超时"))
    out = ask(build_deps(settings=settings, llm=llm, retriever=FixedRetriever(chunks)), question)
    supported(out)
    assert all(value in out["answer"] for value in expected)


def test_packing_paraphrases_have_the_same_supported_answer(settings):
    llm = FixedModel(settings, answer=RuntimeError("不应为带箱定义调用模型"))
    deps = build_deps(settings=settings, llm=llm, retriever=FixedRetriever([MELON, PACKING]))
    answers = []
    for q in ["羊角蜜带箱 5 斤，就是 5 斤净重吗？", "羊角蜜 5 斤带箱是净重吗？", "羊角蜜5斤带箱子，果肉有5斤吗？"]:
        out = ask(deps, q)
        supported(out)
        assert "含包装" in out["answer"]
        assert "不能" in out["answer"] or "不是" in out["answer"]
        answers.append(out["answer"])
    assert len(set(answers)) == 1


@pytest.mark.parametrize("question", [
    "牛排 10 片坏了 2 片，是不是退 20%？",
    "牛排10片坏了2片，是不是退20%",
    "坏了两片牛排，退款比例按数量算吗？",
])
def test_generic_damage_policy_is_deterministic_and_never_asks_for_an_order(settings, question):
    llm = FixedModel(settings, classify="refund_request")
    chunks = chunk_document_file(PROJECT_ROOT / "data/knowledge/退款政策.md")
    chunks = [c for c in chunks if "生鲜类" in c.title]
    out = ask(build_deps(settings=settings, llm=llm, retriever=FixedRetriever(chunks)), question)
    assert out["exit"] is Exit.KNOWLEDGE
    assert out["classify"].method == "rule"
    assert out["tool_calls"] == [] and out["missing_slots"] == []
    assert "订单号" not in out["answer"]
    supported(out)
    assert "不能" in out["answer"] or "不是" in out["answer"]
    if "20%" in question:
        assert "20%" in out["answer"]
    assert "24" in out["answer"] and "80%" in out["answer"]


def test_fixture_product_specs_do_not_depend_on_the_model(settings):
    products = json.loads((PROJECT_ROOT / "data/fixtures/products.json").read_text())["products"]
    product = next(p for p in products if p["sku"] == "SKU-FRESH-002")
    chunk = card(product["name"] + "规格", "；".join(product["sizes"]) + "。", doc=product["name"])
    llm = FixedModel(settings, answer=RuntimeError("固定模拟生成超时"))
    out = ask(build_deps(settings=settings, llm=llm, retriever=FixedRetriever([chunk])), product["name"] + "有哪些规格？")
    supported(out)
    assert all(size in out["answer"] for size in product["sizes"])


def test_price_followup_reuses_full_evidence_after_session_reload(settings):
    # 不虚构报告未给出的红运道价格：用仓库夹具的真实商品/价格重放同一种追问。
    products = json.loads((PROJECT_ROOT / "data/fixtures/products.json").read_text())["products"]
    product = next(p for p in products if p["sku"] == "SKU-HOME-001")
    chunks = chunk_document("护颈枕", "护颈枕", "# 护颈枕\n\n## 规格与价格\n\n"
        "| 商品 | 规格 | 价格 |\n|---|---|---|\n"
        f"| {product['name']} | {'、'.join(product['sizes'])} | {product['price']}元 |")
    init_db(settings=settings)
    deps = build_deps(settings=settings, retriever=FixedRetriever(chunks))
    service = ChatService(graph=build_graph(deps), settings=settings)
    first = service.ask(user_id="u", session_id="s", message="护颈枕有哪些规格？")
    assert first.evidences and not first.handoff
    deps.retriever = FixedRetriever([])
    reopened = ChatService(graph=build_graph(deps), sessions=SessionStore(settings), settings=settings)
    second = reopened.ask(user_id="u", session_id="s", message="MewHome那款多少钱？")
    assert second.evidences, "展示用表头摘要无法代替原始表格证据"
    assert not second.handoff, second.answer
    assert str(product["price"]) in second.answer
    assert second.evidences[0].chunk.table == chunks[0].table


@pytest.mark.parametrize("question", ["那 5 斤的是净重吗？", "羊角蜜 5 斤带箱是净重吗？"])
def test_followup_uses_evidence_even_when_previous_answer_was_a_refusal(settings, question):
    init_db(settings=settings)
    first = result_from_state(ask(build_deps(
        settings=settings, retriever=FixedRetriever([MELON, PACKING]),
    ), "羊角蜜有哪些规格？"), latency_ms=0)
    first.answer = "暂时没能组织出可靠答复，帮你转人工确认。"
    first.handoff = True
    service = ChatService(graph=None, settings=settings)
    service._persist(first)
    deps = build_deps(settings=settings, retriever=FixedRetriever([]), llm=FixedModel(settings, answer=RuntimeError("固定失败")))
    reopened = ChatService(graph=build_graph(deps), settings=settings)
    result = reopened.ask(user_id="u", session_id="issue12", message=question)
    assert result.evidences
    assert not result.handoff, result.answer
    assert "含包装" in result.answer


def test_prior_evidence_is_not_reused_for_a_new_product_or_another_session(settings):
    init_db(settings=settings)
    deps = build_deps(settings=settings, retriever=FixedRetriever([MELON, PACKING]))
    service = ChatService(graph=build_graph(deps), settings=settings)
    service.ask(user_id="u", session_id="s", message="羊角蜜有哪些规格？")
    deps.retriever = FixedRetriever([])
    reopened = ChatService(graph=build_graph(deps), settings=settings)
    for session, question in [("s", "火箭带箱5斤是净重吗？"), ("other", "那5斤的是净重吗？")]:
        result = reopened.ask(user_id="u", session_id=session, message=question)
        assert result.handoff
        assert result.evidences == []
        assert "含包装" not in result.answer


@pytest.mark.parametrize("chunks", [[], [card("苹果重量", "苹果1斤档净重0.8–1.2斤。", doc="苹果")]])
def test_missing_or_wrong_product_evidence_still_refuses(settings, chunks):
    deps = build_deps(settings=settings, retriever=FixedRetriever(chunks), llm=FixedModel(settings, product="黑土豆1斤档"))
    out = ask(deps, "黑土豆1斤档收到0.9斤，一定算少重吗？")
    assert out["handoff"]
    assert out["refusal_reason"] in {RefusalReason.NO_EVIDENCE, RefusalReason.LOW_COVERAGE}
    assert "generate" not in out["path"]


def test_real_generation_failure_has_one_handoff_notice(settings):
    chunk = card("耳机声音说明", "耳机支持双设备连接。", doc="耳机")
    deps = build_deps(settings=settings, retriever=FixedRetriever([chunk]), llm=FixedModel(settings, answer=RuntimeError("固定失败")))
    out = ask(deps, "耳机同时连接多个设备时如何切换声音？")
    assert out["handoff"]
    assert out["refusal_reason"] is not None
    assert out["answer"].count("转人工") + out["answer"].count("转接人工") == 1


def test_answered_question_with_an_unasked_caveat_does_not_handoff(settings):
    chunk = card("红心柚发货地", "红心柚从广西南宁发货，发货地不等于产地。", doc="红心柚")
    llm = FixedModel(settings, answer="红心柚从广西南宁发货；产地无法确认。", verdict={"grounded": True, "coverage": 1, "answered": True})
    out = ask(build_deps(settings=settings, llm=llm, retriever=FixedRetriever([chunk])), "红心柚从哪里发货？")
    supported(out)
    assert "广西南宁" in out["answer"]


@pytest.mark.parametrize("score", [0.1, 0.9])
def test_complete_evidence_rule_is_not_rejected_by_score_alone(settings, score):
    question = "羊角蜜带箱5斤是净重吗？"
    decision = ConfidenceGate(settings).check(
        question, FixedRetriever([PACKING], score=score).retrieve(question),
        Intent.PRODUCT_FAQ, SlotExtraction(product="羊角蜜带箱5斤"),
    )
    assert decision.passed, decision


def test_pipeline_preserves_a_complete_rule_when_reranker_scores_it_low(settings):
    deps = build_deps(settings=settings, llm=FixedModel(settings, answer=RuntimeError("规则题无需模型")))
    deps.retriever.index([MELON, PACKING, POTATO])

    class LowReranker:
        def rerank(self, query, evidences, top_k):
            return [e.model_copy(update={"score": 0.1, "rerank_score": 0.1}) for e in evidences][:top_k]

    deps.retriever.reranker = LowReranker()
    out = ask(deps, "羊角蜜5斤带箱是净重吗？")
    supported(out)
    assert "含包装" in out["answer"]
    assert all(e.chunk.doc_id == "羊角蜜" for e in out["retrieval"].evidences)


@pytest.mark.parametrize("question,chunk", [
    ("黑土豆3斤档收到0.9斤算少重吗？", POTATO),
    ("黑土豆1斤档收到0.9斤怎么赔？", POTATO),
    ("黑土豆1斤档收到0.9斤算少重吗？", card("黑土豆规格与重量", "黑土豆1斤档净重0.8–1.2斤。黑土豆1斤档净重1.0–1.1斤。", doc="黑土豆")),
])
def test_rule_does_not_invent_an_uncovered_tier_remedy_or_conflicting_range(settings, question, chunk):
    deps = build_deps(settings=settings, retriever=FixedRetriever([chunk]), llm=FixedModel(
        settings, answer="无法确认。", verdict={"grounded": True, "coverage": 1},
    ))
    out = ask(deps, question)
    assert out["handoff"] or out["interrupt_prompt"]
    assert "范围内" not in out["answer"]


def test_comma_separated_conflicting_ranges_do_not_pick_the_first_one(settings):
    chunk = card("黑土豆规格与重量", "黑土豆1斤档净重0.8–1.2斤，1斤档净重1.0–1.1斤。", doc="黑土豆")
    deps = build_deps(settings=settings, retriever=FixedRetriever([chunk]), llm=FixedModel(settings, answer="无法确认。"))
    out = ask(deps, "黑土豆1斤档收到0.9斤算少重吗？")
    assert out["handoff"]
    assert "范围内" not in out["answer"]


@pytest.mark.parametrize("streaming", [False, True])
async def test_successful_model_answer_does_not_publish_a_spurious_handoff(settings, streaming):
    text = "耳机双击按钮即可切换声音。"

    class StreamingModel(FixedModel):
        def chat_stream(self, messages, *, on_token, **kwargs):
            response = self.chat(messages, **kwargs)
            for char in response.text:
                on_token(char)
            return response

    llm = StreamingModel(settings, answer=text + "已经帮你转接人工客服进一步处理。", verdict={"grounded": True, "coverage": 1})
    deps = build_deps(settings=settings, llm=llm, retriever=FixedRetriever([card("耳机操作", text, doc="耳机")]))
    init_db(settings=settings)
    service = ChatService(graph=build_graph(deps), settings=settings)
    args = dict(user_id="u", session_id="s", message="耳机如何切换声音？")
    if streaming:
        events = [event async for event in service.stream(**args)]
        done = json.loads(events[-1]["data"])
        tokens = "".join(json.loads(e["data"])["text"] for e in events if e["event"] == "token")
        assert tokens == done["answer"] == text
        assert done["handoff"] is False
    else:
        result = service.ask(**args)
        assert result.answer == text
        assert not result.handoff
    assert service.sessions.load("s", user_id="u")[-1].content == text


def test_named_followup_selects_the_requested_product_row(settings):
    products = json.loads((PROJECT_ROOT / "data/fixtures/products.json").read_text())["products"]
    products = [p for p in products if p["sku"] in {"SKU-HOME-001", "SKU-HOME-002"}]
    requested, other = products[1], products[0]
    chunk = card("家居商品规格与价格", "", doc="家居商品", table=[
        ["商品", "规格", "价格"],
        *[[p["name"], "、".join(p["sizes"]), f"{p['price']}元"] for p in products],
    ])
    init_db(settings=settings)
    deps = build_deps(settings=settings, retriever=FixedRetriever([chunk]))
    service = ChatService(graph=build_graph(deps), settings=settings)
    first = service.ask(user_id="u", session_id="s", message="家居商品有哪些规格？")
    assert not first.handoff
    deps.retriever = FixedRetriever([])
    service = ChatService(graph=build_graph(deps), settings=settings)
    result = service.ask(user_id="u", session_id="s", message=requested["name"] + "那款多少钱？")
    assert not result.handoff
    assert requested["name"] in result.answer and str(requested["price"]) in result.answer
    assert other["name"] not in result.answer


def test_specs_do_not_count_as_answering_an_uncovered_storage_question(settings):
    deps = build_deps(settings=settings, retriever=FixedRetriever([MELON]), llm=FixedModel(
        settings, answer="保存方法无法确认。", verdict={"grounded": True, "coverage": 1},
    ))
    out = ask(deps, "羊角蜜有哪些规格，应该怎么保存？")
    assert out["handoff"] or out["interrupt_prompt"]


def test_price_table_without_a_price_is_not_a_complete_answer(settings):
    chunk = card("苹果价格", "", doc="苹果", table=[["商品", "价格"], ["苹果", "待定"]])
    deps = build_deps(settings=settings, retriever=FixedRetriever([chunk]), llm=FixedModel(settings, answer="无法确认。"))
    out = ask(deps, "苹果多少钱？")
    assert out["handoff"] or out["interrupt_prompt"]
