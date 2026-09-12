"""置信度闸门 + 自评的测试。

这个文件的核心不是"跑通"，是**锁住两条设计约束**：

  1. `check()` 在生成之前跑 —— 用签名断言 + 结构断言 + 行为断言三重锁住。
     这条约束一旦被"顺手加个 answer 参数"破坏，代价是幻觉已经送达用户，
     而任何普通的单元测试都发现不了。所以要专门为它写测试。
  2. 任何异常都降级、不外泄 —— 用故障注入证明，不是靠读代码相信。

其余用例覆盖契约里点名的四种判定结果 + 阈值可配置 + 确定性。
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

import mewhelp.quality.gate as gate_module
from mewhelp.config import get_settings
from mewhelp.domain import Intent, RefusalReason
from mewhelp.llm import LLMResponse
from mewhelp.quality import ConfidenceGate, SelfEvaluator, extract_entities
from mewhelp.schemas import Evidence, KbChunk, RetrievalResult, SelfEvalResult

# ------------------------------------------------------------
# 构造证据的小工具
# ------------------------------------------------------------


def chunk(cid: str, title: str, text: str) -> KbChunk:
    return KbChunk(chunk_id=cid, doc_id="doc-1", title=title, text=text)


def evidence(cid: str, title: str, text: str, score: float) -> Evidence:
    return Evidence(chunk=chunk(cid, title, text), score=score)


def retrieval(*items: Evidence, query: str = "q") -> RetrievalResult:
    return RetrievalResult(query=query, evidences=list(items))


def gate_with(**overrides: Any) -> ConfidenceGate:
    """造一个改了阈值的闸门 —— 不改代码只改配置，这正是消融实验要的能力。"""
    return ConfidenceGate(get_settings().model_copy(update=overrides))


#: 一条内容正常、包含实体的证据
GOOD = evidence(
    "c1",
    "运费政策",
    "生鲜类商品满 99 元包邮，运费按订单金额计算。",
    0.82,
)
GOOD2 = evidence(
    "c2",
    "退款政策 > 生鲜类",
    "生鲜类商品签收后 24 小时内可申请退款，超过 3 天不予受理。",
    0.61,
)
WEAK = evidence("c3", "会员权益", "会员积分可用于抵扣。", 0.18)
IRRELEVANT = evidence("c4", "发票说明", "电子发票在订单完成后开具。", 0.79)


# ============================================================
# 1. 核心判据：四种结论
# ============================================================


class TestNoEvidence:
    def test_empty_retrieval_is_rejected(self) -> None:
        decision = ConfidenceGate().check("运费怎么算", retrieval())
        assert decision.passed is False
        assert decision.reason is RefusalReason.NO_EVIDENCE
        assert decision.confidence == 0.0
        assert decision.message  # 拒答话术不能是空的

    def test_none_retrieval_does_not_crash(self) -> None:
        """检索没跑成功就传了 None —— 闸门必须当"没证据"处理，而不是崩。"""
        decision = ConfidenceGate().check("运费怎么算", None)
        assert decision.reason is RefusalReason.NO_EVIDENCE

    def test_recalled_but_all_below_threshold_is_weak_not_no_evidence(self) -> None:
        """召回了 3 条但全不过线，报的是 weak_evidence 而不是 no_evidence。

        这个区分是飞轮的立身之本：no_evidence 意味着"该补文档"，
        weak_evidence 意味着"文档有、检索没捞上来"，两件事的处理方式完全不同。
        """
        decision = ConfidenceGate().check(
            "运费怎么算",
            retrieval(
                evidence("a", "无关", "凑数", 0.10),
                evidence("b", "无关", "凑数", 0.20),
                evidence("c", "无关", "凑数", 0.30),
            ),
        )
        assert decision.reason is RefusalReason.WEAK_EVIDENCE
        assert decision.checks["recalled_count"] == 3.0
        assert decision.checks["evidence_count"] == 0.0


class TestWeakEvidence:
    def test_low_score_is_rejected(self) -> None:
        decision = ConfidenceGate().check("会员积分怎么用", retrieval(WEAK))
        assert decision.passed is False
        assert decision.reason is RefusalReason.WEAK_EVIDENCE
        assert decision.checks["top_score"] == pytest.approx(0.18)

    def test_below_min_evidence_count_is_rejected(self) -> None:
        """单条强证据 + min_evidence_count=2 → 拒答。阈值可配置的直接体现。"""
        gate = gate_with(min_evidence_count=2)
        decision = gate.check("生鲜类商品运费怎么算", retrieval(GOOD))
        assert decision.passed is False
        assert decision.reason is RefusalReason.WEAK_EVIDENCE
        assert decision.checks["below_min_count"] == 1.0

        lenient = gate_with(min_evidence_count=1)
        assert lenient.check("生鲜类商品运费怎么算", retrieval(GOOD)).passed is True


class TestLowCoverage:
    def test_entities_missing_from_evidence_is_rejected(self) -> None:
        """分数够高、条数也够，但证据完全没提到问题里的实体 → low_coverage。

        这正是"分数高 ≠ 答得对"：`.79` 的发票说明对"生鲜类好不好"这个问题
        一个字都没答上。
        """
        decision = ConfidenceGate().check(
            "生鲜类商品能退吗",
            retrieval(IRRELEVANT, evidence("c5", "配送", "配送时效 3 天。", 0.72)),
            Intent.RETURN_REQUEST,
        )
        assert decision.passed is False
        assert decision.reason is RefusalReason.LOW_COVERAGE
        assert decision.checks["entity_coverage"] < 0.5
        assert "生鲜" in decision.message  # 话术要点出没覆盖到什么

    def test_order_id_does_not_count_into_coverage(self) -> None:
        """订单号进覆盖率分母 = 系统性误拒：知识类证据天然不含订单号。

        "该给单号却没给"由 missing_slot 负责，不该由闸门越权代报。
        """
        decision = ConfidenceGate().check(
            "ORD-1001 这个订单的发票怎么开",
            retrieval(evidence("c6", "发票说明", "电子发票在订单完成后开具。", 0.8)),
        )
        assert decision.checks["order_id_count"] == 1.0
        assert decision.checks["entity_count"] == 0.0
        assert decision.passed is True  # 订单号不该拖累覆盖率

    def test_empty_entity_question_does_not_get_rejected_for_coverage(self) -> None:
        """抽不到实体 → 这条判据失效，而不是判 0 分误拒。"""
        decision = ConfidenceGate().check(
            "请问一下怎么处理",
            retrieval(GOOD, GOOD2),
        )
        assert decision.checks["entity_count"] == 0.0
        assert decision.passed is True


class TestPassed:
    def test_good_evidence_passes_with_four_checks(self) -> None:
        decision = ConfidenceGate().check(
            "生鲜类商品运费怎么算",
            retrieval(GOOD, GOOD2),
            Intent.POLICY_FAQ,
        )
        assert decision.passed is True
        assert decision.reason is None
        assert 0.0 < decision.confidence <= 1.0
        assert decision.message == ""
        # 契约点名的四个判据必须齐全，且都是数值
        for key in ("top_score", "evidence_count", "score_margin", "entity_coverage"):
            assert key in decision.checks, f"缺判据 {key}"
            assert isinstance(decision.checks[key], float)
        assert decision.checks["top_score"] == pytest.approx(0.82)

    def test_checks_are_self_explanatory(self) -> None:
        """checks 必须能独立支撑"为什么判它不过"，所以把输入也一并记下来。"""
        decision = ConfidenceGate().check("生鲜类商品运费怎么算", retrieval(GOOD, GOOD2))
        assert decision.checks["threshold"] == pytest.approx(0.35)
        assert decision.checks["recalled_count"] == 2.0
        assert decision.checks["reject"] == 0.0


# ============================================================
# 2. 单条证据时 margin 的表现
# ============================================================


class TestScoreMargin:
    def test_single_evidence_margin_is_zero(self) -> None:
        """只有一条证据时 margin = 0，而不是"跟 1.0 的差"。

        margin 的语义是"第二名离第一名多远"= "有没有第二条独立证据印证"。
        只有一条时答案是"没有"，就该是 0 —— 一个看着很大的假差距会让
        单条证据被误当成高置信。
        """
        decision = ConfidenceGate().check(
            "生鲜类商品运费怎么算",
            retrieval(GOOD),
        )
        assert decision.checks["evidence_count"] == 1.0
        assert decision.checks["score_margin"] == 0.0

    def test_single_evidence_confidence_is_capped_below_multi(self) -> None:
        """同一条证据，配上一条佐证后 confidence 必须更高 —— 交叉印证有价。"""
        gate = ConfidenceGate()
        solo = gate.check("生鲜类商品运费怎么算", retrieval(GOOD))
        paired = gate.check("生鲜类商品运费怎么算", retrieval(GOOD, GOOD2))
        assert solo.passed and paired.passed
        assert paired.confidence > solo.confidence

    def test_margin_uses_valid_evidence_only(self) -> None:
        """1 条过线 + 2 条噪音，也算"没有交叉印证"，margin 仍是 0。

        噪音不构成印证 —— 拿全部召回去算 margin，会让"一条独苗 + 一堆垃圾"
        看起来像"两条独立证据互相支持"。
        """
        decision = ConfidenceGate().check(
            "生鲜类商品运费怎么算",
            retrieval(
                GOOD,
                evidence("n1", "噪音", "无关内容", 0.33),
                evidence("n2", "噪音", "无关内容", 0.30),
            ),
        )
        assert decision.checks["evidence_count"] == 1.0
        assert decision.checks["score_margin"] == 0.0
        assert decision.checks["recalled_count"] == 3.0

    def test_two_evidence_margin_is_the_real_gap(self) -> None:
        decision = ConfidenceGate().check("生鲜类商品运费怎么算", retrieval(GOOD, GOOD2))
        assert decision.checks["score_margin"] == pytest.approx(0.21)


# ============================================================
# 3. 顺序约束：check() 不依赖任何生成结果
# ============================================================


class TestRunsBeforeGeneration:
    """这一组是整套测试里最重要的一组。见模块 docstring 的时序论证。"""

    def test_signature_has_no_answer(self) -> None:
        params = set(inspect.signature(ConfidenceGate.check).parameters)
        assert params == {"self", "question", "retrieval", "intent", "slots"}
        assert "answer" not in params

    def test_gate_module_has_no_llm_dependency(self) -> None:
        """结构断言：gate.py 里根本没有 LLM 的入口。

        签名断言只能挡住"加个 answer 参数"，挡不住"在函数体里调一次模型"。
        没有 LLM 依赖 = 物理上不可能在 check 里读生成结果。
        """
        assert not hasattr(gate_module, "build_llm")
        assert not hasattr(gate_module, "LLMClient")
        assert not hasattr(ConfidenceGate(get_settings()), "llm")

    def test_gate_never_calls_the_llm_even_if_one_explodes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """把 build_llm 换成"一调就炸"，闸门行为必须完全不变。"""

        def explode(*_a: Any, **_kw: Any) -> Any:
            raise AssertionError("闸门不应触碰 LLM")

        monkeypatch.setattr("mewhelp.llm.build_llm", explode)
        decision = ConfidenceGate().check("生鲜类商品运费怎么算", retrieval(GOOD, GOOD2))
        assert decision.passed is True

    def test_result_identical_when_called_before_and_after_generation(self) -> None:
        """行为断言：先判一次、再"生成"一段回答、再判一次，结论逐字节相同。

        这就是"卡在模型开口之前"的可验证形式 —— 闸门的输出里不存在任何
        受生成影响的自由度。
        """
        gate = ConfidenceGate()
        before = gate.check("生鲜类商品运费怎么算", retrieval(GOOD, GOOD2), Intent.POLICY_FAQ)

        # 模拟生成阶段（在这个模块里完全不可见）
        _generated_answer = "根据平台规则，生鲜类商品满 99 元包邮，运费按订单金额计算。"

        after = gate.check("生鲜类商品运费怎么算", retrieval(GOOD, GOOD2), Intent.POLICY_FAQ)
        assert before.model_dump() == after.model_dump()


# ============================================================
# 4. 确定性与兜底
# ============================================================


class TestDeterminism:
    def test_same_input_same_output(self) -> None:
        gate = ConfidenceGate()
        runs = [
            gate.check("生鲜类商品运费怎么算", retrieval(GOOD, GOOD2)).model_dump()
            for _ in range(8)
        ]
        assert all(r == runs[0] for r in runs)

    def test_evidence_order_does_not_change_conclusion(self) -> None:
        """乱序输入同结论 —— 闸门内部会排序，不能依赖调用方的顺序。"""
        gate = ConfidenceGate()
        a = gate.check("生鲜类商品运费怎么算", retrieval(GOOD, GOOD2))
        b = gate.check("生鲜类商品运费怎么算", retrieval(GOOD2, GOOD))
        assert a.confidence == pytest.approx(b.confidence)
        assert a.checks["top_score"] == b.checks["top_score"]


class TestRobustness:
    def test_nan_score_does_not_silently_vanish(self) -> None:
        """NaN 跟任何数比较都是 False。不显式处理，证据会静默消失且不留痕。"""
        nan_ev = Evidence(chunk=chunk("nan", "运费政策", "满 99 元包邮"), score=float("nan"))
        decision = ConfidenceGate().check("运费怎么算", retrieval(nan_ev))
        assert decision.reason is RefusalReason.WEAK_EVIDENCE
        assert decision.checks["top_score"] == 0.0
        assert decision.checks["recalled_count"] == 1.0

    def test_breaking_retrieval_degrades_instead_of_raising(self) -> None:
        """故障注入：检索对象本身取属性就炸，闸门必须给出降级结论。"""

        class Exploding:
            query = "q"

            @property
            def evidences(self) -> list[Evidence]:
                raise RuntimeError("retrieval exploded")

        gate = ConfidenceGate()
        decision = gate.check("运费怎么算", Exploding())  # type: ignore[arg-type]
        assert decision.passed is False
        assert decision.reason is RefusalReason.NO_EVIDENCE
        assert decision.checks["error"] == 1.0
        assert "RuntimeError" in gate.last_error

    def test_degraded_checks_keep_the_same_keys(self) -> None:
        """降级路径的键位必须与正常路径一致 —— 否则下游在读 checks 时 KeyError。"""
        gate = ConfidenceGate()
        normal = gate.check("运费怎么算", retrieval(GOOD)).checks
        degraded = gate.check("运费怎么算", None).checks
        assert set(normal) == set(degraded)

    def test_intent_is_optional(self) -> None:
        assert ConfidenceGate().check("运费怎么算", retrieval(GOOD)).passed is True


# ============================================================
# 5. 阈值可配置
# ============================================================


class TestConfigurableThreshold:
    def test_raising_threshold_flips_the_verdict(self) -> None:
        default = ConfidenceGate()
        assert default.check("生鲜类商品运费怎么算", retrieval(GOOD, GOOD2)).passed is True

        strict = gate_with(confidence_threshold=0.95)
        decision = strict.check("生鲜类商品运费怎么算", retrieval(GOOD, GOOD2))
        assert decision.passed is False
        assert decision.reason is RefusalReason.WEAK_EVIDENCE
        assert decision.checks["threshold"] == pytest.approx(0.95)

    def test_lowering_threshold_admits_weaker_evidence(self) -> None:
        lenient = gate_with(confidence_threshold=0.1)
        decision = lenient.check("会员积分怎么用", retrieval(WEAK))
        assert decision.passed is True
        assert decision.checks["top_score"] == pytest.approx(0.18)

    def test_threshold_defaults_to_settings_for_every_intent(self) -> None:
        """契约 §2.E：阈值就是 settings.confidence_threshold，不多也不少。

        高风险意图加成默认关闭 —— 任何按契约写的断言都不该被扩展悄悄改掉行为。
        """
        gate = ConfidenceGate()
        for intent in (Intent.POLICY_FAQ, Intent.REFUND_REQUEST, Intent.HUMAN_HANDOFF):
            decision = gate.check("运费怎么算", retrieval(GOOD), intent)
            assert decision.checks["threshold"] == pytest.approx(0.35)

    def test_high_risk_margin_is_opt_in(self) -> None:
        """显式打开后，退款/退货/取消/改地址要更严 —— 错误代价是钱。

        同一份 0.40 分的证据：政策咨询能过，退款申请过不去。
        """
        borderline = evidence("b", "运费政策", "满 99 元包邮。", 0.40)

        default = ConfidenceGate()
        assert default.check("运费怎么算", retrieval(borderline), Intent.REFUND_REQUEST).passed

        strict = ConfidenceGate(high_risk_margin=0.1)
        policy = strict.check("运费怎么算", retrieval(borderline), Intent.POLICY_FAQ)
        refund = strict.check("运费怎么算", retrieval(borderline), Intent.REFUND_REQUEST)
        assert policy.checks["threshold"] == pytest.approx(0.35)
        assert refund.checks["threshold"] == pytest.approx(0.45)
        assert policy.passed is True and refund.passed is False


# ============================================================
# 6. 实体抽取
# ============================================================


class TestExtractEntities:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("生鲜类商品怎么退", ["生鲜"]),
            ("满 99 元包邮吗", ["99元"]),
            ("几天能到，7 天无理由吗", ["7天"]),
            ("ORD-1001 的状态", ["ORD-1001"]),
            ("ord1001 的状态", ["ORD-1001"]),
        ],
    )
    def test_extraction(self, text: str, expected: list[str]) -> None:
        assert extract_entities(text) == expected

    def test_order_id_formats_normalize_to_one_form(self) -> None:
        """`ord1001` 和 `ORD-1001` 必须归一成同一个串，否则覆盖率会算成 0。"""
        assert extract_entities("ord1001") == extract_entities("ORD-1001")

    def test_no_entity_returns_empty(self) -> None:
        assert extract_entities("请问一下怎么处理") == []

    def test_coverage_tolerates_whitespace_differences(self) -> None:
        """证据里是 "7 天"、抽出来是 "7天" —— 归一化后必须算命中。"""
        decision = ConfidenceGate().check(
            "生鲜类商品几天能到",
            retrieval(evidence("t", "配送时效", "生鲜类商品配送时效 7 天。", 0.8)),
        )
        assert decision.checks["entity_coverage"] == 1.0
        assert decision.passed is True


# ============================================================
# 7. SelfEvaluator
# ============================================================


class FakeLLM:
    """按预设剧本回答，或按预设方式失败。`task` 要断言成 self_eval。"""

    model = "fake"

    def __init__(self, payload: Any = None, *, exc: Exception | None = None) -> None:
        self.payload = payload
        self.exc = exc
        self.tasks: list[str] = []

    def chat(self, messages, **kwargs: Any) -> LLMResponse:
        self.tasks.append(kwargs.get("task", "generic"))
        if self.exc is not None:
            raise self.exc
        import json

        text = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return LLMResponse(text=text, model=self.model)


CONTEXT = "[1] [退款政策 > 生鲜类]\n生鲜类商品签收后 24 小时内可申请退款，超过 3 天不予受理。"
GROUNDED_ANSWER = "生鲜类商品签收后 24 小时内可以申请退款，超过 3 天不予受理。"


class TestSelfEvaluatorHappyPath:
    def test_grounded_answer_passes(self) -> None:
        result = SelfEvaluator(FakeLLM({"grounded": True, "coverage": 1.0})).evaluate(
            GROUNDED_ANSWER, CONTEXT
        )
        assert isinstance(result, SelfEvalResult)
        assert result.grounded is True
        assert result.coverage == pytest.approx(1.0)
        assert result.unsupported_claims == []

    def test_fabricated_numbers_are_caught_by_rules_even_if_model_says_ok(self) -> None:
        """关键用例：模型说"没问题"，规则说"48 小时证据里没有"。

        取更保守的一侧 → 判不通过。模型自评本身也是模型，不能当唯一依据。
        """
        llm = FakeLLM({"grounded": True, "coverage": 1.0})
        result = SelfEvaluator(llm).evaluate(
            "生鲜类商品签收后 48 小时内可以申请退款，超过 7 天不予受理。", CONTEXT
        )
        assert result.grounded is False
        assert "48 小时" in result.unsupported_claims
        assert "7 天" in result.unsupported_claims
        assert "不一致" in result.note  # 分歧必须留痕，否则调参没线索

    def test_model_too_strict_also_wins(self) -> None:
        """反向不一致时同样保守处理 —— 取舍是单向的，不能只朝有利方向偏。"""
        llm = FakeLLM(
            {"grounded": False, "coverage": 0.0, "unsupported_claims": ["生鲜类是活体"]}
        )
        result = SelfEvaluator(llm).evaluate(GROUNDED_ANSWER, CONTEXT)
        assert result.grounded is False
        assert "生鲜类是活体" in result.unsupported_claims

    def test_task_is_self_eval(self) -> None:
        llm = FakeLLM({"grounded": True, "coverage": 1.0})
        SelfEvaluator(llm).evaluate(GROUNDED_ANSWER, CONTEXT)
        assert llm.tasks == ["self_eval"]

    def test_prompt_fences_the_answer(self) -> None:
        """`<answer>` 标签是接口的一部分，不是格式偏好。

        缺了它，MockLLMClient 会把整段 prompt（含证据）当成回答，
        把证据里的数字也算成"回答里编的"，自评恒不通过。
        """
        llm = FakeLLM({"grounded": True, "coverage": 1.0})
        SelfEvaluator(llm).evaluate(GROUNDED_ANSWER, CONTEXT)
        # FakeLLM 没存 messages，用一个会记录的实现再跑一次
        recorded: list[Any] = []

        class Recorder(FakeLLM):
            def chat(self, messages, **kwargs):  # type: ignore[override]
                recorded.append(messages)
                return super().chat(messages, **kwargs)

        SelfEvaluator(Recorder({"grounded": True, "coverage": 1.0})).evaluate(
            GROUNDED_ANSWER, CONTEXT
        )
        blob = str(recorded[0])
        assert "<answer>" in blob and "</answer>" in blob


class TestSelfEvaluatorDegradation:
    def test_chat_exception_degrades_to_rules_without_raising(self) -> None:
        llm = FakeLLM(exc=RuntimeError("upstream 502"))
        result = SelfEvaluator(llm).evaluate(GROUNDED_ANSWER, CONTEXT)
        assert result.grounded is True  # 规则那路判定它是干净的
        assert "降级" in result.note
        assert "RuntimeError" in result.note

    def test_chat_exception_still_catches_fabrication(self) -> None:
        """降级不等于放行：模型挂了，规则那路照样要拦住编造的数字。"""
        llm = FakeLLM(exc=TimeoutError("timeout"))
        result = SelfEvaluator(llm).evaluate("签收后 480 小时内可退款。", CONTEXT)
        assert result.grounded is False
        assert "480 小时" in result.unsupported_claims

    @pytest.mark.parametrize(
        "garbage",
        ["", "不知道", "我觉得可以吧", '{"unclosed": ', "```json\nnot json\n```"],
    )
    def test_unparseable_response_degrades(self, garbage: str) -> None:
        llm = FakeLLM(garbage)
        result = SelfEvaluator(llm).evaluate(GROUNDED_ANSWER, CONTEXT)
        assert isinstance(result, SelfEvalResult)
        assert result.grounded is True  # 规则那路兜住
        assert "降级" in result.note

    def test_wrapped_payload_is_unwrapped(self) -> None:
        """模型爱把结果包一层；多探一层比整体降级划算。"""
        llm = FakeLLM({"self_eval": {"grounded": True, "coverage": 0.9, "note": "ok"}})
        result = SelfEvaluator(llm).evaluate(GROUNDED_ANSWER, CONTEXT)
        assert result.grounded is True
        assert result.coverage == pytest.approx(0.9)
        assert "降级" not in result.note

    def test_sloppy_types_are_coerced(self) -> None:
        """模型把 grounded 写成 "true"、coverage 写成 85 是常态。

        85 不矫正会变成 85.0 —— Pydantic 不报错，覆盖率 8500%，静默错。
        """
        llm = FakeLLM({"grounded": "true", "coverage": 85, "unsupported_claims": None})
        result = SelfEvaluator(llm).evaluate(GROUNDED_ANSWER, CONTEXT)
        assert result.grounded is True
        assert result.coverage == pytest.approx(0.85)
        assert result.unsupported_claims == []

    def test_empty_context_refuses_to_bless_the_answer(self) -> None:
        """没有证据时规则只能证明"某个数字没找到"，证明不了"整段没依据"——

        而后者恰恰是最要命的，所以必须显式拦。
        """
        llm = FakeLLM({"grounded": True, "coverage": 1.0})
        result = SelfEvaluator(llm).evaluate("生鲜类商品可以退。", "")
        assert result.grounded is False
        assert result.coverage == 0.0

    def test_empty_answer_is_not_grounded(self) -> None:
        result = SelfEvaluator(FakeLLM({"grounded": True, "coverage": 1.0})).evaluate(
            "", CONTEXT
        )
        assert result.grounded is False

    def test_deterministic(self) -> None:
        llm = FakeLLM({"grounded": True, "coverage": 1.0})
        evaluator = SelfEvaluator(llm)
        runs = [evaluator.evaluate(GROUNDED_ANSWER, CONTEXT).model_dump() for _ in range(5)]
        assert all(r == runs[0] for r in runs)
