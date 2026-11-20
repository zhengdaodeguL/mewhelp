"""NLU 层测试：指代消解 + 意图识别 + 槽位抽取。

测试的组织顺序刻意和"出事的代价"对齐：
  1. 消解/槽位这类**能不能正确理解业务实体**的用例（错了会退错单、改错地址）
  2. 意图分流的九类命中（错了会走错出口）
  3. 三层架构的行为契约（能省则省、异常不外泄）

第 3 组最关键：一个"永远调模型"的实现也能让前两组全绿，所以必须显式
断言"规则命中时 LLM 没被调用过"。这是省成本和降延迟的实际证据，不是口号。
"""

from __future__ import annotations

import pytest

from mewhelp.domain import INTENT_SPECS, Exit, Intent
from mewhelp.llm import LLMError, LLMResponse, MockLLMClient
from mewhelp.nlu import CoreferenceResolver, IntentClassifier
from mewhelp.schemas import SlotExtraction, Turn

# ---------------------------------------------------------------- 固定装置

def turn(role: str, content: str) -> Turn:
    return Turn(role=role, content=content)


#: 项目里最典型的多轮场景：上一轮用户报了订单号
HISTORY_WITH_ORDER = [
    turn("user", "我买的羊毛衫有点问题"),
    turn("assistant", "请问订单号是多少？"),
    turn("user", "订单 ORD-1001"),
    turn("assistant", "已找到订单 ORD-1001，是已签收状态。"),
]

HISTORY_WITH_PRODUCT = [
    turn("user", "我买的那款连衣裙到了"),
    turn("assistant", "好的，请问有什么可以帮您？"),
]


@pytest.fixture()
def llm() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture()
def resolver(llm) -> CoreferenceResolver:
    return CoreferenceResolver(llm=llm)


@pytest.fixture()
def classifier(llm) -> IntentClassifier:
    return IntentClassifier(llm=llm)


def _tasks(llm: MockLLMClient) -> list[str]:
    return [c["task"] for c in llm.calls]


# ================================================================
# 1. 指代消解
# ================================================================

def test_resolve_pronoun_to_order_id(resolver, llm):
    """"那它能退吗" —— 文章里点名的场景，必须补成带 ORD-1001 的完整问题。

    这句没有句内证据（"它"在句首，前后都没说是什么东西），所以允许走一层
    模型判定；关键是**结果**必须补对，而且补出来的实体在历史里查得到，
    引用的轮次也要标对（审计要能回答"这个单号哪来的"）。
    """
    result = resolver.resolve("那它能退吗", HISTORY_WITH_ORDER)

    assert result.rewritten is True
    assert "ORD-1001" in result.resolved
    assert "它" not in result.resolved
    assert result.substitutions == {"它": "订单 ORD-1001"}
    assert result.referenced_turns == [4]          # 单号出现在第 4 轮
    assert result.original == "那它能退吗"
    assert _tasks(llm) == ["resolve"]              # 只花一次，没有多余的 classify


def test_resolve_pronoun_to_order_id_by_rule_skips_llm():
    """同一句话在模型不可用时必须仍能补出来（规则兜底），且不额外调用模型。

    规则层不在这里"再想想"还有一个理由：把指代留给下游检索，"那它能退吗"
    这种半截话一定检索不到任何东西 —— 补一个来自历史的具体实体远好过不补。
    代价是可能补错类型（把商品补成订单），所以子流程仍会用槽位再校验一次。
    """

    class Boom(MockLLMClient):
        def chat(self, *a, **kw):
            raise LLMError("upstream 502")

    llm = Boom()
    resolver = CoreferenceResolver(llm=llm)
    result = resolver.resolve("那它能退吗", HISTORY_WITH_ORDER)

    assert result.rewritten is True
    assert "ORD-1001" in result.resolved
    assert result.substitutions == {"它": "订单 ORD-1001"}
    assert llm.calls == []          # 异常被吞掉，且没有重试风暴


def test_resolve_midsentence_pronoun_by_rule():
    """句中指代有句内证据（"这单"自带量词），纯规则搞定，一次模型都不调用。

    "订单 ORD-1001 是上个月买的，这单现在能改地址吗" 里，"改地址"是诉求，
    "订单 ORD-1001"才是主体 —— 语义域必须只看指代词之前的部分，
    否则"地址"会把这句话判成地址域。
    """
    llm = MockLLMClient()
    resolver = CoreferenceResolver(llm=llm)
    result = resolver.resolve("订单 ORD-1001 是上个月买的，这单现在能改地址吗", [])

    assert result.rewritten is True
    assert "ORD-1001 这单" in result.resolved
    assert result.substitutions == {"这单": "ORD-1001 这单"}
    assert result.referenced_turns == []      # 实体来自本句，不是历史
    assert llm.calls == []


def test_resolve_midsentence_entity_from_history_by_rule():
    """句中指代、本句没有实体时从历史取，同样不调模型。"""
    llm = MockLLMClient()
    resolver = CoreferenceResolver(llm=llm)
    result = resolver.resolve("这单能退吗", [turn("user", "订单 ORD-1001")])

    assert result.rewritten is True
    assert result.substitutions == {"这单": "订单 ORD-1001 这单"}
    assert result.referenced_turns == [1]
    assert llm.calls == []


def test_resolve_no_pronoun_keeps_original(resolver, llm):
    """没有指代词时不许硬改 —— 乱改写比不改写更伤检索。"""
    result = resolver.resolve("退货流程是什么", HISTORY_WITH_ORDER)

    assert result.rewritten is False
    assert result.resolved == "退货流程是什么"
    assert result.substitutions == {}
    assert result.referenced_turns == []
    assert llm.calls == []


@pytest.mark.parametrize("as_dict", [False, True], ids=["turn", "dict"])
@pytest.mark.parametrize(
    "notice",
    [
        "转接人工客服。",
        "已经帮你转接人工客服进一步处理。",
        "已为你转接人工客服，稍后会有同事继续跟进。",
        "这个问题我帮你转人工处理，稍后会有客服同事联系你。",
    ],
    ids=["short", "appended", "notice", "fallback"],
)
def test_handoff_history_is_not_an_entity(resolver, classifier, llm, notice, as_dict):
    """NLU 读取历史时去掉转接句，保留业务事实和引用轮次，不修改原记录。"""
    history = [turn("user", "请问尺码"), turn("assistant", f"连衣裙有S、M、L。\n\n{notice}")]
    if as_dict:
        history = [t.model_dump() for t in history]
    before = repr(history)
    result = resolver.resolve("这个尺码怎么选", history)
    slots = classifier.extract_slots("这个尺码怎么选", history)
    assert result.substitutions == {"这个": "连衣裙 这个"}
    assert result.referenced_turns == [2]
    assert slots.product == "连衣裙"
    assert repr(history) == before
    assert all(notice not in str(c["messages"]) for c in llm.calls)


def test_resolve_model_does_not_receive_handoff_history(resolver, llm):
    """规则找不到指代对象而调用模型时，转接句同样不能进入 prompt。"""
    history = [turn("assistant", "已经帮你转接人工客服进一步处理。")]
    result = resolver.resolve("换成红薯呢，它有哪些规格？", history)
    assert result.resolved == "换成红薯呢，它有哪些规格？"
    assert _tasks(llm) == ["resolve"]
    assert all("转接人工客服" not in str(c["messages"]) for c in llm.calls)


def test_resolve_size_question_prefers_product_over_order(resolver, llm):
    """"这个尺码"要补商品名，不能补订单号。

    这是"只在语义上说得通时才替换"的核心用例：同一句带"这个"的话，
    域不同补的东西就必须不同。补成订单号会让下游去查订单，而用户在问商品。
    历史里最近的实体是第 3 轮的羊毛衫（第 4 轮的 ORD-1001 属于另一个线程）。
    """
    history = HISTORY_WITH_PRODUCT + HISTORY_WITH_ORDER
    result = resolver.resolve("这个尺码怎么选", history)

    assert result.rewritten is True
    assert "羊毛衫" in result.resolved
    assert "ORD-1001" not in result.resolved
    assert result.substitutions == {"这个": "羊毛衫 这个"}


def test_resolve_nearest_product_wins_over_order(resolver):
    """问"这个能退吗"且上一轮在谈商品 → 补商品。

    售后动作默认指向订单，但用户嘴里最近提到的是那款连衣裙。补订单号虽然
    更"常规"，却会把用户真正问的东西丢掉；订单号仍会被 extract_slots 从
    历史里抽出来，高风险子流程不会因此缺槽位。
    """
    history = [turn("user", "我买的那款连衣裙到了"), turn("assistant", "好的，请问有什么可以帮您？")]
    result = resolver.resolve("这个能退吗", history)

    assert result.rewritten is True
    assert "连衣裙" in result.resolved
    assert "ORD" not in result.resolved


def test_resolve_determiner_not_expanded(resolver, llm):
    """"这个连衣裙能退吗"自带中心词，再补一次会把句子改坏。"""
    result = resolver.resolve("这个连衣裙能退吗", HISTORY_WITH_PRODUCT)

    assert result.rewritten is False
    assert result.resolved == "这个连衣裙能退吗"


def test_resolve_unknown_domain_falls_through_to_llm():
    """句中指代但前半句判不出语义域时交给模型，而不是猜一个。"""
    llm = MockLLMClient()
    resolver = CoreferenceResolver(llm=llm)
    resolver.resolve("上面的东西它怎么回事", [turn("user", "订单 ORD-1001")])

    assert _tasks(llm) == ["resolve"]


def test_resolve_no_entity_in_history_left_alone(resolver, llm):
    """历史里没有可补的实体 → 不许凭空造一个订单号。"""
    result = resolver.resolve("那它能退吗", [turn("user", "你好")])

    assert result.rewritten is False
    assert result.resolved == "那它能退吗"
    assert "ORD" not in result.resolved


def test_resolve_uses_model_when_rule_gives_up(resolver):
    """第二层确实接上了 mock 的分任务推理：句首指代 + 历史里有单号。"""
    result = resolver.resolve("那它能退吗", [turn("user", "MOCK 历史 ORD-1001")])
    # mock 的 resolve 只看历史里的订单号 + 当前句的"退"字，应该补出来
    assert result.rewritten is True
    assert "ORD-1001" in result.resolved


def test_resolve_llm_exception_returns_fallback():
    """异常注入：chat 抛异常、且规则也判不出来时必须降级成原文，绝不冒到调用方。"""

    class Boom(MockLLMClient):
        def chat(self, *a, **kw):
            raise LLMError("upstream 502")

    resolver = CoreferenceResolver(llm=Boom())
    # "上面那个"在句首 → 走模型；历史里也没有任何可补实体 → 只能原样返回
    result = resolver.resolve("上面那个怎么样了", [turn("user", "随便什么")])

    assert result.rewritten is False
    assert result.resolved == "上面那个怎么样了"
    assert result.substitutions == {}


def test_resolve_llm_returns_garbage():
    """模型返回不是 JSON 的废话 → 规则兜底接住，不崩。"""

    class Garbage(MockLLMClient):
        def chat(self, *a, **kw):
            from mewhelp.llm import LLMResponse

            return LLMResponse(text="我不太确定你在说什么。")

    resolver = CoreferenceResolver(llm=Garbage())
    result = resolver.resolve("那它能退吗", HISTORY_WITH_ORDER)

    # 模型没给可用结果，但历史里明确有单号 —— 规则兜底必须接住
    assert result.rewritten is True
    assert "ORD-1001" in result.resolved
    assert result.resolved != "那它能退吗"


def test_resolve_garbage_and_no_entity_returns_original():
    """模型没用、历史里也没有可补实体 → 返回原文（第三层兜底）。"""

    class Garbage(MockLLMClient):
        def chat(self, *a, **kw):
            from mewhelp.llm import LLMResponse

            return LLMResponse(text="你说啥")

    resolver = CoreferenceResolver(llm=Garbage())
    result = resolver.resolve("它怎么回事", [turn("user", "你好")])

    assert result.rewritten is False
    assert result.resolved == "它怎么回事"
    assert result.substitutions == {}


def test_resolve_empty_question():
    resolver = CoreferenceResolver(llm=MockLLMClient())
    result = resolver.resolve("", HISTORY_WITH_ORDER)
    assert result.resolved == ""
    assert result.rewritten is False


def test_resolve_is_deterministic(resolver):
    """同输入同输出 —— 确定性是硬性要求，专门钉一条。"""
    a = resolver.resolve("那它能退吗", HISTORY_WITH_ORDER)
    b = resolver.resolve("那它能退吗", HISTORY_WITH_ORDER)
    assert a.model_dump() == b.model_dump()


def test_resolve_accepts_dict_history(resolver):
    """反序列化过的 history 是 dict（SQLite / API body），不能因此崩掉。"""
    history = [{"role": "user", "content": "订单 ORD-2002"}]
    result = resolver.resolve("那它能退吗", history)
    assert "ORD-2002" in result.resolved


# ================================================================
# 2. 意图识别：九类各一例
# ================================================================

@pytest.mark.parametrize(
    "text,expected",
    [
        ("帮我查一下订单状态", Intent.ORDER_QUERY),
        ("我的快递到哪了", Intent.LOGISTICS_QUERY),
        ("我要申请退款", Intent.REFUND_REQUEST),
        ("我要退货，七天无理由怎么走", Intent.RETURN_REQUEST),
        ("帮我取消订单", Intent.CANCEL_ORDER),
        ("收货地址写错了帮我改一下", Intent.CHANGE_ADDRESS),
        ("这个尺码怎么选", Intent.PRODUCT_FAQ),
        ("运费怎么算，包邮吗", Intent.POLICY_FAQ),
        ("我要找人工客服", Intent.HUMAN_HANDOFF),
    ],
)
def test_classify_nine_intents(classifier, text, expected):
    result = classifier.classify(text)
    assert result.intent is expected, f"{text!r} → {result.intent} (matched={result.matched})"
    assert result.confidence >= 0.5
    assert result.matched, "命中必须有可复核的关键词"


def test_classify_rule_layer_does_not_call_llm(classifier, llm):
    """省钱证据：规则能定的意图一次模型都不调。"""
    classifier.classify("我要申请退款")
    assert llm.calls == []


def test_classify_long_keyword_weight_wins():
    """长度加权口径的定点验证："退款"压过"退货"里的"退"。"""
    clf = IntentClassifier(llm=MockLLMClient())
    result = clf.classify("我要退款")
    assert result.intent is Intent.REFUND_REQUEST
    assert "退款" in result.matched


# ============================================================
# 咨询 vs 请求：问规则的话不许触发动钱的流程
#
# 这是整份分类器里代价最不对称的一条：判成"请求"最多多问一次订单号，
# 判成"咨询"却是用户说了"我要退款"而系统回一段政策。
# 但反方向也不便宜 —— 「退货运费谁出？」被当成退货申请，系统会伸手向
# 用户要订单号，而他只是想问运费。下面两组用例把两个方向都钉住。
# ============================================================


@pytest.mark.parametrize(
    "text",
    [
        "你们家七天无理由是怎么规定的？",
        "七天无理由怎么规定的",
        "退货运费谁出？",
        "退货的运费谁承担",
        "退款要多久到账",
        "退款多长时间到账",
        "退货流程怎么走",
        "退款政策是什么",
        "生鲜类可以退款吗",
    ],
)
def test_policy_questions_do_not_enter_high_risk(classifier, text):
    """问规则的句子必须落到 policy_faq（knowledge 出口）。

    这些句子里大多带着"退货""退款"这种**动作词** —— 规则层会先把它认成
    高风险意图。区分靠的是 `_is_question` 加 `_demote_question_to_policy`：
    祈使标记（"帮我""我要"）和指向自己那单的说法（"我的""这单"）一票否决，
    剩下的看是不是在问定义、归属、时限。
    """
    result = classifier.classify(text)
    assert result.intent is Intent.POLICY_FAQ, (
        f"{text!r} → {result.intent}（matched={result.matched}）"
    )
    assert result.confidence >= 0.35, "低到闸门以下就等于答不出来"


@pytest.mark.parametrize(
    "text,expected",
    [
        # 带祈使标记：明确说了要做什么，句尾有"吗"也是请求
        ("我要退货，七天无理由怎么走", Intent.RETURN_REQUEST),
        ("能帮我退款吗", Intent.REFUND_REQUEST),
        ("不要了，帮我退掉", Intent.RETURN_REQUEST),
        ("麻烦帮我申请一下退货", Intent.RETURN_REQUEST),
        # 指向自己那一单：问的是这笔单子能不能操作，不是品类规则
        ("我的订单能退吗", Intent.RETURN_REQUEST),
        ("订单能撤吗", Intent.CANCEL_ORDER),
        ("这个订单我不要了", Intent.RETURN_REQUEST),
        ("收货地址写错了帮我改一下", Intent.CHANGE_ADDRESS),
    ],
)
def test_real_requests_still_enter_high_risk(classifier, text, expected):
    """反方向：真请求不许被"咨询"这道闸门挡掉。

    这组比上一组更重要 —— 用户明确说要退款，系统答一段政策，
    是"没办成事"，不是"答错了话"。
    """
    result = classifier.classify(text)
    assert result.intent is expected, f"{text!r} → {result.intent}"


def test_policy_reading_outranks_the_action_word(classifier):
    """"退货运费谁出？"里政策候选必须压过"退货"这个动作词。

    这里踩过一个坑：POLICY_FAQ 命中了"运费"，但因为 RETURN_REQUEST 带动作词，
    打分阶段给"运费"加了一道话题降权（×0.5）只剩 0.20，而"退货"降级后还有 0.44。
    降级函数改了排序却没改胜负 —— "谁出"这个再明确不过的咨询信号被抵消掉了。
    """
    from mewhelp.nlu.intent import _is_question

    assert _is_question("退货运费谁出？") is True
    scored = classifier._score("退货运费谁出？")
    assert scored[0][0] is Intent.POLICY_FAQ, f"第一名是 {scored[0][0]}，分数 {scored[:2]}"


def test_model_action_intent_is_demoted_on_policy_questions():
    """模型那一路也要被同一道业务铁律拦住。

    `_demote_question_to_policy` 只在**规则候选**上生效，"退款要多久到账"
    这种规则层不敢独断（覆盖度不够）的句子会被交给模型，而模型看着满句的
    "退款"会答 refund_request —— 一句咨询就这样走进了高风险出口。
    所以改判必须落在代码里，不能指望提示词里的 few-shot 每次都灵。
    """

    class ActionHappy(MockLLMClient):
        """无论问什么都答"要退货"的模型。"""

        def chat(self, messages, **kw):
            if kw.get("task") == "classify":
                self.calls.append({"task": "classify", "messages": list(messages)})
                return LLMResponse(
                    text='{"intent": "refund_request", "confidence": 0.95, "matched": ["退款"]}'
                )
            return super().chat(messages, **kw)

    clf = IntentClassifier(llm=ActionHappy())
    result = clf.classify("退款要多久到账")
    assert result.intent is Intent.POLICY_FAQ
    assert result.method == "demoted", "改判要能在 trace 里一眼看出来"


def test_demotion_does_not_touch_real_requests():
    """反方向同样要钉住：模型报了动作意图、句子又是真请求时，不许改判。"""

    class ActionHappy(MockLLMClient):
        def chat(self, messages, **kw):
            if kw.get("task") == "classify":
                self.calls.append({"task": "classify", "messages": list(messages)})
                return LLMResponse(
                    text='{"intent": "cancel_order", "confidence": 0.95, "matched": ["取消"]}'
                )
            return super().chat(messages, **kw)

    clf = IntentClassifier(llm=ActionHappy())
    result = clf.classify("帮我取消一下那笔单子")
    assert result.intent is Intent.CANCEL_ORDER
    assert result.method != "demoted"


def test_transfer_to_human_is_never_demoted_to_policy(classifier):
    """「可以转人工吗」必须是转人工，不能被"咨询降级"吞掉。

    这条曾经真的写错过：降级集合用的是 `_ACTION_ONLY`，而它把 HUMAN_HANDOFF
    也算成"动作"（"转人工"确实是用户要做的事）。于是"可以"两个字一命中咨询
    判据，用户明确要人工就变成了一段政策正文 —— 而且 handoff=false。
    比答错更糟：他以为有人会来，其实没有。
    """
    for text in ("转人工可以吗", "可以转人工吗", "可以找客服吗", "能转人工吗"):
        result = classifier.classify(text)
        assert result.intent is Intent.HUMAN_HANDOFF, f"{text!r} → {result.intent}"


def test_policy_question_with_no_rule_evidence_still_refuses_to_act():
    """规则层一个词都没命中时，"咨询不许动手"依然成立。

    "能补吗"命中了能愿结构（判定为咨询），但关键词表里一个词都命不中 ——
    此时若把模型给的 return_request 原样放行，一句咨询会直接走进高风险子流程；
    当轮文本里恰好带单号时，系统会真的登记退货。
    宁可退到一个低置信的咨询出口，让闸门去表达"答不上来"，也不能替用户退货。
    """

    class ActionHappy(MockLLMClient):
        def chat(self, messages, **kw):
            if kw.get("task") == "classify":
                self.calls.append({"task": "classify", "messages": list(messages)})
                return LLMResponse(
                    text='{"intent": "return_request", "confidence": 0.86, "matched": []}'
                )
            return super().chat(messages, **kw)

    from mewhelp.nlu.intent import _is_question

    text = "能补吗"
    assert _is_question(text), "前置条件：这句被判成咨询"
    clf = IntentClassifier(llm=ActionHappy())
    assert clf._score(text) == [], "前置条件：规则层没有任何候选"

    result = clf.classify(text)
    assert result.intent is Intent.POLICY_FAQ, f"→ {result.intent}"
    assert result.method == "demoted"
    assert result.confidence < 0.35, "低到闸门线以下，让'答不上来'由闸门表达"


def test_classify_tie_goes_to_model(llm):
    """规则层打平时不许用表顺序定胜负，必须多花一次模型调用。"""
    clf = IntentClassifier(llm=llm)
    result = clf.classify("我要退货退款")
    # 这句"退货"和"退款"各命中一词，规则层判不了 → 必须出现过 classify 调用
    assert "classify" in _tasks(llm)
    assert result.intent in {Intent.RETURN_REQUEST, Intent.REFUND_REQUEST}


def test_classify_unrelated_text_is_fallback(classifier):
    """完全无关的话 → fallback 标记 + 低置信（交给兜底出口）。

    "今天天气不错**适合**出去散步"里那个"适合"是商品咨询的关键词之一 ——
    这类"必须依附对象才有意义"的弱关键词不能单独支撑一个意图，否则闲聊
    会被答成商品咨询。所以只命中弱词时置信度被封顶，必然标 fallback。
    """
    result = classifier.classify("今天天气不错适合出去散步")

    assert result.method == "fallback"
    assert result.confidence < 0.45
    assert result.confidence < classifier.settings.confidence_threshold


def test_classify_weak_keyword_with_object_still_works(classifier):
    """弱关键词配上具体对象时不再算弱 —— 不能为了挡闲聊把真问题也挡掉。"""
    result = classifier.classify("这个材质适合敏感肌吗")
    assert result.intent is Intent.PRODUCT_FAQ
    assert result.confidence >= 0.5
    assert set(result.matched) >= {"材质", "适合"}


def test_classify_empty_text(classifier):
    result = classifier.classify("")
    assert result.method == "fallback"
    assert result.confidence < 0.35


def test_classify_llm_exception_returns_fallback():
    """异常注入：classify 通道炸了也不能崩。"""

    class Boom(MockLLMClient):
        def chat(self, *a, **kw):
            raise LLMError("timeout")

    clf = IntentClassifier(llm=Boom())
    result = clf.classify("我要退货退款")   # 打平句，必然走模型
    assert result.method == "fallback"
    assert result.confidence < 0.55


def test_classify_llm_bad_intent_name_ignored():
    """模型编了个不存在的意图名 → 当作没答，退回规则结果。"""

    class Liar(MockLLMClient):
        def chat(self, *a, **kw):
            from mewhelp.llm import LLMResponse

            return LLMResponse(text='{"intent": "complaint", "confidence": 0.99}')

    clf = IntentClassifier(llm=Liar())
    result = clf.classify("我要退货退款")
    assert result.intent in {Intent.RETURN_REQUEST, Intent.REFUND_REQUEST}


def test_classify_method_names_model_channel():
    """接真模型风格客户端时 method 应该是 llm 而不是 mock。"""

    class FakeRemote(MockLLMClient):
        pass

    FakeRemote.__name__ = "RemoteLLMClient"
    FakeRemote.model = "remote"
    clf = IntentClassifier(llm=FakeRemote())
    result = clf.classify("我要退货退款")
    assert result.method == "llm"


# ================================================================
# 3. 槽位抽取
# ================================================================

def test_slots_order_id_from_current_text(classifier):
    slots = classifier.extract_slots("订单 ORD-1234 退款", [])
    assert slots.order_id == "ORD-1234"


def test_slots_order_id_from_history(classifier):
    """当前句没说单号，但上一轮给过 —— 必须能从历史补上。

    这条直接决定高风险子流程要不要中断：抽不到就会反复追问用户，
    而用户上一轮明明已经答过了。
    """
    slots = classifier.extract_slots("那它能退吗", HISTORY_WITH_ORDER)
    assert slots.order_id == "ORD-1001"


def test_slots_current_text_beats_history(classifier):
    """用户改口说新单号时，必须用新的 —— 用对了才退对单。"""
    slots = classifier.extract_slots("算了，退订单 ORD-2002 吧", HISTORY_WITH_ORDER)
    assert slots.order_id == "ORD-2002"


def test_slots_product_and_address_and_reason(classifier):
    slots = classifier.extract_slots("我刚买的那款连衣裙色差太大了，我要退货", [])
    assert slots.product == "连衣裙"
    assert slots.reason == "色差"

    addr = classifier.extract_slots("收货地址改成杭州市西湖区文三路 100 号", [])
    assert addr.address is not None
    assert "杭州" in addr.address


def test_slots_no_false_order_id_from_house_number(classifier):
    """裸数字必须有"订单/单号"引导才认，否则门牌号会被当成订单号。"""
    slots = classifier.extract_slots("我家住在 1001 号，能改地址吗", [])
    assert slots.order_id is None


def test_slots_missing_for_drives_interrupt(classifier):
    """missing_for 的返回值决定中断 —— 两个方向都要验。"""
    no_order = classifier.extract_slots("我要退款", [])
    assert "order_id" in no_order.missing_for(Intent.REFUND_REQUEST)

    with_order = classifier.extract_slots("我要退款，订单 ORD-1001", [])
    assert with_order.missing_for(Intent.REFUND_REQUEST) == []

    # 改地址除了订单号还要地址
    no_addr = classifier.extract_slots("订单 ORD-1001 帮我改地址", [])
    assert no_addr.missing_for(Intent.CHANGE_ADDRESS) == ["address"]

    # 知识类意图本来就不需要订单号，别误报中断
    assert SlotExtraction().missing_for(Intent.POLICY_FAQ) == []


def test_slots_address_from_history(classifier):
    """改地址场景：地址在上一轮说过，这轮只说"就那个地址"。

    注意"那个地址"本身**不能**被当成地址值 —— 否则槽位看起来填好了、
    missing_for 不报警，改地址流程会拿着一个假地址去执行。
    """
    history = [turn("user", "收货地址改成杭州市西湖区文三路 100 号")]
    slots = classifier.extract_slots("订单 ORD-1001，还是寄到那个地址", history)
    assert slots.order_id == "ORD-1001"
    assert slots.address is not None and "杭州" in slots.address
    assert slots.missing_for(Intent.CHANGE_ADDRESS) == []


def test_slots_placeholder_address_is_not_a_slot(classifier):
    """"那个地址"不是地址值 —— 必须报缺槽位，让流程去追问而不是瞎执行。"""
    slots = classifier.extract_slots("订单 ORD-1001，寄到那个地址", [])
    assert slots.address is None
    assert slots.missing_for(Intent.CHANGE_ADDRESS) == ["address"]


def test_slots_llm_exception_keeps_rule_result():
    """异常注入：slots 通道炸了也要保住规则抽到的订单号（它是高风险流程的命根子）。"""

    class Boom(MockLLMClient):
        def chat(self, *a, **kw):
            raise LLMError("timeout")

    clf = IntentClassifier(llm=Boom())
    slots = clf.extract_slots("订单 ORD-1234", [])
    assert slots.order_id == "ORD-1234"


def test_slots_model_hallucinated_null_is_cleaned():
    """模型返回字符串 "null" 时不能当成填好的槽位。

    Pydantic 会照单全收 "null" 这个字符串，于是 missing_for 不再报警，
    中断机制静默失效 —— 这是最隐蔽的一类 bug，所以专门钉一条。
    """

    class Nuller(MockLLMClient):
        def chat(self, *a, **kw):
            from mewhelp.llm import LLMResponse

            return LLMResponse(
                text='{"order_id": "null", "address": "无", "reason": null, "product": null}'
            )

    clf = IntentClassifier(llm=Nuller())
    slots = clf.extract_slots("能改地址吗", [])
    assert slots.order_id is None
    assert slots.address is None
    assert slots.missing_for(Intent.CHANGE_ADDRESS) == ["order_id", "address"]


def test_slots_rule_and_resolve_agree_on_order_id():
    """消解补出来的单号与槽位抽出来的单号必须一致 —— 两份实现共用 entities，
    这条断言就是"别各写一套正则"的回归防护。"""
    llm = MockLLMClient()
    resolver = CoreferenceResolver(llm=llm)
    classifier = IntentClassifier(llm=llm)

    resolved = resolver.resolve("那它能退吗", HISTORY_WITH_ORDER)
    slots = classifier.extract_slots(resolved.resolved, HISTORY_WITH_ORDER)

    assert slots.order_id == "ORD-1001"
    assert slots.missing_for(Intent.REFUND_REQUEST) == []


# ================================================================
# 4. 回归：中文假阳性（"应该"里的"该"、"其中"里的"其"）
# ================================================================

@pytest.mark.parametrize(
    "question",
    ["应该怎么退款", "其中哪个能退", "其他订单能退吗"],
)
def test_no_pronoun_false_positive(resolver, question):
    result = resolver.resolve(question, HISTORY_WITH_ORDER)
    assert result.rewritten is False, f"{question!r} 被误改写为 {result.resolved!r}"
    assert result.resolved == question


def test_resolve_uses_topic_context_without_llm():
    """"最近在谈什么"可以由调用方作为 context 传进来。

    为什么需要这个口子：memory 层会把 `window_turns` 之外的轮次压成一句摘要，
    "上一轮在谈羊毛衫"这种关键线索就丢了。调用方手上还有原文时传进来，
    规则层直接命中，连模型都不用调 —— 这也是"两层架构省成本"最直接的体现。
    历史为空、context 里有实体时，纯规则就能给出正确答案。
    """
    llm = MockLLMClient()
    resolver = CoreferenceResolver(llm=llm)
    result = resolver.resolve("那它能退吗", [], context="我买的羊毛衫有点问题")

    assert result.rewritten is True
    assert "羊毛衫" in result.resolved
    assert result.substitutions == {"它": "羊毛衫"}
    assert llm.calls == []


def test_resolve_context_can_point_to_order():
    """context 指向订单时补订单号 —— 同一个指代词，靠上下文决定补什么。"""
    llm = MockLLMClient()
    resolver = CoreferenceResolver(llm=llm)
    result = resolver.resolve("那它能退吗", [], context="订单 ORD-1001 帮我查一下")

    assert result.rewritten is True
    assert "ORD-1001" in result.resolved


def test_resolve_no_cross_domain_borrowing():
    """"这个订单"自报订单域，就不能借上文里的商品名来补。

    跨域补全会产出"羊毛衫 这个订单能退吗"这种自相矛盾的句子，而且下游
    完全看不出来 —— 它只拿到一句通顺的话。这类静默矛盾必须挡在规则层。
    """
    llm = MockLLMClient()
    resolver = CoreferenceResolver(llm=llm)
    result = resolver.resolve(
        "这个订单能退吗", [], context="我买的羊毛衫有点问题"
    )

    assert result.rewritten is False, f"被跨域补全: {result.resolved!r}"
    assert result.resolved == "这个订单能退吗"


# ================================================================
# 5. 工程约束：确定性、离线可跑、无模型也能用
# ================================================================

def test_offline_no_llm_injected_still_works():
    """不注入 llm 时走 build_llm()，默认 mock —— 不依赖网络和 API Key。"""
    resolver = CoreferenceResolver()
    classifier = IntentClassifier()

    assert classifier.classify("我要退款").intent is Intent.REFUND_REQUEST
    assert resolver.resolve("那它能退吗", HISTORY_WITH_ORDER).rewritten is True


def test_classify_and_slots_are_deterministic(classifier):
    """同输入同输出 —— 确定性是硬性要求，两个接口都钉一条。"""
    a = classifier.classify("我要退货退款").model_dump()
    b = classifier.classify("我要退货退款").model_dump()
    assert a == b

    slots_a = classifier.extract_slots("那它能退吗", HISTORY_WITH_ORDER).model_dump()
    slots_b = classifier.extract_slots("那它能退吗", HISTORY_WITH_ORDER).model_dump()
    assert slots_a == slots_b


def test_rule_layer_rescues_official_examples():
    """契约里 INTENT_SPECS.examples 是"这些话必须能分对"的承诺。

    有两类例外，都是**有意为之**，不是缺陷：

    1. 「我上周买的东西到哪了」的落点是物流而不是查订单（问的是"到哪了"），
       判成 logistics_query 是更准的分流。

    2. 提问式咨询：「东西坏了能退钱吗」「这个能退吗」被归到 policy_faq。
       这是问/要分离的结果 —— 用户在问"这种情形能不能退"，不是在提交退款。
       判成 refund_request 会让系统去追问订单号，甚至在用户只想了解规则时
       就发起一笔真实退款。宁可答一段政策，也不能越权动钱。

    这条测试的价值是"覆盖一旦意外变化立刻暴露"，所以例外清单写死在这里。
    """
    classifier = IntentClassifier(llm=MockLLMClient())
    diffs = []
    for spec in INTENT_SPECS:
        for example in spec.examples:
            got = classifier.classify(example).intent
            if got is not spec.intent:
                diffs.append((spec.intent.value, example, got.value))

    expected = [
        ("order_query", "我上周买的东西到哪了", "logistics_query"),
        ("refund_request", "东西坏了能退钱吗", "policy_faq"),
        ("return_request", "这个能退吗", "policy_faq"),
    ]
    assert sorted(diffs) == sorted(expected), f"规则层的覆盖出现意外变化: {diffs}"


def test_official_examples_all_route_to_a_sane_exit():
    """比"分对哪个意图"更重要的性质：每个官方例子都要落到合理的出口。

    意图标签本身是有歧义的（"这个能退吗"到底是咨询还是申请，取决于上下文），
    但出口不能有歧义 —— 它决定了系统会不会去动用户的钱。
    所以这条测试断言的是出口集合，而不是标签字面。
    """
    from mewhelp.domain import INTENT_TO_EXIT

    classifier = IntentClassifier(llm=MockLLMClient())
    allowed = {
        "order_query": {Exit.BUSINESS_TOOL, Exit.KNOWLEDGE},
        "logistics_query": {Exit.BUSINESS_TOOL},
        "refund_request": {Exit.HIGH_RISK, Exit.KNOWLEDGE},
        "return_request": {Exit.HIGH_RISK, Exit.KNOWLEDGE},
        "cancel_order": {Exit.HIGH_RISK, Exit.KNOWLEDGE},
        "change_address": {Exit.HIGH_RISK, Exit.KNOWLEDGE},
        "product_faq": {Exit.KNOWLEDGE},
        "policy_faq": {Exit.KNOWLEDGE},
        "human_handoff": {Exit.HANDOFF},
    }
    for spec in INTENT_SPECS:
        for example in spec.examples:
            got = classifier.classify(example).intent
            exit_ = INTENT_TO_EXIT.get(got, Exit.FALLBACK)
            assert exit_ in allowed[spec.intent.value], (
                f"{example!r} 期望出口 {allowed[spec.intent.value]}，实际 {exit_.value}"
            )


@pytest.mark.parametrize(
    "text",
    [
        "红心柚从哪里发货？",
        "生鲜收到货24小时内有品质问题退多少？",
        "七天无理由退货从哪天开始算？",
        "退货需要把商品寄回吗？",
    ],
)
def test_通用售后问题不索要订单号(text):
    """没有订单指向或明确操作词的通用咨询必须走知识出口。"""
    result = IntentClassifier(llm=MockLLMClient()).classify(text)
    assert result.intent in {Intent.PRODUCT_FAQ, Intent.POLICY_FAQ}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("我的订单从哪里发货？", Intent.ORDER_QUERY),
        ("ORD-1007 退货需要把商品寄回吗？", Intent.RETURN_REQUEST),
        ("帮我退货，需要把商品寄回吗？", Intent.RETURN_REQUEST),
        ("我要退货，七天无理由从哪天开始算？", Intent.RETURN_REQUEST),
    ],
)
def test_具体订单和明确操作仍走业务流程(classifier, text, expected):
    assert classifier.classify(text).intent is expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("申请退货，需要把商品寄回吗？", Intent.RETURN_REQUEST),
        ("转人工，红心柚从哪里发货？", Intent.HUMAN_HANDOFF),
    ],
)
def test_通用问法不抢占明确申请和转人工(classifier, text, expected):
    assert classifier.classify(text).intent is expected


@pytest.mark.parametrize("question", ["那3斤的是哪个店的？", "那个是哪个店的？"])
def test_商品追问承接上轮已确认答复(resolver, question):
    history = [
        turn("user", "苹果有哪些规格，分别是哪个店的？"),
        turn("assistant", "苹果3斤是红运道旗舰店，5斤是果园旗舰店。"),
    ]
    result = resolver.resolve(question, history)
    assert result.rewritten is True
    assert "苹果" in result.resolved
    assert "3斤是红运道旗舰店" in result.resolved
    assert result.resolved.endswith(question)
    assert result.referenced_turns == [1, 2]


@pytest.mark.parametrize(
    "question",
    ["这个连衣裙多少钱？", "那个羊角蜜带箱5斤，就是5斤净重吗？"],
)
def test_商品追问自带商品名时不借用上轮话题(resolver, question):
    history = [
        turn("user", "苹果有哪些规格，分别是哪个店的？"),
        turn("assistant", "苹果3斤是红运道旗舰店，5斤是果园旗舰店。"),
    ]
    result = resolver.resolve(question, history)
    assert result.rewritten is False
    assert result.resolved == question
    assert "苹果" not in result.resolved
    assert result.referenced_turns == []
    assert result.substitutions == {}


@pytest.mark.parametrize(
    "answer",
    [
        "苹果13斤是红运道旗舰店。",
        "苹果3斤无法确认店铺，帮你转人工。",
        "订单 ORD-1007 的苹果3斤来自红运道旗舰店。",
    ],
)
def test_商品规格追问不借用未确认或订单上下文(resolver, answer):
    result = resolver.resolve(
        "那3斤的是哪个店的？",
        [turn("user", "苹果有哪些规格？"), turn("assistant", answer)],
    )
    assert result.rewritten is False


def test_商品规格追问没有历史时保留原文(resolver):
    question = "那3斤的是哪个店的？"
    assert resolver.resolve(question, []).resolved == question


@pytest.mark.parametrize(
    "text,expected",
    [
        ("羊角蜜5斤带箱子，果肉有5斤吗？", "羊角蜜5斤带箱"),
        ("这个超轻便携保温杯多少钱？", "超轻便携保温杯"),
    ],
)
def test_商品名按词语边界保留首字(classifier, text, expected):
    assert classifier.extract_slots(text, []).product == expected
