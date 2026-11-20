"""意图识别 + 槽位抽取。

同样是分层：规则高速通道 → 模型（task="classify" / "slots"）→ 兜底。
每一层都必须能独立给结果，因为下游的 route 靠 `intent` 决定走哪个出口，
而高风险出口会真的动用户的钱。

规则层的打分口径（刻意做成可解释的，因为面试/论文都要能一句话讲清楚）：

    单个关键词权重  w      = 0.30 + 0.10 * (len(kw) - 1)，上限 1.0
    意图原始得分    raw    = min(1.0, Σ w(命中的关键词))，多命中 ×1.05
    动作词加权      句中出现**动作词**的意图 raw ×= 2.0（上限仍为 1.0）
    话题词降权      句中有动作候选时，只命中话题词的意图 raw ×= 0.5
    弱关键词封顶    只命中"适合/几天"这类依附性词 → conf ≤ 0.30
    覆盖度          cover  = best / (best + second)
    置信度          conf   = round(0.50 * best + 0.50 * cover, 4)
    开门条件        conf ≥ 0.55 **且** cover ≥ 0.75 才返回 method="rule"

为什么用长度加权：中文里长词天然更具体。"退款"两个字基本锁死了意图，
而"退"一个字既能是退款也能是退货还是退票；如果两者同权，一句"我要退款"
里"退款"和"退"会被算成两次独立证据，反而把 cancel_order 里的"取消"抬上去。
长度加权是"信息量"最廉价的近似 —— 不需要模型、不需要词频统计、完全确定性。

为什么要有"动作词 / 话题词"这两档权重：这是本层最关键的一条。
"帮我取消订单" 里 "订单" 是**话题**（在聊什么），"取消" 是**动作**（要干什么）。
两者同权时它们打成 0.5 : 0.5，规则层只能推给模型 —— 而这句话的意图其实现行
可见。动作词是用户诉求的落点，天然比话题词高一个层级，所以给动作候选加倍、
给纯话题候选减半（只在句子里真的存在动作候选时才减，避免压低本来就没歧义
的句子，比如"这个尺码怎么选"）。

为什么要看覆盖度：只看绝对分会让"我要退货退款"这种多意图句子两个意图都拿
高分，而它们其实在互相解释同一句话。用 second/best 的比值衡量"第二名能解释
多少"，比写死一个固定差值稳，也不需要为每类意图调参。打平时不许用
INTENT_SPECS 的书写顺序来定胜负 —— 那等于把"改一行排列"变成"改一次分流"。

关键词表不可能完备（"我不想买了""能撤吗"这种口语说法穷举不完），所以补了
一层轻量正则兜底（_EXTRA_PATTERNS）。它的命中**不参与关键词计数**，只用来
判断"这个意图有没有被表达出来"，理由见 _score 里的注释。
"""

from __future__ import annotations

import logging
import re

from ..config import get_settings
from ..domain import HIGH_RISK_INTENTS, INTENT_KEYWORDS, INTENT_LABELS, INTENT_SPECS, Intent
from ..llm import as_bool, build_llm, extract_json
from ..knowledge import is_damage_policy_question, product_attribute_question, product_subject
from ..schemas import ClassifyResult, SlotExtraction, Turn
from . import entities as E

logger = logging.getLogger(__name__)

#: 关键词表不保证完备（"我不想买了""能撤吗"这种转折/口语说法穷举不完），
#: 而规则通道又必须在这些说法上命中才谈得上"省一次调用"。所以补一张
#: 轻量正则兜底：命中即算该意图的一个额外关键词。
_EXTRA_PATTERNS: dict[Intent, tuple[str, ...]] = {
    Intent.CANCEL_ORDER: (r"(?:不|别)(?:想|要)(?:买|要)了", r"能撤吗", r"撤(?:掉|了|销)"),
    Intent.RETURN_REQUEST: (r"退(?:这|那|它)(?:个|单|款)?", r"能退吗", r"可以退吗"),
    Intent.REFUND_REQUEST: (r"退(?:我|回)?钱", r"返(?:我)?钱", r"退(?:运费|邮费|差价|税)"),
    Intent.LOGISTICS_QUERY: (
        r"到哪(?:了|儿)", r"几号(?:能)?到", r"什么(?:时候|时)?(?:能)?(?:送到|到货|到)",
    ),
}
_EXTRA_COMPILED: dict[Intent, tuple[re.Pattern[str], ...]] = {
    intent: tuple(re.compile(p) for p in pats) for intent, pats in _EXTRA_PATTERNS.items()
}

#: 补充关键词。`domain.INTENT_KEYWORDS` 是冻结契约不能改，但它在真实说法上
#: 有缺口：官方例子里的"换成公司地址"一个词都不命中，直接掉进 fallback。
#: 这里以**追加**（不改写原表）的方式补齐，冻结契约原封不动，规则层也能用。
#: 只补"漏词"，不补"改判" —— 判据口径仍归 _score 那一套。
_EXTRA_KEYWORDS: dict[Intent, tuple[str, ...]] = {
    Intent.ORDER_QUERY: ("这单", "那单"),
    Intent.CHANGE_ADDRESS: ("换成", "改成", "改一下地址"),
    Intent.LOGISTICS_QUERY: ("送到", "到货", "何时到"),
}

#: "弱关键词"：必须有对象才能确定意图的属性/程度词。
#: "适合"单独出现既可能是"这个材质适合敏感肌吗"，也可能是"今天天气适合散步"；
#: "几天"可能是政策（"几天无理由"）也可能是物流。只命中这类词时**不认**这个意图 ——
#: 这是把闲聊挡在门外唯一可行的办法，因为关键词表本身是按意图列词、不区分强弱。
_WEAK_KEYWORDS: dict[Intent, tuple[str, ...]] = {
    Intent.PRODUCT_FAQ: ("适合", "能不能用", "多大", "参数"),
    Intent.POLICY_FAQ: ("几天",),
    Intent.ORDER_QUERY: ("买了",),
}
#: 只有弱证据时的置信度上限 —— 必须**严格低于** settings.confidence_threshold
#: （默认 0.35），否则一句闲聊会被闸门判成"够格直接回答"
_WEAK_ONLY_CONF_CAP = 0.30

#: 单个关键词的基准权重。定成 0.30 是为了让"命中一个两字词"刚好跨过
#: settings.confidence_threshold（默认 0.35），不至于每条规则都要求多词命中。
_BASE_W = 0.30
_LEN_W = 0.10
_MAX_W = 1.0
#: 多关键词加成 —— 加得很小，因为"多命中一个词"并不等于线性增加确定性
_MULTI_BONUS = 1.05
#: 规则通道开门线：低于它就把决定权交给模型
_RULE_CONF_MIN = 0.55
#: 覆盖度开门线：best / (best + second) 低于它，说明这句话被两个意图共同
#: 解释，规则层不该独断 —— 交给模型或者标 fallback
_RULE_COVER_MIN = 0.75
#: 动作候选的加权与纯话题候选的降权（口径见模块 docstring）
_ACTION_BOOST = 2.0
_TOPIC_DISCOUNT = 0.5

#: 规则层判不出来时给模型用的 few-shot —— 中文意图的边界（退款 vs 退货 vs
#: 取消）用文字描述很难说清，给例子最快。这段提示词在接真模型时才起作用，
#: 但它属于"业务知识"，放在代码里而不是配置文件里，是为了让改意图的人一眼看到。
_FEWSHOT = (
    ("那它能退吗", "return_request"),
    ("钱什么时候退回来", "refund_request"),
    # 和上一条的差别就在"我那笔钱"和"退款这件事"：问时限、问规则的一律 policy_faq。
    # 没有这条例子时，模型对「退款要多久到账」会给 refund_request —— 字面没错，
    # 但它会把一句咨询推进高风险出口去要订单号。代码里还有一道 `_demoted` 兜底，
    # 这里的例子是为了让模型**第一遍就答对**，而不是每次都靠兜底纠偏。
    ("退款要多久到账", "policy_faq"),
    ("这个尺码偏小吗", "product_faq"),
    ("运费怎么算", "policy_faq"),
    ("别发了", "cancel_order"),
)


class IntentClassifier:
    """规则优先的意图分类器。"""

    def __init__(self, llm=None, settings=None) -> None:
        self.settings = settings or get_settings()
        self.llm = llm if llm is not None else build_llm(self.settings)

    # ------------------------------------------------------------------
    # 意图识别
    # ------------------------------------------------------------------
    def classify(self, text: str) -> ClassifyResult:
        text = text or ""
        if not text.strip():
            return self._fallback("空文本")

        try:
            scored = self._score(text)
            rule = self._rule_result(scored, text)
            if rule is not None:
                return rule

            model = self._model_result(text)
            if model is None:
                return self._best_effort(scored) if scored else self._fallback("规则与模型都没有结果")
            # 规则层只拿到弱证据（"适合""几天"这类必须依附对象的词）、而模型给出的
            # 又是同一个意图时，说明模型也只是复用了同一条弱证据 —— 那它仍然是弱的，
            # 不该因为"模型说了"就升级成可信结论。模型若给出**不同**的意图，
            # 就是它看到了词表没有的线索，听它的。
            if scored and _weak_evidence_only(scored[0], text) and model.intent is scored[0][0]:
                return self._best_effort(scored)
            # 咨询句里模型回了"要动手"的意图 —— 不听它的，按规则降级成政策咨询。
            #
            # 「退款要多久到账」在模型眼里满是"退款"，它给 refund_request 是个
            # 合理的字面判断；但那句话一个祈使词都没有、问的是规则，不是自己那单。
            # `_demote_question_to_policy` 只降得住**规则候选**，模型这一路它够不着 ——
            # 于是一句咨询照样命中高风险出口，系统伸手向用户要订单号。
            #
            # 这道判断必须落在代码里而不是提示词里：它是一条**业务铁律**
            # （咨询不许触发动手的流程），不能指望模型每次都听话。
            # 提示词里的 few-shot 只负责让模型第一遍就答对，答不对时由这里兜住。
            if _is_question(text) and model.intent in _DEMOTABLE_INTS:
                return self._demoted(scored, model)
            if model.method != "fallback" and model.confidence >= _RULE_CONF_MIN:
                return model
            # 模型也只给出低置信（甚至自己标了 fallback）时，规则层的词表直觉
            # 通常更贴近业务 —— 但整体降级为 fallback，不假装这是可靠判断。
            return self._best_effort(scored) if scored else model
        except Exception as exc:  # noqa: BLE001 —— 异常不外泄
            return self._fallback(str(exc))

    def _demoted(
        self,
        scored: list[tuple[Intent, float, list[str], float]],
        model: ClassifyResult,
    ) -> ClassifyResult:
        """把模型给出的高风险动作意图改判成政策咨询。

        置信度沿用规则层降级后的那个候选（`_score` 里的合成项），而不是
        照搬模型的分数 —— 模型报的 0.95 是它对"这句话里有'退款'"的把握，
        跟"该不该动钱"没关系，照搬会把一次改判包装成一次高置信命中。

        标 `method="demoted"` 而不是复用 "rule"/"fallback"：trace 里要能一眼
        看出"这条是被改判的"，否则排查时会以为是模型自己改的口径。
        """
        if not scored:
            # 规则层一个词都没命中（"生鲜的东西坏了能赔吗"就是），没有可依据的
            # 降级口径 —— 但**"咨询不许动手"是硬约束**，不能因为规则层没词就放行：
            # 那样模型给的 return_request 会直接走进高风险子流程，把一句咨询
            # 当成退货申请真跑起来（当轮带单号时还会真的登记退货）。
            #
            # 退到咨询出口，置信度取 `_fallback` 那一档（0.10）。这不是判断，
            # 是"不敢动" —— 答不出来该由闸门和转人工去表达，而不是替用户退货。
            logger.warning(
                "咨询句里模型给了 %s，但规则层无候选可降级，退到 %s",
                model.intent.value,
                _QUESTION_INTENT.value,
            )
            return ClassifyResult(
                intent=_QUESTION_INTENT,
                confidence=_DEMOTED_NO_EVIDENCE_CONF,
                label=INTENT_LABELS[_QUESTION_INTENT],
                matched=[_QUESTION_SYNTHETIC_MATCH],
                method="demoted",
            )
        demoted = self._best_effort(scored)
        return ClassifyResult(
            intent=demoted.intent,
            confidence=demoted.confidence,
            label=demoted.label,
            matched=demoted.matched,
            method="demoted",
        )

    def _score(self, text: str) -> list[tuple[Intent, float, list[str], float]]:
        """返回 [(意图, 置信度, 命中词, 有效得分)]，按置信度降序。"""
        # 通用问法只在没有具体订单、操作申请或转人工诉求时优先。
        if not (
            E.find_order_id(text)
            or _OWN_ORDER_RE.search(text)
            or _IMPERATIVE_RE.search(text)
            or "申请" in text
            or any(kw in text for kw in INTENT_KEYWORDS[Intent.HUMAN_HANDOFF])
        ):
            if is_damage_policy_question(text):
                return [(Intent.POLICY_FAQ, 0.90, ["售后比例咨询"], 1.0)]
            if product_attribute_question(text):
                matched = [kw for kw in INTENT_KEYWORDS[Intent.PRODUCT_FAQ] if kw in text]
                return [(Intent.PRODUCT_FAQ, 0.90, matched or ["商品属性咨询"], 1.0)]
            for pattern, intent in (
                (_ORIGIN_QUESTION_RE, Intent.PRODUCT_FAQ),
                (_AFTERSALES_RULE_RE, Intent.POLICY_FAQ),
            ):
                if match := pattern.search(text):
                    return [(intent, 0.90, [match.group(0)], 1.0)]
        # 先算出每个意图的命中证据，再统一判断"谁带动作词" —— 两档权重必须
        # 基于同一份证据，否则"动作候选"的判定会和打分口径脱节。
        hits: list[tuple[Intent, list[str], bool, bool]] = []
        for spec in INTENT_SPECS:
            rule_kws = INTENT_KEYWORDS[spec.intent] + _EXTRA_KEYWORDS.get(spec.intent, ())
            matched = [kw for kw in rule_kws if kw in text]
            extra = [
                m.group(0)
                for p in _EXTRA_COMPILED.get(spec.intent, ())
                if not matched and (m := p.search(text))
            ]
            if not matched and not extra:
                continue
            # 正则兜底只用来判断"这个意图有没有被表达出来"，不参与关键词计数。
            # 命中的是用户**原话片段**（如"我不想买了"），进 matched 才能复核。
            action_backed = bool(extra) or _is_action_backed(spec.intent, text)
            weak_only = (
                not extra
                and bool(matched)
                and all(kw in _WEAK_KEYWORDS.get(spec.intent, ()) for kw in matched)
            )
            hits.append((spec.intent, matched or extra, action_backed, weak_only))

        if not hits:
            return []
        has_action_candidate = any(action for _i, _m, action, _w in hits)

        eff: list[tuple[Intent, float, list[str], bool]] = []
        for intent, matched, action_backed, weak_only in hits:
            score = min(_MAX_W, sum(self._kw_weight(kw) for kw in matched))
            if len(matched) > 1:
                score = min(_MAX_W, score * _MULTI_BONUS)
            if action_backed:
                score = min(_MAX_W, score * _ACTION_BOOST)
            elif has_action_candidate:
                score *= _TOPIC_DISCOUNT
            eff.append((intent, score, matched, weak_only))
        eff.sort(key=lambda x: (-x[1], x[0].value))

        best = eff[0][1]
        second = eff[1][1] if len(eff) > 1 else 0.0
        # 覆盖度只由最高分那一个享有（其它名次拿 0），保证排序稳定且可解释
        cover = best / (best + second)

        out: list[tuple[Intent, float, list[str], float]] = []
        for rank, (intent, score, matched, weak_only) in enumerate(eff):
            conf = 0.50 * score + 0.50 * (cover if rank == 0 else 0.0)
            # 只有弱证据的意图封顶：允许它参与排序（万一没有别的候选），
            # 但封顶值低于开门线，所以最终一定会被交给模型或标 fallback。
            if weak_only:
                conf = min(conf, _WEAK_ONLY_CONF_CAP)
            out.append((intent, round(conf, 4), matched, score))
        # 最后一道：把"问规则"和"提请求"分开。
        # 放在打分之后、返回之前，是为了不动上面那套排序口径 ——
        # 高风险动作的降级是一条独立的业务安全规则，不是打分的一部分。
        return _demote_question_to_policy(out, text)

    @staticmethod
    def _kw_weight(kw: str) -> float:
        return min(_MAX_W, _BASE_W + _LEN_W * (len(kw) - 1))

    def _rule_result(
        self, scored: list[tuple[Intent, float, list[str], float]], text: str
    ) -> ClassifyResult | None:
        """唯一高分意图 → 规则命中；打平或不够高 → None（交给模型）。"""
        if not scored:
            return None
        intent, conf, matched, score = scored[0]
        if conf < _RULE_CONF_MIN:
            return None
        second = scored[1][3] if len(scored) > 1 else 0.0
        cover = score / (score + second)
        if cover < _RULE_COVER_MIN:
            # 打平的情况允许一个例外："帮我取消订单"里"订单"是**话题**、
            # "取消"是**动作**。当只有一个候选带动作词时，动作就是用户的落点，
            # 让带动作的那个胜出，而不是把判断推给模型。
            action_backed = [
                item for item in (scored[0], scored[1]) if _is_action_backed(item[0], text)
            ]
            if not (len(action_backed) == 1 and _is_action_backed(intent, text)):
                return None
        return ClassifyResult(
            intent=intent,
            confidence=conf,
            label=INTENT_LABELS[intent],
            matched=matched,
            method="rule",
        )

    def _model_result(self, text: str) -> ClassifyResult | None:
        prompt = [
            {
                "role": "system",
                "content": (
                    "你是电商客服的意图分类器。把用户的话归到下面九类之一："
                    + "、".join(f"{s.intent.value}({s.label})" for s in INTENT_SPECS)
                    + "。示例："
                    + "；".join(f'"{q}" → {i}' for q, i in _FEWSHOT)
                    + "。只输出 JSON："
                    '{"intent": str, "confidence": float, "matched": [str], "fallback": bool}'
                ),
            },
            {"role": "user", "content": text},
        ]
        resp = self.llm.chat(prompt, temperature=0.0, json_mode=True, task="classify")
        data = resp.json()
        if not isinstance(data, dict):
            return None
        try:
            intent = Intent(str(data.get("intent")))
        except ValueError:
            return None  # 模型编了个不存在的意图名 —— 当作没答

        conf = _clamp(data.get("confidence"), 0.0, 1.0)
        # 必须过 `as_bool`：模型常把布尔写成字符串，而字符串 "false" 是真值。
        # 直接 `if data.get("fallback")` 会把"我很确定"读成"我拿不准"，
        # 于是这次判断被丢掉、换成规则层那个本来就没过开门线的弱候选 ——
        # 出口都可能跟着换掉，而日志里一切正常。同文件其它解析点
        # （confidence 有 _clamp、intent 有 Intent(str(...))）都做了归一，就这里漏了。
        method = "fallback" if as_bool(data.get("fallback")) else _method_of(self.llm)
        matched = [str(m) for m in (data.get("matched") or []) if m]
        return ClassifyResult(
            intent=intent,
            confidence=conf,
            label=INTENT_LABELS[intent],
            matched=matched,
            method=method,
        )

    def _best_effort(self, scored: list[tuple[Intent, float, list[str], float]]) -> ClassifyResult:
        """规则没过线、模型也不可用时的最好猜测。

        必须标 method="fallback"：这个字段下游会用来判断"这个意图可不可信"，
        谎报成 rule/llm 会让低置信意图被当成正常结果继续走高风险流程。
        """
        if not scored:
            return self._fallback("没有任何意图关键词命中")
        intent, conf, matched, _score = scored[0]
        return ClassifyResult(
            intent=intent,
            confidence=conf,
            label=INTENT_LABELS[intent],
            matched=matched,
            method="fallback",
        )

    @staticmethod
    def _fallback(error: str) -> ClassifyResult:
        """彻底猜不出来的时候。

        默认给 policy_faq 而不是随便挑一个：它是 knowledge 出口，下游会用
        知识库检索 + 置信度闸门兜底 —— 闸门过不了就拒答转人工。相比之下
        给一个高风险意图会让系统去找用户要订单号，属于无中生有的骚扰。
        """
        logger.warning("intent 兜底: %s", error)
        return ClassifyResult(
            intent=Intent.POLICY_FAQ,
            confidence=0.10,
            label=INTENT_LABELS[Intent.POLICY_FAQ],
            matched=[],
            method="fallback",
        )

    # ------------------------------------------------------------------
    # 槽位抽取
    # ------------------------------------------------------------------
    def extract_slots(
        self, text: str, history: list[Turn] | None = None, *, intent: Intent | None = None,
    ) -> SlotExtraction:
        """抽槽位。**当前句优先，历史兜底**。

        顺序不能反：用户改主意的情况很常见（上一轮说 ORD-1001，这一轮说
        "算了，还是退 ORD-2002 吧"），如果历史优先，系统会一直拿着旧单号
        去操作 —— 而这是会真实退钱的路径，错单比查不到更严重。
        """
        text = text or ""
        turns = list(E.iter_turns(history))
        try:
            slots = SlotExtraction(
                order_id=E.find_order_id(text) or E.nearest_entity(turns, E.find_order_id)[0],
                address=E.find_address(text) or E.nearest_entity(turns, E.find_address)[0],
                reason=E.find_reason(text) or E.nearest_entity(turns, E.find_reason)[0],
                product=E.find_product(text) or E.nearest_entity(turns, E.find_product)[0],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("slots 规则层异常，降级为空槽位: %s", exc)
            slots = SlotExtraction()

        # 按这次路由的实际需要补槽位。高风险流程会重新从当轮原话取必填值，
        # 不消费这里的模型槽位；商品属性有明确主体时仍须经过检索/闸门校验。
        if intent in HIGH_RISK_INTENTS or intent is Intent.HUMAN_HANDOFF:
            return slots
        if intent in {Intent.ORDER_QUERY, Intent.LOGISTICS_QUERY} and slots.order_id:
            return slots
        if intent is Intent.PRODUCT_FAQ and product_attribute_question(text):
            subject = product_subject(text)
            # 属性主体规则不是通用实体识别器。“换成红薯呢，它有哪些规格”
            # 等复合问句仍走原来的抽取，不能把整段话写进 product 槽位。
            if (re.fullmatch(r"[一-龥A-Za-z0-9·_-]{2,32}", subject)
                    and not re.search(r"换成|改成|[呢吗吧它他她我你]", subject)):
                return slots.model_copy(update={"product": subject})

        # 历史里没抽到、或抽到的东西不可信时才问模型。
        # 四个槽位都齐了还调模型是纯浪费；而高风险流程真正在乎的 order_id
        # 一旦缺了，多花一次调用去别的说法里找一找是值得的。
        if slots.order_id and slots.address and slots.product and slots.reason:
            return slots
        try:
            enriched = self._model_slots(text, turns)
        except Exception as exc:  # noqa: BLE001
            logger.warning("slots 模型层异常，保留规则结果: %s", exc)
            return slots

        # 合并口径：规则抽到的一律保留（正则可核验），模型只补规则漏掉的空位。
        # 反过来让模型覆盖正则结果，会出现"模型幻觉出一个订单号"的路径。
        return SlotExtraction(
            order_id=slots.order_id or _clean(enriched.get("order_id")),
            address=slots.address or _clean(enriched.get("address")),
            reason=slots.reason or _clean(enriched.get("reason")),
            product=slots.product or _clean(enriched.get("product")),
        )

    def _model_slots(self, text: str, turns: list[tuple[int, str, str]]) -> dict:
        lines = [f"{role}: {t}" for _i, role, t in turns]
        content = "已有历史：\n" + ("\n".join(lines) or "（无）") + "\n当前这句：\n" + text
        prompt = [
            {
                "role": "system",
                "content": (
                    "从对话里抽取槽位。只抽用户明确说过的信息，不要猜。只输出 JSON："
                    '{"order_id": str|null, "address": str|null, '
                    '"reason": str|null, "product": str|null}'
                ),
            },
            {"role": "user", "content": content},
        ]
        resp = self.llm.chat(prompt, temperature=0.0, json_mode=True, task="slots")
        data = extract_json(resp.text)
        return data if isinstance(data, dict) else {}


# ----------------------------------------------------------------


#: 话题词 vs 动作词 —— 判断一个意图的命中里有没有"用户真正要做的动作"。
#: 取消订单的"取消"、退款的"退款"是动作；查订单的"订单"、商品咨询的"尺码"
#: 是话题。区分它们，是为了在"帮我取消订单"这种动作+话题的句子里，
#: 不让话题（订单）把动作（取消）压掉。
_ACTION_ONLY: dict[Intent, tuple[str, ...]] = {
    Intent.CANCEL_ORDER: ("取消", "撤销", "撤单", "别发", "不想要", "能撤", "撤吗"),
    Intent.REFUND_REQUEST: ("退款", "退钱", "返钱", "退费"),
    Intent.RETURN_REQUEST: ("退货", "退掉", "寄回", "不要了", "七天无理由"),
    Intent.CHANGE_ADDRESS: ("改地址", "修改地址", "换地址", "地址写错", "改收货"),
    Intent.HUMAN_HANDOFF: ("人工", "客服", "投诉", "转接", "真人"),
}


#: 疑问标记 —— 判断用户是在"问"还是在"要"。
#:
#: 这个区分不是语言学的洁癖，是**业务安全的必需品**：高风险出口会真的动钱。
#: 「生鲜类可以退款吗」和「我要退款」如果都走高风险流程，前者会被当成
#: 一个缺订单号的退款申请而中断追问 —— 用户只是想了解规则，系统却伸手
#: 向他要订单号；更糟的是当轮恰好带了订单号时，一句咨询会直接触发真退款。
#:
#: 所以：带疑问标记、且没有祈使动作的句子，视为**规则咨询**而不是**请求**。
#: 疑问标点。**不要**把单个"么"放进来 —— 它会命中"什么/怎么/这么/那么"，
#: 于是"什么时候能送到""七天无理由怎么退"这类明确的请求全被误判成咨询。
#: 疑问语气词只认语义完整的"吗/呢"和问号。
_QUESTION_MARK_RE = re.compile(r"吗|呢|[?？]")

#: 征求意见的能愿结构 —— 「可以退款吗」「能否退货」。
#:
#: 单独列出来，是因为"吗"有时候会被口语省掉：「可以退款」和「我要退款」
#: 只差一个字，语义却一个是问规则、一个是提请求。抓到"可以/能否"这层
#: 结构，比只看句尾标点稳得多。
#:
#: 注意这里只收"能+具体动作"的固定搭配（能退/能用/能开），
#: 不收泛化的"能不能" —— 泛化形式会把"订单能撤吗""能改地址吗"
#: 这类真实请求一并卷进来降权。
_MODAL_RE = re.compile(r"可以|能否|是否|能退|能用|能开|能补|行不行")

#: "…是什么/怎么算/有哪些" —— 询问定义而不是提交请求。
#: 「退款政策是什么」没有问号也没有"吗"，但它显然是在问，不是在退。
#:
#: 这张表**必须覆盖到"用户在问规则"的全部常见句式**，漏一个就出一次事故：
#: 漏掉"谁出"，「退货运费谁出？」就会命中"退货"这个**动作词**，
#: 以 0.8 的置信度走进高风险出口，系统伸手向用户要订单号 ——
#: 而他只是想问运费谁付。这不是判错一个词，是把一句咨询当成了动钱的操作。
#:
#: 分四类收词（都只收"没有动作词也能确定是在问"的说法）：
#:   定义式 —— 是什么 / 怎么算 / 怎么规定
#:   归属式 —— 谁出 / 谁承担（问责任归属，不是问"能不能给我免了"）
#:   时长式 —— 多久 / 多长时间 / 要几天（问规则里的时限）
#:   流程式 —— 怎么走 / 什么流程
#:
#: 刻意**不收**单个"几天"：它既可能是政策（"几天无理由"）也可能是物流
#: （"几天能到"），在 `_WEAK_KEYWORDS` 里已经因此被标成弱词，
#: 这里再收一次等于把那条判定绕过去。
_DEFINITION_RE = re.compile(
    r"是什么|有哪些|怎么算|什么规则|什么政策|什么条件|如何计算"
    r"|怎么规定|如何规定|怎么定的|怎么说的|什么标准|什么流程|怎么走"
    r"|谁出|谁承担|谁付|谁负责|谁来出"
    r"|多久|多长时间|要几天|多少天|几个工作日"
)

#: 祈使标记 —— 出现这些就是明确在提要求，即便句尾带"吗"也不算咨询。
#: 「能帮我退款吗」是请求，不是咨询 —— 单看"吗"会判错，必须靠这条兜住。
_IMPERATIVE_RE = re.compile(r"帮我|给我|麻烦|替我|请帮|我要|我想|烦请")

#: 明确指向"我这一单"的说法。带上它就是在处理自己的订单，不是问规则。
#: 「订单能撤吗」「这单能退吗」问的是自己那笔单子能不能操作，
#: 和「生鲜类可以退款吗」这种问品类规则的句子不是一回事。
_OWN_ORDER_RE = re.compile(r"我的|我这|这单|那一单|那单|这订单|订单能|订单可以")

#: 只识别通用属性/规则问法，具体订单和明确操作请求仍由原有分类路径处理。
_ORIGIN_QUESTION_RE = re.compile(r"发货地|从(?:哪[里儿]|哪个(?:地方|城市)|什么地方)发货")
_AFTERSALES_RULE_RE = re.compile(
    r"(?:退货|退款|无理由|品质问题|质量问题).*"
    r"(?:从哪天|哪天开始|开始算|退多少|需要.*(?:寄回|退回)|是否.*寄回)"
)


#: 咨询类意图的归属：问"能不能退"落到规则政策出口，由知识库回答。
_QUESTION_INTENT = Intent.POLICY_FAQ


def _is_question(text: str) -> bool:
    """用户在问规则，还是在提请求？

    判定顺序（任一步不通过就直接判为"请求"）：

      1. 带祈使标记（"帮我""我要"）-> 请求。用户明确说了要做什么，
         哪怕句尾有"吗"也是请求，不能当咨询打发掉。
      2. 带显式订单号或"我的/这单"等说法 -> 请求。问的是具体那笔单子
         能不能操作，不是品类规则。
      3. 命中定义式问法（"是什么""有哪些"）-> 咨询。
      4. 命中能愿结构（"可以退款吗"）-> 咨询。

    宁可把边界情况判成请求（多走一次高风险流程去追问订单号），
    也不要把真实请求判成咨询（用户明确说了要退款，系统却答一段政策）。
    """
    if _IMPERATIVE_RE.search(text):
        return False
    if E.find_order_id(text) or _OWN_ORDER_RE.search(text):
        return False
    return bool(_DEFINITION_RE.search(text) or _MODAL_RE.search(text))


def _demote_question_to_policy(
    scored: list[tuple[Intent, float, list[str], float]], text: str
) -> list[tuple[Intent, float, list[str], float]]:
    """把"咨询型"的高风险候选降级成规则政策咨询。

    这里有个容易写错的地方：**降权之后必须重新算置信度**。

    上面那套打分把"关键词得分"和"覆盖度"各占一半（0.50/0.50）。如果只把
    候选乘个系数、却留着旧置信度，排序变了而分数没变 —— 胜出者带着
    降权前的分数被交给下游。高风险出口看的就是这个分数，后果是
    「生鲜类可以退款吗」拿着 0.98 的置信度走进退款流程。

    所以这里显式重算：走折扣的那个候选由折扣后的分数推导，没走折扣的
    沿用原置信度，保证"分数和排序永远自洽"。
    """
    if not _is_question(text):
        return scored

    out: list[tuple[Intent, float, list[str], float]] = []
    for intent, conf, matched, score in scored:
        if intent in _DEMOTABLE_INTS:
            discounted = score * _QUESTION_DISCOUNT
            # 覆盖度按"最高分"重新给，规则与打分口径保持一致
            out.append((intent, round(discounted, 4), matched, discounted))
        else:
            out.append((intent, conf, matched, score))

    # 咨询句里"政策"就是正解，它必须**不低于**合成下限 —— 否则胜负是由折扣的
    # 先后顺序决定的，而不是由"这是一句咨询"这个已经做出的判断决定的。
    #
    # 会踩的坑：「退货运费谁出？」里 POLICY_FAQ 命中了"运费"，但 `_score` 因为
    # RETURN_REQUEST 带着动作词（"退货"）而给它加了一道话题降权（×0.5），只剩
    # 0.20；"退货"降级后倒还有 0.44。"谁出"这个明确的咨询信号，就这样被
    # "运费"那个词的降权抵消掉了 —— 降级函数改了排序，唯独没改胜负。
    others = [t for t in out if t[0] is not _QUESTION_INTENT]
    existing = next((t for t in out if t[0] is _QUESTION_INTENT), None)
    top = max((c for _i, c, _m, _s in others), default=0.0)
    floor = round(min(0.90, max(0.62, top + 0.05)), 4)

    if existing is None:
        # 一个政策词都没命中（比如"能退吗"里没有"政策"），补一条合成候选
        out = others + [(_QUESTION_INTENT, floor, [_QUESTION_SYNTHETIC_MATCH], 1.0)]
    elif existing[1] < floor:
        # 命中了政策词，但被话题降权压住了 —— 抬到下限。
        # 把合成标记追加进 matched，审计里能看出"这条的分不全是关键词给的"。
        out = others + [
            (_QUESTION_INTENT, floor, [*existing[2], _QUESTION_SYNTHETIC_MATCH], 1.0)
        ]
    # existing[1] >= floor：关键词本来就命中得够强，维持原样，不拿合成项改写它
    out.sort(key=lambda x: (-x[1], x[0].value))
    return out


#: 合成候选的 matched 标记 —— 让审计里能看出一眼"这条不是关键词命中的"
_QUESTION_SYNTHETIC_MATCH = "（提问式咨询）"

#: 改判时规则层一个候选都没有（无从降级）给的置信度。
#: 和 `_fallback` 同档，**必须低于 settings.confidence_threshold**。
_DEMOTED_NO_EVIDENCE_CONF = 0.10

#: 哪些意图在咨询句里要被降级 —— **只有会动钱的四条**。
#:
#: 刻意不用 `_ACTION_ONLY`。那个集合回答的是"命中里有没有动作词"，
#: 它把 HUMAN_HANDOFF 也算成动作（"转人工"确实是用户要做的事），
#: 用它来判"要不要降级"就错了一格：转人工既不动钱也不是咨询，
#: 用户说「可以转人工吗」就是要人工。曾经真的这么写过，代价是
#: 用户明确要人工、系统回一段《开票时效与限制》政策正文，而且 handoff=false ——
#: 比答错更糟：他以为有人会来，其实没有。
_DEMOTABLE_INTS: frozenset[Intent] = HIGH_RISK_INTENTS

#: 提问式咨询的降权系数
_QUESTION_DISCOUNT = 0.55


# ----------------------------------------------------------------


def _is_action_backed(intent: Intent, text: str) -> bool:
    """这个意图的命中里有没有动作词（而不是只中了话题词）。"""
    return any(kw in text for kw in _ACTION_ONLY.get(intent, ()))


def _weak_evidence_only(top: tuple[Intent, float, list[str], float], text: str) -> bool:
    """规则层的最高分候选是不是只靠"弱关键词"撑着。"""
    intent, _conf, matched, _score = top
    weak = _WEAK_KEYWORDS.get(intent, ())
    return bool(matched) and all(kw in weak for kw in matched) and not any(
        p.search(text) for p in _EXTRA_COMPILED.get(intent, ())
    )


def _clamp(value: object, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return low


def _method_of(llm) -> str:
    """按客户端类型给通道命名。

    简单类名判断是故意的：不能靠 "import MockLLMClient" 去 isinstance，
    否则接真模型时这个模块会多一条对 mock 的硬依赖。
    """
    return "mock" if "Mock" in type(llm).__name__ else "llm"


def _clean(value: object) -> str | None:
    """模型偶尔会返回 "null"/"无"/指代占位 —— 统一归一成 None。

    Pydantic 会把字符串 "null" 原样收下，于是槽位看起来是"填好的"，
    missing_for() 不再报警，中断/恢复机制整个静默失效。这比抽不到槽位
    严重得多 —— 抽不到只是多问一句，假槽位会真的去执行一个错误操作。
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "无", "没有", "未知", "未提及", "none."}:
        return None
    if E.is_placeholder(text):
        return None
    return text
