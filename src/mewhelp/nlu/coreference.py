"""指代消解：把"那它能退吗"补成一句能独立看懂的话。

这是整条链路里"省一次模型调用、又最不容易被替代"的一步。分层与每层的边界：

  ① 量词自证类型（"这单"=订单、"这款"=商品）
      补什么完全确定，规则直接改写，一次模型调用都不花。
  ② 句内 / 上下文原文 / 历史里有实体
      句内实体优先，其次是调用方传进来的上下文原文（按话题域锁定），
      再次是按语义域从历史取最近实体。这三档全是纯规则。
  ③ 句首裸指代且三档都空（"那它能退吗"）
      只能交给模型判一次（task="resolve"）。模型不可用时退回"语义域假设 +
      历史最近实体"—— 补一个可能不完美的实体，好过把半截话丢给检索。
  ④ 兜底
      以上都没结果 → 返回**原文**，绝不抛异常。指代没补全顶多让检索差一点；
      抛异常会把整条请求打断，用户什么都拿不到。

语义域约束是做替换的前提，也是这一层真正的价值：问"这个尺码"时历史里
明明有订单号，但把"这个"补成订单号会让下游检索到完全无关的文档 ——
补错比不补更糟，因为它看起来是成功的，还不报错。

三条反直觉但重要的规则：
  * "当前句自己说清楚了就不补"：句内已有实体时，指代词八成是限定词
    （"这个连衣裙能退吗"），再替换只会把句子改坏。
  * "更近的商品可以盖过订单号"：售后动作默认指向订单，但如果用户在更近的
    一轮里单独提过某个商品，他嘴里的"它"更可能是那个商品。订单号不会因此
    丢掉 —— extract_slots 仍会从历史里把它抽出来交给高风险子流程。
  * "不许跨域借实体"：句子自报的域说了算，用户说"这个订单"时，哪怕上文在
    谈商品，也不能补成"羊毛衫 这个订单能退吗" —— 这种自相矛盾的句子下游
    完全看不出来（它只拿到一句通顺的话），属于最隐蔽的一类错。
"""

from __future__ import annotations

import logging
import re

from ..config import get_settings
from ..llm import build_llm
from ..schemas import ResolveResult, Turn
from . import entities as E
from .verbs import ORDER_ACTIONS

logger = logging.getLogger(__name__)

#: 指代词池。长的排前面 —— "上面那个"必须先于"那个"命中，否则会被截成残缺替换。
#: 刻意**不收**"该"：它在中文里绝大多数时候是"应该/该单号"的一部分，
#: 收进来的收益远小于"应订单 ORD-1001能退"这种垃圾改写的代价。
PRONOUNS: tuple[str, ...] = (
    "上面那个", "刚才那个", "上面这单", "刚才这单", "上面那单", "刚才那单",
    "这个", "那个", "这单", "那单", "这款", "那款", "这件", "那件",
    "它", "其",
)

#: 指示代词开头 —— 用来识别"这个 X"里的限定词用法
_DEMONSTRATIVE = frozenset({"这", "那"})

#: **自带类型信息**的指代词：量词本身就说清了补什么（"这单"=订单、"这款"=商品）。
#: 有它就不必问模型 —— 这是规则层能白捡的一批确定性判断。
#: "上面那个/它/这个" 这类不带量词的无法自证类型，句首出现时交给模型。
_SELF_TYPED: dict[str, str] = {
    "这单": "order",
    "那单": "order",
    "上面这单": "order",
    "刚才这单": "order",
    "上面那单": "order",
    "刚才那单": "order",
    "这款": "product",
    "那款": "product",
    "这件": "product",
    "那件": "product",
}

#: 单字指代词的误伤防护。中文没有词边界，"其中"里的"其"若被当成指代词，
#: 会产出"订订单 ORD-1001中能退"这种垃圾句子，而且不报错。
_PREV_BLOCK: dict[str, str] = {"其": "尤极与何"}
_NEXT_BLOCK: dict[str, str] = {"它": "们", "其": "中他实次余它"}

#: 语义域：由当前问句里的动词/名词决定"该补什么类型的实体"。
#: 表里没有的域 = 规则层判不出来 → 交给模型，而不是瞎补。
_DOMAIN_ORDER = "order"
_DOMAIN_PRODUCT = "product"
_DOMAIN_ADDRESS = "address"

_DOMAIN_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # 商品域排在最前：属性词是**唯一**能一眼看出"这句话在谈商品"的信号，
    # 而且它是最窄的（"尺码""材质"几乎不会出现在别的话题里）。
    # 把它排在订单域后面会让"这个尺码能退吗"被"退"抢走，补成订单号 ——
    # 用户问的是商品属性，检索订单信息等于答非所问。
    (_DOMAIN_PRODUCT, (
        "商品", "产品", "东西", "尺码", "尺寸", "材质", "面料", "颜色", "色号",
        "型号", "参数", "规格", "保修", "重量", "品牌", "多大", "口径", "容量",
    )),
    # 订单/售后域：靠"动作词"判定（见 nlu/verbs.py）。为什么不用话题词：
    # "运费怎么算"和"我要退运费"共享"运费"，但前者是政策咨询，
    # 指代对象根本不该被塞进一个订单号。
    (_DOMAIN_ORDER, ORDER_ACTIONS),
    # 地址域
    (_DOMAIN_ADDRESS, (
        "地址", "收货地址", "寄到", "送到", "改地址", "写错", "公司地址",
    )),
)

#: 每个域按优先级排列的实体抽取器
_DOMAIN_FINDERS = {
    _DOMAIN_ORDER: (E.find_order_id,),
    _DOMAIN_PRODUCT: (E.find_product,),
    _DOMAIN_ADDRESS: (E.find_address,),
}

#: 判"这段上下文在谈什么"用的**实体名词**（不是动作词）。
#: 顺序即优先级：同时提到商品和订单时，商品更可能是被指代的那个 ——
#: "我买的羊毛衫有点问题"里，"羊毛衫"才是话题主语，"问题"不是实体。
_TOPIC_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (_DOMAIN_PRODUCT, (
        "商品", "产品", "东西", "尺码", "材质", "颜色", "型号", "坏了", "有点问题",
    )),
    (_DOMAIN_ADDRESS, ("地址", "寄到", "送到")),
    (_DOMAIN_ORDER, ("订单", "单号", "快递", "物流", "发货")),
)

#: 补全后的实体前缀 —— 让补出来的话读起来像人话（"订单 ORD-1001"）
_DOMAIN_PREFIX = {_DOMAIN_ORDER: "订单 "}

#: 指代短语里的量词。替换时保留它，才能把"那单能退吗"补成
#: "订单 ORD-1001 那单能退吗"而不是"…1001 能退吗" —— 前者语义没有丢失。
_DEMONSTRATIVE_MEASURE_RE = re.compile(r"^[那这](单|订单|笔|款|个|件)$")


def _replacement_for(pronoun: str, entity: str, domain: str, prefix: str) -> str:
    """指代词 → 补全文本。

    量词型指代词（"那单"）保留成同位语："订单 ORD-1001 那单能退吗"。
    直接删掉它虽然也通顺，但会丢掉"说的是哪一单"的语气；中文用户
    更习惯前者。substitutions 里记的就是这个最终替换文本，审计能直接对上。
    """
    if _DEMONSTRATIVE_MEASURE_RE.match(pronoun):
        return f"{prefix}{entity} {pronoun}"
    return prefix + entity


class CoreferenceResolver:
    """规则优先的指代消解器。"""

    def __init__(self, llm=None, settings=None) -> None:
        self.settings = settings or get_settings()
        # 允许 llm=None：规则层已经能兜住大部分场景，没有模型的调用方
        # （比如只跑规则层做数据标注的脚本）不该被强制注入一个客户端。
        self.llm = llm if llm is not None else build_llm(self.settings)

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def resolve(
        self,
        question: str,
        history: list[Turn] | None = None,
        context: str = "",
    ) -> ResolveResult:
        """三层入口。任何一层给出确定结论就返回，异常一律降级成原文。

        `context` 是可选参数：被指代的那一轮原文。为什么需要它 —— history 可能
        已经被 memory 层压成摘要了（`window_turns` 之外只剩一句话的 summary），
        "最近在谈的是商品还是订单"这个关键信息会丢。调用方手上有原文时传进来，
        规则层就能少一次瞎猜。不传也不影响正确性，只是可能多问一次模型。
        """
        original = question or ""
        turns = list(history or [])
        # 空问题没有东西可补，原样返回，让下游按兜底出口处理
        if not original.strip():
            return self._noop(original)

        try:
            # 第一层：量词自带类型（"这单"=订单、"这款"=商品）时补什么完全确定，
            # 连语义域都不用猜，一次模型调用都省掉。
            typed = self._self_typed_domain(original)
            picked = self._pick_pronoun(original)
            if picked is None:
                # 没有真指代词：要么这句话本来就能独立看懂（"退货流程是什么"），
                # 要么指代词是限定词（"这个连衣裙能退吗"）。都是正常结论，不调模型。
                return self._noop(original)

            pronoun, pos = picked
            # 句首指代只能靠整句的语义线索判域（"这个尺码"的线索在"这个"之后）；
            # 句中指代必须只看指代词**之前**的部分 —— 句尾的诉求词（"能改地址吗"）
            # 属于下一句的诉求，不是指代对象（"订单 ORD-1001 …能改地址吗"）。
            domain = typed or self._domain_of(original[:pos] or original)
            if domain is None:
                # 句首裸指代（"它""这个"）且句内毫无线索。此时先看被指代那一轮的
                # 原文 —— 有原文就是纯规则能定的事，不值得花一次模型调用。
                topic = self._topic_context(original, context)
                if topic is not None:
                    return self._rewrite(original, pronoun, topic[1], None, domain=topic[0])
                # 连原文都没有 → 只有模型能判它指什么。模型不可用时规则用语义域
                # 假设兜底：宁可补得不够完美，也不能把"那它能退吗"整句丢给检索
                # （半截话必定检索不到东西）。
                return self._resolve_bare_pronoun(original, turns, pronoun, context)

            rule = self._resolve_pronoun(original, turns, pronoun, pos, domain, context)
            if rule is not None:
                return rule
            return self._resolve_by_llm(original, turns, context)
        except Exception as exc:  # noqa: BLE001 —— 契约要求任何异常都收敛成降级结果
            return self._fallback(original, exc)

    @staticmethod
    def _self_typed_domain(question: str) -> str | None:
        """句子的指代对象能不能由量词自己说清。"""
        for pron, domain in _SELF_TYPED.items():
            if pron in question:
                return domain
        return None

    def _pick_pronoun(self, question: str) -> tuple[str, int] | None:
        """找出第一个**真的在指代**的指代词，返回 (词, 位置)。

        过滤三类假阳性：单字词被裹在别的词里（"其中"）、限定词用法
        （"这个连衣裙"）、以及指代短语本身（"那个地址"里的"那个"）。
        """
        for p in PRONOUNS:
            pos = question.find(p)
            if pos < 0:
                continue
            prev = question[pos - 1] if pos > 0 else ""
            nxt = question[pos + len(p)]
            # prev/nxt 可能为空串，而 `"" in "任意字符串"` 恒为 True，
            # 不显式挡一下会把句首/句尾的合法指代词错误地滤掉。
            if prev and prev in _PREV_BLOCK.get(p, ""):
                continue
            if nxt and nxt in _NEXT_BLOCK.get(p, ""):
                continue
            if p[0] in _DEMONSTRATIVE and self._is_determiner(question, pos, len(p)):
                continue
            return p, pos
        return None

    @staticmethod
    def _is_determiner(question: str, pos: int, pron_len: int) -> bool:
        """"这个/那个"后面直接跟着一个像商品名的词 → 它是限定词，不是代词。

        这是"这个连衣裙能退吗"和"这个几号到"的分界线：前者自身完整，
        后者必须靠历史补全。判错的代价不对称 —— 误判成限定词只是少补一次
        （下游检索质量略降），误判成代词会把好好的句子改成"连衣裙能退吗"。
        """
        tail = question[pos + pron_len :]
        if not tail:
            return False
        name = E.find_product(tail)
        return bool(name) and tail.startswith(name)

    @staticmethod
    def _domain_of(text: str) -> str | None:
        """按语义域线索判"用户在要求做什么"。

        只看指代词**之前**的部分（调用方截好）—— 句尾的诉求词（"能改地址吗"）
        属于下一句的诉求，不是指代对象；混在一起会让"订单 ORD-1001 …能改地址吗"
        被判成地址域。
        """
        for domain, cues in _DOMAIN_CUES:
            if any(cue in text for cue in cues):
                return domain
        return None

    def _topic_context(self, question: str, context: str) -> tuple[str, str] | None:
        """文本里能直接找到"话题实体"时返回 (域, 实体)，否则 None。

        这里刻意把"上下文原文"也做成规则层能消费的证据：调用方（图里的 resolve
        节点）手上有 memory 摘要之外的原文时，靠它就够了，不必为一个字符串替换
        付一次模型往返。找不到就返回 None，让流程继续走模型/兜底。

        **一句话自报的域优先，且不允许跨域取实体**：用户说"这个订单能退吗"时，
        哪怕上文在谈商品，也不能补成"羊毛衫 这个订单能退吗"。跨域补全会产出
        语义自相矛盾的句子，而且下游是看不出来的（它只拿到一句通顺的话）。
        """
        if self._topic_domain(question):
            topic = self._topic_domain(question)
            entity = self._find_in(question, topic)
            return (topic, entity) if entity else None
        if context:
            topic = self._topic_domain(context)
            if topic:
                entity = self._find_in(context, topic)
                if entity:
                    return topic, entity
        return None
    def _resolve_pronoun(
        self,
        question: str,
        history: list[Turn],
        pronoun: str,
        pos: int,
        domain: str,
        context: str = "",
    ) -> ResolveResult | None:
        """按语义域找实体并替换。没有可用实体时返回 None（交给模型或原样返回）。"""
        head = question[:pos]
        entity = self._find_in(head, domain)
        in_sentence = entity is not None
        turn_idx: int | None = None
        if not entity:
            entity, turn_idx = self._find_in_history(history, domain)

        # 本句或上下文原文里有"话题实体"时优先用它（先于历史）—— 摘要里
        # "用户咨询了退款"这种话是判断不了指代对象的，而原文可以。
        if not in_sentence:
            topic = self._topic_context(question, context)
            if topic is not None and topic[0] != domain:
                return self._rewrite(question, pronoun, topic[1], None, domain=topic[0])

        # 售后动作默认作用于订单，但如果用户在**更近的一轮**里单独提过某个商品，
        # 他嘴里的"它"更可能指那个商品（"我买的那款连衣裙到了" → "这个能退吗"）。
        # 订单号不会因此丢掉：extract_slots 仍会从历史里把它抽出来交给高风险子流程。
        if domain == _DOMAIN_ORDER and not in_sentence:
            prod, prod_idx = self._find_in_history(history, _DOMAIN_PRODUCT)
            if prod and prod_idx and (turn_idx is None or prod_idx > turn_idx):
                return self._rewrite(
                    question, pronoun, prod, prod_idx, domain=_DOMAIN_PRODUCT
                )

        if not entity:
            return None
        # 实体已经在本句前半部分出现过时，不再重复补"订单 "前缀 ——
        # 否则会得到"订单 ORD-1001 …，订单 ORD-1001 这单能改地址吗"这种车轱辘话。
        prefix = "" if in_sentence else _DOMAIN_PREFIX.get(domain, "")
        return self._rewrite(
            question, pronoun, entity, turn_idx if turn_idx else None,
            domain=domain, prefix=prefix,
        )

    @staticmethod
    def _topic_domain(text: str) -> str | None:
        """判断一段文本在谈哪类实体（不看动作词，只看有没有提到实体名词）。

        与 _domain_of 的区别：_domain_of 判的是"用户在要求做什么"（所以用动作词，
        避免把"运费怎么算"当成退款）；这里判的是"这段上下文在谈什么"（所以用
        实体名词，因为"我买的羊毛衫有点问题"里根本没有动作词）。
        """
        for domain, cues in _TOPIC_CUES:
            if any(cue in text for cue in cues):
                return domain
        return None

    @staticmethod
    def _find_in(text: str, domain: str) -> str | None:
        """句内找实体（只需要值，不需要轮次）。"""
        for finder in _DOMAIN_FINDERS[domain]:
            value = finder(text)
            if value:
                return value
        return None

    @staticmethod
    def _find_in_history(
        history: list[Turn], domain: str
    ) -> tuple[str | None, int | None]:
        turns = list(E.iter_turns(history))
        for finder in _DOMAIN_FINDERS[domain]:
            entity, turn_idx = E.nearest_entity(turns, finder)
            if entity:
                return entity, turn_idx
        return None, None

    @staticmethod
    def _noop(question: str) -> ResolveResult:
        """明确的"不该改写"。

        注意这**不是**降级：规则通道判定这句话本来就能独立看懂
        （"退货流程是什么"、"这个连衣裙能退吗"）是正常结论。
        """
        return ResolveResult(original=question, resolved=question, rewritten=False)

    @staticmethod
    def _rewrite(
        question: str,
        pronoun: str,
        entity: str,
        turn_idx: int | None,
        *,
        domain: str = "order",
        prefix: str | None = None,
    ) -> ResolveResult:
        """执行替换并落成结果。替换不动（文本没变）时如实返回 rewritten=False。"""
        head = _DOMAIN_PREFIX.get(domain, "") if prefix is None else prefix
        replacement = _replacement_for(pronoun, entity, domain, head)
        resolved = question.replace(pronoun, replacement)
        if domain == "order":
            # 原句已带"订单"时会变成"订单订单 ORD-1001"，合并成一个
            resolved = re.sub(r"订单\s*订单\s*ORD", "订单 ORD", resolved)
        if resolved == question:
            return CoreferenceResolver._noop(question)
        return ResolveResult(
            original=question,
            resolved=resolved,
            rewritten=True,
            substitutions={pronoun: replacement},
            referenced_turns=[turn_idx] if turn_idx else [],
        )

    def _resolve_bare_pronoun(
        self, question: str, history: list[Turn], pronoun: str, context: str = ""
    ) -> ResolveResult:
        """句首裸指代（"那它能退吗"）且上下文原文也没给出实体时的最后一道规则。

        走到这里说明规则层确实没有依据了 —— 先请模型判一次（mock 与真模型
        都走 task="resolve" 这条约定接口）。模型不可用时才退回"语义域假设 +
        历史最近实体"：补一个可能不完美的实体，远好过把半截话丢给检索，
        那样必定一无所获。
        """
        try:
            result = self._resolve_by_llm(question, history, context)
        except Exception as exc:  # noqa: BLE001 —— 模型层失败不能吃掉规则兜底的机会
            logger.warning("coreference 模型层失败，改用规则兜底: %s", exc)
            result = self._noop(question)
        if result.rewritten:
            return result
        # 兜底顺序：先看被指代那一轮的原文（最可靠的线索），再按域假设找实体。
        entity, turn_idx, domain = self._fallback_entity(
            question, history, self._domain_of(question) or _DOMAIN_ORDER, context
        )
        if not entity:
            return result
        return self._rewrite(
            question, pronoun, entity, turn_idx if turn_idx else None, domain=domain
        )

    def _fallback_entity(
        self, question: str, history: list[Turn], prefer: str, context: str = ""
    ) -> tuple[str | None, int | None, str]:
        """模型不可用时，按"句内优先、域假设优先"的顺序找一个能补的实体。

        顺序刻意如此：**用户这句话里已经写出来的实体**是最强证据（"订单 ORD-1001
        是上个月买的，它能退吗"），历史只是次选。异常路径上宁可补一个来自本句的
        实体，也不要因为"判不出域"就整句放弃。
        """
        domains = [prefer] + [d for d in (_DOMAIN_PRODUCT, _DOMAIN_ORDER, _DOMAIN_ADDRESS) if d != prefer]
        # 本句/上下文原文里能直接找到实体时优先取 —— 它比域假设准得多
        topic = self._topic_context(question, context)
        if topic is not None:
            return topic[1], None, topic[0]
        for domain in domains:
            entity = self._find_in(question, domain)
            if entity:
                return entity, None, domain
        for domain in domains:
            entity, turn_idx = self._find_in_history(history, domain)
            if entity:
                return entity, turn_idx, domain
        return None, None, prefer

    # ------------------------------------------------------------------
    # 第二层：模型
    # ------------------------------------------------------------------
    def _resolve_by_llm(
        self, question: str, history: list[Turn], context: str = ""
    ) -> ResolveResult:
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "你是指代消解器。把用户这一句里的指代词替换成历史里出现过的实体，"
                    "补成一句可以独立看懂的问题。只在语义上说得通时才替换，"
                    "没把握就原样返回。只输出 JSON："
                    '{"resolved": str, "rewritten": bool, "substitutions": {代词: 实体}}'
                ),
            }
        ]
        for _idx, role, text in E.iter_turns(history):
            messages.append({"role": role, "content": text})
        if context:
            messages.append({"role": "system", "content": f"用户最近在说：{context}"})
        messages.append({"role": "user", "content": question})

        resp = self.llm.chat(messages, temperature=0.0, json_mode=True, task="resolve")
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 —— 解析失败等于模型没给答案，
            # 这不是"异常外泄"，所以不往上抛：让调用方继续走规则兜底。
            logger.warning("coreference 模型输出无法解析: %s", exc)
            return self._noop(question)
        if not isinstance(data, dict):
            return self._noop(question)

        resolved = str(data.get("resolved") or question)
        subs = {
            str(k): str(v)
            for k, v in (data.get("substitutions") or {}).items()
            if isinstance(k, str) and v
        }
        if not (data.get("rewritten") and resolved != question):
            return self._noop(question)
        turns = list(E.iter_turns(history))
        # 模型给的补全也要标出引用了第几轮，否则审计时无法回溯"它从哪来"
        refs = [i for i in (E.turn_of(turns, v) for v in subs.values()) if i]
        return ResolveResult(
            original=question,
            resolved=resolved,
            rewritten=True,
            substitutions=subs,
            referenced_turns=refs,
        )

    # ------------------------------------------------------------------
    # 第三层：兜底
    # ------------------------------------------------------------------
    def _fallback(self, question: str, exc: Exception) -> ResolveResult:
        """降级：原样返回。

        宁可"没消解"也不要"消解到一半再抛异常" —— 原文至少还能继续检索，
        异常会让整条请求失败。异常信息走 logger（ResolveResult 契约里没有
        error 字段，不能往里塞私货），既不丢线索也不污染对外数据模型。
        """
        logger.warning("coreference 兜底，returned original question: %s", exc)
        return self._noop(question)
