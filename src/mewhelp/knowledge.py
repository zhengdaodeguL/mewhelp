"""可直接核验的知识问答规则。

只从本轮证据摘录属性/表格，或对证据中的净重区间作比较；不维护商品答案表。
返回值同时标记实际用到的片段，让检索、闸门和生成使用同一份覆盖判定。
无法确认主体、条件或区间有冲突时返回 None，继续走常规校验。
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from .schemas import Evidence, KbChunk, Turn


_NUM = r"\d+(?:\.\d+)?"
_UNIT = r"(?:公斤|千克|kg|斤|克|g)"
_MEASURE = re.compile(_NUM + r"\s*(?:" + _UNIT + r"|两|只|片|盒|袋|箱|cm|mm)", re.I)
_ATTRIBUTE = re.compile(r"规格|净重|重量|果肉|价格|多少钱|哪些款|型号|材质|保存|保质期|产地|发货地")
_PROPERTY_SUFFIX = re.compile(r"(?:商品)?(?:规格|重量|包装|价格|赔付|发货地|保存|说明|参数|资料).*$")
_GENERIC_TOPICS = {"商品", "商品与尺码", "售后", "售后规则", "规则", "退款政策", "退货政策", "物流与配送"}
_TIER = re.compile(r"(?P<n>" + _NUM + r")(?P<u>" + _UNIT + r")(?:档|装|规格)?", re.I)
_RECEIVED = re.compile(
    r"(?:收到|实收|实际(?:净重|称重)|称重)[为是了:：]*"
    r"(?P<n>" + _NUM + r")(?P<u>" + _UNIT + r")", re.I,
)
_RANGE = re.compile(
    r"(?P<low>" + _NUM + r")(?P<lu>" + _UNIT + r")?"
    r"(?:[-–—~～至到])(?P<high>" + _NUM + r")(?P<u>" + _UNIT + r")", re.I,
)
_GRAMS = {
    "公斤": Decimal(1000), "千克": Decimal(1000), "kg": Decimal(1000),
    "斤": Decimal(500), "克": Decimal(1), "g": Decimal(1),
}
_FRESH = re.compile(r"生鲜|牛排|肉|鱼|虾|蟹|果|蔬|菜|薯|土豆|羊角蜜|乳|奶")


@dataclass(frozen=True)
class EvidenceAnswer:
    text: str
    chunk_ids: tuple[str, ...]
    kind: str


def normalized(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or "")).casefold()


def question_focus(question: str) -> str:
    # 兼容旧指代消解格式：旧答复只是背景，不能把旧问题当作当前诉求。
    return (question or "").rsplit("追问：", 1)[-1]


def is_damage_policy_question(question: str) -> bool:
    q = normalized(question_focus(question))
    return bool(
        re.search(r"坏|损|变质|腐|化冻|品质|质量|售后|违约", q)
        and re.search(r"退|赔", q)
        and re.search(r"%|比例|多少|按.*算|是不是|是否|就是|算吗", q)
    )


def product_attribute_question(question: str) -> bool:
    return bool(_ATTRIBUTE.search(question) or "少重" in question)


def product_subject(question: str) -> str:
    """只提取属性问句的主体，去掉量重、包装和问法；不是通用的实体识别器。"""
    q = normalized(question_focus(question))
    q = re.sub(r"^(?:请问|问一下|你们的?|这个|那个|这款|那款|它|那|这)+", "", q)
    q = re.split(
        r"有(?:什么|哪些)|是(?:不是|否)|(?:的)?(?:规格|净重|重量|果肉|价格|多少钱"
        r"|型号|材质|保存|保质期|产地|发货地)|(?:带|含|连)箱|收到|实收|" + _NUM + _UNIT,
        q, maxsplit=1,
    )[0]
    q = re.sub(r"(?:那款|这款|那个|这个|那|这)$", "", q).strip(" ，,：:。?？的")
    # 诸如“那5斤的”没有商品名，“坏单包”也不是商品。
    if not q or re.match(r"^[\d\W]", q) or is_damage_policy_question(question):
        return ""
    return q


def _topics(chunk: KbChunk) -> list[str]:
    values = [*(chunk.breadcrumb[:1]), chunk.doc_id, chunk.title]
    out = []
    for value in values:
        name = _PROPERTY_SUFFIX.sub("", normalized(value).split(">", 1)[0]).strip("_ -：:")
        if len(name) >= 2 and re.search(r"[一-龥]", name) and name not in _GENERIC_TOPICS:
            out.append(name)
    return list(dict.fromkeys(out))


def _belongs(question: str, chunk: KbChunk) -> bool:
    q = normalized(question)
    subject = product_subject(question)
    for topic in _topics(chunk):
        if topic not in q:
            continue
        # “猪肉梅花肉”不能靠共同的“猪肉”套用五花肉的资料。
        detail = subject.replace(topic, "", 1) if topic in subject else ""
        detail = detail.strip(" ，,：:")
        if not detail or detail in normalized(chunk.prompt_text()):
            return True
    return False


def _render(chunk: KbChunk, rows: Sequence[Sequence[str]] | None = None) -> str:
    if chunk.is_table and chunk.table:
        head = chunk.table[0]
        return "\n".join(
            "；".join(f"{key}：{value}" for key, value in zip(head, row)) + "。"
            for row in (rows if rows is not None else chunk.table[1:])
        )
    return chunk.text.strip().replace("**", "").replace("__", "")


def _answer(items: Sequence[tuple[Evidence, str]], kind: str) -> EvidenceAnswer | None:
    if not items:
        return None
    texts = list(dict.fromkeys(text for _, text in items if text.strip()))
    text = "\n\n".join(texts)
    if not text or len(text) > 4000:
        return None
    return EvidenceAnswer(text, tuple(dict.fromkeys(e.chunk_id for e, _ in items)), kind)


def _price_rows(question: str, chunk: KbChunk) -> list[list[str]]:
    """名称追问按原表的行定位，不能把同表的另一款价格一起当作该款答复。"""
    subject = product_subject(question)
    for topic in _topics(chunk):
        subject = subject.replace(topic, "", 1)
    subject = subject.strip(" ，,：:")
    rows = chunk.table[1:] if chunk.table else []
    if subject and subject not in normalized(chunk.title):
        rows = [row for row in rows if subject in normalized(" ".join(row))]
    columns = [i for i, h in enumerate(chunk.table[0]) if re.search(r"价格|售价|单价", h)]
    if not all(
        i < len(row) and re.search(_NUM, row[i])
        and not re.search(r"待定|待确认|未知|暂无|无法确认", row[i])
        for row in rows for i in columns
    ):
        return []
    return rows


def _weight_answer(q: str, candidates: Sequence[Evidence]) -> EvidenceAnswer | None:
    if not re.search(r"少重|足重|够重|范围|正常", q) or re.search(r"怎么赔|如何赔|赔多少|退多少", q):
        return None
    tier, received = _TIER.search(q), _RECEIVED.search(q)
    if tier is None or received is None or tier.start() >= received.start():
        return None
    tier_weight = Decimal(tier["n"]) * _GRAMS[tier["u"]]
    weight = Decimal(received["n"]) * _GRAMS[received["u"]]
    found = []
    for ev in candidates:
        c = ev.chunk
        if c.is_table and c.table and "净重" in "".join(c.table[0]):
            parts = [(normalized(" ".join(row)), _render(c, [row])) for row in c.table[1:]]
        else:
            parts = [(normalized(part), part.strip()) for part in re.split(r"[。；;\n]", c.text) if "净重" in part]
        for part, original in parts:
            intervals = list(_RANGE.finditer(part))
            # 区间的端点不是另一档规格；逐个检查区间，不能只取一句中的第一个。
            specs = [
                spec for spec in _TIER.finditer(part)
                if not any(r.start() <= spec.start() < r.end() for r in intervals)
            ]
            for interval in intervals:
                spec = next((s for s in reversed(specs) if s.end() <= interval.start()), None)
                if spec is None or Decimal(spec["n"]) * _GRAMS[spec["u"]] != tier_weight:
                    continue
                low = Decimal(interval["low"]) * _GRAMS[interval["lu"] or interval["u"]]
                high = Decimal(interval["high"]) * _GRAMS[interval["u"]]
                if low <= high:
                    found.append((ev, original, low, high))
    # 同一档位出现不同区间，不能按检索排序挑一个。
    if not found or len({(lo, hi) for _, _, lo, hi in found}) != 1:
        return None
    ev, original, low, high = found[0]
    actual = received["n"] + received["u"]
    if low <= weight <= high:
        conclusion = f"{actual}在上述净重范围内，不能仅因低于标称重量就认定少重。"
    elif weight < low:
        conclusion = f"{actual}低于上述净重范围的下限。"
    else:
        conclusion = f"{actual}高于上述净重范围的上限，不是低于下限的少重情况。"
    return _answer([(ev, original.rstrip("。") + "。\n" + conclusion)], "weight_range")


def _damage_answer(q: str, evidences: Sequence[Evidence]) -> EvidenceAnswer | None:
    percentages = set(re.findall(_NUM + r"%", q))
    # 百分比的含义由原文定义，不能把服务违约赔付变成商品坏损退款比例。
    definitions = [
        (ev, _render(ev.chunk)) for ev in evidences
        if percentages.intersection(re.findall(_NUM + r"%", normalized(ev.chunk.text)))
        and re.search(r"未履行.*服务|服务.*未履行", normalized(ev.chunk.text))
        and re.search(r"不是|不等于|不代表", normalized(ev.chunk.text))
    ]
    if definitions:
        return _answer(definitions, "policy_definition")

    # 仓库政策表按状态/时间列退款比例，不能拿坏损数量的除法代替它。
    if not _FRESH.search(q):
        return None
    items = []
    for ev in evidences:
        c = ev.chunk
        if not (c.is_table and c.table and "生鲜" in " ".join([c.title, *c.breadcrumb])):
            continue
        if "退款比例" not in c.table[0] or "申请时间窗" not in c.table[0]:
            continue
        rows = [row for row in c.table[1:] if any("部分变质" in cell for cell in row)]
        if rows:
            proposed = "（" + "、".join(sorted(percentages)) + "）" if percentages else ""
            items.append((ev, f"不能仅按坏损数量比例{proposed}认定退款比例。资料按商品状态和申请时间区分：\n" + _render(c, rows)))
    return _answer(items, "damage_policy")


def answer_from_evidence(question: str, evidences: Sequence[Evidence]) -> EvidenceAnswer | None:
    """已覆盖才返回答案；分数高低不改变原文中明确规则的含义。"""
    q = normalized(question_focus(question))
    items = sorted((e for e in evidences if math.isfinite(e.score)), key=lambda e: e.chunk_id)
    if is_damage_policy_question(q):
        return _damage_answer(q, items)
    # 摘录规格不能被当作同时答完“规格和保存方式”等复合问题。
    # 这些属性不在下面的规则覆盖范围，交给带当前问题的生成/自评处理。
    if re.search(r"保存|保质期|材质|尺码|产地|发货|运费|物流", q):
        return None
    candidates = [e for e in items if _belongs(question, e.chunk)]
    if not candidates:
        return None
    if "少重" in q or "范围" in q or "足重" in q:
        return _weight_answer(q, candidates)
    if re.search(r"净重|果肉", q):
        return _answer([
            (e, _render(e.chunk)) for e in candidates
            if re.search(r"带箱|含箱|连箱", normalized(e.chunk.text))
            and "包装" in e.chunk.text
            and re.search(r"(?:不是|不等于|不能说|不代表).*?(?:净重|果肉)", normalized(e.chunk.text))
        ], "packing")
    if re.search(r"多少钱|价格", q):
        prices = []
        for e in candidates:
            c = e.chunk
            if c.is_table and c.table and any(re.search(r"价格|售价|单价", h) for h in c.table[0]):
                if rows := _price_rows(question, c):
                    prices.append((e, _render(c, rows)))
            elif re.search(r"价格|售价|报价", c.title) and re.search(r"[¥￥]" + _NUM + "|" + _NUM + r"元", normalized(c.text)):
                prices.append((e, _render(c)))
        return _answer(prices, "price")
    if re.search(r"规格|哪些款|型号", q) and not re.search(r"净重|赔|退|少重|能否|适合", q):
        return _answer([
            (e, _render(e.chunk)) for e in candidates
            if (re.search(r"规格|型号", e.chunk.title) or (e.chunk.table and any("规格" in h for h in e.chunk.table[0])))
            and _MEASURE.search(_render(e.chunk))
        ], "specification")
    return None


def previous_evidence(question: str, history: Sequence[Turn]) -> tuple[str, list[Evidence]]:
    """沿用最近一轮同商品证据，拒答正文不作为知识；跨主题/跨会话不借用。"""
    if not product_attribute_question(question):
        return "", []
    if history and history[-1].metadata.get("quoted"):
        # 显式引用由原有消解逻辑处理，不能让最近一轮资料盖过被引用的主题。
        return "", []
    last = next((t for t in reversed(history) if t.role == "assistant"), None)
    if last is None:
        return "", []
    evidence = []
    for raw in last.metadata.get("retrieval_evidence", []):
        try:
            evidence.append(Evidence.model_validate(raw))
        except (ValueError, TypeError):
            continue
    if not evidence:
        return "", []
    q, subject = normalized(question), product_subject(question)
    matching = [e for e in evidence if _belongs(question, e.chunk)]
    if not matching and subject:
        # “红运道那款”一类带名称的追问，名称必须真的在上一轮原始证据里。
        anchor = re.split(r"那款|这款|那个|这个", subject, maxsplit=1)[0]
        matching = [e for e in evidence if len(anchor) >= 2 and anchor in normalized(e.chunk.prompt_text())]
    if not matching and not subject and re.match(r"^(?:那|这|它)", q):
        if len({e.chunk.doc_id for e in evidence}) == 1:
            matching = evidence
    if not matching:
        return "", []
    docs = {e.chunk.doc_id for e in matching}
    if len(docs) != 1:
        return "", []
    related = [e for e in evidence if e.chunk.doc_id in docs]
    topics = _topics(related[0].chunk)
    return (topics[0] if topics else ""), related
