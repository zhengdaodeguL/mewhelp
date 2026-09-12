"""置信度闸门 —— 卡在模型开口之前的那道检查。

## 为什么必须在生成之前

`ConfidenceGate.check()` 的签名里**只有 `question / retrieval / intent / slots`，
没有 `answer`**。这是刻意的，不是还没写到。

顺序不同，代价差一个数量级：

  * **先判后答**：预测都没发出去，最坏结果是多一次转人工。用户从没看到错的东西。
  * **先答后判**：模型已经把话说完了，用户已经读到了。此时再"发现"证据不足，
    你只能追加一句"以上仅供参考"—— 幻觉已经送达，撤回不存在。

所以这个模块的全部输入都是"检索阶段就已经拿到的东西"：零模型调用、零副作用、
微秒级。它便宜到可以每次都跑，也贵到没它就要出事。

## 判据

`checks` 字典逐条可解释，事故复盘时要能指着某一条说"是它拦下来的"：

  top_score        最高证据分（精排分）归一化到 0~1
  evidence_count   有效证据条数（分数**过阈值**的才算 —— 召回总会返回 top_k 条，
                   哪怕库里毫无相关内容，"最不像无关"的那条也会被返回。
                   不过阈值的就是噪音，不能拿来凑置信度）
  score_margin     最高分与次高分的差；只有一条证据时记 0 —— 没有第二条来交叉印证，
                   这个"一致"是假的，不能当成高置信的理由
  entity_coverage  问题里的关键实体（品类/时间/金额）在有效证据里出现的比例
  avg_score        有效证据的平均分：区分"一条独苗"和"一片全都够格"

另外记几个只作审计、不参与决策的量：`recalled_count`（未经阈值过滤的召回条数，
用来分辨"没召回"还是"召回了但都不够格"）、`entity_count`、`order_id_count`、
`threshold`（这一次实际生效的阈值）。

## confidence 的算法

    base       = 0.40*top_score + 0.30*margin_score + 0.30*count_score
    confidence = base * (0.70 + 0.30*entity_coverage)

  * `margin_score` 不直接用差值，而是过一道饱和映射 `m/(m+0.25)`。
    差值本身没有绝对意义：精排分差 0.2，在一种打分口径下是"碾压"，
    在另一种口径下是"五五开"。判相对量级比判绝对值稳。
  * `count_score` 用对数增长（`log1p(n)/log1p(3)`，3 条饱和）。第 2 条证据的价值
    远大于第 4 条 —— 从 1 到 2 是"有了交叉印证"这个质变。线性增长会奖励
    "召回一堆"而不是"召回得准"。
  * 折价系数下限 0.7 而不是 0：实体覆盖不足时闸门**不该自己拒**，拒答原因要由
    `low_coverage` 这条规则单独给出，原因才可解释。折价只负责把 confidence
    压下去，让下游（高风险审核、飞轮优先级）也跟着谨慎一点。

阈值只来自 `settings.confidence_threshold` 与 `settings.min_evidence_count`，
模块内不写死 —— 调阈值不改代码，消融实验才有意义。
"""

from __future__ import annotations

import math
import re

from ..config import Settings, get_settings
from ..domain import HIGH_RISK_INTENTS, Intent, RefusalReason
from ..schemas import GateDecision, RetrievalResult, SlotExtraction

# ------------------------------------------------------------
# 可调常量（集中在这里，是为了让"闸门有多严"一眼可数）
# ------------------------------------------------------------

#: 有效证据条数达到这个量，`count_score` 饱和。再多的证据只是重复。
_COUNT_SATURATION = 3

#: margin 的饱和尺度：margin 等于它时 `margin_score` 恰好 0.5。
#: 取 0.25 是因为精排分（归一化后）落在 0~0.25 的差距区间最能区分
#: "明显更相关"和"都差不多"。
_MARGIN_SCALE = 0.25

#: 只有一条证据时 margin_score 的上限。没有第二条证据 = 没有交叉印证，
#: 此时"最高分很高"只说明那条证据自己分高，不说明它答对了问题。
_SINGLE_EVIDENCE_MARGIN_CAP = 0.35

#: 实体覆盖率低于它就拒答。取 0.5 而不是更高：一个问题里的实体常常分散在
#: 不同证据里（一条表答时效、一条答品类），要求全落进同一批证据会误杀。
_COVERAGE_MIN = 0.5

#: base 分的权重：最高分最重要，但不独裁 —— 0.6 的权重交给"有没有交叉印证"
#: 和"够格的证据多不多"。
_W_TOP, _W_MARGIN, _W_COUNT = 0.40, 0.30, 0.30

#: 实体折价的底线。见模块 docstring。
_COVERAGE_FLOOR = 0.7

#: 高风险意图（退款/退货/取消/改地址）的阈值加成 —— **默认 0，即默认不生效**。
#:
#: 理由是保持与 CONTRACT.md §2.E 的严格一致：阈值就是
#: `settings.confidence_threshold`，不多也不少。默认关闭，任何一条按契约写的
#: 断言（"0.40 分在 0.35 阈值下应当通过"）都不会被这个扩展悄悄改掉行为。
#:
#: 打开它是个真实需求：同一份证据在"运费怎么算"上够用，在"这笔能退吗"上就不够，
#: 因为后者的错误代价是钱。所以留了这条通道 —— 由
#: `ConfidenceGate(high_risk_margin=...)` 显式开启，或从
#: `settings.high_risk_confidence_margin` 读（Settings 目前没有这个字段，
#: 用 getattr 取是为了不改动冻结的 config.py）。
_DEFAULT_HIGH_RISK_MARGIN = 0.0

# ------------------------------------------------------------
# 实体抽取：确定性规则，零依赖
# ------------------------------------------------------------
#
# 为什么不用模型抽实体：`check()` 是同步的、在生成之前跑的，在这里挂一次
# 模型调用等于把"最便宜的一道检查"变成"第二贵的调用"。用正则抽不全没关系
# （抽不到就退化成只按分数判），抽得稳才有意义。

_RE_ORDER_ID = re.compile(r"\bORD[-\s]?(\d{4,8})\b", re.I)
_RE_DAYS = re.compile(r"(\d{1,3})\s*(?:个)?\s*天")
_RE_YUAN = re.compile(r"(\d+(?:\.\d+)?)\s*元")

#: 品类后缀词表。"生鲜"这类词没有形态特征，只能靠"{0~4 个汉字}+后缀"兜住。
#: 这张表故意写得窄：**抽错比抽不到更糟** —— 抽错会把覆盖率压到 0，
#: 让闸门误拒一个本来答得出来的问题；抽不到只是让这条判据失效，
#: 那一份权重自然流回 top_score。
_PRODUCT_SUFFIXES = (
    "手机", "耳机", "电脑", "笔记本", "平板", "相机", "手表", "手环", "音箱",
    "衣服", "外套", "裤子", "裙子", "鞋", "背包", "食品", "零食", "水果",
    "生鲜", "图书", "玩具", "化妆品", "面膜", "洗发水", "家电", "冰箱",
    "洗衣机", "空调", "充电器", "数据线", "键盘", "鼠标",
)
_RE_PRODUCT = re.compile("|".join(r"[一-龥]{0,4}" + re.escape(s) for s in _PRODUCT_SUFFIXES))

#: 单独出现的后缀词（"书"、"鞋"）是泛指不是实体，丢掉。
_MIN_PRODUCT_LEN = 2

#: 实体类别。分类的目的只有一个：不同意图下各类实体的重要程度不同
#: （见 `_KIND_WEIGHTS`）。
_KIND_PRODUCT = "product"
_KIND_TIME = "time"
_KIND_MONEY = "money"
_KIND_ORDER = "order"

#: 按意图给实体类别配权重。
#:
#: 政策类问题的答案就是硬数字（几天内、多少钱、打几折），用户冲着这个数字来，
#: 数字没落在证据里就是答非所问；商品类问题的答案是属性（材质、尺码、怎么用），
#: 品类对不上整段就跑偏。同一份证据在两类问题上的"够用"标准本来就不一样，
#: 用一个不加权的覆盖率会把这两种情况抹平。
_KIND_WEIGHTS: dict[Intent, dict[str, float]] = {
    Intent.POLICY_FAQ: {_KIND_TIME: 1.0, _KIND_MONEY: 1.0, _KIND_PRODUCT: 0.4},
    Intent.PRODUCT_FAQ: {_KIND_PRODUCT: 1.0, _KIND_TIME: 0.6, _KIND_MONEY: 0.8},
}
_DEFAULT_KIND_WEIGHT = 1.0


def _normalize_order_id(raw: str) -> str:
    """`ord 1001` / `ORD1001` → `ORD-1001`。

    不归一化，用户打 `ord1001` 而证据写 `ORD-1001`，覆盖率会算成 0 并误拒。
    这是最典型的"规则没错、口径没统一"造成的假故障。
    """
    return "ORD-" + raw


def _extract_kinds(text: str) -> list[tuple[str, str]]:
    """抽出 (类别, 实体值) 列表，保序去重。"""
    if not text:
        return []

    pairs: list[tuple[str, str]] = []
    for m in _RE_ORDER_ID.finditer(text):
        pairs.append((_KIND_ORDER, _normalize_order_id(m.group(1))))
    for m in _RE_PRODUCT.finditer(text):
        word = m.group(0)
        if len(word) >= _MIN_PRODUCT_LEN:
            pairs.append((_KIND_PRODUCT, word))
    for m in _RE_DAYS.finditer(text):
        pairs.append((_KIND_TIME, f"{m.group(1)}天"))
    for m in _RE_YUAN.finditer(text):
        # 去掉小数尾巴：证据写 "99 元"、问题写 "99.0 元" 不该算两回事
        raw = m.group(1)
        num = str(int(float(raw))) if raw.replace(".", "").isdigit() else raw
        pairs.append((_KIND_MONEY, f"{num}元"))

    # 保序去重：审计日志里"抽到了哪些实体"读起来要跟问题同一个顺序
    return list(dict.fromkeys(pairs))


def extract_entities(text: str) -> list[str]:
    """公开的实体抽取：订单号 / 品类 / 时间 / 金额。纯确定性，无副作用。"""
    return [value for _, value in _extract_kinds(text)]


def _coverage_entities(question: str, slots: SlotExtraction | None) -> list[tuple[str, str]]:
    """参与覆盖率计算的问题侧实体（含槽位补充）。

    两条刻意的取舍：

    1. **订单号留在列表里，但在算覆盖率之前被剔除**（见 `_check`）。它是事务性
       标识，知识类证据（政策、FAQ）里天然不会出现它；把它算进覆盖率分母，
       等于"只要用户报了个单号就一定低覆盖"，系统性误拒。而"该给单号却没给"
       已经有专门的出口 —— `SlotExtraction.missing_for()` 报 `missing_slot`，
       不该由闸门越权代报。留在列表里是为了 `order_id_count` 这个审计量能数出来。
    2. **槽位里的实体要补进来**。多轮对话里用户说的是"那它能退吗"，实体在上一轮
       被指代消解进了 `slots`。只看 `question` 会抽空，这些实体就白丢了。
    """
    pairs = list(_extract_kinds(question))
    if slots is not None:
        if slots.product:
            pairs.append((_KIND_PRODUCT, slots.product))
        if slots.order_id:
            pairs.append((_KIND_ORDER, _normalize_order_id(slots.order_id)))
    # 保序去重：同一实体的不同来源（问题里 + 槽位里）不该重复计入分母
    return list(dict.fromkeys(pairs))


def _weighted_coverage(
    pairs: list[tuple[str, str]], haystack: str, intent: Intent | None
) -> tuple[float, list[str]]:
    """按类别加权算覆盖率，返回 (覆盖率, 未覆盖到的实体列表)。

    没有实体时返回 (0.0, [])。调用方要自己判"空" —— 覆盖率 0 和"没有可覆盖的
    东西"是两件事，混在一起会让无实体的正常问题被误拒。
    """
    if not pairs:
        return 0.0, []

    weights = _KIND_WEIGHTS.get(intent) if isinstance(intent, Intent) else None
    total = 0.0
    hit_weight = 0.0
    missing: list[str] = []

    for kind, value in pairs:
        w = (weights or {}).get(kind, _DEFAULT_KIND_WEIGHT)
        total += w
        # 两侧都压掉空白再比：`haystack` 由调用方传入前已 squash，
        # 这里保证实体侧也 squash，否则槽位里的 "生鲜 类" 永远匹配不上 "生鲜类"。
        if _squash(value) in haystack:
            hit_weight += w
        else:
            missing.append(value)

    if total <= 0:
        return 0.0, missing
    return hit_weight / total, missing


# ------------------------------------------------------------
# 闸门
# ------------------------------------------------------------


class ConfidenceGate:
    """生成前的置信度闸门。无模型调用、无副作用、同输入同输出。"""

    def __init__(self, settings: Settings | None = None, high_risk_margin: float | None = None) -> None:
        self.settings = settings or get_settings()
        #: 高风险意图的阈值加成。默认从 Settings 取（取不到就是 0，即不生效），
        #: 见 `_DEFAULT_HIGH_RISK_MARGIN` 的说明。
        self.high_risk_margin = (
            float(getattr(self.settings, "high_risk_confidence_margin", _DEFAULT_HIGH_RISK_MARGIN))
            if high_risk_margin is None
            else float(high_risk_margin)
        )
        #: 最近一次 `check()` 的判据 —— 给测试和审计用，业务逻辑不读它。
        #: 只留最后一条：闸门本身是纯函数，攒历史只会给并行请求"串味"留机会。
        self.last_checks: dict[str, float] = {}
        self._last_error = ""

    # --------------------------------------------------------
    def check(
        self,
        question: str,
        retrieval: RetrievalResult | None,
        intent: Intent | None = None,
        slots: SlotExtraction | None = None,
    ) -> GateDecision:
        """判断"这份证据配不配得上一句回答"。

        签名里没有 `answer` —— 见模块 docstring。任何想在这里读生成结果的改动，
        都是把这道检查的时序搞反了。

        永不抛异常：检索结果畸形（None、分数是 NaN、evidences 为空）一律降级成
        "证据不足"而不是崩掉整条链路。闸门崩了就等于闸门不存在 ——
        "检测幻觉的组件自己把自己搞挂"是最没面子的一种故障。
        """
        try:
            return self._check(question, retrieval, intent, slots)
        except Exception as exc:  # noqa: BLE001 — 闸门必须永远给出结论
            self._last_error = f"{type(exc).__name__}: {exc}"
            checks = self._skeleton_checks(threshold=self._effective_threshold(intent))
            checks["error"] = 1.0
            self.last_checks = checks
            return GateDecision.reject(
                0.0,
                RefusalReason.NO_EVIDENCE,
                checks,
                message="抱歉，检索环节出了点问题，我这边没有可靠依据，帮你转人工核实。",
            )

    # --------------------------------------------------------
    def _check(
        self,
        question: str,
        retrieval: RetrievalResult | None,
        intent: Intent | None,
        slots: SlotExtraction | None,
    ) -> GateDecision:
        threshold = self._effective_threshold(intent)

        raw: list[tuple[float, str]] = []
        for ev in (retrieval.evidences if retrieval else None) or []:
            raw.append((_safe_float(ev.score), _evidence_text(ev)))

        # 降序排序：所有判据都建立在"最高分/次高分/有效条数"上，顺序不稳会让
        # 同输入不同输出。sorted 稳定，同分时保持检索给的顺序 —— 仍是确定性的。
        raw.sort(key=lambda pair: pair[0], reverse=True)

        # 有效证据 = 分数过阈值的那些。`top_score` 取的是**全部召回**里的最高分，
        # 不是有效证据里的最高分 —— 这两个数只在"至少有一个人过线"时才相等，
        # 而它们不等的那种情况恰恰是最需要区分开的一种：
        #   raw 为空          → 库里真的没有 → no_evidence
        #   raw 不空但全不过线 → 有声噪但不足以回答 → weak_evidence
        # 把这两种都算成 no_evidence，会把"知识缺失"和"检索没做好"混为一谈，
        # 而飞轮上这两种问题的处理方式完全不同（一个补文档，一个调检索）。
        valid = [pair for pair in raw if pair[0] >= threshold]
        texts = [text for _, text in valid]

        top_score = raw[0][0] if raw else 0.0
        # margin 只在有效证据之间算。用全部召回算，会出现"1 条过线 +
        # 2 条噪音"被算成有交叉印证的情况 —— 噪音不构成印证。
        margin = _margin(valid)
        avg_score = (sum(s for s, _ in valid) / len(valid)) if valid else 0.0

        cov_pairs = _coverage_entities(question or "", slots)
        order_count = sum(1 for kind, _ in cov_pairs if kind == _KIND_ORDER)
        cov_pairs = [(k, v) for k, v in cov_pairs if k != _KIND_ORDER]
        # 证据侧先压掉空白再匹配：证据写 "7 天"、问题抽出的实体是 "7天"，
        # 不归一化就会把"一致"判成"没覆盖"，进而误拒一个本来答得出来的问题。
        coverage, missing = _weighted_coverage(cov_pairs, _squash(" ".join(texts)), intent)

        checks = {
            "top_score": round(_clamp01(top_score), 4),
            "evidence_count": float(len(valid)),
            "score_margin": round(margin, 4),
            "entity_coverage": round(coverage, 4),
            "avg_score": round(avg_score, 4),
            # -- 以下只作审计，不参与决策 --
            "recalled_count": float(len(raw)),
            "entity_count": float(len(cov_pairs)),
            "order_id_count": float(order_count),
            "threshold": round(threshold, 4),
        }

        confidence = _confidence(top_score, len(valid), margin, coverage)
        min_count = max(1, int(self.settings.min_evidence_count))

        # ---------------- 拒答规则（顺序即优先级） ----------------
        # 顺序是有意义的：先报"一条都没有"，再报"有但都不够格"，接着"够格但太少"，
        # 最后才是"够格但不贴题"。这样用户拿到的解释永远是最贴近根因的那一条，
        # 而不是"最先撞上的那条"。
        if not raw:
            checks["reject"] = 1.0
            decision = GateDecision.reject(
                0.0,
                RefusalReason.NO_EVIDENCE,
                checks,
                message="抱歉，知识库里没有检索到与这个问题相关的资料，我不猜，帮你转人工核实一下。",
            )
        elif top_score < threshold:
            checks["reject"] = 1.0
            decision = GateDecision.reject(
                confidence,
                RefusalReason.WEAK_EVIDENCE,
                checks,
                message=(
                    f"抱歉，我找到的资料相关性只有 {top_score:.2f}（低于 "
                    f"{threshold:.2f}），不足以支撑一个确定答复，帮你转人工确认。"
                ),
            )
        elif len(valid) < min_count:
            checks["reject"] = 1.0
            checks["below_min_count"] = 1.0
            decision = GateDecision.reject(
                confidence,
                RefusalReason.WEAK_EVIDENCE,
                checks,
                message=(
                    f"抱歉，只找到 {len(valid)} 条够格的资料，"
                    f"不到 {min_count} 条的最低要求，帮你转人工确认。"
                ),
            )
        elif cov_pairs and coverage < _COVERAGE_MIN:
            checks["reject"] = 1.0
            checks["missing_entity_count"] = float(len(missing))
            decision = GateDecision.reject(
                confidence,
                RefusalReason.LOW_COVERAGE,
                checks,
                message=(
                    "抱歉，我检索到的资料里没有覆盖到："
                    + "、".join(missing[:3])
                    + "。信息对不上我不敢乱答，帮你转人工确认。"
                ),
            )
        else:
            checks["reject"] = 0.0
            decision = GateDecision.ok(confidence, checks)

        self.last_checks = checks
        return decision

    # --------------------------------------------------------
    def _effective_threshold(self, intent: Intent | None) -> float:
        """本次实际生效的证据分阈值。

        默认等于 `settings.confidence_threshold`（契约 §2.E 的要求）。
        `high_risk_margin > 0` 时对高风险意图加价：同一份证据在"运费怎么算"上够用，
        在"这笔钱能退吗"上就不够 —— 错误代价不一样，门槛就不该一样。
        """
        base = float(self.settings.confidence_threshold)
        if self.high_risk_margin > 0 and isinstance(intent, Intent) and intent in HIGH_RISK_INTENTS:
            return _clamp01(base + self.high_risk_margin)
        return base

    def _skeleton_checks(self, threshold: float) -> dict[str, float]:
        """降级路径用的骨架 —— 键位与正常路径完全一致。

        键位一致是硬要求：下游读 `checks["top_score"]` 时不该因为这次走了降级
        路径就 KeyError。"兜底"和"兜底并保持一致"是两件事。
        """
        return {
            "top_score": 0.0,
            "evidence_count": 0.0,
            "score_margin": 0.0,
            "entity_coverage": 0.0,
            "avg_score": 0.0,
            "recalled_count": 0.0,
            "entity_count": 0.0,
            "order_id_count": 0.0,
            "threshold": round(threshold, 4),
            "reject": 1.0,
        }

    @property
    def last_error(self) -> str:
        """最近一次降级的错误摘要（没出过错就是空串）。"""
        return self._last_error


# ------------------------------------------------------------
# 内部工具
# ------------------------------------------------------------


def _evidence_text(ev) -> str:
    """把一条证据摊平成可做子串匹配的文本。

    标题要一起算：品类常常只出现在标题里（"退款政策 > 生鲜类"），
    只看正文会把覆盖率算低，进而误拒。
    """
    chunk = getattr(ev, "chunk", None)
    if chunk is None:
        return str(ev)
    parts = [getattr(chunk, "title", "") or "", getattr(chunk, "text", "") or ""]
    table = getattr(chunk, "table", None)
    if table:
        parts.append(" ".join(" ".join(row) for row in table))
    return " ".join(parts)


def _safe_float(value) -> float:
    """分数来自任何一路检索，NaN / None / 字符串都可能出现。

    NaN 特别危险：它跟任何数比较都是 False，会让 `score >= threshold` 静默为假，
    证据凭空消失且不留痕。这里直接压成 0，让"这条无效"显式化。
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return f


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _margin(ranked: list[tuple[float, str]]) -> float:
    """最高分与次高分之差。

    只有一条时返回 0.0 —— **不是**返回"跟阈值之差"或"跟 1.0 之差"。
    margin 的语义是"第二名离第一名多远"，也就是"有没有第二条独立证据印证同一个
    结论"。只有一条时这个问题的答案是"没有"，数值就该是 0，
    而不是一个看着挺大的假差距。
    """
    if len(ranked) < 2:
        return 0.0
    return max(0.0, ranked[0][0] - ranked[1][0])


def _count_score(n: int) -> float:
    """证据条数 → 0~1，对数饱和。见模块 docstring 里为什么不用线性。"""
    if n <= 0:
        return 0.0
    return _clamp01(math.log1p(n) / math.log1p(_COUNT_SATURATION))


def _squash(text: str) -> str:
    """去掉所有空白，用于宽松子串匹配。"""
    return re.sub(r"\s+", "", text or "")


def _confidence(
    top_score_raw: float,
    valid_count: int,
    margin: float,
    coverage: float,
) -> float:
    """综合置信度。公式与理由见模块 docstring。

    `top_score_raw` 是**全部召回**里的最高分（与 `checks["top_score"]` 同源），
    不是有效证据里的最高分 —— 这样一个"没人过线"的结果不会因为某条刚好卡在
    边缘而被算出一个非零的 confidence。
    """
    if valid_count <= 0:
        return 0.0

    top_score = _clamp01(top_score_raw)

    if valid_count == 1:
        # 单条证据：margin 恒为 0，这里再显式封一次顶。原因是有别处（比如离线
        # 评估脚本）可能绕过 `_margin` 直接传 margin 进来，而"单条证据 = 无交叉
        # 印证"这个结论不该依赖调用方把参数传对。
        margin_score = min(_SINGLE_EVIDENCE_MARGIN_CAP, margin / (_MARGIN_SCALE + margin))
    else:
        margin_score = margin / (_MARGIN_SCALE + margin)

    base = _W_TOP * top_score + _W_MARGIN * margin_score + _W_COUNT * _count_score(valid_count)
    discounted = base * (_COVERAGE_FLOOR + (1.0 - _COVERAGE_FLOOR) * _clamp01(coverage))
    return round(_clamp01(discounted), 4)
