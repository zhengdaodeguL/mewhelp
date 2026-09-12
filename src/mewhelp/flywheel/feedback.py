"""数据飞轮的入口分类与"审核前分诊"。

这个模块只做两件小事，但它们是整个飞轮能被人用起来的前提：

1. `source_for()` —— 把「拒答原因 + 用户点踩」翻译成三个入口之一。没有它，
   落池的记录就只是一堆没有来源标签的问题，运营看不出"哪个环节在漏"。
2. `triage_hint()` —— 用落池时存的检索快照，判断这条到底该"补文档"还是
   "调检索"。

第 2 点是这个模块存在的核心理由。人工审核最容易犯的错，是把所有没答上来的
问题都当成"缺知识"去写文档 —— 结果文档越写越多，检索还是捞不出来。事实是
失败至少分两种：

  * 知识库里根本没有这条知识（真缺知识）→ 补文档 / 补 FAQ，`backfill_kb`
  * 知识库有，但没被检索到（召回或排序失败）→ 调 embedding / 精排 / 阈值，
    补文档完全无用，`tune_retrieval`

这两种情况的处理动作完全不同，而区分它们的唯一依据就是落池那一刻的
`RetrievalResult` 快照。这就是 `FlywheelRecord.retrieval_snapshot`
必须存在的理由 —— 事后再去重跑检索是复现不出当时的现场的（知识库已经变了、
会话上下文没了）。
"""

from __future__ import annotations

from ..config import Settings, get_settings
from ..domain import FlywheelSource, RefusalReason
from ..schemas import FlywheelRecord

# ============================================================
# 分诊结论
# ============================================================
#: 一条证据都没召回 —— 最可能是真的没有这条知识（也可能是问法完全对不上）
TRIAGE_BACKFILL_KB = "backfill_kb"
#: 召回了但分数撑不起回答 —— 知识在库里，是检索/排序的问题
TRIAGE_TUNE_RETRIEVAL = "tune_retrieval"
#: 证据分数够了却没通过 —— 问题在生成或闸门判据，不在检索
TRIAGE_REVIEW_ANSWER = "review_answer"
#: 没有快照，无法判定 —— 这本身就是"必须存快照"的反面教材
TRIAGE_UNKNOWN = "unknown"

#: 分诊结论 → 给审核人的中文说明（前端/审核台直接展示）
TRIAGE_LABELS: dict[str, str] = {
    TRIAGE_BACKFILL_KB: "一条证据都没召回：大概率知识库真缺这条，优先补文档",
    TRIAGE_TUNE_RETRIEVAL: "召回了但分数不足：知识在库里，先调检索参数/embedding，别急着写文档",
    TRIAGE_REVIEW_ANSWER: "证据分数够却没通过：查生成与闸门判据，检索侧不用动",
    TRIAGE_UNKNOWN: "无检索快照，无法判定是缺知识还是没检索到",
}

#: 这三种拒答原因属于"知识侧"问题，补文档/调检索才有意义。
#: MISSING_SLOT（缺订单号）和 HIGH_RISK_UNAPPROVED（高风险待批）不进这个集合 ——
#: 它们是流程问题，写一百篇文档也解决不了"用户没给订单号"。
_KNOWLEDGE_REASONS: frozenset[RefusalReason] = frozenset(
    {
        RefusalReason.NO_EVIDENCE,
        RefusalReason.WEAK_EVIDENCE,
        RefusalReason.LOW_COVERAGE,
    }
)


def source_for(
    reason: RefusalReason | None, thumbs_down: bool
) -> FlywheelSource:
    """推导这条记录该归到飞轮的哪个入口。

    优先级是"用户显式信号 > 自评 > 静默拒答"，理由是信息量：
      * 点踩是用户亲自告诉你有问题，即使这条当时被拒答了，也应该按点踩走
        （点踩链路会额外带上 comment，是三个入口里唯一有用户原话的）；
      * 自评没过是模型自己承认没把握，比闸门拒答更接近答案质量问题；
      * 其余一律归 `refused`（含 `no_evidence` / `weak_evidence` /
        `low_coverage`）。

    注意 `RefusalReason` 有 6 个取值而 `FlywheelSource` 只有 3 个，这是故意的：
    `missing_slot` / `high_risk_unapproved` 也会落成 `refused`，因为它们对用户
    而言同样是"没拿到答案"。审核台要过滤掉它们时用 `is_knowledge_gap()`
    判断，而不是让入口分类去猜。
    """
    if thumbs_down:
        return FlywheelSource.THUMBS_DOWN
    if reason is RefusalReason.SELF_EVAL_FAILED:
        return FlywheelSource.SELF_EVAL
    # 兜底给 refused：这是最常见的入口，宁可把来源标粗也不能标错成"自评失败"
    return FlywheelSource.REFUSED


def is_knowledge_gap(reason: RefusalReason | None) -> bool:
    """这条拒答是不是"知识侧"问题 —— 决定要不要写文档。

    单独抽成一个函数而不是塞进 `source_for()`，是因为入口分类（三个枚举）
    和分诊（要不要补文档）是两个维度：一条 `refused` 记录完全可能是缺订单号，
    补知识库毫无用处。
    """
    return reason in _KNOWLEDGE_REASONS


def is_knowledge_candidate(record: FlywheelRecord) -> bool:
    """这条记录值不值得用"补文档 / 调检索"的思路处理。

    除了知识侧的三类拒答，用户点踩也算 —— 点踩的记录可能压根没拒答
    （答案给出去了但被否），这时 `refusal_reason` 是空的，却恰恰是最需要
    人工看一眼的那批。缺槽位、高风险待批则明确排除：那是流程问题，
    分诊表里混进它们只会让"该写几篇文档"这个数字虚高。
    """
    return is_knowledge_gap(record.refusal_reason) or record.source is FlywheelSource.THUMBS_DOWN


def triage_hint(
    record: FlywheelRecord, settings: Settings | None = None
) -> str:
    """用落池时的检索快照给这条记录分诊，返回 `TRIAGE_*` 之一。

    判据只有一条线：快照里有没有证据、最高分够不够 `confidence_threshold`。
    刻意不用"重跑一次检索"来判断 —— 知识库在审核时已经变了，重跑出来的是
    今天的现场，不是当时的现场，那会把"没检索到"误判成"真缺知识"。

    例外：thumbs_down 的记录即使有满分的证据也可能被点踩（答案主次颠倒、
    口径错、语气差），所以这种分诊只回答"要不要动检索和知识库"，
    不回答"这条是不是 bug"。
    """
    snapshot = record.retrieval_snapshot
    if snapshot is None:
        return TRIAGE_UNKNOWN
    if not snapshot.evidences:
        # 一条都没召回。注意这里不能直接断言"真缺知识"：也可能是问法太怪
        # 导致向量和 BM25 都没命中，所以中文说明里写成"大概率"。
        return TRIAGE_BACKFILL_KB
    threshold = (settings or get_settings()).confidence_threshold
    if snapshot.top_score < threshold:
        return TRIAGE_TUNE_RETRIEVAL
    return TRIAGE_REVIEW_ANSWER


def triage_label(hint: str) -> str:
    """分诊结论 → 中文说明。未知结论返回空串而不是抛异常。"""
    return TRIAGE_LABELS.get(hint, "")
