"""数据飞轮。

对外只暴露四样东西，上层（workflow/graph.py 的 flywheel 节点、api/app.py 的
审核路由）不需要知道内部细节：

    FlywheelSource / source_for       —— 入口分类
    FlywheelStore                     —— 落池 / 排队 / 审核 / 回灌
    build_record()                    —— 从一次失败问答的现场信息组装记录
    apply_review_to_kb()              —— 审核通过后把答案回灌知识库

`build_record()` 单独放在这里，是为了让"落池要带哪些字段"这件事只有一个
答案 —— 图节点、API 反馈接口、离线脚本三处都在落池，各写一遍必然会漏掉
`retrieval_snapshot`，而漏掉它飞轮就退化成一个普通的问题收集表。
"""

from __future__ import annotations

from typing import Any

from ..config import Settings, get_settings
from ..domain import FlywheelSource, RefusalReason
from ..schemas import FlywheelRecord, KbChunk, RetrievalResult
from .feedback import (
    TRIAGE_BACKFILL_KB,
    TRIAGE_LABELS,
    TRIAGE_REVIEW_ANSWER,
    TRIAGE_TUNE_RETRIEVAL,
    TRIAGE_UNKNOWN,
    is_knowledge_candidate,
    is_knowledge_gap,
    source_for,
    triage_hint,
    triage_label,
)
from .store import FlywheelStore, dedup_key, make_record_id, normalize_question

__all__ = [
    "FlywheelStore",
    "FlywheelSource",
    "FlywheelRecord",
    "build_record",
    "source_for",
    "dedup_key",
    "normalize_question",
    "make_record_id",
    "apply_review_to_kb",
    "is_knowledge_gap",
    "is_knowledge_candidate",
    "triage_hint",
    "triage_label",
    "TRIAGE_BACKFILL_KB",
    "TRIAGE_TUNE_RETRIEVAL",
    "TRIAGE_REVIEW_ANSWER",
    "TRIAGE_UNKNOWN",
    "TRIAGE_LABELS",
]


def build_record(
    *,
    session_id: str,
    question: str,
    source: FlywheelSource | None = None,
    refusal_reason: RefusalReason | None = None,
    thumbs_down: bool = False,
    retrieval: RetrievalResult | None = None,
    resolved_question: str = "",
    intent: Any = None,
) -> FlywheelRecord:
    """组装一条待落池的记录。

    两个刻意的行为：

    * `source` 不给就由 `source_for()` 推导 —— 让"入口分类"只有一个实现，
      调用方不需要记住 `refusal_reason` 和 `thumbs_down` 的优先级。
    * `retrieval` 直接挂到 `retrieval_snapshot`。调用方传的就是**当下这一次**
      的检索结果对象，不做任何裁剪或降采样 —— 快照的价值就在于完整：
      `dense_hits` / `sparse_hits` / `retrieved_ids` 这些中间态，正是事后判断
      "是没召回到还是召回了没排上来"的依据。
    """
    return FlywheelRecord(
        # 这里给个临时 id，真正的 id 由 store 按去重键重新推导（保证确定性）
        record_id=make_record_id(dedup_key(question, resolved_question)),
        session_id=session_id,
        question=question,
        resolved_question=resolved_question,
        intent=intent,
        source=source or source_for(refusal_reason, thumbs_down),
        refusal_reason=refusal_reason,
        retrieval_snapshot=retrieval,
    )


def apply_review_to_kb(
    store: FlywheelStore,
    record: FlywheelRecord,
    retriever: Any | None = None,
    *,
    settings: Settings | None = None,
) -> KbChunk | None:
    """审核通过后把答案回灌知识库，返回写进去的 `KbChunk`。

    `retriever` 是检索层的 `RetrievalPipeline`（或用例里任何有
    `index(chunks)` 的对象），只鸭子类型调用它的 `index`，不 import 检索层 ——
    飞轮不该因为检索层换实现而改代码，也避免测试时必须拉起整个检索栈。

    幂等靠两道锁：`to_kb_chunk()` 的确定性 chunk_id，和 `mark_backfilled()`
    的去重登记。跑两次回灌不会让知识库多一条。
    """
    chunk = store.to_kb_chunk(record)
    if chunk is None:
        return None
    if not store.mark_backfilled(chunk, record.record_id):
        # 已经回灌过：返回同一条，但不再写库
        return chunk
    if retriever is not None and hasattr(retriever, "index"):
        try:
            retriever.index([chunk])
        except Exception as exc:  # pragma: no cover
            # 写库失败不能把审核动作回滚 —— 记录仍是 approved，
            # 下次跑回灌脚本会重试（chunk_id 不变，重试天然幂等）
            store.errors.append(f"apply_review_to_kb index failed: {exc!r}")
    return chunk
