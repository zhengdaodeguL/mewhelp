"""数据飞轮的落池 / 排队 / 审核 / 回灌。

飞轮的闭环是这四步：

    线上失败 → 落池(带检索快照) → 按被问频次排队审核 → 审核通过的答案回灌知识库
                     ↑                                        ↓
                     └────────── 下一个人再问同样的问题时答得上来了 ────┘

这一步比"调模型"值钱得多：Prompt 调一次只对当下有效，飞轮转一圈是永久收益。

三个设计决定值得单独讲：

**① 同题去重，`ask_count` 累加。**
人工审核是稀缺资源，100 条记录里有 60 条是同一个问题，就必须先审那一条。
所以 `record()` 对"同一道题"做合并而不是插新行，`pending()` 按 `ask_count`
降序 —— 排队顺序本身就是"运营优先级"。去重键是问题的**归一化文本**
（去空白、去标点、转小写、统一全角），并对 `None` 也归一化，避免
"None" 和 "" 被当成两道不同的题。

**② 落池必须带检索快照。**
`retrieval_snapshot` 在合并时**只补空、不覆盖**：第一次落池的现场才是现场，
后来的重试（可能因为当时知识库已经更新过）盖掉它，就等于把证据毁掉。
合并时也尽量补齐 source/intent/session，让第一次没带上的信息能补上，
但绝不覆盖已有值。

**③ 存储走 `_rows` 这个 dict 而不是列表。**
将来换 SQLAlchemy（`storage/models.py` 的 `FlywheelRow`）时，只要照着这几个
方法的语义实现一遍就行 —— 接口里没有任何"内存"的味道：没有游标、没有下标，
返回的全是 Pydantic 模型或其副本。

可选落盘（JSONL）一是为了演示"重启不丢"，二是为了让人工审核的结果在
数据飞轮的下一环（微调数据构造）里能被别的进程读到。落盘用追加而不是覆盖，
进程崩了也只丢最后一行；读的时候对坏行直接跳过，绝不让一行脏数据搞挂整个池子。
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from ..config import Settings, get_settings
from ..domain import FlywheelSource
from ..schemas import FlywheelRecord, KbChunk

__all__ = [
    "FlywheelStore",
    "dedup_key",
    "normalize_question",
    "make_record_id",
    "STATUS_PENDING",
    "STATUS_APPROVED",
    "STATUS_REJECTED",
    "STATUS_MERGED",
]

#: 状态取值（与 `FlywheelRecord.status`、`storage.models.FlywheelRow.status` 对齐）
STATUS_PENDING: Literal["pending"] = "pending"
STATUS_APPROVED: Literal["approved"] = "approved"
STATUS_REJECTED: Literal["rejected"] = "rejected"
STATUS_MERGED: Literal["merged"] = "merged"

_PUNCT_RE = re.compile(r"[\s　]+")
#: 中英文常见标点 + 全角符号。逐个删掉，避免"改地址？"和"改地址"被当成两道题。
_STRIP_CHARS_RE = re.compile(
    r"[?？!！。，,、；;：:\"'“”‘’（）()\[\]【】<>《》~～\-_+=*/\\|@#$%^&…·`]+"
)
_FULLWIDTH_MAP = {
    ord(f): ord(h)
    for f, h in zip(
        "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
        "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz",
    )
}


def normalize_question(text: str | None) -> str:
    """问题归一化 —— 去重键的原材料。

    做四件事：全角转半角、统一小写、删掉所有标点和空白。删标点而不是规范化
    标点，是因为用户打"运费怎么算"和"运费怎么算？"、「运费怎么算。。。」
    在业务上是同一个问题，多一条重复记录就多浪费一次人工审核。

    容忍 `None`：调用方从状态里取 `resolved_question` 时它可能是空的，
    让归一化自己兜住比在每个调用点写 `or ""` 更不容易漏。
    """
    if not text:
        return ""
    text = str(text).translate(_FULLWIDTH_MAP).lower()
    text = _STRIP_CHARS_RE.sub("", text)
    return _PUNCT_RE.sub("", text)


def make_record_id(question_key: str, session_id: str = "", salt: str = "") -> str:
    """由去重键生成确定性 `record_id`。

    用哈希而不是 uuid4：同题重复落池要复用同一条记录，id 必须可复现 ——
    调试时能直接由问题算出 id，人工审核结果回填也不会对不上号。
    `salt` 是给"同题、不同会话、要分别建条"这种少数场景留的口子。
    """
    raw = f"{question_key}\x1f{session_id}\x1f{salt}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"fw_{digest}"


def dedup_key(question: str | None, resolved_question: str | None = None) -> str:
    """公开的去重键计算 —— 与 `FlywheelStore` 内部用的**必须是同一个函数**。

    为什么要把这个看起来像内部细节的东西暴露出来：`storage/models.py` 的
    `FlywheelRow` 上有一条 `UniqueConstraint("question_key")`，写库那侧要自己
    算这个键。如果两边各写一遍归一化规则（哪怕只差一个标点处理），就会出现
    "内存里合并成一条、写库时唯一约束报冲突"这种最难查的不一致。
    所以规则只能有一份，谁要落库都调这里。

    优先用补全后的问题：半截话的原文千奇百怪，拿原文当键几乎永远命中不了。
    """
    return normalize_question(resolved_question) or normalize_question(question)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _stable(*parts: str) -> str:
    """确定性短哈希。

    不能用 Python 内置 `hash()` —— CPython 对 str 的哈希带进程级随机盐，
    换一次进程同样的输入会得到不同的值，飞轮就会变成"每次重启知识库多一堆
    重复条目"。sha1 在这里既确定又够短。
    """
    raw = "\x1f".join(p or "" for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class FlywheelStore:
    """问题池 + 审核台。默认纯内存，可选 JSONL 落盘。

    线程安全按"单进程无并发写"假设（mock 演示和 pytest 都满足）；真上多进程
    时换掉 `_rows` 的实现即可，外面这几个方法的签名不用动。
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        path: Path | str | None = None,
        persist: bool | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        # 默认不落盘：pytest 和演示都不该往仓库里写文件，要落盘就显式给 path
        self.path = Path(path) if path else None
        self.persist = bool(persist and self.path)
        #: record_id -> FlywheelRecord。用 dict 而不是 list，因为三个查询都
        #: 是按键取，而"按 ask_count 排序"只在 pending 里做一次。
        self._rows: dict[str, FlywheelRecord] = {}
        #: 归一化问题 -> record_id。同题合并全靠它，O(1) 找到已有记录。
        self._index: dict[str, str] = {}
        #: 模块内部兜底留痕（不抛给调用方，但要能被诊断）
        self.errors: list[str] = []
        #: 已回灌过的 chunk_id -> record_id。做幂等键，重复回灌同一问题不再产条目。
        self._kb_index: dict[str, str] = {}

    # ========================================================
    # ① 落池
    # ========================================================

    def record(self, rec: FlywheelRecord) -> FlywheelRecord:
        """落池一条记录；同题则合并进已有记录，`ask_count += 1`。

        合并时保留的字段遵循"现场优先"：第一次落池的检索快照和拒答原因是
        当时的真相，后来的重复提问不能覆盖它。反过来，为空的可选字段会被
        后来的值补上（比如第一次没带 resolved_question）。
        """
        try:
            key = self._dedup_key(rec)
            existing_id = self._index.get(key)
            if existing_id and existing_id in self._rows:
                merged = self._merge(self._rows[existing_id], rec)
                self._rows[existing_id] = merged
                self._append_jsonl("merge", merged)
                return merged.model_copy(deep=True)

            row = rec.model_copy(deep=True)
            # 同题第二次落池时调用方很可能沿用原来的 record_id（或又随机一个），
            # 统一由去重键推导，保证 id 稳定可复现。
            row.record_id = make_record_id(key, row.session_id)
            if row.record_id in self._rows:  # 极端碰撞：加会话再哈希一次
                row.record_id = make_record_id(key, row.session_id, salt="dup")
            if not row.resolved_question:
                row.resolved_question = row.question
            self._rows[row.record_id] = row
            self._index[key] = row.record_id
            self._append_jsonl("add", row)
            return row.model_copy(deep=True)
        except Exception as exc:  # pragma: no cover - 兜底路径，正常不会走到
            # 异常不外泄：落池失败也不能让用户的回答流程崩掉。
            # 返回一条标记过的记录，调用方看得出来"这次没存上"。
            self.errors.append(f"record failed: {exc!r}")
            fallback = rec.model_copy(deep=True)
            # 兜底 id 也要确定：整条链路承诺"同输入同输出"，一个失败分支里
            # 混进随机值，就足以让复现问题变成不可能。
            fallback.record_id = make_record_id(
                normalize_question(rec.question), rec.session_id, salt="failed"
            )
            fallback.status = STATUS_PENDING
            return fallback

    def _dedup_key(self, rec: FlywheelRecord) -> str:
        """去重键 —— 和公开的 `dedup_key()` 是同一份规则，不要在这里另写一遍。

        用 `resolved_question` 优先是刻意的：「那它能退吗」补全成「七天无理由
        可以退货吗」之后，才和其它用户的同一问题对得上。
        """
        return dedup_key(rec.question, rec.resolved_question)

    def _merge(self, old: FlywheelRecord, new: FlywheelRecord) -> FlywheelRecord:
        """同题合并。这份合并规则就是"飞轮的运营逻辑"本身。"""
        merged = old.model_copy(deep=True)
        merged.ask_count = (old.ask_count or 0) + max(1, new.ask_count or 1)
        # 快照只补空不覆盖：第一次落池的现场才是现场。
        if merged.retrieval_snapshot is None and new.retrieval_snapshot is not None:
            merged.retrieval_snapshot = new.retrieval_snapshot
        if merged.refusal_reason is None and new.refusal_reason is not None:
            merged.refusal_reason = new.refusal_reason
        if merged.intent is None and new.intent is not None:
            merged.intent = new.intent
        if not merged.resolved_question and new.resolved_question:
            merged.resolved_question = new.resolved_question
        # 来源升级：用户点踩是比静默拒答更强的信号，值得改写来源标签。
        if self._source_rank(new.source) > self._source_rank(merged.source):
            merged.source = new.source
        # 已审核的记录被再次问到，说明审核时给的那条答案没解决问题 ——
        # 打回待审，让运营有机会看到"回灌后又被问了"。
        if merged.status in (STATUS_APPROVED, STATUS_REJECTED, STATUS_MERGED):
            merged.status = STATUS_PENDING
            merged.answer = None
            merged.reviewer = None
            merged.reviewed_at = None
        return merged

    @staticmethod
    def _source_rank(source: Any) -> int:
        """来源信号强度。数字越大越"值得优先处理"。"""
        value = getattr(source, "value", source)
        return {"refused": 1, "self_eval": 2, "thumbs_down": 3}.get(str(value), 0)

    # ========================================================
    # ② 排队
    # ========================================================

    def pending(self, limit: int = 50) -> list[FlywheelRecord]:
        """待审记录，按 `ask_count` 降序 —— 先审被问得最多的。

        并列时按 `(created_at, record_id)` 再排一次。这一步不是可有可无的
        修饰：没有它，并列的记录顺序取决于 dict 的插入顺序，同一份数据在
        两次运行里能排出两个顺序，"确定性"就没了。
        """
        try:
            rows = [r for r in self._rows.values() if r.status == STATUS_PENDING]
            rows.sort(
                key=lambda r: (-(r.ask_count or 0), r.created_at, r.record_id)
            )
            if limit is not None and limit >= 0:
                rows = rows[:limit]
            return [r.model_copy(deep=True) for r in rows]
        except Exception as exc:  # pragma: no cover
            self.errors.append(f"pending failed: {exc!r}")
            return []

    def get(self, record_id: str) -> FlywheelRecord | None:
        """按 id 取一条（副本）。审核台和 API 层都要用。"""
        row = self._rows.get(record_id)
        return row.model_copy(deep=True) if row else None

    def all(self) -> list[FlywheelRecord]:
        """全部记录（副本），按落池时间升序。导出微调数据时用。"""
        rows = sorted(self._rows.values(), key=lambda r: (r.created_at, r.record_id))
        return [r.model_copy(deep=True) for r in rows]

    def __len__(self) -> int:
        return len(self._rows)

    # ========================================================
    # ③ 审核
    # ========================================================

    def review(
        self,
        record_id: str,
        approved: bool,
        reviewer: str,
        answer: str | None = None,
    ) -> FlywheelRecord:
        """人工审核。approved → `approved`，否则 → `rejected`。

        找不到记录时**不抛异常**，返回一条 `pending` 的占位记录并在 `errors`
        里留痕：审核动作是人在前端点的，接口层不能因为一次点击就 500。
        """
        try:
            row = self._rows.get(record_id)
            if row is None:
                self.errors.append(f"review: record not found {record_id}")
                return FlywheelRecord(
                    record_id=record_id,
                    session_id="",
                    question="",
                    source=_UNKNOWN_SOURCE,
                    status=STATUS_PENDING,
                )
            updated = row.model_copy(deep=True)
            updated.status = STATUS_APPROVED if approved else STATUS_REJECTED
            updated.reviewer = reviewer
            updated.reviewed_at = _utcnow()
            if approved:
                # 审核人给了答案就用审核人的（人以证据为准）；没给就把模型
                # 当时的回答或证据原文留档，别让 approved 的记录没有可回灌的正文。
                updated.answer = answer or row.answer or self._fallback_answer(row)
            else:
                updated.answer = answer or row.answer
            self._rows[record_id] = updated
            self._append_jsonl("review", updated)
            return updated.model_copy(deep=True)
        except Exception as exc:  # pragma: no cover
            self.errors.append(f"review failed: {exc!r}")
            row = self._rows.get(record_id)
            if row is not None:
                return row.model_copy(deep=True)
            return FlywheelRecord(
                record_id=record_id,
                session_id="",
                question="",
                source=_UNKNOWN_SOURCE,
            )

    @staticmethod
    def _fallback_answer(rec: FlywheelRecord) -> str | None:
        """审核通过但没给答案时，用快照里最像答案的证据顶上。

        这样做而不是留空，是因为 `to_kb_chunk()` 需要正文才建得出条目；
        留空会让"审核通过"变成一句空话，回灌环节静默失败。
        """
        snapshot = rec.retrieval_snapshot
        if snapshot and snapshot.evidences:
            return snapshot.evidences[0].chunk.text
        return None

    # ========================================================
    # ④ 回灌 —— 审核通过的记录变知识库条目
    # ========================================================

    def to_kb_chunk(self, record: FlywheelRecord | str) -> KbChunk | None:
        """把审核通过的记录转成 `KbChunk`，交给上层写进向量库。

        设计要点：
          * `chunk_id` 由**归一化问题**哈希得来，不看 `record_id`。这样同一个
            问题即使因为并发/重放产生了不同 id 的记录，落到知识库里也只有一条
            （`ask_count` 20 的问题不会变成 20 条重复 FAQ）。
          * 标题前缀固定为"用户高频问题"，方便在知识库里把飞轮产出的条目
            整体筛出来、整体回滚。
          * 未通过审核的记录返回 `None` 而不是抛异常 —— 调用方可以无脑遍历
            全部记录调这个方法。
        """
        try:
            rec = self.get(record) if isinstance(record, str) else record
            if rec is None:
                self.errors.append(f"to_kb_chunk: record not found {record}")
                return None
            if rec.status != STATUS_APPROVED:
                return None
            body = (rec.answer or "").strip()
            if not body:
                self.errors.append(f"to_kb_chunk: approved but no answer {rec.record_id}")
                return None
            question = (rec.resolved_question or rec.question or "").strip()
            chunk_id = self.kb_chunk_id(question)
            keywords = [k for k in (rec.intent.value if rec.intent else "", "flywheel") if k]
            return KbChunk(
                chunk_id=chunk_id,
                doc_id="flywheel_faq",
                title=f"用户高频问题 · {question}" if question else "用户高频问题",
                breadcrumb=["用户高频问题", question or "未命名"],
                text=f"Q: {question}\nA: {body}" if question else body,
                keywords=keywords,
                is_table=False,
                char_len=len(body) + len(question),
                source_path="flywheel://reviewed",
            )
        except Exception as exc:  # pragma: no cover
            self.errors.append(f"to_kb_chunk failed: {exc!r}")
            return None

    @staticmethod
    def kb_chunk_id(question: str) -> str:
        """确定性 chunk_id。同问题永远同一 id → 重复审核不产生重复条目。"""
        return f"fw_faq_{_stable(normalize_question(question))[:16]}"

    def approved_chunks(self) -> list[KbChunk]:
        """全部已通过记录的知识库条目（按 chunk_id 去重）。

        去重放在这里而不是指望上层：运营很可能为了修一句话把同一条记录
        反复审核几次，没有这层去重，向量库里就会出现多份内容略有差异的
        同一问题，检索时互相挤占 top-k。
        """
        seen: dict[str, KbChunk] = {}
        try:
            for rec in self._rows.values():
                if rec.status != STATUS_APPROVED:
                    continue
                chunk = self.to_kb_chunk(rec)
                if chunk is not None:
                    seen.setdefault(chunk.chunk_id, chunk)
        except Exception as exc:  # pragma: no cover
            self.errors.append(f"approved_chunks failed: {exc!r}")
        return list(seen.values())

    def mark_backfilled(self, chunk: KbChunk, record_id: str = "") -> bool:
        """登记"这条已写进向量库"。重复登记返回 False。

        这就是回灌的幂等闸门，和 `to_kb_chunk` 的确定性 chunk_id 一起，
        保证跑两次回灌脚本不会让知识库翻倍。
        顺带把来源 `record_id` 记下来，之后能追溯"某条知识是哪个用户问题换来的"。
        """
        if chunk.chunk_id in self._kb_index:
            return False
        self._kb_index[chunk.chunk_id] = record_id
        return True

    @property
    def backfilled_ids(self) -> set[str]:
        return set(self._kb_index)

    # ========================================================
    # 统计
    # ========================================================

    def stats(self) -> dict:
        """各状态计数 + 几个"飞轮转得怎么样"的指标。

        除了 `total/pending/approved/rejected/merged`，额外给 `by_source`
        和 `top_questions`：前者回答"哪个环节在漏"（拒答多还是点踩多），
        后者就是审核排队的依据。`triage` 计数回答"该写文档还是该调检索" ——
        这个数字直接决定下一周的工作安排。
        """
        try:
            # 分诊函数在 feedback 里，而 feedback 反向 import 本模块的模型会成环，
            # 所以放在函数内 import。必须放在最前面：函数体里出现赋值（import 也是
            # 赋值）会把名字变成局部变量，写在用到它之后就是 UnboundLocalError。
            from .feedback import is_knowledge_candidate, triage_hint

            rows = list(self._rows.values())
            by_status = {
                STATUS_PENDING: 0,
                STATUS_APPROVED: 0,
                STATUS_REJECTED: 0,
                STATUS_MERGED: 0,
            }
            by_source = {"refused": 0, "self_eval": 0, "thumbs_down": 0}
            by_reason: dict[str, int] = {}
            refusal_side = 0
            for r in rows:
                by_status[r.status] = by_status.get(r.status, 0) + 1
                src = getattr(r.source, "value", r.source)
                by_source[str(src)] = by_source.get(str(src), 0) + 1
                if r.refusal_reason is not None:
                    key = r.refusal_reason.value
                    by_reason[key] = by_reason.get(key, 0) + 1
                    # 这两个原因不是知识侧的，写文档解决不了，单独算出来
                    if key in ("missing_slot", "high_risk_unapproved"):
                        refusal_side += 1
            triage_counts: dict[str, int] = {}
            for r in rows:
                # 只统计"知识侧"的待审记录：缺槽位、高风险待批就算堆成山，
                # 也和"该不该写文档/调检索"无关，混进来只会让决策数字失真。
                if r.status != STATUS_PENDING or not is_knowledge_candidate(r):
                    continue
                hint = triage_hint(r, self.settings)
                triage_counts[hint] = triage_counts.get(hint, 0) + 1
            top = sorted(
                rows, key=lambda r: (-(r.ask_count or 0), r.created_at, r.record_id)
            )[:5]
            return {
                "total": len(rows),
                "pending": by_status[STATUS_PENDING],
                "approved": by_status[STATUS_APPROVED],
                "rejected": by_status[STATUS_REJECTED],
                "merged": by_status[STATUS_MERGED],
                "by_status": by_status,
                "by_source": by_source,
                "by_refusal_reason": by_reason,
                # 用户侧原因（缺槽位/高风险待批）计入 refused 但属于流程问题
                "refused_process_side": refusal_side,
                "triage": triage_counts,
                "top_questions": [
                    {"question": r.question, "ask_count": r.ask_count} for r in top
                ],
                "backfilled": len(self._kb_index),
                "errors": len(self.errors),
            }
        except Exception as exc:  # pragma: no cover
            self.errors.append(f"stats failed: {exc!r}")
            return {
                "total": 0, "pending": 0, "approved": 0, "rejected": 0, "merged": 0,
                "by_status": {}, "by_source": {}, "by_refusal_reason": {},
                "refused_process_side": 0, "triage": {}, "top_questions": [],
                "backfilled": 0, "errors": len(self.errors),
            }

    # ========================================================
    # 可选 JSONL 落盘（不依赖 SQLAlchemy —— storage 模块由别人实现）
    # ========================================================

    def _append_jsonl(self, op: str, rec: FlywheelRecord) -> None:
        """追加一行**状态快照**（不是事件）。

        每次改动后把该记录的完整最新状态追加一行，恢复时按 record_id 取最后
        一行。这样做的三个好处：
          * 回放不会重复计数 —— 事件流回放时 "add + merge" 会把 ask_count
            算两遍，快照日志天然幂等；
          * 崩在写一半只丢最后一行，上一行的状态还能用；
          * 一行的内容就是一条记录，`FlywheelRow` 的字段一一对得上，
            将来迁到数据库时这层可以整个删掉。

        落盘失败只记 error，不影响内存里的流程 —— 审核动作不能因为磁盘满就丢。
        """
        if not self.persist or self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(
                {"op": op, "record": rec.model_dump(mode="json")},
                ensure_ascii=False,
            )
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception as exc:
            self.errors.append(f"persist failed: {exc!r}")

    def load_jsonl(self, path: Path | str | None = None) -> int:
        """从 JSONL 恢复。按 record_id 取最后一行，返回恢复的记录条数。

        直接写入 `_rows` 而不是走 `record()`：`record()` 的语义是"线上又发生
        了一次提问"，会把 `ask_count` 再加一；而恢复要的是"还原当时的状态"，
        两者混在一起就会越重启、频次越高。

        坏行（写到一半、手改坏的）整行跳过，不让一行脏数据搞挂整个池子。
        """
        target = Path(path) if path else self.path
        if target is None or not Path(target).exists():
            return 0
        latest: dict[str, FlywheelRecord] = {}
        try:
            with Path(target).open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                        rec = FlywheelRecord.model_validate(payload["record"])
                    except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
                        self.errors.append(f"load_jsonl skipped line: {exc!r}")
                        continue
                    latest[rec.record_id] = rec  # 后写的覆盖先写的
        except Exception as exc:
            self.errors.append(f"load_jsonl failed: {exc!r}")
            return 0

        for rec in latest.values():
            self._rows[rec.record_id] = rec
            key = self._dedup_key(rec)
            if key and key not in self._index:
                self._index[key] = rec.record_id
        return len(latest)


#: 审核/回灌时找不到记录，用这个来源占位。取 `refused` 是因为它语义最弱，
#: 不会让下游误以为"有用户点踩"。
_UNKNOWN_SOURCE = FlywheelSource.REFUSED
