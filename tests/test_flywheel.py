"""数据飞轮的测试。

覆盖点对应 CONTRACT.md 2.G 和这一模块的四个设计决定：
来源分类、同题去重、频次排序、审核与回灌幂等。全部纯内存 + 纯 mock，
不依赖网络/DB/API Key，且同输入同输出（没有 random、没有真实时钟参与分支）。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mewhelp.domain import FlywheelSource, RefusalReason
from mewhelp.flywheel import (
    FlywheelStore,
    apply_review_to_kb,
    build_record,
    dedup_key,
    is_knowledge_gap,
    normalize_question,
    source_for,
    triage_hint,
)
from mewhelp.flywheel.feedback import (
    TRIAGE_BACKFILL_KB,
    TRIAGE_REVIEW_ANSWER,
    TRIAGE_TUNE_RETRIEVAL,
    TRIAGE_UNKNOWN,
)
from mewhelp.schemas import Evidence, FlywheelRecord, KbChunk, RetrievalResult

T0 = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


# ============================================================
# 构造工具
# ============================================================


def make_ev(chunk_id: str, score: float, text: str = "正文") -> Evidence:
    chunk = KbChunk(
        chunk_id=chunk_id,
        doc_id="d1",
        title=f"标题-{chunk_id}",
        breadcrumb=["退款政策", "生鲜类"],
        text=text,
        keywords=["退款"],
        char_len=len(text),
    )
    return Evidence(chunk=chunk, score=score, retriever="hybrid_rerank", rerank_score=score)


def make_retrieval(
    *,
    query: str = "生鲜能退吗",
    scores: tuple[float, ...] = (0.82, 0.61),
    dense_hits: int = 5,
    sparse_hits: int = 4,
    retrieved_ids: tuple[str, ...] = ("c1", "c2"),
) -> RetrievalResult:
    """造一个完整的检索快照 —— 中间态字段都填上，用来验证快照没被裁剪。"""
    evidences = [make_ev(f"c{i + 1}", s) for i, s in enumerate(scores)]
    return RetrievalResult(
        query=query,
        evidences=evidences,
        dense_hits=dense_hits,
        sparse_hits=sparse_hits,
        fused_hits=len(evidences),
        reranked_hits=len(evidences),
        retrieved_ids=list(retrieved_ids),
        strategy="hybrid_rerank",
        latency_ms=12.5,
    )


def make_record(
    question: str,
    *,
    session_id: str = "s1",
    source: FlywheelSource = FlywheelSource.REFUSED,
    reason: RefusalReason | None = RefusalReason.WEAK_EVIDENCE,
    retrieval: RetrievalResult | None = None,
    ask_count: int = 1,
    created_at: datetime | None = None,
    record_id: str | None = None,
) -> FlywheelRecord:
    return FlywheelRecord(
        record_id=record_id or f"tmp_{normalize_question(question)}",
        session_id=session_id,
        question=question,
        source=source,
        refusal_reason=reason,
        retrieval_snapshot=retrieval,
        ask_count=ask_count,
        created_at=created_at or T0,
    )


# ============================================================
# 1. 三个入口的来源分类
# ============================================================


@pytest.mark.parametrize(
    "reason, thumbs_down, expected",
    [
        # 检索证据弱被拒答 → refused
        (RefusalReason.WEAK_EVIDENCE, False, FlywheelSource.REFUSED),
        (RefusalReason.NO_EVIDENCE, False, FlywheelSource.REFUSED),
        (RefusalReason.LOW_COVERAGE, False, FlywheelSource.REFUSED),
        # 生成前自评没过 → self_eval
        (RefusalReason.SELF_EVAL_FAILED, False, FlywheelSource.SELF_EVAL),
        # 用户点踩 → thumbs_down
        (None, True, FlywheelSource.THUMBS_DOWN),
        (RefusalReason.WEAK_EVIDENCE, True, FlywheelSource.THUMBS_DOWN),
        # 点踩优先级最高：即使当时是自评没过，也按用户信号归口
        (RefusalReason.SELF_EVAL_FAILED, True, FlywheelSource.THUMBS_DOWN),
    ],
)
def test_source_for_classification(reason, thumbs_down, expected):
    assert source_for(reason, thumbs_down) == expected


def test_source_for_maps_all_refusal_reasons_to_three_entries():
    """6 个拒答原因全部有归口，不能漏 —— 漏一个就会静默丢记录。"""
    got = {source_for(r, False) for r in RefusalReason}
    assert got <= {FlywheelSource.REFUSED, FlywheelSource.SELF_EVAL}
    assert len(got) == 2


def test_build_record_derives_source_and_keeps_snapshot():
    snap = make_retrieval()
    rec = build_record(
        session_id="s1",
        question="生鲜能退吗",
        refusal_reason=RefusalReason.WEAK_EVIDENCE,
        retrieval=snap,
    )
    assert rec.source is FlywheelSource.REFUSED
    assert rec.retrieval_snapshot is not None
    assert len(rec.retrieval_snapshot.evidences) == 2

    down = build_record(session_id="s1", question="生鲜能退吗", thumbs_down=True)
    assert down.source is FlywheelSource.THUMBS_DOWN


def test_is_knowledge_gap_separates_process_problems():
    """缺槽位/高风险待批也会落成 refused，但它们不是"缺知识"。"""
    assert is_knowledge_gap(RefusalReason.NO_EVIDENCE) is True
    assert is_knowledge_gap(RefusalReason.MISSING_SLOT) is False
    assert is_knowledge_gap(RefusalReason.HIGH_RISK_UNAPPROVED) is False
    # 两者在入口分类上无法区分，这正是必须再看 refusal_reason 的原因
    assert source_for(RefusalReason.MISSING_SLOT, False) is FlywheelSource.REFUSED


# ============================================================
# 2. 同题去重 / ask_count
# ============================================================


def test_same_question_recorded_twice_merges_and_counts():
    store = FlywheelStore()
    r1 = store.record(make_record("运费怎么算？"))
    r2 = store.record(make_record("运费怎么算"))

    assert r1.record_id == r2.record_id
    assert r2.ask_count == 2
    assert len(store) == 1
    assert store.stats()["total"] == 1


def test_dedup_key_prefers_resolved_question():
    """半截话补全后要与完整问法合并 —— 否则飞轮统计不到真实频次。"""
    store = FlywheelStore()
    store.record(make_record("那它能退吗", session_id="s1"))
    merged = store.record(
        FlywheelRecord(
            record_id="tmp_x",
            session_id="s2",
            question="那它能退吗",
            resolved_question="生鲜商品能退货吗",
            source=FlywheelSource.REFUSED,
        )
    )
    # 第二次换了原文但 resolved 不同，因此是新的一条
    assert merged.ask_count == 1
    assert len(store) == 2

    same = store.record(
        FlywheelRecord(
            record_id="tmp_y",
            session_id="s3",
            question="这个可以退掉吗",
            resolved_question="那它能退吗",
            source=FlywheelSource.REFUSED,
        )
    )
    assert same.ask_count == 2
    assert len(store) == 2


def test_normalize_question_kills_case_punctuation_fullwidth():
    assert normalize_question("  运费 怎么算？！ ") == normalize_question("运费怎么算")
    assert normalize_question("ABC") == normalize_question("ａｂｃ")
    assert normalize_question(None) == ""


def test_dedup_key_is_the_storage_seam():
    """storage/models.py 的 FlywheelRow 有 question_key 唯一约束，写库那侧要
    自己算这个键 —— 两边必须是同一份规则，否则会出现"内存里合并了、写库冲突"。"""
    from mewhelp.storage.models import FlywheelRow

    assert "question_key" in FlywheelRow.__table__.columns
    assert dedup_key("运费怎么算？") == dedup_key("  运费怎么算 ")
    # resolved 优先：半截话补全后要和完整问法落成同一条
    assert dedup_key("那它能退吗", "生鲜能退吗") == dedup_key("生鲜能退吗")
    assert dedup_key("那它能退吗", "") == dedup_key("那它能退吗")
    # 内部去重用的就是这一个函数（同题两次落池 id 相同 → 必然同一键）
    store = FlywheelStore()
    a = store.record(make_record("运费怎么算？"))
    b = store.record(make_record("运费怎么算"))
    assert a.record_id == b.record_id
    assert dedup_key(a.question) == dedup_key(b.question)


def test_merge_does_not_overwrite_first_snapshot():
    """快照只补空不覆盖：第一次落池的现场才是审核要看的现场。"""
    store = FlywheelStore()
    first = make_retrieval(scores=(0.9,), retrieved_ids=("old1",))
    store.record(make_record("发票怎么开", retrieval=first))
    second = make_retrieval(scores=(0.1,), retrieved_ids=("new1",))
    merged = store.record(make_record("发票怎么开", retrieval=second))

    assert merged.ask_count == 2
    assert merged.retrieval_snapshot is not None
    assert merged.retrieval_snapshot.retrieved_ids == ["old1"]


def test_merge_escalates_source_to_thumbs_down():
    store = FlywheelStore()
    store.record(make_record("包邮吗", source=FlywheelSource.REFUSED))
    merged = store.record(make_record("包邮吗", source=FlywheelSource.THUMBS_DOWN))
    assert merged.source is FlywheelSource.THUMBS_DOWN


def test_repeat_ask_moves_reviewed_record_back_to_pending():
    """已回灌又被问到 → 说明那条答案没解决问题，打回待审。"""
    store = FlywheelStore()
    rec = store.record(make_record("会员怎么退", retrieval=make_retrieval()))
    store.review(rec.record_id, approved=True, reviewer="alice", answer="人工答案")
    again = store.record(make_record("会员怎么退"))
    assert again.status == "pending"
    assert again.answer is None
    assert again.ask_count == 2


# ============================================================
# 3. pending() 按 ask_count 降序
# ============================================================


def test_pending_sorted_by_ask_count_desc():
    store = FlywheelStore()
    # A 问 1 次、B 问 3 次、C 问 2 次；created_at 故意设成 A 最早，
    # 用来证明排序依据是频次而不是时间。
    store.record(make_record("A问题", created_at=T0))
    for _ in range(3):
        store.record(make_record("B问题", created_at=T0 + timedelta(minutes=1)))
    for _ in range(2):
        store.record(make_record("C问题", created_at=T0 + timedelta(minutes=2)))

    rows = store.pending()
    assert [r.question for r in rows] == ["B问题", "C问题", "A问题"]
    assert [r.ask_count for r in rows] == [3, 2, 1]


def test_pending_ties_are_deterministic():
    """并列时按 (created_at, record_id) 兜底，保证同数据两次运行同顺序。

    两次插入用**相反的**遍历顺序（模拟请求到达顺序不同），只要 created_at
    相同，待审队列就必须一致 —— 否则运营看到的队列顺序每次都不一样。
    """
    questions = ["问题甲", "问题乙", "问题丙"]
    store = FlywheelStore()
    for i, q in enumerate(questions):
        store.record(make_record(q, created_at=T0 + timedelta(seconds=i)))

    store2 = FlywheelStore()
    for i, q in enumerate(reversed(questions)):
        # reversed 后下标要换算回原来的秒数，created_at 才一一对应
        store2.record(
            make_record(q, created_at=T0 + timedelta(seconds=questions.index(q)))
        )

    first = [r.question for r in store.pending()]
    second = [r.question for r in store2.pending()]
    assert first == second == questions  # 同 created_at → 按时间升序

    # 连 created_at 也完全一样时，退化到 record_id 兜底，仍然确定
    store3 = FlywheelStore()
    for q in questions:
        store3.record(make_record(q, created_at=T0))
    for q in reversed(questions):
        store3.record(make_record(q, created_at=T0))
    third = [r.record_id for r in store3.pending()]
    assert third == sorted(third)
    assert set(third) == {r.record_id for r in store.pending()}


def test_pending_limit_and_excludes_reviewed():
    store = FlywheelStore()
    for i in range(5):
        store.record(make_record(f"q{i}", created_at=T0 + timedelta(seconds=i)))
    rows = store.pending()
    assert store.pending(limit=2) == rows[:2]
    assert len(rows) == 5

    store.review(rows[0].record_id, approved=False, reviewer="bob")
    assert len(store.pending()) == 4
    assert all(r.status == "pending" for r in store.pending())


# ============================================================
# 4. 审核
# ============================================================


def test_review_approved_fills_reviewer_and_reviewed_at():
    store = FlywheelStore()
    rec = store.record(make_record("保修多久", retrieval=make_retrieval()))
    out = store.review(rec.record_id, approved=True, reviewer="alice", answer="保修两年。")

    assert out.status == "approved"
    assert out.reviewer == "alice"
    assert out.reviewed_at is not None
    assert out.answer == "保修两年。"
    # 读回来的也要是新状态（不是只改了返回的副本）
    assert store.get(rec.record_id).status == "approved"


def test_review_rejected_marks_status():
    store = FlywheelStore()
    rec = store.record(make_record("这个是不是 bug", retrieval=make_retrieval()))
    out = store.review(rec.record_id, approved=False, reviewer="bob")
    assert out.status == "rejected"
    assert out.reviewer == "bob"
    assert store.stats()["rejected"] == 1


def test_review_unknown_record_does_not_raise():
    """异常不外泄：前端点错一次不能 500。"""
    store = FlywheelStore()
    out = store.review("fw_not_exist", approved=True, reviewer="alice")
    assert out.status == "pending"
    assert store.errors  # 但必须留痕，否则会变成静默失败


# ============================================================
# 5. 检索快照完整保留
# ============================================================


def test_retrieval_snapshot_preserved_intact():
    store = FlywheelStore()
    snap = make_retrieval(scores=(0.82, 0.61), dense_hits=7, sparse_hits=3)
    rec = store.record(make_record("生鲜能退吗", retrieval=snap))

    got = store.get(rec.record_id).retrieval_snapshot
    assert got is not None
    assert [e.chunk_id for e in got.evidences] == ["c1", "c2"]
    assert got.evidences[0].chunk.text == "正文"
    assert got.evidences[0].chunk.breadcrumb == ["退款政策", "生鲜类"]
    assert got.top_score == pytest.approx(0.82)
    # 中间态字段也要在，否则事后无法判断"没召回到"还是"没排上来"
    assert (got.dense_hits, got.sparse_hits) == (7, 3)
    assert got.retrieved_ids == ["c1", "c2"]
    assert got.strategy == "hybrid_rerank"

    # pending 返回的是副本，但快照内容一样要完整
    pending = store.pending()[0]
    assert [e.chunk_id for e in pending.retrieval_snapshot.evidences] == ["c1", "c2"]


def test_snapshot_survives_review_and_returned_copies_are_isolated():
    store = FlywheelStore()
    rec = store.record(make_record("尺码怎么选", retrieval=make_retrieval()))
    out = store.review(rec.record_id, approved=True, reviewer="alice", answer="看尺码表")
    assert out.retrieval_snapshot is not None
    assert len(out.retrieval_snapshot.evidences) == 2

    # 返回副本：外部改动不能污染池子里的现场
    out.retrieval_snapshot.evidences.clear()
    assert len(store.get(rec.record_id).retrieval_snapshot.evidences) == 2


# ============================================================
# 6. 分诊（快照的用处）
# ============================================================


def test_triage_uses_snapshot_to_split_document_vs_retrieval():
    store = FlywheelStore()
    # 一条证据都没召回 → 补文档
    no_ev = store.record(make_record("冷门问题", retrieval=make_retrieval(scores=())))
    assert triage_hint(no_ev) == TRIAGE_BACKFILL_KB
    # 召回了但分数低于阈值(0.35) → 调检索
    weak = store.record(make_record("弱召回问题", retrieval=make_retrieval(scores=(0.2, 0.1))))
    assert triage_hint(weak) == TRIAGE_TUNE_RETRIEVAL
    # 分数够却没通过 → 查生成与闸门
    strong = store.record(make_record("强召回问题", retrieval=make_retrieval(scores=(0.9,))))
    assert triage_hint(strong) == TRIAGE_REVIEW_ANSWER
    # 没有快照 → 判不了，这本身就是"必须存快照"的证据
    blind = store.record(make_record("没快照", retrieval=None))
    assert triage_hint(blind) == TRIAGE_UNKNOWN

    assert store.stats()["triage"] == {
        TRIAGE_BACKFILL_KB: 1,
        TRIAGE_TUNE_RETRIEVAL: 1,
        TRIAGE_REVIEW_ANSWER: 1,
        TRIAGE_UNKNOWN: 1,
    }


# ============================================================
# 7. 回灌：chunk_id 确定性
# ============================================================


def test_to_kb_chunk_id_is_deterministic():
    store = FlywheelStore()
    rec = store.record(make_record("生鲜能退吗", retrieval=make_retrieval()))
    store.review(rec.record_id, approved=True, reviewer="alice", answer="生鲜不支持七天无理由。")

    c1 = store.to_kb_chunk(rec.record_id)
    c2 = store.to_kb_chunk(store.get(rec.record_id))
    assert c1 is not None and c2 is not None
    assert c1.chunk_id == c2.chunk_id
    # chunk_id 只由问题文本决定，和 store 实例、record_id 都无关 ——
    # 这样并发/重放产生的不同 record_id 落到知识库里也是同一条。
    assert c1.chunk_id == FlywheelStore.kb_chunk_id("生鲜能退吗")
    assert c1.chunk_id == FlywheelStore.kb_chunk_id("生鲜能退吗？")  # 归一化后同题
    # 换一个 store 实例（模拟重启后换进程）重建同样的问题，id 必须一致
    other = FlywheelStore()
    rec2 = other.record(make_record("生鲜能退吗", retrieval=make_retrieval()))
    other.review(rec2.record_id, approved=True, reviewer="cd", answer="生鲜不支持七天无理由。")
    assert other.to_kb_chunk(rec2.record_id).chunk_id == c1.chunk_id
    # 未审核的记录即使手里拿着对象也不产出条目
    assert other.to_kb_chunk(rec) is None
    # 不同问题必须得到不同 id（碰撞会让两个问题的答案互相覆盖）
    assert FlywheelStore.kb_chunk_id("包邮吗") != c1.chunk_id


def test_repeat_review_does_not_produce_duplicate_chunks():
    store = FlywheelStore()
    rec = store.record(make_record("发票怎么开", retrieval=make_retrieval()))
    store.review(rec.record_id, approved=True, reviewer="alice", answer="在订单页申请。")
    first = store.to_kb_chunk(rec.record_id)
    # 同一问题被第二次问、又被审核一次
    store.record(make_record("发票怎么开，我要报销"))
    store.record(make_record("发票怎么开"))
    store.review(rec.record_id, approved=True, reviewer="bob", answer="在订单页申请电子发票。")
    second = store.to_kb_chunk(rec.record_id)
    assert first.chunk_id == second.chunk_id

    chunks = store.approved_chunks()
    assert len(chunks) == 1


def test_to_kb_chunk_only_for_approved():
    store = FlywheelStore()
    rec = store.record(make_record("未审核问题", retrieval=make_retrieval()))
    assert store.to_kb_chunk(rec.record_id) is None
    store.review(rec.record_id, approved=False, reviewer="bob")
    assert store.to_kb_chunk(rec.record_id) is None
    assert store.approved_chunks() == []


def test_apply_review_to_kb_writes_once():
    """回灌幂等：跑两次不会让向量库多一条。"""

    class FakeRetriever:
        def __init__(self):
            self.written: list[KbChunk] = []

        def index(self, chunks):
            self.written.extend(chunks)
            return len(chunks)

    store = FlywheelStore()
    retriever = FakeRetriever()
    rec = store.record(make_record("快递几天到", retrieval=make_retrieval()))
    store.review(rec.record_id, approved=True, reviewer="alice", answer="一般 3 天。")

    a = apply_review_to_kb(store, store.get(rec.record_id), retriever)
    b = apply_review_to_kb(store, store.get(rec.record_id), retriever)
    assert a is not None and a.chunk_id == b.chunk_id
    assert len(retriever.written) == 1
    assert store.stats()["backfilled"] == 1


# ============================================================
# 8. stats()
# ============================================================


def test_stats_counts_each_status():
    store = FlywheelStore()
    a = store.record(make_record("甲", source=FlywheelSource.REFUSED))
    b = store.record(make_record("乙", source=FlywheelSource.SELF_EVAL))
    c = store.record(make_record("丙", source=FlywheelSource.THUMBS_DOWN))
    d = store.record(make_record("丁"))
    store.record(make_record("甲"))  # 甲被问两次

    store.review(a.record_id, approved=True, reviewer="alice", answer="答案甲")
    store.review(b.record_id, approved=False, reviewer="bob")

    stats = store.stats()
    assert stats["total"] == 4
    assert stats["pending"] == 2  # 丙、丁
    assert stats["approved"] == 1
    assert stats["rejected"] == 1
    assert stats["merged"] == 0
    assert stats["by_status"] == {"pending": 2, "approved": 1, "rejected": 1, "merged": 0}
    assert stats["by_source"] == {"refused": 2, "self_eval": 1, "thumbs_down": 1}
    assert stats["top_questions"][0] == {"question": "甲", "ask_count": 2}
    assert c.status == "pending" and d.status == "pending"


def test_stats_isolates_process_side_refusals():
    """缺槽位/高风险待批计入 refused 来源，但不算"缺知识"。"""
    store = FlywheelStore()
    store.record(
        build_record(
            session_id="s1",
            question="帮我取消订单",
            refusal_reason=RefusalReason.MISSING_SLOT,
        )
    )
    snap = make_retrieval()
    store.record(
        build_record(
            session_id="s1",
            question="生鲜能退吗",
            refusal_reason=RefusalReason.WEAK_EVIDENCE,
            retrieval=snap,
        )
    )
    stats = store.stats()
    assert stats["by_refusal_reason"]["missing_slot"] == 1
    assert stats["refused_process_side"] == 1
    # 只有"生鲜能退吗"是知识侧问题，会被分诊
    assert stats["triage"] == {TRIAGE_REVIEW_ANSWER: 1}


# ============================================================
# 9. 可选 JSONL 落盘
# ============================================================


def test_jsonl_roundtrip_restores_ask_count_and_status(tmp_path):
    path = tmp_path / "flywheel.jsonl"
    store = FlywheelStore(path=path, persist=True)
    rec = store.record(make_record("包邮吗", retrieval=make_retrieval()))
    store.record(make_record("包邮吗"))
    store.record(make_record("能开发票吗"))
    store.review(rec.record_id, approved=True, reviewer="alice", answer="满 99 包邮。")

    assert path.exists()

    restored = FlywheelStore(path=path)
    loaded = restored.load_jsonl()
    # 落盘是状态快照日志：add/add/merge/add/review 共 5 行，按 record_id 取最后一条，
    # 所以恢复出来是 2 条记录（坏行才该被跳过）
    assert loaded == 2
    rows = {r.question: r for r in restored.all()}
    assert set(rows) == {"包邮吗", "能开发票吗"}
    assert rows["包邮吗"].ask_count == 2
    assert rows["包邮吗"].status == "approved"
    assert rows["包邮吗"].reviewer == "alice"
    assert rows["包邮吗"].answer == "满 99 包邮。"
    # 快照也要能挺过重启 —— 否则重启一次飞轮就退化成问题收集表
    assert len(rows["包邮吗"].retrieval_snapshot.evidences) == 2
    assert rows["能开发票吗"].status == "pending"
    # 再恢复一次不能把频次翻倍（快照日志幂等，事件流会在这里出错）
    assert restored.load_jsonl() == 2
    assert {r.question: r.ask_count for r in restored.all()} == {"包邮吗": 2, "能开发票吗": 1}


def test_load_jsonl_skips_dirty_lines(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"op":"add","record": "not-json\n', encoding="utf-8")
    store = FlywheelStore()
    assert store.load_jsonl(path) == 0
    assert store.errors  # 坏行留痕，但不炸


def test_persist_default_off_writes_nothing(tmp_path):
    store = FlywheelStore()
    store.record(make_record("不落盘"))
    assert not list(tmp_path.iterdir())
