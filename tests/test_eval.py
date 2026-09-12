"""评估体系测试。

重点验证三件事：
  1. 指标口径正确（Recall 的分母是"应有"不是"实际返回"）
  2. 分桶真的分开了（表格类全错、总分还行，这种情况必须能被看出来）
  3. 拒答桶走的是另一套判定（它没有相关块，硬算 recall 是错的）
"""

from __future__ import annotations

import json

import pytest

from mewhelp.eval import (
    BUCKETS,
    EvalReport,
    GoldenItem,
    bucket_summary,
    evaluate_strategy,
    hit_rate,
    load_golden,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    render_ablation_table,
    render_bucket_table,
    save_report,
)
from mewhelp.schemas import Evidence, KbChunk, RetrievalResult


# ============================================================
# 指标口径
# ============================================================


def test_recall_denominator_is_relevant_not_retrieved():
    """Recall 的分母必须是"标注的相关块总数"。

    写成 len(retrieved) 的话，只返回 1 条且命中的系统会拿满分 —— 这是
    实现这个指标时最常见的错误，所以专门钉一条测试。
    """
    # 应召回 4 块，只召回了 1 块 -> recall = 0.25，不是 1.0
    assert recall_at_k(["a"], ["a", "b", "c", "d"], 5) == pytest.approx(0.25)
    # 前排命中 2 块
    assert recall_at_k(["a", "b", "x", "y"], ["a", "b", "c", "d"], 5) == pytest.approx(0.5)


def test_recall_respects_k_cutoff():
    """排在第 6 的命中的不算 Recall@5 —— 否则 @5 这个记号没有意义。"""
    assert recall_at_k(["x", "y", "z", "w", "v", "a"], ["a"], 5) == 0.0
    assert recall_at_k(["a", "y", "z", "w", "v"], ["a"], 5) == 1.0


def test_recall_empty_relevant_is_zero():
    assert recall_at_k(["a"], [], 5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b"], ["a"]) == 1.0
    assert reciprocal_rank(["x", "a"], ["a"]) == 0.5
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_hit_rate_is_binary_unlike_recall():
    """Recall 会被多块题稀释，HitRate 不会 —— 两个都要有。"""
    # 应召回 4 块，命中 1 块
    assert recall_at_k(["a"], ["a", "b", "c", "d"], 5) == pytest.approx(0.25)
    assert hit_rate(["a"], ["a", "b", "c", "d"], 5) == 1.0


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c", "d"], ["a", "b"], 5) == pytest.approx(0.5)
    assert precision_at_k([], ["a"], 5) == 0.0


# ============================================================
# 分桶汇总
# ============================================================


def _res(chunk_ids: list[str], scores: list[float] | None = None, latency: float = 10.0):
    scores = scores or [0.9 - i * 0.1 for i in range(len(chunk_ids))]
    return RetrievalResult(
        query="q",
        evidences=[
            Evidence(
                chunk=KbChunk(chunk_id=cid, doc_id="d", title="t", text="x"), score=s
            )
            for cid, s in zip(chunk_ids, scores)
        ],
        retrieved_ids=list(chunk_ids),
        latency_ms=latency,
    )


ITEMS = [
    GoldenItem(qid="F1", bucket="fact_lookup", question="运费怎么算", expected_chunk_ids=["c1"]),
    GoldenItem(qid="F2", bucket="fact_lookup", question="怎么开发票", expected_chunk_ids=["c2"]),
    GoldenItem(qid="T1", bucket="table_lookup", question="尺码对照", expected_chunk_ids=["t1"]),
    GoldenItem(qid="T2", bucket="table_lookup", question="退款比例表", expected_chunk_ids=["t2"]),
    GoldenItem(qid="N1", bucket="no_answer", question="你们卖火箭吗", expected_chunk_ids=[]),
    GoldenItem(qid="N2", bucket="no_answer", question="外星能发货吗", expected_chunk_ids=[]),
]


def test_bucket_isolation_surfaces_localized_failure():
    """核心场景：总分还行，但表格桶全军覆没 —— 这个必须在分桶里看得出来。

    这正是分桶存在的理由。只看总分会得出"效果还不错"的错误结论。
    """
    hits = {"F1": ["c1"], "F2": ["c2"], "T1": ["wrong"], "T2": ["wrong2"]}

    def retrieve(q, history):
        qid = {"运费怎么算": "F1", "怎么开发票": "F2", "尺码对照": "T1", "退款比例表": "T2"}.get(q)
        if qid is None:
            return _res([], [], latency=5.0)
        return _res(hits[qid], latency=5.0)

    rep = evaluate_strategy(ITEMS, retrieve, strategy="hybrid_rerank")

    assert rep.bucket("fact_lookup").recall_at_5 == pytest.approx(1.0)
    assert rep.bucket("table_lookup").recall_at_5 == pytest.approx(0.0)
    # 总分被表格桶拉低，但没到 0 —— 只看总分就会漏掉"表格全错"
    assert 0.4 < rep.recall_at_5 < 0.6
    assert rep.n_scored == 4  # no_answer 桶不计入召回类指标


def test_no_answer_bucket_uses_refusal_not_recall():
    """拒答桶没有相关块，硬算 recall 会得到 0（错误地惩罚正确拒答的系统）。

    它走的是另一套判定：检索侧最高分够不够低。
    """

    def retrieve(q, history):
        if q in ("你们卖火箭吗", "外星能发货吗"):
            return _res([], [], latency=3.0)  # 什么都没召回 -> 应该拒答
        return _res(["c1"], latency=3.0)

    rep = evaluate_strategy(ITEMS, retrieve, strategy="hybrid_rerank")
    assert rep.n_refusal == 2
    assert rep.refusal_accuracy == pytest.approx(1.0)
    assert rep.bucket("no_answer").refusal_accuracy == pytest.approx(1.0)
    # 拒答题的 recall 字段不参与全局统计
    assert rep.bucket("no_answer").recall_at_5 == 0.0


def test_low_score_evidence_counts_as_refusal():
    """召回了但分数很低 -> 检索侧没有底气 -> 应该算拒答成功。"""

    def retrieve(q, history):
        if q in ("你们卖火箭吗", "外星能发货吗"):
            return _res(["junk"], scores=[0.05], latency=3.0)
        return _res(["c1"], latency=3.0)

    rep = evaluate_strategy(ITEMS, retrieve, strategy="hybrid")
    assert rep.refusal_accuracy == pytest.approx(1.0)


def test_retrieve_error_does_not_kill_the_run():
    """一题出错不该让整轮评估崩掉 —— 否则一节烂数据就废掉一次全量实验。"""

    def retrieve(q, history):
        if q == "尺码对照":
            raise RuntimeError("向量库超时")
        return _res(["c1", "c2", "t1", "t2"], latency=1.0)

    rep = evaluate_strategy(ITEMS, retrieve, strategy="dense")
    assert any("T1" in e for e in rep.errors)
    assert rep.n_scored == 3  # 剩下三题照常统计


def test_latency_percentiles():
    times = {f"q{i}": 10.0 * i for i in range(1, 11)}
    items = [
        GoldenItem(qid=f"Q{i}", bucket="fact_lookup", question=f"q{i}", expected_chunk_ids=["c"])
        for i in range(1, 11)
    ]

    def retrieve(q, history):
        return _res(["c"], latency=times[q])

    rep = evaluate_strategy(items, retrieve, strategy="hybrid")
    assert rep.latency_p50_ms == pytest.approx(55.0, abs=6)
    assert rep.latency_p95_ms >= 90.0


def test_ablation_table_renders_all_strategies():
    """四种策略的对照表 —— 项目里最该被截图的那张。"""

    def retrieve(q, history):
        return _res(["c1"], latency=4.0)

    reps = [
        evaluate_strategy(ITEMS, retrieve, strategy=s)
        for s in ("dense", "sparse", "hybrid", "hybrid_rerank")
    ]
    table = render_ablation_table(reps)
    for s in ("dense", "sparse", "hybrid", "hybrid_rerank"):
        assert s in table
    assert table.count("\n") == 5  # 表头 + 分隔 + 4 行
    assert "Recall@5" in table


def test_bucket_table_marks_refusal_column():
    def retrieve(q, history):
        return _res(["c1"], latency=1.0)

    rep = evaluate_strategy(ITEMS, retrieve, strategy="hybrid")
    table = render_bucket_table(rep)
    assert "no_answer" in table
    assert "拒答准确率" in table


# ============================================================
# 加载 / 保存 / 分桶覆盖
# ============================================================


def test_load_golden_roundtrip(tmp_path):
    p = tmp_path / "g.jsonl"
    rows = [
        {"qid": "Q1", "bucket": "fact_lookup", "question": "运费", "expected_chunk_ids": ["c1"]},
        {"qid": "Q2", "bucket": "no_answer", "question": "火箭", "expected_chunk_ids": []},
    ]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    items = load_golden(p)
    assert len(items) == 2
    assert items[1].is_no_answer is True
    assert items[0].is_no_answer is False


def test_load_golden_reports_line_number_on_bad_json(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"qid":"Q1","bucket":"fact_lookup","question":"x"}\n{not json}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="2"):
        load_golden(p)


def test_save_report_roundtrip(tmp_path):
    def retrieve(q, history):
        return _res(["c1"], latency=1.0)

    rep = evaluate_strategy(ITEMS, retrieve, strategy="hybrid")
    out = tmp_path / "reports" / "hybrid.json"
    save_report(rep, out)
    back = json.loads(out.read_text(encoding="utf-8"))
    assert back["strategy"] == "hybrid"
    assert len(back["buckets"]) == len(BUCKETS)
    assert back["per_item"]


def test_bucket_summary_counts_every_bucket():
    counts = bucket_summary(ITEMS)
    assert set(counts) == set(BUCKETS)
    assert counts["fact_lookup"] == 2
    assert counts["table_lookup"] == 2
    assert counts["no_answer"] == 2
    assert counts["multi_turn"] == 0
