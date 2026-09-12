"""RAG 评估体系。

这个模块存在的理由，是让"效果还不错"这句话变得不可说。

三个刻意的设计：

1. **评估集分桶**，不是一个总分。分桶才能回答"哪一类问题答不好" ——
   总分 0.9 但表格类全错，和总分 0.9 且各桶均匀，是完全不同的两件事。

2. **指标口径写死在这里**，不在报告里。Recall@5 的 @5 是 5，分母是标注的
   相关块总数 —— 换个实现算出来的数不一样，那就不是同一个指标了。

3. **消融对照**：dense / sparse / hybrid / hybrid_rerank 四种策略跑同一套题，
   差异才是"混合检索 + 重排到底值不值"的证据。
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, Field

#: 五个桶。分桶维度是"失败原因"，不是"话题" —— 这样每个桶的分数直接对应一个改进方向。
BUCKETS: dict[str, str] = {
    "fact_lookup": "事实查询：答案在单个片段里",
    "multi_condition": "多条件：需要跨片段拼条件，切分错了就会漏",
    "multi_turn": "多轮指代：必须先消解才能检索",
    "table_lookup": "表格结构化：需要保住表头才有意义",
    "no_answer": "知识库无答案：应当正确拒答，不是硬编",
}

#: 四种检索策略（消融实验的列）
STRATEGIES: tuple[str, ...] = ("dense", "sparse", "hybrid", "hybrid_rerank")


class GoldenItem(BaseModel):
    """一道评估题。"""

    qid: str
    bucket: str
    question: str
    expected_intent: str | None = None
    #: 标注的相关块。no_answer 桶里为空 —— 空正是它要考的东西。
    expected_chunk_ids: list[str] = Field(default_factory=list)
    golden_answer: str = ""
    #: 多轮题的历史（user/assistant 交替）
    history: list[dict[str, str]] = Field(default_factory=list)

    @property
    def is_no_answer(self) -> bool:
        return self.bucket == "no_answer" or not self.expected_chunk_ids


# ============================================================
# 指标
# ============================================================


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Recall@k = 前 k 个里命中的相关块数 / 标注的相关块总数。

    分母是"应有"的块数，不是 len(retrieved)。这是最容易写错的地方：
    用后者的话，只召回 1 条且命中的系统会拿到满分。
    """
    if not relevant:
        return 0.0
    hit = len(set(retrieved[:k]) & set(relevant))
    return hit / len(set(relevant))


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    top = list(retrieved[:k])
    if not top:
        return 0.0
    return len(set(top) & set(relevant)) / len(top)


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """RR 的倒数：第一个命中排在第几。衡量"答案是不是排在最前面"。"""
    rel = set(relevant)
    for i, cid in enumerate(retrieved, 1):
        if cid in rel:
            return 1.0 / i
    return 0.0


def hit_rate(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """前 k 里有没有命中（二值）。Recall 会被多块题稀释，HitRate 不会。"""
    if not relevant:
        return 0.0
    return 1.0 if set(retrieved[:k]) & set(relevant) else 0.0


class BucketScore(BaseModel):
    bucket: str
    n: int
    recall_at_5: float = 0.0
    mrr: float = 0.0
    hit_rate_at_5: float = 0.0
    precision_at_5: float = 0.0
    #: no_answer 桶不看 recall，看的是"该拒答时拒没拒"
    refusal_accuracy: float | None = None


class EvalReport(BaseModel):
    strategy: str
    #: 全局（no_answer 桶不计入召回类指标 —— 它没有相关块）
    recall_at_5: float = 0.0
    mrr: float = 0.0
    hit_rate_at_5: float = 0.0
    precision_at_5: float = 0.0
    n_scored: int = 0
    #: 分桶
    buckets: list[BucketScore] = Field(default_factory=list)
    #: 拒答桶的单独成绩
    refusal_accuracy: float = 0.0
    n_refusal: int = 0
    #: 每题的明细，出问题时能直接翻到那一题
    per_item: list[dict[str, Any]] = Field(default_factory=list)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    errors: list[str] = Field(default_factory=list)

    def bucket(self, name: str) -> BucketScore | None:
        return next((b for b in self.buckets if b.bucket == name), None)


def evaluate_strategy(
    items: Sequence[GoldenItem],
    retrieve_fn,
    *,
    strategy: str,
    k: int = 5,
) -> EvalReport:
    """跑一种策略，返回完整报告。

    `retrieve_fn(question, history) -> RetrievalResult` —— 传函数进来而不是
    传 Pipeline，是为了让评估能在不启动向量库的情况下用假实现跑通。
    """
    per_item: list[dict[str, Any]] = []
    errors: list[str] = []
    latencies: list[float] = []

    for it in items:
        try:
            res = retrieve_fn(it.question, it.history)
        except Exception as exc:  # noqa: BLE001 — 评估不该因为一题崩掉整轮
            errors.append(f"{it.qid}: {exc}")
            per_item.append({"qid": it.qid, "bucket": it.bucket, "error": str(exc)})
            continue

        retrieved = list(res.retrieved_ids)
        latencies.append(res.latency_ms)

        if it.is_no_answer:
            # 拒答题：没有相关块，考的是"检索侧有没有给出高置信证据"。
            # 这里用"最高分是否低于阈值"作为拒答代理 —— 精确的拒答判定在闸门层，
            # 这里只评估检索侧给了多少底气。
            top = res.top_score
            refused = top < 0.35
            per_item.append(
                {
                    "qid": it.qid,
                    "bucket": it.bucket,
                    "top_score": round(top, 4),
                    "refused": refused,
                    "retrieved": retrieved[:k],
                }
            )
            continue

        per_item.append(
            {
                "qid": it.qid,
                "bucket": it.bucket,
                "recall_at_5": round(recall_at_k(retrieved, it.expected_chunk_ids, k), 4),
                "mrr": round(reciprocal_rank(retrieved, it.expected_chunk_ids), 4),
                "hit_rate_at_5": round(hit_rate(retrieved, it.expected_chunk_ids, k), 4),
                "precision_at_5": round(precision_at_k(retrieved, it.expected_chunk_ids, k), 4),
                "retrieved": retrieved[:k],
                "expected": it.expected_chunk_ids,
            }
        )

    # ---- 汇总：全局只统计有相关块的题 ----
    scored = [r for r in per_item if not r.get("error") and "recall_at_5" in r]
    refusal_rows = [r for r in per_item if r.get("bucket") == "no_answer" and "refused" in r]

    def _mean(rows, key) -> float:
        vals = [r[key] for r in rows if key in r]
        return round(statistics.fmean(vals), 4) if vals else 0.0

    # ---- 汇总：分桶 ----
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in scored:
        by_bucket[r["bucket"]].append(r)

    buckets: list[BucketScore] = []
    for name in BUCKETS:
        rows = by_bucket.get(name, [])
        if name == "no_answer":
            ref = [r for r in refusal_rows if r["bucket"] == name]
            if ref:
                buckets.append(
                    BucketScore(
                        bucket=name,
                        n=len(ref),
                        refusal_accuracy=round(statistics.fmean(1.0 if r["refused"] else 0.0 for r in ref), 4),
                    )
                )
            else:
                buckets.append(BucketScore(bucket=name, n=0, refusal_accuracy=None))
            continue
        if not rows:
            buckets.append(BucketScore(bucket=name, n=0))
            continue
        buckets.append(
            BucketScore(
                bucket=name,
                n=len(rows),
                recall_at_5=_mean(rows, "recall_at_5"),
                mrr=_mean(rows, "mrr"),
                hit_rate_at_5=_mean(rows, "hit_rate_at_5"),
                precision_at_5=_mean(rows, "precision_at_5"),
            )
        )

    refused_vals = [1.0 if r["refused"] else 0.0 for r in refusal_rows]
    return EvalReport(
        strategy=strategy,
        recall_at_5=_mean(scored, "recall_at_5"),
        mrr=_mean(scored, "mrr"),
        hit_rate_at_5=_mean(scored, "hit_rate_at_5"),
        precision_at_5=_mean(scored, "precision_at_5"),
        n_scored=len(scored),
        buckets=buckets,
        refusal_accuracy=round(statistics.fmean(refused_vals), 4) if refused_vals else 0.0,
        n_refusal=len(refusal_rows),
        per_item=per_item,
        latency_p50_ms=round(statistics.median(latencies), 2) if latencies else 0.0,
        latency_p95_ms=round(_percentile(latencies, 95), 2) if latencies else 0.0,
        errors=errors,
    )


def _percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(round((p / 100) * len(s) + 0.5)) - 1))
    return s[idx]


# ============================================================
# 加载 / 保存
# ============================================================


def load_golden(path: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                items.append(GoldenItem(**json.loads(line)))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"{path}:{lineno} 解析失败: {exc}") from exc
    return items


def save_report(report: EvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def bucket_summary(items: Sequence[GoldenItem]) -> dict[str, int]:
    counts: dict[str, int] = {b: 0 for b in BUCKETS}
    for it in items:
        counts[it.bucket] = counts.get(it.bucket, 0) + 1
    return counts


def render_ablation_table(reports: Sequence[EvalReport]) -> str:
    """把四种策略渲染成一张对照表 —— 这是项目里最该被截图的那张图。"""
    head = "| 策略 | Recall@5 | MRR | HitRate@5 | Precision@5 | P50(ms) | 拒答准确率 |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    rows = [
        f"| {r.strategy} | {r.recall_at_5:.4f} | {r.mrr:.4f} | {r.hit_rate_at_5:.4f} "
        f"| {r.precision_at_5:.4f} | {r.latency_p50_ms:.1f} | {r.refusal_accuracy:.4f} |"
        for r in reports
    ]
    return "\n".join([head, sep, *rows])


def render_bucket_table(report: EvalReport) -> str:
    head = "| 桶 | 题数 | Recall@5 | MRR | HitRate@5 | 拒答准确率 |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: |"
    rows = []
    for b in report.buckets:
        # 两个桶各看各的指标：召回桶不看拒答率，拒答桶不看召回。
        # 同一格里混着放两种口径，表格就没法读了。
        if b.bucket == "no_answer":
            recall = mrr = hit = "—"
            refusal = f"{b.refusal_accuracy:.4f}" if b.refusal_accuracy is not None else "—"
        else:
            recall, mrr, hit = f"{b.recall_at_5:.4f}", f"{b.mrr:.4f}", f"{b.hit_rate_at_5:.4f}"
            refusal = "—"
        rows.append(f"| {b.bucket} | {b.n} | {recall} | {mrr} | {hit} | {refusal} |")
    return "\n".join([head, sep, *rows])
