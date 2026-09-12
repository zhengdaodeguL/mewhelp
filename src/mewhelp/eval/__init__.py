"""RAG 评估体系：评估集 + 指标口径 + 消融对照。

指标定义在 metrics.py，这里是包的入口。
关键是让"效果还不错"这句话变得不可说 —— 每个结论都要有口径和数字。
"""

from .metrics import (
    BUCKETS,
    STRATEGIES,
    BucketScore,
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

__all__ = [
    "BUCKETS",
    "STRATEGIES",
    "BucketScore",
    "EvalReport",
    "GoldenItem",
    "bucket_summary",
    "evaluate_strategy",
    "hit_rate",
    "load_golden",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "render_ablation_table",
    "render_bucket_table",
    "save_report",
]
