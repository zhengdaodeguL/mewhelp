"""消融实验：同一套 300 题，四种检索策略各跑一遍。

这个脚本要回答的问题只有一个：

    "混合检索 + 重排"这一堆工程投入，到底比单路稠密 / 单路 BM25 强多少？

所以四遍跑的必须是**同一批题、同一个检索器、同一个知识库**，只换 strategy。
任何一项不同，表里的差就不是策略的差。

## 多轮题为什么必须在这里做指代消解

`golden.jsonl` 里的 multi_turn 题，question 是半截话（"那它签收后多久内还能退？"），
真正的限定条件在 `history` 里（"我上次买的那台手机"）。**如果直接把 question
丢给检索，这个桶测的就不是检索能力，而是"半截话能不能撞上词"** —— 分数会
低得毫无意义，而且低的原因也不是策略差异，是指代信息根本没进查询。

所以这里按知识库线上链路的同样顺序先跑 `CoreferenceResolver`：
    question + history → resolved → pipeline.retrieve(resolved, strategy)
消解是**四种策略共用**的一步，不进任何一路，这样消融对照才干净。

（脚本末尾附了一份"消解 / 不消解"的对照，用来量化这一步到底值多少分。）

用法：
    cd E:/临时/projects/mewhelp
    PYTHONPATH=src PYTHONIOENCODING=utf-8 python eval/run_eval.py
    PYTHONPATH=src PYTHONIOENCODING=utf-8 python eval/run_eval.py --no-diagnostic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mewhelp.config import get_settings  # noqa: E402
from mewhelp.eval import (  # noqa: E402
    BUCKETS,
    EvalReport,
    GoldenItem,
    bucket_summary,
    evaluate_strategy,
    load_golden,
    render_ablation_table,
    render_bucket_table,
    save_report,
)
from mewhelp.nlu.coreference import CoreferenceResolver  # noqa: E402
from mewhelp.retrieval.pipeline import build_pipeline  # noqa: E402
from mewhelp.schemas import Turn  # noqa: E402

GOLDEN = ROOT / "data" / "eval" / "golden.jsonl"
KB_DIR = ROOT / "data" / "knowledge"
ARTIFACTS = ROOT / "eval" / "artifacts"

STRATEGIES: tuple[str, ...] = ("dense", "sparse", "hybrid", "hybrid_rerank")

#: 与 metrics.evaluate_strategy 里拒答代理判据保持同一个数（写在这里是为了让报告
#: 能从同一处引用它，而不是在正文里再抄一遍 0.35）
REFUSAL_THRESHOLD = 0.35


def make_retrieve_fn(pipeline, resolver, strategy: str, *, resolve: bool = True):
    """构造 `retrieve_fn(question, history) -> RetrievalResult`。

    契约要求的是"检索函数"，所以指代消解被包在函数体里而不是外挂在评估循环上 ——
    评估模块因此完全不需要知道 NLU 的存在，四种策略也拿到完全一致的前处理。
    """

    def retrieve(question: str, history: list[dict[str, str]]):
        query = question
        if resolve and history:
            # dict → Turn：golden.jsonl 里 history 是裸 dict（可读、可 diff），
            # 而解析器按 Turn 契约工作。转换只做一次，放在这里。
            turns = [Turn(role=str(t.get("role", "user")), content=str(t.get("content", ""))) for t in history]
            query = resolver.resolve(question, turns).resolved
        return pipeline.retrieve(query, strategy=strategy)

    return retrieve


def run_all(items: list[GoldenItem], pipeline, resolver, strategies=STRATEGIES) -> list[EvalReport]:
    reports: list[EvalReport] = []
    for strategy in strategies:
        fn = make_retrieve_fn(pipeline, resolver, strategy)
        rep = evaluate_strategy(items, fn, strategy=strategy)
        reports.append(rep)
    return reports


def run_multi_turn_diagnostic(items: list[GoldenItem], pipeline, resolver) -> dict:
    """量化"指代消解"这一步在 multi_turn 桶上的增量。

    只跑默认线上策略（hybrid_rerank）。两个数字之差就是消解这一步的贡献 ——
    这也是"multi_turn 桶必须消解"这句话唯一的证据来源，不测就不该写进报告。
    """
    sub = [it for it in items if it.bucket == "multi_turn"]
    with_resolve = evaluate_strategy(
        sub, make_retrieve_fn(pipeline, resolver, "hybrid_rerank", resolve=True),
        strategy="hybrid_rerank_resolved",
    )
    without = evaluate_strategy(
        sub, make_retrieve_fn(pipeline, resolver, "hybrid_rerank", resolve=False),
        strategy="hybrid_rerank_raw",
    )
    return {
        "n": len(sub),
        "with_resolution": {
            "recall_at_5": with_resolve.recall_at_5,
            "mrr": with_resolve.mrr,
            "hit_rate_at_5": with_resolve.hit_rate_at_5,
            "precision_at_5": with_resolve.precision_at_5,
        },
        "without_resolution": {
            "recall_at_5": without.recall_at_5,
            "mrr": without.mrr,
            "hit_rate_at_5": without.hit_rate_at_5,
            "precision_at_5": without.precision_at_5,
        },
        "delta_recall_at_5": round(with_resolve.recall_at_5 - without.recall_at_5, 4),
        "delta_mrr": round(with_resolve.mrr - without.mrr, 4),
    }


def render_markdown(reports: list[EvalReport], diagnostic: dict | None, items: list[GoldenItem]) -> str:
    counts = bucket_summary(items)
    lines = [
        "# 检索消融实验",
        "",
        f"评估集：`data/eval/golden.jsonl`，共 {len(items)} 题，五桶各 "
        f"{counts['fact_lookup']} 题。知识库：`data/knowledge/` 6 篇文档，切分后 70 个 chunk。",
        "",
        "四种策略跑的是同一批题、同一个检索器（`MockEmbedder` + `BM25` + `MockReranker`，",
        "全离线、确定性）。多轮题在检索前统一经 `CoreferenceResolver` 做指代消解，",
        "该步骤为四种策略共用，不计入任何一路。",
        "",
        "## 总表",
        "",
        render_ablation_table(reports),
        "",
        "列说明：Recall@5 的分母是标注的相关块总数（多条件题有 2～3 个相关块，",
        "只命中一个就是 0.5）；拒答准确率只对 `no_answer` 桶有意义，判据是检索侧最高分 "
        f"< {REFUSAL_THRESHOLD}。",
        "",
    ]
    for rep in reports:
        lines += [
            f"## 分桶明细：{rep.strategy}",
            "",
            f"（有相关块的题 {rep.n_scored} 道；拒答题 {rep.n_refusal} 道；"
            f"P50 {rep.latency_p50_ms:.1f} ms，P95 {rep.latency_p95_ms:.1f} ms）",
            "",
            render_bucket_table(rep),
            "",
        ]
    if diagnostic:
        d = diagnostic
        lines += [
            "## 多轮桶：指代消解值多少分",
            "",
            "| 口径 | Recall@5 | MRR | HitRate@5 | Precision@5 |",
            "| --- | ---: | ---: | ---: | ---: |",
            "| 消解后检索 | {r:.4f} | {m:.4f} | {h:.4f} | {p:.4f} |".format(
                r=d["with_resolution"]["recall_at_5"], m=d["with_resolution"]["mrr"],
                h=d["with_resolution"]["hit_rate_at_5"], p=d["with_resolution"]["precision_at_5"],
            ),
            "| 直接用半截话检索 | {r:.4f} | {m:.4f} | {h:.4f} | {p:.4f} |".format(
                r=d["without_resolution"]["recall_at_5"], m=d["without_resolution"]["mrr"],
                h=d["without_resolution"]["hit_rate_at_5"], p=d["without_resolution"]["precision_at_5"],
            ),
            "",
            f"差值：Recall@5 {d['delta_recall_at_5']:+.4f}，MRR {d['delta_mrr']:+.4f}"
            f"（n={d['n']}，策略 hybrid_rerank）。",
            "",
        ]
    lines += ["## 怎么读这张表（口径限制）", ""] + [f"{c}\n" for c in CAVEATS]
    return "\n".join(lines) + "\n"


#: 这张表**不能**被读成什么。写在脚本里而不是报告里，是为了让任何一个重跑这个
#: 脚本的人（包括半年后的我）都先看到限制，再去引用那几个小数。
CAVEATS = [
    "1. **不要把绝对分当成线上效果**。知识库只有 70 个 chunk，`fusion_top_k=10` "
    "的候选池已经覆盖全库的 1/7，Recall@5 天然偏高；换成上千块的库，"
    "同一套题的数字会明显下降。",
    "2. **dense 这一路是被 mock 拖累的**。`MockEmbedder` 是确定性哈希词袋，"
    "没有真正的语义泛化能力，所以 dense 输给 BM25 只能说明"
    "「在这个 mock 环境下词面匹配更强」，不能说明真实向量模型也这样。",
    "3. **拒答准确率只在 hybrid_rerank 上可读**。dense / hybrid 的分数是 RRF 的"
    "相对分（每条除以本查询最高分），排第一的那条恒等于 1.0，于是 `top_score` "
    "永远大于阈值 0.35，拒答判据在这两条策略上恒判「不拒答」。sparse 能拿到 "
    "0.98 是因为部分拒答题在 BM25 上真的一条都匹配不到（返回空 → 最高分 0）。"
    "**这是指标口径与打分口径不匹配，不是检索质量问题**："
    "`evaluate_strategy` 里的 `top < 0.35` 代理判据只对「绝对分」策略成立。",
    "4. **hybrid_rerank 的绝对分偏乐观**。评估集生成时做过一次标定"
    "（`eval/build_golden.py --check` 要求每条标注在 hybrid_rerank 下至少有一个"
    "相关块落在前 5），所以它是唯一被对齐过的策略。策略之间的**相对**差异不受影响 "
    "—— 同一批题对四条路是同等难度的。",
    "5. **Precision@5 低不是检索差**。240 道召回题里有 99 道标了 2～3 个相关块，"
    "其余只标 1 个，5 个返回位置里必然有 4 个「未标注」——"
    "分母决定的上限。读这张表时看 Recall 和 MRR，别读 Precision 的绝对值。",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-diagnostic", action="store_true", help="跳过消解对照（省一轮检索）")
    args = ap.parse_args()

    items = load_golden(GOLDEN)
    print(f"评估集 {len(items)} 题：{bucket_summary(items)}")

    settings = get_settings()
    pipeline = build_pipeline(settings)
    n_chunks = pipeline.load_from_dir(KB_DIR)
    print(f"知识库已建索引：{n_chunks} chunks；{pipeline.stats()}")

    resolver = CoreferenceResolver(settings=settings)
    reports = run_all(items, pipeline, resolver)

    table = render_ablation_table(reports)
    print("\n==== 四策略对照 ====")
    print(table)
    for rep in reports:
        print(f"\n---- {rep.strategy} 分桶 ----")
        print(render_bucket_table(rep))
        if rep.errors:
            print(f"  errors: {rep.errors}")

    diagnostic = None if args.no_diagnostic else run_multi_turn_diagnostic(items, pipeline, resolver)
    if diagnostic:
        print("\n==== multi_turn 桶：指代消解的增量 ====")
        print(f"  消解后   : {diagnostic['with_resolution']}")
        print(f"  半截话直搜: {diagnostic['without_resolution']}")
        print(f"  增量 Recall@5 {diagnostic['delta_recall_at_5']:+.4f} / MRR {diagnostic['delta_mrr']:+.4f}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    # 每份报告单独存一份，便于定位"哪个策略的哪道题错了"
    for rep in reports:
        save_report(rep, ARTIFACTS / f"{rep.strategy}.json")
    # 汇总一份：四份报告 + 复现实验所需的元信息
    (ARTIFACTS / "ablation.json").write_text(
        json.dumps(
            {
                "golden": str(GOLDEN.relative_to(ROOT)).replace("\\", "/"),
                "knowledge_dir": str(KB_DIR.relative_to(ROOT)).replace("\\", "/"),
                "n_items": len(items),
                "n_chunks": n_chunks,
                "bucket_counts": bucket_summary(items),
                "refusal_threshold": REFUSAL_THRESHOLD,
                "embedder": pipeline.stats()["embedder"],
                "reranker": pipeline.stats()["reranker"],
                "store": pipeline.stats()["store"],
                "multi_turn_diagnostic": diagnostic,
                "reports": [r.model_dump() for r in reports],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (ARTIFACTS / "ablation.md").write_text(render_markdown(reports, diagnostic, items), encoding="utf-8")

    # CONTRACT.md 的 K 节把 data/eval/ablation.json 声明为"脚本产出的对照结果"，
    # 所以额外往那里也放一份。两份内容完全一致，由本脚本一次性写出，不存在
    # 手改其中一份导致漂移的可能。
    contract_path = ROOT / "data" / "eval" / "ablation.json"
    contract_path.write_text((ARTIFACTS / "ablation.json").read_text(encoding="utf-8"), encoding="utf-8")

    print(f"\n产物已写入 {ARTIFACTS}：ablation.json / ablation.md / 每策略一份报告")
    print(f"（另按 CONTRACT.md K 节写了一份到 {contract_path.relative_to(ROOT)}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
