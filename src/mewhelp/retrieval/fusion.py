"""RRF（Reciprocal Rank Fusion）融合。

为什么是 RRF 而不是"加权求和分数"：稠密的余弦分和 BM25 的分数量纲根本不可比，
归一化也救不了 —— 它们的分布形状不同（余弦挤在 0.6~1.0，BM25 可以是 0~30 的
任意值）。RRF 直接扔掉分数、只看排名，天然免疫量纲问题，并且实测在混合检索上
常常打赢一堆调参后的加权方案。

    score(d) = Σ_r  w_r / (k + rank_r(d))

`k` 是平滑常数（原论文 60）：它压住"第 1 名和第 2 名"的差距，让多路都靠前的
文档胜出，而不是被某一路的第一名一票否决。这正是我们要的性质 ——
两路都排第 3 的文档，通常比只有一路排第 1 的更靠谱。
"""

from __future__ import annotations


def rrf_fuse(
    rankings: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """把多路排名融成一个排名。

    Args:
        rankings: 每一路的 id 列表，**已按相关性降序**（第 0 个最相关）。
        k: RRF 平滑常数，默认 60（原论文默认值）。
        weights: 每一路的权重，长度需与 rankings 一致；缺省等权。

    Returns:
        [(id, fused_score)]，按分数降序。分数相同则按 id 升序 —— 保证同输入同输出，
        否则并行测试会随机失败，消融实验的表也会对不上。
    """
    if not rankings:
        return []

    # k 必须 ≥ 1：k=0 时 rank=0 会除零，且"第 0 名"的权重会趋于无穷，
    # 融合退化成"取第一路的第一名"，失去融合的意义。
    k = max(1, int(k))

    if weights is None:
        weights = [1.0] * len(rankings)

    scores: dict[str, float] = {}
    # 同一路内重复 id 只按最好名次计一次 —— 重复出现往往意味着数据有脏，
    # 让它重复加分等于放大脏数据的权重。
    for path_idx, ids in enumerate(rankings):
        w = weights[path_idx] if path_idx < len(weights) else 1.0
        if not w:
            continue
        seen: set[str] = set()
        for rank, cid in enumerate(ids, start=1):
            if cid in seen:
                continue
            seen.add(cid)
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)

    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def weighted_fuse(
    rankings: list[list[str]],
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """按权重分配"名次分"的线性融合，作为 RRF 的对照组。

    保留它纯粹是为了消融实验能回答"为什么选 RRF 而不是更简单的加权投票"。
    """
    if not rankings:
        return []
    if weights is None:
        weights = [1.0] * len(rankings)

    scores: dict[str, float] = {}
    n = max(len(r) for r in rankings) or 1
    for path_idx, ids in enumerate(rankings):
        w = weights[path_idx] if path_idx < len(weights) else 1.0
        for rank, cid in enumerate(ids, start=1):
            # 名次越靠前分越高，线性衰减到 0
            scores[cid] = scores.get(cid, 0.0) + w * (n - rank + 1) / n
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
