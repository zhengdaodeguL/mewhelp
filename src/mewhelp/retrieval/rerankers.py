"""两种精排器：离线 mock 与真实 bge-reranker-v2-m3。

精排存在的意义：双塔召回（Embedder）把 query 和文档**分别**编码，
两者直到最后一刻才算一次余弦 —— 模型没有机会看到"生鲜"和"24 小时"
出现在同一句话里。Cross-Encoder 把 query 和候选拼成一条输入一起过模型，
能捕捉词级交互，所以它在"多条件规则"这类问题上明显更强，也正是本项目
知识库最容易出错的地方（条件写在相邻段落）。

`MockReranker` 用**伪 IDF 加权的查询词覆盖度**近似这个能力：
覆盖度是按候选集内部统计出来的 IDF 加权的 —— 一个罕见词（"生鲜"）被命中，
比一个到处都有的词（"商品"）被命中更有说服力。
"""

from __future__ import annotations

import math

from ..config import Settings, get_settings
from ..schemas import Evidence
from .base import BackendUnavailableError, Reranker
from .chunking import tokenize


class MockReranker(Reranker):
    """确定性的词覆盖度精排器。

    它比 Embedder 强在**看得到全部候选**：IDF 是相对候选集算的，
    所以"这一条比那一条更对题"这件事可以被显式地判出来。
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def rerank(self, query: str, evidences: list[Evidence], top_k: int) -> list[Evidence]:
        if not evidences or top_k <= 0:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            # 查询里没有任何实词（"啊这" / "???"）：没有判据就别乱动顺序，
            # 保持上游（融合）的排序 —— 这是"降级但不破坏"的正确姿势。
            return list(evidences[:top_k])

        # ---- 候选集内部统计 IDF ----
        cand_tokens: list[set[str]] = []
        for ev in evidences:
            cand_tokens.append(self._chunk_tokens(ev))
        df: dict[str, int] = {}
        for toks in cand_tokens:
            for t in set(q_tokens):
                if t in toks:
                    df[t] = df.get(t, 0) + 1
        n = len(evidences)

        def idf(t: str) -> float:
            # 平滑 IDF：df=0 时取最大值（该词在候选集里没人命中，命中它的那条极有价值）
            return math.log((n + 1) / (df.get(t, 0) + 1)) + 1.0

        total_w = sum(idf(t) for t in set(q_tokens)) or 1.0
        prior_max = max((ev.score for ev in evidences), default=0.0) or 1.0

        # 标题命中加权：查询词出现在标题里（"生鲜类"、"异常件"）几乎等价于
        # "这条证据就是讲这个的"。实测把 P@1 从 0.33 提到 0.50 —— 权重取 0.2：
        # 再大就会让"标题偶尔撞词"的无关条款压过正文真正对题的证据。
        title_bonus = 0.2
        scored: list[Evidence] = []
        for ev, toks in zip(evidences, cand_tokens):
            hit = {t for t in set(q_tokens) if t in toks}
            coverage = sum(idf(t) for t in hit) / total_w
            title_toks = self._title_tokens(ev)
            t_hit = hit & title_toks
            title_cov = (len(t_hit) / len(hit)) if hit else 0.0
            base = (1 - title_bonus) * coverage + title_bonus * title_cov
            # 上游融合分保留 10% 权重：精排只看词重合，对"同义改写"是无感的，
            # 保留一点上游信号可以让精排退化成"微调"而不是"推翻"。
            prior = ev.score / prior_max
            score = 0.9 * base + 0.1 * prior
            scored.append(
                ev.model_copy(update={"score": round(score, 6), "rerank_score": round(score, 6), "retriever": "rerank"})
            )

        # 同分时按 chunk_id 升序 —— 消融实验的表必须可复现
        scored.sort(key=lambda e: (-e.score, e.chunk_id))
        return scored[:top_k]

    # ------------------------------------------------------------------
    @staticmethod
    def _title_tokens(ev: Evidence) -> set[str]:
        c = ev.chunk
        return set(tokenize(f"{c.title} {' '.join(c.breadcrumb)}"))

    @staticmethod
    def _chunk_tokens(ev: Evidence) -> set[str]:
        """候选文档的可匹配词集合。

        把 breadcrumb 和 keywords 也算进去：一条证据的标题（"退款政策 > 生鲜类"）
        往往就是查询里最关键的限定词，只用正文会漏掉这个信号。
        """
        c = ev.chunk
        parts = [c.text, c.title, " ".join(c.breadcrumb), " ".join(c.keywords)]
        if c.is_table and c.table:
            parts.append(" ".join(" ".join(r) for r in c.table))
        toks = set(tokenize(" ".join(p for p in parts if p)))
        # 表格单元格里的短词（"24 小时"被 jieba 切成 "24" + "小时"）也需要能匹配上
        return toks


_DEFAULT_BGE_RERANKER = "BAAI/bge-reranker-v2-m3"


class BgeReranker(Reranker):
    """真实 bge-reranker-v2-m3 交叉编码精排。

    与 BgeEmbedder 同样的 lazy import 策略：不装 torch 的人跑 mock 路径时，
    一行可选依赖的代码都不会被执行到。
    """

    def __init__(self, model_name: str | None = None, settings: Settings | None = None):
        s = settings or get_settings()
        self.settings = s
        self.model_name = model_name or s.reranker_model or _DEFAULT_BGE_RERANKER
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from FlagEmbedding import FlagReranker  # type: ignore
        except Exception as e:  # pragma: no cover
            raise BackendUnavailableError(
                f"BgeReranker 需要 FlagEmbedding/torch（{type(e).__name__}: {e}）。"
                '请先安装：python -m pip install -e ".[bge]"'
            ) from e
        try:  # pragma: no cover
            self._model = FlagReranker(self.model_name, use_fp16=True)
        except Exception as e:  # pragma: no cover
            raise BackendUnavailableError(
                f"加载 {self.model_name} 失败（{type(e).__name__}: {e}）。"
            ) from e
        return self._model

    def rerank(self, query: str, evidences: list[Evidence], top_k: int) -> list[Evidence]:  # pragma: no cover
        if not evidences or top_k <= 0:
            return []
        model = self._load()
        pairs = [[query, ev.chunk.prompt_text()] for ev in evidences]
        scores = model.compute_score(pairs, normalize=True)
        if isinstance(scores, (int, float)):
            scores = [float(scores)]
        out = [
            ev.model_copy(update={"score": float(s), "rerank_score": float(s), "retriever": "rerank"})
            for ev, s in zip(evidences, scores)
        ]
        out.sort(key=lambda e: (-e.score, e.chunk_id))
        return out[:top_k]


def build_reranker(settings: Settings | None = None) -> Reranker:
    s = settings or get_settings()
    if s.reranker_mode == "bge":
        return BgeReranker(settings=s)
    return MockReranker(settings=s)
