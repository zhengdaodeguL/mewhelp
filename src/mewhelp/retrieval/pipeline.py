"""检索流水线：稠密 + 稀疏（BM25）双路召回 → RRF 融合 → Cross-Encoder 精排。

四种策略是**消融实验的唯一入口**，所以每次检索必须把中间态如实填进
`RetrievalResult`：`dense_hits / sparse_hits / fused_hits / reranked_hits /
retrieved_ids`，以及每条证据的 `dense_rank / sparse_rank / rerank_score`。
只填最终结果的话，"混合检索到底比单路强多少"这个问题就永远答不了。

设计取舍：

* **BM25 索引常驻内存**（`rank_bm25.BM25Okapi`）。它是纯 Python 的，
  重建成本随语料线性增长，知识库在线服务期间不能每次查询都重建 ——
  所以只在 `index()` 之后标脏、第一次检索时才重建（lazy rebuild）。
* **两路召回并行不起来就不并行**。进程内 numpy 点积是微秒级，
  为它开线程池的开销比收益大；真正的瓶颈在网络 LLM 调用上。
* **异常一律降级**。向量库连不上时，hybrid 退化成纯 BM25 仍然能给出
  可用答案 —— 检索层静默降级比整条链路 500 要好得多，这也是为什么
  错误被记在 `last_error` 里而不是抛出去。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from ..config import Settings, get_settings
from ..schemas import Evidence, KbChunk, RetrievalResult
from .base import Embedder, Reranker, VectorStore
from .chunking import chunk_document_file, tokenize

logger = logging.getLogger(__name__)

#: 支持的四条策略（名字会被写进消融实验的表头，不能改）
STRATEGIES: tuple[str, ...] = ("dense", "sparse", "hybrid", "hybrid_rerank")

DEFAULT_STRATEGY = "hybrid_rerank"


class _SimpleBM25:
    """`rank_bm25` 缺席时的兜底 BM25，逐项对齐 `BM25Okapi` 的口径。

    存在的意义：不让"一个纯 Python 小依赖没装上"变成整个检索模块不可用。

    刻意复刻 BM25Okapi 的三个细节，而不是用教科书版公式：
      * idf = log(N - df + 0.5) - log(df + 0.5)，允许负值；
      * 负 idf 被抬到 epsilon * 平均 idf 的下限（ATIRE 变体的防负处理）；
      * 打分里乘的是**查询词在该文档中的词频**，不是二值命中。
    口径一旦不同，"装了 rank_bm25"和"没装"两种环境就会给出不同的排序，
    消融实验表里的数字将无法跨环境复现 —— 那比模块不可用更糟。
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus = corpus
        self.corpus_size = len(corpus)
        self.doc_len = [len(d) for d in corpus]
        self.avgdl = (sum(self.doc_len) / self.corpus_size) if self.corpus_size else 0.0

        self.doc_freqs: list[dict[str, int]] = []
        nd: dict[str, int] = {}
        for doc in corpus:
            freqs: dict[str, int] = {}
            for w in doc:
                freqs[w] = freqs.get(w, 0) + 1
            self.doc_freqs.append(freqs)
            for w in freqs:
                nd[w] = nd.get(w, 0) + 1

        self.idf: dict[str, float] = {}
        idf_sum = 0.0
        negative: list[str] = []
        for w, df in nd.items():
            idf = _bm25_idf_raw(df, self.corpus_size)
            self.idf[w] = idf
            idf_sum += idf
            if idf < 0:
                negative.append(w)
        self.average_idf = (idf_sum / len(self.idf)) if self.idf else 0.0
        eps = self.epsilon * self.average_idf
        for w in negative:
            self.idf[w] = eps

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        if not self.corpus_size or self.avgdl <= 0:
            return scores
        for q in query:
            idf = self.idf.get(q, 0.0)
            if not idf:
                continue
            for i, doc in enumerate(self.doc_freqs):
                qf = doc.get(q, 0)
                if not qf:
                    continue
                denom = qf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * qf * (self.k1 + 1) / denom
        return scores


def _bm25_idf_raw(df: int, n: int) -> float:
    """BM25Okapi 的原始 idf（可负）。"""
    import math

    return math.log(n - df + 0.5) - math.log(df + 0.5)


class RetrievalPipeline:
    """检索层的门面。

    依赖注入三个后端（embedder / store / reranker），任意一个都可以换成
    假的或换掉实现 —— 这正是"能跑消融实验"的前提。
    """

    def __init__(
        self,
        embedder: Embedder | None,
        store: VectorStore,
        reranker: Reranker | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.embedder = embedder
        self.store = store
        self.reranker = reranker

        #: chunk_id -> KbChunk，检索回来只有 id，要在这里还原成原文
        self._chunks: dict[str, KbChunk] = {}
        self._bm25 = None
        self._bm25_dirty = True
        #: 最近一次失败的原因（降级路径要能被看见，不能悄悄吞掉）
        self.last_error: str | None = None
        #: 是否已经从 store 回捞过原文（lazy，避免构造期做重活）
        self._hydrated = False

    # ==================================================================
    # 建库
    # ==================================================================
    def index(self, chunks: list[KbChunk]) -> int:
        """写入知识库。返回成功的条数。

        同 chunk_id 覆盖：知识库重建时不会留下旧版本的"幽灵证据"。
        """
        if not chunks:
            return 0
        vectors: list[list[float]] = []
        if self.embedder is not None:
            try:
                vectors = self.embedder.encode([c.prompt_text() for c in chunks])
            except Exception as e:  # embedder 挂了也不能让建库失败
                self._record_error(f"embedder.encode 失败：{type(e).__name__}: {e}")
                vectors = []

        written = 0
        if vectors:
            try:
                written = self.store.upsert(chunks, vectors)
            except Exception as e:
                self._record_error(f"store.upsert 失败：{type(e).__name__}: {e}")
                written = 0
        # 即使向量库写失败，也把原文留在内存里 —— BM25 单路仍然可用。
        # 这就是"降级但可用"：宁可召回质量下降，也不要一条都搜不到。
        for c in chunks:
            self._chunks[c.chunk_id] = c
        self._bm25_dirty = True
        return written if vectors else 0

    def load_from_dir(self, path: Path | str, reset: bool = True) -> int:
        """读目录下所有 `*.md`，切分、建索引。返回 chunk 总数。

        `reset=True` 让这个操作幂等：反复调用不会累积重复 chunk，
        评估脚本可以随便重跑。
        """
        root = Path(path)
        if not root.is_dir():
            self._record_error(f"知识库目录不存在：{root}")
            return 0

        if reset:
            try:
                self.store.delete_all()
            except Exception as e:
                self._record_error(f"store.delete_all 失败：{type(e).__name__}: {e}")
            self._chunks.clear()
            self._bm25 = None
            self._bm25_dirty = True

        all_chunks: list[KbChunk] = []
        # 排序保证不同文件系统下建库顺序一致（chunk_id 与编号有关）
        for md in sorted(root.glob("*.md"), key=lambda p: p.name):
            try:
                chunks = chunk_document_file(md)
            except Exception as e:
                self._record_error(f"切分 {md.name} 失败：{type(e).__name__}: {e}")
                continue
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0
        self.index(all_chunks)
        return len(all_chunks)

    # ==================================================================
    # 检索
    # ==================================================================
    def retrieve(self, query: str, *, strategy: str = DEFAULT_STRATEGY) -> RetrievalResult:
        """按指定策略检索。

        strategy 四选一：
          * `dense`         —— 只用稠密向量（衡量"语义"这一路的贡献）
          * `sparse`        —— 只用 BM25（衡量"关键词"这一路的贡献）
          * `hybrid`        —— 两路 + RRF（衡量融合的增量）
          * `hybrid_rerank` —— hybrid + 交叉编码精排（默认，线上路径）
        """
        t0 = time.perf_counter()
        # 策略名非法时退回默认值：宁愿按默认路径回答，也不要因为一个拼写错误拒答
        if strategy not in STRATEGIES:
            self._record_error(f"未知 strategy={strategy!r}，回退到 {DEFAULT_STRATEGY}")
            strategy = DEFAULT_STRATEGY

        # 向量库里可能已经有内容（复用同一个 store 再挂一条 pipeline），
        # 这时必须把原文捞回来 —— 否则只能查到 id、拿不到文本，检索会"静默变空"。
        # 放在这里而不是 __init__：构造期做 I/O 会让依赖注入变慢且可能抛错。
        self._hydrate_from_store()

        result = RetrievalResult(query=query, strategy=strategy)
        q = (query or "").strip()
        if not q or not self._chunks:
            result.latency_ms = _ms(t0)
            return result

        use_dense = strategy in ("dense", "hybrid", "hybrid_rerank")
        use_sparse = strategy in ("sparse", "hybrid", "hybrid_rerank")

        dense_ranked: list[tuple[str, float]] = []
        sparse_ranked: list[tuple[str, float]] = []

        if use_dense and self.embedder is not None:
            dense_ranked = self._dense_search(q)
        if use_sparse:
            sparse_ranked = self._sparse_search(q)

        result.dense_hits = len(dense_ranked)
        result.sparse_hits = len(sparse_ranked)

        # ---- 融合 ----
        rankings = [ids for ids in ([i for i, _ in dense_ranked] if use_dense and dense_ranked else None,
                                    [i for i, _ in sparse_ranked] if use_sparse and sparse_ranked else None) if ids is not None]
        if not rankings:
            result.latency_ms = _ms(t0)
            return result

        from .fusion import rrf_fuse  # 延迟导入：保持 import 图简单，便于单测替换

        fused = rrf_fuse(rankings, k=int(self.settings.rrf_k))
        result.fused_hits = len(fused)

        dense_pos = {cid: r for r, (cid, _) in enumerate(dense_ranked, 1)}
        sparse_pos = {cid: r for r, (cid, _) in enumerate(sparse_ranked, 1)}
        prior_max = fused[0][1] if fused else 1.0

        evidences: list[Evidence] = []
        for cid, raw_score in fused[: int(self.settings.fusion_top_k)]:
            chunk = self._chunks.get(cid)
            if chunk is None:
                # 向量库里有、内存里没有：说明索引和原文不同步，只能跳过
                continue
            evidences.append(
                Evidence(
                    chunk=chunk,
                    # RRF 的绝对分是 1/(k+rank) 量级的小数，直接当置信度用会
                    # 永远低于闸门阈值。归一化成"相对本查询最好一条"的比例分，
                    # 让阈值 0.35 这类配置在语义上稳定。
                    score=round(raw_score / prior_max, 6) if prior_max else 0.0,
                    # 如实标来源策略：消融实验的界面要能一眼看出"这条是 dense 路
                    # 捞上来的还是融合出来的"，写死成 hybrid 会让表格说谎。
                    retriever=strategy,
                    dense_rank=dense_pos.get(cid),
                    sparse_rank=sparse_pos.get(cid),
                )
            )

        # ---- 精排 ----
        if strategy == "hybrid_rerank" and self.reranker is not None and evidences:
            try:
                evidences = self.reranker.rerank(q, evidences, top_k=int(self.settings.rerank_top_k))
            except Exception as e:
                # 精排挂了就保留融合结果 —— 降级路径必须也能出答案
                self._record_error(f"reranker.rerank 失败：{type(e).__name__}: {e}")
                evidences = evidences[: int(self.settings.rerank_top_k)]
        else:
            evidences = evidences[: int(self.settings.rerank_top_k)]

        result.evidences = evidences
        result.reranked_hits = len(evidences)
        # retrieved_ids 取"当前排序的前 fusion_top_k 个"：评估集算 Recall@k
        # 时要看到完整的候选池，不能只看最终喂给模型的 5 条。
        result.retrieved_ids = _top_ids(evidences, fused, int(self.settings.fusion_top_k))
        result.latency_ms = _ms(t0)
        return result

    # ------------------------------------------------------------------
    def _dense_search(self, query: str) -> list[tuple[str, float]]:
        try:
            vec = self.embedder.encode([query])[0]
            return self.store.search(vec, top_k=int(self.settings.dense_top_k))
        except Exception as e:
            self._record_error(f"稠密检索失败：{type(e).__name__}: {e}")
            return []

    def _sparse_search(self, query: str) -> list[tuple[str, float]]:
        bm25 = self._ensure_bm25()
        if bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        ids = list(self._chunks.keys())  # 与 _bm25 的语料顺序严格一致
        try:
            scores = bm25.get_scores(tokens)
        except Exception as e:
            self._record_error(f"BM25 打分失败：{type(e).__name__}: {e}")
            return []
        pairs = [(cid, float(s)) for cid, s in zip(ids, scores) if s > 0]
        # 同分时按 chunk_id 升序：保证同输入同输出
        pairs.sort(key=lambda kv: (-kv[1], kv[0]))
        return pairs[: int(self.settings.sparse_top_k)]

    def _ensure_bm25(self):
        """lazy 构建 BM25 索引（+ 兜底实现）。

        jieba 分词是纯 Python 的，10 万条 chunk 大约要几秒 —— 放在检索热路径
        之外，只在小节变更后重建一次。
        """
        if not self._bm25_dirty and self._bm25 is not None:
            return self._bm25
        chunk_ids = list(self._chunks.keys())
        if not chunk_ids:
            self._bm25 = None
            self._bm25_dirty = False
            return None
        corpus = [self._bm25_doc(self._chunks[cid]) for cid in chunk_ids]
        self._bm25 = _build_bm25(corpus)
        self._bm25_dirty = False
        return self._bm25

    @staticmethod
    def _bm25_doc(chunk: KbChunk) -> list[str]:
        """BM25 的"文档"要包含标题和表格内容。

        只用正文会漏掉两类关键信号：标题里的限定词（"生鲜类"）和表格单元格
        —— 而表格里的数字恰恰是用户最常问的（"退多少比例"）。
        """
        parts = [chunk.title, " ".join(chunk.breadcrumb), chunk.text]
        if chunk.is_table and chunk.table:
            parts.append(" ".join(" ".join(r) for r in chunk.table))
        return tokenize(" ".join(p for p in parts if p))

    # ------------------------------------------------------------------
    def _record_error(self, msg: str) -> None:
        """记错但不抛 —— 检索层的失败必须降级成"少召回几条"，不是 500。"""
        self.last_error = msg
        logger.warning("[retrieval] %s", msg)

    def _hydrate_from_store(self) -> None:
        """从向量库回捞原文（每个实例只做一次）。

        故意用 duck typing 而不是给 ABC 加方法：契约里的 `VectorStore` 只有四个
        方法，为了一个便利功能去改公共接口，会让所有实现都被迫跟着改。
        """
        if self._hydrated:
            return
        self._hydrated = True
        all_chunks = getattr(self.store, "all_chunks", None)
        if not callable(all_chunks):
            return
        try:
            for c in all_chunks():
                self._chunks.setdefault(c.chunk_id, c)
            if self._chunks:
                self._bm25_dirty = True
        except Exception as e:  # pragma: no cover - 纯防御
            self._record_error(f"回捞历史 chunk 失败：{type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    def count(self) -> int:
        return len(self._chunks)

    def get_chunk(self, chunk_id: str) -> KbChunk | None:
        return self._chunks.get(chunk_id)

    def stats(self) -> dict:
        """给 /api/health 和评估脚本用的自检信息。"""
        return {
            "chunks": len(self._chunks),
            "store": type(self.store).__name__,
            "store_count": _safe(self.store.count),
            "embedder": type(self.embedder).__name__ if self.embedder else None,
            "reranker": type(self.reranker).__name__ if self.reranker else None,
            "bm25": self._bm25 is not None,
            "last_error": self.last_error,
        }


# ======================================================================
# 工具函数
# ======================================================================


def _build_bm25(corpus: list[list[str]]):
    """优先用 rank_bm25；不可用时退回内置实现。"""
    try:
        from rank_bm25 import BM25Okapi  # type: ignore

        return BM25Okapi(corpus, k1=1.5, b=0.75, epsilon=0.25)
    except Exception as e:  # pragma: no cover - 依赖已装
        logger.warning("[retrieval] rank_bm25 不可用（%s），改用内置 BM25", e)
        return _SimpleBM25(corpus)


def _top_ids(evidences: list[Evidence], fused: list[tuple[str, float]], n: int) -> list[str]:
    """候选池顺序：最终排序优先，再用融合结果补足到 n 条。"""
    ids: list[str] = [e.chunk_id for e in evidences]
    seen = set(ids)
    for cid, _ in fused:
        if len(ids) >= n:
            break
        if cid not in seen:
            ids.append(cid)
            seen.add(cid)
    return ids[:n]


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 3)


def _safe(fn):
    try:
        return fn()
    except Exception as e:  # pragma: no cover - 自检不能反过来把健康检查搞挂
        return f"error: {type(e).__name__}"


def build_pipeline(settings: Settings | None = None) -> RetrievalPipeline:
    """按配置装配一条完整的检索流水线。"""
    from .embedders import build_embedder
    from .rerankers import build_reranker
    from .stores import build_store

    s = settings or get_settings()
    return RetrievalPipeline(
        embedder=build_embedder(s),
        store=build_store(dim=s.embedding_dim, settings=s),
        reranker=build_reranker(s),
        settings=s,
    )
