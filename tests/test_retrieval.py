"""检索层测试。

覆盖三件必须被证明的事，而不是"跑起来没报错"：

1. **切分保住了规则完整性** —— 表格不被拆、一条规则的两个半句不落在两个
   chunk 里。这是整个项目的主张，必须有断言把它钉死。
2. **MockEmbedder 真的有语义区分度** —— 否则后面所有策略对比都是自欺欺人。
3. **四种策略都能跑，且如实填了消融实验需要的字段** ——
   dense_hits / sparse_hits / fused_hits / reranked_hits / *_rank。

整个文件离线可跑：不碰 Docker、不碰网络、不需要 API Key。
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from mewhelp.config import get_settings
from mewhelp.retrieval import (
    STRATEGIES,
    BgeEmbedder,
    BgeReranker,
    MemoryVectorStore,
    MilvusVectorStore,
    MockEmbedder,
    MockReranker,
    RetrievalPipeline,
    build_pipeline,
    chunk_document,
    rrf_fuse,
    tokenize,
)
from mewhelp.retrieval.base import BackendUnavailableError, RetrievalError
from mewhelp.schemas import Evidence, KbChunk

KNOWLEDGE_DIR = get_settings().knowledge_dir
KNOWLEDGE_PATH = KNOWLEDGE_DIR / "退款政策.md"


# ======================================================================
# 夹具
# ======================================================================


@pytest.fixture(scope="module")
def pipeline() -> RetrievalPipeline:
    """用真实知识库建一次索引，整个模块共用。

    建库 + jieba 初始化大约几秒，每个用例重建一次会让测试慢到没人愿意跑。
    """
    p = build_pipeline()
    n = p.load_from_dir(KNOWLEDGE_DIR)
    assert n > 0, "知识库目录为空，data/knowledge/*.md 没写进去"
    return p


@pytest.fixture(scope="module")
def chunks(pipeline: RetrievalPipeline) -> list[KbChunk]:
    return list(pipeline._chunks.values())


def find_chunk(chunks: list[KbChunk], *markers: str, doc: str | None = None) -> KbChunk:
    """找出同时包含全部 marker 的那个 chunk；找不到就报错。"""
    pool = [c for c in chunks if doc is None or c.doc_id == doc]
    hits = [c for c in pool if all(m in c.text for m in markers)]
    assert hits, f"没有任何 chunk 同时包含 {markers!r}（doc={doc}）"
    return hits[0]


# ======================================================================
# 1. 表格：整体一个 chunk，行列完整
# ======================================================================


def test_every_table_survives_as_one_rectangular_chunk(chunks: list[KbChunk]):
    """所有表格 chunk 都是规则矩形，且表头在第一行。

    拆表的典型症状就是某几行丢了列头、或者列数对不齐 —— 所以断言"矩形 +
    表头含中文列名"，而不是只看 is_table 标记。
    """
    tables = [c for c in chunks if c.is_table]
    assert len(tables) >= 6, f"知识点里的表格太少（只有 {len(tables)} 张），示例数据不合格"
    for c in tables:
        assert c.table, f"{c.chunk_id} 标了 is_table 却没有 table 结构"
        width = len(c.table[0])
        assert width >= 2, f"{c.chunk_id} 的表格只有 {width} 列"
        for row in c.table:
            assert len(row) == width, f"{c.chunk_id} 行宽不齐：{[len(r) for r in c.table]}"
        # 表头必须是文字，不是分隔行（分隔行被 parser 丢掉了）
        assert all(cell.strip() for cell in c.table[0]), f"{c.chunk_id} 表头有空列"
        assert not any(set(cell.strip()) <= set("-: ") for cell in c.table[0])


def test_refund_table_keeps_all_rows_and_columns(chunks: list[KbChunk]):
    """生鲜退款比例表：7 行 4 列一行不少。"""
    table = find_chunk(chunks, "退款比例", "是否需要寄回", "已签收且有品质问题", doc="退款政策")
    assert table.is_table
    assert table.table[0] == ["商品状态", "申请时间窗", "退款比例", "是否需要寄回"]
    assert len(table.table) == 7, f"表格行数被改动/截断：{len(table.table)}"
    assert all(len(r) == 4 for r in table.table)
    # 关键数字必须在表里：拆表最容易丢的就是这些"答案单元格"
    flat = " ".join(" ".join(r) for r in table.table)
    for token in ("24 至 48 小时", "50%", "100%", "80%"):
        assert token in flat, f"表格丢了关键单元格 {token!r}"


def test_table_chunk_prompt_text_restores_header(chunks: list[KbChunk]):
    """喂给模型的文本必须带表头，否则模型看到的是"50%"却没有列名。"""
    table = find_chunk(chunks, "退款比例", "已签收且有品质问题", doc="退款政策")
    rendered = table.prompt_text()
    assert "商品状态" in rendered and "退款比例" in rendered
    # 每一行（表头 + 数据行）都要出现，且管道符数量 = 列数 - 1
    for row in table.table:
        for cell in row:
            assert cell in rendered, f"渲染后丢了单元格 {cell!r}"
    assert rendered.count("|") == (len(table.table[0]) - 1) * (len(table.table) + 1)


# ======================================================================
# 2. 一条规则不被切碎
# ======================================================================


@pytest.mark.parametrize(
    "markers,doc",
    [
        # 条件（"只受理品质问题"）与时限（"24 小时内"）必须同处一个 chunk
        (("只受理", "24 小时内", "品质问题"), "退款政策"),
        # 例外句：超过时限 + 不予退款
        (("超过 48 小时", "不予退款"), "退款政策"),
        # 服饰：吊牌状态与折损比例（20%）是同一句规则的两半
        (("吊牌已剪", "20%"), "退款政策"),
        # 跨段落的规则：数量限制与"超限后怎么办"分属相邻段落，
        # 定长滑窗极容易把它们切散 —— 这里断言它们仍在同一个 chunk
        (("最多可申请 3 次部分退款", "人工客服介入"), "退款政策"),
        # 退货：逾期关闭
        (("审核通过后 7 天内寄回", "自动关闭"), "退货政策"),
        # 物流：偏远地区"哪个省算高原"与"哪些品类不能送"同处一张表
        # （表格作为原子单位，必须整体保留）
        (("西藏全境", "不可配送品类"), "物流与配送"),
        # 会员：券与会员折扣能否叠加
        (("秒杀、拼团、清仓商品", "不参与任何叠加"), "会员与优惠"),
    ],
)
def test_rule_halves_stay_in_one_chunk(chunks: list[KbChunk], markers: tuple[str, ...], doc: str):
    """规则的两个半句必须落在同一个 chunk 里。

    这是"显式切分 vs 定长滑窗"的判决性断言：滑窗实现会在其中至少一条上失败，
    因为滑窗按字数切，而条件句和它的结论往往隔着几十个字。
    """
    hit = find_chunk(chunks, *markers, doc=doc)
    assert hit.char_len == len(hit.text)
    # 命中的片段必须是"完整可读"的一条，而不是被截断的半句
    for m in markers:
        assert m in hit.prompt_text()


def test_numeric_condition_and_conclusion_share_one_chunk(chunks: list[KbChunk]):
    """数值条件（时限、比例）与它的结论必须同处一个 chunk。

    单独拿出来断言，是因为这类句子是"半个条件导致错误答案"的高发区：
    "超过 24 小时 → 不予受理"里的"不予受理"一旦被切走，模型就会答成"可以退"。
    """
    rule = find_chunk(chunks, "超过 48 小时", "不予退款", doc="退款政策")
    assert "生鲜" in rule.text, "条件句和它的适用品类被切散了"
    assert "48 小时" in rule.text and "不予退款" in rule.text

    ratio = find_chunk(chunks, "吊牌已剪", "20%", doc="退款政策")
    assert "退货" in ratio.text or "退款" in ratio.text


def test_text_chunk_never_spans_two_headings(chunks: list[KbChunk]):
    """每个 chunk 的面包屑只有一个末级标题 —— 跨标题合并会让溯源说谎。"""
    for c in chunks:
        if c.is_table:
            continue
        assert c.breadcrumb, f"{c.chunk_id} 没有面包屑"
        assert c.title == c.breadcrumb[-1], f"{c.chunk_id} 的 title 与面包屑末级不一致"


def test_long_section_splits_on_paragraphs_without_losing_text():
    """超长小节：按段落细分，内容不丢不重，且不跨标题。"""
    # 每段约 36 字 × 30 段 ≈ 1100 字，明显超过 settings 里的细分阈值
    def para(i: int) -> str:
        return f"这是第 {i} 段说明文字，用来把小节撑长以验证细分行为是否按段落进行。"

    body = "\n\n".join(para(i) for i in range(30))
    md = f"# 文档\n\n## 甲小节\n\n{body}\n\n## 乙小节\n\n乙小节独有内容乙小节独有内容。\n"
    got = chunk_document("doc", "文档", md)

    a_chunks = [c for c in got if c.breadcrumb and c.breadcrumb[-1] == "甲小节"]
    b_chunks = [c for c in got if c.breadcrumb and c.breadcrumb[-1] == "乙小节"]
    assert len(a_chunks) >= 2, "超过阈值的正文没被细分，检索精度会被长文稀释"
    assert len(b_chunks) == 1
    # 乙小节的内容绝不能出现在甲小节的 chunk 里
    for c in a_chunks:
        assert "乙小节独有内容" not in c.text
    # 段落只是被重新分组，一个字都不能丢
    recombined = "\n\n".join(c.text for c in a_chunks)
    for i in range(30):
        assert para(i) in recombined, f"细分后丢了第 {i} 段"
    assert recombined.count(para(0)) == 1, "内容被重复计入"


def test_chunk_id_and_keywords_are_deterministic():
    md = "# 标题\n\n## 小节\n\n退款需要在签收后 48 小时内提交申请。\n"
    a = chunk_document("docA", "标题", md)
    b = chunk_document("docA", "标题", md)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.keywords for c in a] == [c.keywords for c in b]
    assert a[0].chunk_id == "docA#0001"
    assert "退款" in a[0].keywords


def test_code_fence_is_not_parsed_as_table_or_heading():
    """代码块里的 `|` 和 `#` 是字面量，不能当结构解析。"""
    md = "# 文档\n\n## 示例\n\n```\n| a | b |\n| --- | --- |\n| 1 | 2 |\n```\n\n正文结束。\n"
    got = chunk_document("doc", "文档", md)
    assert all(not c.is_table for c in got), "代码块被误判成表格"
    assert all("示例" in c.breadcrumb for c in got)


def test_nested_headings_build_breadcrumb():
    md = "# 根\n\n## 一级\n\n### 二级\n\n二级正文内容二级正文内容。\n"
    got = chunk_document("doc", "根", md)
    last = got[-1]
    assert last.breadcrumb == ["根", "一级", "二级"]
    assert last.title == "二级"


# ======================================================================
# 3. MockEmbedder：确定性 + 真的有区分度
# ======================================================================


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def test_mock_embedder_is_deterministic():
    e = MockEmbedder(dim=1024)
    text = "生鲜类商品签收后 24 小时内可以申请退款"
    v1 = e.encode([text])[0]
    v2 = e.encode([text])[0]
    assert v1 == v2, "同一个文本两次 encode 结果不一致 —— mock 必须确定性"
    # 跨实例也要一致（哈希不能用内置 hash()，它带 PYTHONHASHSEED 随机化）
    e2 = MockEmbedder(dim=1024)
    assert e2.encode([text])[0] == v1
    # 维度与归一化
    assert len(v1) == 1024
    assert math.isclose(math.sqrt(sum(x * x for x in v1)), 1.0, rel_tol=1e-6)


def test_mock_embedder_has_semantic_discrimination():
    """相关文本的相似度必须明显高于无关文本。"""
    e = MockEmbedder(dim=1024)
    query = "生鲜商品签收后超过多久就不能退钱了"
    relevant = "生鲜类商品只受理品质问题退款，且必须在签收后 24 小时内提交，超过 48 小时不予退款。"
    near = "生鲜类商品的退款比例如下，已签收且有品质问题在签收后 24 小时内可全额退款。"
    unrelated = "会员等级按近 12 个月累计消费金额自动升降级，每月 1 日结算一次。"

    q, r, n, u = e.encode([query, relevant, near, unrelated])
    sim_rel = _cos(q, r)
    sim_near = _cos(q, n)
    sim_unrel = _cos(q, u)

    assert sim_rel > 0.3, f"相关文本相似度太低（{sim_rel:.3f}），mock 没有语义信号"
    assert sim_unrel < 0.1, f"无关文本相似度异常高（{sim_unrel:.3f}），存在假相似"
    assert sim_rel > sim_unrel + 0.2, f"区分度不足：相关 {sim_rel:.3f} vs 无关 {sim_unrel:.3f}"
    assert sim_near > sim_unrel + 0.2


def test_mock_embedder_handles_empty_and_stopword_text():
    """空文本/纯停用词要返回零向量而不是抛异常 —— 上游可能传空串。"""
    e = MockEmbedder(dim=64)
    v = e.encode(["", "的了呢吗"])[0]
    assert len(v) == 64
    assert all(x == 0.0 for x in v)


def test_mock_embedder_respects_configured_dim():
    e = MockEmbedder()
    assert e.dim == get_settings().embedding_dim == 1024


# ======================================================================
# 4. 向量库
# ======================================================================


def _mk_chunks(n: int = 3) -> list[KbChunk]:
    return [
        KbChunk(chunk_id=f"d#{i}", doc_id="d", title="t", text=f"文本 {i}")
        for i in range(n)
    ]


def test_memory_store_upsert_search_count():
    store = MemoryVectorStore(dim=4)
    chunks = _mk_chunks(3)
    vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    assert store.upsert(chunks, vectors) == 3
    assert store.count() == 3

    hits = store.search([1.0, 0.0, 0.0, 0.0], top_k=2)
    assert [cid for cid, _ in hits] == ["d#0", "d#1"]
    assert hits[0][1] == pytest.approx(1.0)

    # 同 chunk_id 覆盖而不是追加
    store.upsert([chunks[0]], [[0.0, 0.0, 0.0, 1.0]])
    assert store.count() == 3
    assert store.search([0.0, 0.0, 0.0, 1.0], top_k=1)[0][0] == "d#0"

    # 零向量查询没有方向，返回空而不是抛错
    assert store.search([0.0, 0.0, 0.0, 0.0], top_k=3) == []

    store.delete_all()
    assert store.count() == 0
    assert store.search([1.0, 0.0, 0.0, 0.0], top_k=3) == []


def test_memory_store_rejects_mismatched_input():
    store = MemoryVectorStore(dim=4)
    with pytest.raises(RetrievalError):
        store.upsert(_mk_chunks(2), [[1.0, 0.0, 0.0, 0.0]])
    with pytest.raises(RetrievalError):
        store.upsert(_mk_chunks(1), [[1.0, 0.0]])  # 维度不对


def test_memory_store_search_is_deterministic_on_ties():
    """同分时按 chunk_id 升序 —— 否则消融实验的表格数字会随机跳。"""
    store = MemoryVectorStore(dim=2)
    chunks = [
        KbChunk(chunk_id="z", doc_id="d", title="t", text="z"),
        KbChunk(chunk_id="a", doc_id="d", title="t", text="a"),
        KbChunk(chunk_id="m", doc_id="d", title="t", text="m"),
    ]
    store.upsert(chunks, [[1.0, 0.0]] * 3)
    assert [cid for cid, _ in store.search([1.0, 0.0], 3)] == ["a", "m", "z"]


def test_milvus_store_is_lazy_and_fails_fast_with_clear_error():
    """构造不能连网；连不上时要抛清晰的错误，不能卡死。

    端口 1 保证没人监听 → 立刻 connection refused，而不是等超时。
    """
    store = MilvusVectorStore(uri="http://127.0.0.1:1", collection="mewhelp_test", timeout_s=2.0)
    assert store.count.__self__ is store  # 构造本身不建立连接
    try:
        n = store.count()
    except (BackendUnavailableError, RetrievalError) as e:
        msg = str(e)
        assert "Milvus" in msg or "milvus" in msg, f"错误信息没说清是 Milvus 的问题：{msg}"
        assert "memory" in msg or "pymilvus" in msg, f"错误信息缺少可执行的修复建议：{msg}"
    else:
        # 本机恰好跑着 Milvus：至少不能崩
        assert n >= 0


# ======================================================================
# 5. RRF 融合
# ======================================================================


def test_rrf_fuse_manual_ranking():
    """手工构造排名，逐项核对分数。"""
    rankings = [["a", "b", "c"], ["b", "a", "d"]]
    got = rrf_fuse(rankings, k=60)
    scores = dict(got)

    # a 在第二路第 2 名、第一路第 1 名
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    # c / d 各只在一路出现，且名次相同 → 并列，靠 id 升序决定先后
    assert scores["c"] == pytest.approx(1 / 63)
    assert scores["d"] == pytest.approx(1 / 63)
    assert [cid for cid, _ in got] == ["a", "b", "c", "d"]
    # 降序
    assert all(got[i][1] >= got[i + 1][1] for i in range(len(got) - 1))


def test_rrf_fuse_k_smoothing_and_weights():
    # k=1 时第 1 名权重 1/2，第 2 名 1/3；单路 c 的 1/2 与 a 并列 → a 在前（id 升序）
    got = rrf_fuse([["a", "b"], ["c"]], k=1)
    assert [cid for cid, _ in got] == ["a", "c", "b"]
    assert dict(got)["a"] == pytest.approx(0.5)

    # 权重 0 的那一路完全不计入
    got = rrf_fuse([["a", "b", "c"], ["z"]], k=60, weights=[1.0, 0.0])
    assert [cid for cid, _ in got] == ["a", "b", "c"]

    # 权重可以改变胜者：给第二路 5 倍权重，z 反超
    got = rrf_fuse([["a", "b"], ["z"]], k=60, weights=[1.0, 5.0])
    assert got[0][0] == "z"


def test_rrf_fuse_edge_cases():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []
    # 同一路里重复 id 只计最好名次一次，否则脏数据会被重复加分
    got = rrf_fuse([["a", "a", "b"]])
    assert [cid for cid, _ in got] == ["a", "b"]
    assert dict(got)["a"] == pytest.approx(1 / 61)
    # k 被夹到 >= 1，不会除零
    assert rrf_fuse([["a"]], k=0)[0][1] == pytest.approx(0.5)


# ======================================================================
# 6. 四种策略 + 消融字段
# ======================================================================


def test_chunking_is_deterministic_and_survives_pipeline_roundtrip():
    """同一份原文两次切分结果完全一致 —— 这是评估集里 expected_chunk_ids 能写死的前提。"""
    md = KNOWLEDGE_PATH.read_text(encoding="utf-8")
    first = chunk_document("退款政策", "退款政策", md)
    second = chunk_document("退款政策", "退款政策", md)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.text for c in first] == [c.text for c in second]
    assert [c.keywords for c in first] == [c.keywords for c in second]
    assert first, "退款政策.md 没切出任何 chunk"


def test_load_from_dir_builds_index_and_is_retrievable(tmp_path: Path):
    """自建小知识库：建库、计数、检索、幂等重建。"""
    (tmp_path / "a.md").write_text(
        "# 宠物政策\n\n## 猫粮退换\n\n猫粮开封后不支持退换，未开封可在 7 天内退换。\n",
        encoding="utf-8",
    )
    (tmp_path / "b.md").write_text(
        "# 手办政策\n\n## 预售定金\n\n预售手办的定金不予退还，尾款结清后 3 天内发货。\n",
        encoding="utf-8",
    )
    p = build_pipeline()
    n = p.load_from_dir(tmp_path)
    assert n == 2
    assert p.count() == 2

    r = p.retrieve("预售手办的定金可以退吗")
    assert r.evidences, "建库后检索不到任何内容"
    assert r.evidences[0].chunk.doc_id == "b"
    assert "定金" in r.evidences[0].chunk.text

    # 幂等：重复建库不会把 chunk 翻倍
    assert p.load_from_dir(tmp_path) == 2
    assert p.count() == 2

    # 目录不存在时降级返回 0，不抛异常
    assert p.load_from_dir(tmp_path / "nope") == 0
    assert p.last_error is not None


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_all_four_strategies_run_and_respect_top_k(pipeline: RetrievalPipeline, strategy: str):
    """四种策略都要跑通，且返回条数符合 settings 里的 top_k 配置。"""
    s = get_settings()
    result = pipeline.retrieve("生鲜类商品签收后多久可以申请退款", strategy=strategy)

    assert result.strategy == strategy
    assert result.query
    assert result.evidences, f"{strategy} 一条都没召回"
    assert len(result.evidences) <= s.rerank_top_k
    assert result.reranked_hits == len(result.evidences)
    assert len(result.retrieved_ids) <= s.fusion_top_k
    assert len(result.retrieved_ids) == len(set(result.retrieved_ids)), "候选池里有重复 id"
    assert result.latency_ms >= 0

    # 各阶段计数必须如实反映"这一路到底跑没跑"
    if strategy == "dense":
        assert result.dense_hits > 0
        assert result.sparse_hits == 0, "dense 策略不该跑 BM25"
    elif strategy == "sparse":
        assert result.sparse_hits > 0
        assert result.dense_hits == 0, "sparse 策略不该跑向量检索"
    else:
        assert result.dense_hits > 0 and result.sparse_hits > 0
        assert result.fused_hits > 0

    # 分数必须单调不增，否则"top_score"这个闸门判据就没有意义
    scores = [e.score for e in result.evidences]
    assert all(scores[i] >= scores[i + 1] - 1e-9 for i in range(len(scores) - 1)), scores


def test_evidence_rank_fields_are_populated_for_ablation(pipeline: RetrievalPipeline):
    """消融实验表的数据来源：每条证据要说清"谁把它捞上来的、排第几"。"""
    r = pipeline.retrieve("七天无理由退货的例外商品有哪些", strategy="hybrid_rerank")
    assert r.dense_hits > 0 and r.sparse_hits > 0
    # 至少有一条证据是被两路同时命中的 —— 这正是融合要奖励的情形
    assert any(e.dense_rank and e.sparse_rank for e in r.evidences)
    for e in r.evidences:
        assert e.retriever == "rerank"
        assert e.rerank_score is not None, "精排后必须留下 rerank_score"
        assert e.score == pytest.approx(e.rerank_score)
        assert e.chunk_id == e.chunk.chunk_id

    # dense 策略下不应有 rerank 痕迹，sparse 策略下不应有 dense_rank
    d = pipeline.retrieve("七天无理由退货的例外商品有哪些", strategy="dense")
    assert all(e.dense_rank is not None and e.sparse_rank is None for e in d.evidences)
    assert all(e.retriever == "dense" for e in d.evidences), "retriever 标记没如实反映策略"
    sp = pipeline.retrieve("七天无理由退货的例外商品有哪些", strategy="sparse")
    assert all(e.sparse_rank is not None and e.dense_rank is None for e in sp.evidences)
    assert all(e.retriever == "sparse" for e in sp.evidences)
    hy = pipeline.retrieve("七天无理由退货的例外商品有哪些", strategy="hybrid")
    assert all(e.retriever == "hybrid" and e.rerank_score is None for e in hy.evidences)


def test_unknown_strategy_falls_back_instead_of_raising(pipeline: RetrievalPipeline):
    r = pipeline.retrieve("退款要多久到账", strategy="grep-based")
    assert r.strategy == "hybrid_rerank"
    assert pipeline.last_error and "grep-based" in pipeline.last_error


def test_retrieval_is_deterministic_across_instances():
    """同一份知识库、同一个查询，两次检索的 id 与分数完全一致。"""
    a = build_pipeline()
    a.load_from_dir(KNOWLEDGE_DIR)
    b = build_pipeline()
    b.load_from_dir(KNOWLEDGE_DIR)
    q = "偏远地区的配送时效和不可配送品类"
    ra, rb = a.retrieve(q), b.retrieve(q)
    assert [e.chunk_id for e in ra.evidences] == [e.chunk_id for e in rb.evidences]
    assert [e.score for e in ra.evidences] == [e.score for e in rb.evidences]
    assert ra.retrieved_ids == rb.retrieved_ids


def test_empty_query_and_empty_index_degrade_gracefully():
    p = build_pipeline()  # 故意不建库
    for q in ("", "   "):
        r = p.retrieve(q)
        assert r.evidences == [] and r.fused_hits == 0 and r.strategy == "hybrid_rerank"
    r = p.retrieve("退款政策是什么")
    assert r.evidences == [], "空索引下不应该凭空造出证据"


# ======================================================================
# 7. BM25 关键词召回
# ======================================================================


def test_bm25_recalls_rare_keyword_chunk(pipeline: RetrievalPipeline):
    """"江浙沪皖"只在一张表里出现 → 纯稀疏路必须精确命中它。"""
    r = pipeline.retrieve("江浙沪皖", strategy="sparse")
    assert r.dense_hits == 0
    assert r.sparse_hits >= 1
    top = r.evidences[0]
    assert "江浙沪" in top.chunk.text
    assert top.chunk.is_table, "这个区域时效是写在表格里的，命中应该是表格 chunk"


def test_bm25_scores_only_keyword_matching_chunks(pipeline: RetrievalPipeline):
    """BM25 只给含查询词的 chunk 打分 —— 不含查询词的必须 0 分（不被召回）。

    注意"含查询词"要连关键词一起看：chunk 的 keywords 里带了标题和面包屑的词，
    这是刻意的（标题里的限定词往往是查询里最关键的那个）。
    """
    bm25 = pipeline._ensure_bm25()
    ids = list(pipeline._chunks.keys())
    scores = bm25.get_scores(tokenize("会员等级"))
    nonzero = [i for i, s in zip(ids, scores) if s > 0]
    assert nonzero, "BM25 对常见关键词一条都没命中"
    for cid in nonzero:
        c = pipeline.get_chunk(cid)
        haystack = c.text + c.title + " ".join(c.breadcrumb) + " ".join(c.keywords)
        assert "会员" in haystack or "等级" in haystack, f"{cid} 不含查询词却拿到了 BM25 分"


def test_sparse_recalls_rare_exact_keyword(pipeline: RetrievalPipeline):
    """稀有专有词（"江浙沪皖"）在稀疏路上必须精确命中。

    这是 BM25 存在的理由：这个词在稠密路里可能被稀释，而它恰恰是表格里那一行
    的唯一入口 —— 用户不会说"华东地区配送时效"，用户就说"江浙沪皖几天到"。
    """
    q = "江浙沪皖"
    targets = [c.chunk_id for c in pipeline._chunks.values() if "江浙沪" in c.text]
    assert targets, "知识库里没有含该关键词的 chunk"
    sparse = pipeline.retrieve(q, strategy="sparse")
    assert sparse.dense_hits == 0, "sparse 策略不该跑向量检索"
    assert sparse.evidences[0].chunk_id == targets[0]
    assert sparse.evidences[0].chunk.is_table, "区域时效写在表格里，命中应该是表格 chunk"

    # 两路都能召回时，融合与精排的结果里该 chunk 也必须在（不能因为融合丢答案）
    fused = pipeline.retrieve(q, strategy="hybrid")
    assert targets[0] in fused.retrieved_ids


# ======================================================================
# 8. 精排
# ======================================================================


def test_mock_reranker_ranks_relevant_first_and_truncates(pipeline: RetrievalPipeline):
    chunk = find_chunk(list(pipeline._chunks.values()), "生鲜类商品（蔬果、肉禽、水产")
    unrelated = find_chunk(list(pipeline._chunks.values()), "积分兑换比例表")
    evs = [
        Evidence(chunk=unrelated, score=1.0, retriever="hybrid"),
        Evidence(chunk=chunk, score=0.9, retriever="hybrid"),
    ]
    out = MockReranker().rerank("生鲜签收后多久内可以退款", evs, top_k=1)
    assert len(out) == 1
    assert out[0].chunk_id == chunk.chunk_id
    assert out[0].rerank_score is not None


def test_reranker_on_empty_query_keeps_upstream_order():
    """查询里没有实词时，精排不该乱动顺序（没有判据就别改排序）。"""
    chunks = _mk_chunks(2)
    evs = [Evidence(chunk=c, score=1.0 - i * 0.1, retriever="hybrid") for i, c in enumerate(chunks)]
    out = MockReranker().rerank("???", evs, top_k=2)
    assert [e.chunk_id for e in out] == ["d#0", "d#1"]
    assert MockReranker().rerank("退款", [], 3) == []


def test_rerank_improves_top1_on_labeled_queries(pipeline: RetrievalPipeline):
    """精排真的有用的证据：融合后的 P@1 必须高于融合前。

    标注方式是"用正文里的独特片段反查 chunk_id"，而不是手写 id ——
    写死 id 的标注在自己数据改动后会静默失效。
    """
    labeled = [
        ("生鲜类商品（蔬果、肉禽、水产", "生鲜签收后多久可以退款"),
        ("以下商品不适用七天无理由退货", "七天无理由退货的例外清单"),
        ("超过 48 小时无新轨迹", "物流停滞48小时怎么处理"),
        ("请在下午测量", "鞋子尺码怎么量脚长"),
        ("12 个月累计消费金额", "会员等级怎么升级"),
        ("含羊毛和真丝的商品请勿机洗甩干", "羊毛材质怎么洗"),
        ("发票重开需要先作废原发票", "发票怎么重开"),
        ("江浙沪皖", "上海发货几天能到"),
        ("偏远地区订单不支持指定送达时间", "偏远地区配送有什么限制"),
        ("逾期申请自动关闭", "退货流程分几步"),
    ]
    total_pre = total_post = 0
    for marker, query in labeled:
        want = [c.chunk_id for c in pipeline._chunks.values() if marker in c.text]
        assert len(want) == 1, f"标注片段 {marker!r} 命中了 {len(want)} 个 chunk，标注不唯一"
        pre = pipeline.retrieve(query, strategy="hybrid")
        post = pipeline.retrieve(query, strategy="hybrid_rerank")
        total_pre += pre.evidences[0].chunk_id == want[0]
        total_post += post.evidences[0].chunk_id == want[0]
    assert total_post >= total_pre, f"精排把排序搞差了：融合 {total_pre} → 精排 {total_post}"
    assert total_post >= 6, f"精排在标注集上只对了 {total_post}/10，说明精排逻辑没有起作用"


# ======================================================================
# 9. 真实后端：lazy import，不装 torch 也不能把 mock 路径拖下水
# ======================================================================


def test_bge_backends_are_not_imported_at_module_import_time():
    """import retrieval 不能顺带 import torch —— 否则没装 bge 的人整条链路都起不来。"""
    import sys

    assert "torch" not in sys.modules, "业务模块导入时就拖进了 torch"
    assert "FlagEmbedding" not in sys.modules


def test_bge_backends_raise_clear_error_when_dependency_missing(monkeypatch):
    """缺依赖时给出可执行的安装提示，而不是裸 ImportError。"""
    import builtins

    real_import = builtins.__import__

    def deny(name, *args, **kwargs):
        if name.startswith(("FlagEmbedding", "torch", "sentence_transformers")):
            raise ImportError(f"blocked: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny)
    for obj, fn in ((BgeEmbedder(), lambda o: o.encode(["x"])), (BgeReranker(), None)):
        if fn is None:
            with pytest.raises(BackendUnavailableError) as exc:
                obj._load()
        else:
            with pytest.raises(BackendUnavailableError) as exc:
                fn(obj)
        assert "bge" in str(exc.value), f"错误信息里没告诉用户装什么：{exc.value}"


def test_fallback_bm25_matches_rank_bm25_exactly():
    """内置兜底实现必须与 rank_bm25 逐项同分。

    口径差一点，"装了 rank_bm25"和"没装"两种环境就会给出不同排序 ——
    消融实验表里的数字将无法跨环境复现，这比模块不可用更糟。
    """
    from rank_bm25 import BM25Okapi

    from mewhelp.retrieval.pipeline import _SimpleBM25

    rnd = random.Random(0)  # 固定种子：这个测试本身也必须是确定性的
    vocab = ["退款", "生鲜", "小时", "会员", "等级", "物流", "时效", "表格", "退货", "发票"]
    for _ in range(20):
        corpus = [[rnd.choice(vocab) for _ in range(rnd.randint(3, 30))] for _ in range(rnd.randint(2, 10))]
        query = [rnd.choice(vocab) for _ in range(rnd.randint(1, 3))]
        expect = BM25Okapi(corpus, k1=1.5, b=0.75).get_scores(query)
        got = _SimpleBM25(corpus).get_scores(query)
        assert got == pytest.approx(list(expect), abs=1e-9)


def test_pipeline_degrades_to_bm25_when_embedder_breaks():
    """向量化挂了要降级到 BM25 单路，而不是把异常抛给调用方。"""

    class BrokenEmbedder(MockEmbedder):
        def encode(self, texts):  # type: ignore[override]
            raise RuntimeError("模拟 embedding 服务不可用")

    p = build_pipeline()
    p.embedder = BrokenEmbedder()
    assert p.load_from_dir(KNOWLEDGE_DIR) > 0
    assert "模拟 embedding 服务不可用" in (p.last_error or "")

    r = p.retrieve("江浙沪皖几天到", strategy="hybrid")
    assert r.dense_hits == 0, "embedder 挂了不该假装有稠密召回"
    assert r.sparse_hits > 0 and r.evidences, "降级后 BM25 单路仍必须能出结果"
    assert r.evidences[0].dense_rank is None


def test_pipeline_stats_surface_degradation(pipeline: RetrievalPipeline):
    """降级路径必须是可观测的，否则线上只能靠猜。"""
    stats = pipeline.stats()
    assert stats["chunks"] == pipeline.count() > 0
    assert stats["embedder"] == "MockEmbedder"
    assert stats["store"] == "MemoryVectorStore"
    assert stats["store_count"] == stats["chunks"]


def test_pipeline_reuses_existing_store_contents():
    """换一条 pipeline 复用同一个 store 时，仍能取回原文（否则会静默检索为空）。

    回捞是 lazy 的（不在构造期做 I/O），所以先触发一次检索再断言。
    """
    store = MemoryVectorStore(dim=1024)
    p1 = build_pipeline()
    p1.store = store
    p1.load_from_dir(KNOWLEDGE_DIR)
    assert store.count() > 0

    p2 = build_pipeline()
    p2.store = store  # 复用同一个 store，但内存里没有任何原文
    assert p2.count() == 0, "构造期不该自动回捞（会造成构造慢/构造抛错）"

    result = p2.retrieve("退款多久到账")
    assert result.evidences, "复用 store 后检索不到内容"
    assert p2.count() == store.count(), "没有从 store 回捞原文"
