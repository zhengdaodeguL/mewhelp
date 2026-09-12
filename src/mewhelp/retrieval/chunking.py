"""结构化切分：按标题层级 + 表格切，不做定长滑窗。

这是整个项目最核心的一条工程主张：

    一条规则被从中间劈开，模型只看到半个条件，就会给出错误答案。

电商规则几乎都是"条件 + 例外"的结构 —— "生鲜类商品签收后 24 小时内可申请，
超过 24 小时不予受理"。定长滑窗（比如每 500 字一刀）恰好会把"24 小时"和
"不予受理"切到两个 chunk 里，检索到前半段时，模型会自信地告诉你"可以退"。

所以切分的**第一原则是语义边界优先于长度边界**：
  1. 标题是天然的边界（`#` ~ `######`），跨标题绝不合并且绝不切开；
  2. 表格是天然的原子单位，整张表一个 chunk（拆了就丢列头，单元格变成孤儿）；
  3. 只有当一个小节的正文本身太长（超过阈值）时，才在**段落**边界上再细分 ——
     宁可让 chunk 长一点，也不要切断一句话。

chunk_id 的口径是 `doc_id#序号`（四位补零）。用序号而不是内容哈希，是因为
序号可读、可直接定位（"退款政策#0004 就是生鲜那张表"），且在**文档内容不变**
时天然稳定 —— 评估集里写死的 expected_chunk_ids 因此不会随重建索引而失效。
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from ..config import get_settings
from ..schemas import KbChunk

#: 正文超过这个字数才考虑细分。定得太小会重新引入"规则被劈开"的问题，
#: 定得太大则单个 chunk 稀释掉检索精度。800 字 ≈ 一条完整规则 + 它的例外说明。
DEFAULT_MAX_CHARS = 800

#: 细分时的目标下限：宁可稍微超一点，也不要在段落中间断开。
_MIN_CHARS = 200

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
#: 分隔行：| --- | :---: | 这种
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
#: 代码围栏：表格/标题语法在代码块里是字面量，不能当结构解析
_FENCE_RE = re.compile(r"^\s*(```|~~~)")

_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t　]+")

#: 中文停用词 + 一些客服场景的高频虚词。
#: 停用词表存在的意义：BM25 里"的/了/可以"这些词在每篇文档里都出现，
#: IDF 接近 0，留着只会稀释真正区分度高的实词。
STOPWORDS: frozenset[str] = frozenset(
    """
的 了 是 在 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好
自己 这 那 他 她 它 我们 你们 他们 这个 那个 这些 那些 什么 怎么 如何 为什么
吗 呢 吧 啊 呀 哦 嗯 请问 请 可以 可 能 能够 需要 应该 必须 但是 但 因为 所以
如果 若 以及 或者 或 与 及 并 并且 而且 而 还是 只是 就是 还有 例如 比如 等 等等
对于 关于 根据 按照 按 通过 由于 为了 已 已经 将 把 被 让 给 从 向 对 以 之 其
此 该 本 各 每 任何 所有 一些 多少 几个 时 时候 后 之后 前 之前 中 内 外 下 里
一个 一种 进行 使用 提供 支持 相关 情况 问题 方式 方法 内容 部分 时间 之后
""".split()
)

#: jieba 分词结果的过滤规则：纯符号/纯数字不进入关键词。
_MEANINGFUL_RE = re.compile(r"^[一-龥a-zA-Z][一-龥a-zA-Z0-9\-]*$")


@lru_cache(maxsize=1)
def _tokenizer():  # pragma: no cover - 依赖 jieba 的全局初始化
    import jieba

    jieba.setLogLevel(60)  # 关掉 "Building prefix dict" 的 stderr 噪音
    jieba.initialize()
    return jieba


def tokenize(text: str) -> list[str]:
    """分词并过滤停用词/单字/纯符号。

    中英混排的知识库里，英文不能按字符切 —— 所以对纯 ASCII 片段走简单
    词边界切分，中文片段才交给 jieba。混着切会让 "refund" 变成 "r/e/f..."。
    """
    if not text:
        return []
    text = _HTML_RE.sub(" ", text)
    raw = _tokenizer().lcut(text)
    out: list[str] = []
    for w in raw:
        w = w.strip().lower()
        if not w or w in STOPWORDS:
            continue
        if not _MEANINGFUL_RE.match(w):
            continue
        # 单字多为虚词或偶然字，保留会带来大量噪声匹配
        if len(w) < 2:
            continue
        out.append(w)
    return out


def extract_keywords(text: str, top_n: int = 12) -> list[str]:
    """取高频实词作为关键词。

    排序用 (词频, 词) 而不是只按词频 —— 词频相同时必须有一个确定性的
    二级排序键，否则同一份文档两次切分可能给出不同的关键词列表。
    """
    toks = tokenize(text)
    if not toks:
        return []
    freq: dict[str, int] = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[: max(1, top_n)]]


def _clean_heading(text: str) -> str:
    """去掉标题里的 markdown 装饰（加粗、链接、行内代码）。

    面包屑是要展示给用户和模型的，带 `**` 只会污染 prompt。
    """
    text = _IMG_RE.sub("", text)
    text = _LINK_RE.sub(r"\1", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    return _WS_RE.sub(" ", text).strip()


def _split_table_row(line: str) -> list[str]:
    """拆一行 markdown 表格。两侧的外框竖线要去掉，单元格内容去空白。"""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _make_chunk(
    *,
    doc_id: str,
    title: str,
    breadcrumb: list[str],
    text: str,
    source_path: str,
    seq: int,
    table: list[list[str]] | None = None,
    is_table: bool = False,
) -> KbChunk:
    """统一构造入口，保证所有 chunk 的字段口径一致。"""
    body = text.strip()
    return KbChunk(
        chunk_id=f"{doc_id}#{seq:04d}",
        doc_id=doc_id,
        title=title or (breadcrumb[-1] if breadcrumb else doc_id),
        breadcrumb=list(breadcrumb),
        text=body,
        keywords=extract_keywords(body if not is_table else body + " " + _flatten(table)),
        table=table,
        is_table=is_table,
        char_len=len(body),
        source_path=source_path or None,
    )


def _flatten(table: list[list[str]] | None) -> str:
    if not table:
        return ""
    return " ".join(" ".join(r) for r in table)


def _split_paragraphs(text: str) -> list[str]:
    """按空行切段落。段内换行保留 —— 列表项的换行是有意义的。"""
    blocks = re.split(r"\n\s*\n", text)
    return [b for b in (b.strip() for b in blocks) if b]


def _pack_paragraphs(paras: list[str], max_chars: int) -> list[str]:
    """把段落贪心装箱成不超过 max_chars 的若干块。

    关键取舍：**单个段落再长也不切**。宁可这一块超标，也不要在一句话中间
    断开 —— 那句被断开的话往往正是"24 小时后不予受理"这样的条件句。
    """
    out: list[str] = []
    buf: list[str] = []
    size = 0
    for p in paras:
        plen = len(p)
        if buf and size + plen + 2 > max_chars and size >= _MIN_CHARS:
            out.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(p)
        size += plen + 2
    if buf:
        out.append("\n\n".join(buf))
    return out


def chunk_document(
    doc_id: str,
    title: str,
    markdown: str,
    source_path: str = "",
) -> list[KbChunk]:
    """把一篇 Markdown 知识库文档切成有序的 `KbChunk` 列表。

    算法（单遍扫描，不回头）：
      1. 维护面包屑栈：遇到 n 级标题就把栈截到 n-1 级再压入；
      2. 正文按标题边界分段累加，遇到下一个标题、表格或文档结束就落盘；
      3. 表格整体作为一个 chunk（含表头），前后正文各自独立成 chunk；
      4. 落盘时若正文超阈值，才在段落边界细分。

    Args:
        doc_id: 文档标识，chunk_id 的前缀。必须稳定（改了它评估集就全废）。
        title: 文档标题（H1）。会作为面包屑的根。
        markdown: 原文。
        source_path: 来源文件路径，用于溯源展示。

    Returns:
        顺序与原文一致的 chunk 列表。空文档返回 []。
    """
    max_chars = int(getattr(get_settings(), "chunk_max_chars", DEFAULT_MAX_CHARS) or DEFAULT_MAX_CHARS)
    lines = (markdown or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")

    chunks: list[KbChunk] = []
    seq = 0

    def emit(text: str, breadcrumb: list[str]) -> None:
        """落盘一小节正文（必要时按段落细分）。"""
        nonlocal seq
        for part in _pack_paragraphs(_split_paragraphs(text), max_chars):
            if not part.strip():
                continue
            seq += 1
            chunks.append(
                _make_chunk(
                    doc_id=doc_id,
                    title=breadcrumb[-1] if breadcrumb else (title or doc_id),
                    breadcrumb=breadcrumb,
                    text=part,
                    source_path=source_path,
                    seq=seq,
                )
            )

    def emit_table(rows: list[list[str]], breadcrumb: list[str]) -> None:
        nonlocal seq
        if not rows:
            return
        seq += 1
        # 表格的 text 用 markdown 形式存一份，便于不需要结构的地方直接展示；
        # table 字段保留二维结构，渲染时补回表头。
        header = rows[0]
        md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
        md_lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
        caption = breadcrumb[-1] if breadcrumb else (title or doc_id)
        chunks.append(
            _make_chunk(
                doc_id=doc_id,
                title=caption,
                breadcrumb=breadcrumb,
                text="\n".join(md_lines),
                source_path=source_path,
                seq=seq,
                table=[list(r) for r in rows],
                is_table=True,
            )
        )

    breadcrumb: list[str] = []
    if title:
        breadcrumb = [title]

    buf: list[str] = []
    i = 0
    n = len(lines)
    in_fence = False

    while i < n:
        line = lines[i]

        # --- 代码块整块跳过结构解析：里面的 # 和 | 都是字面量 ---
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            buf.append(line)
            i += 1
            continue
        if in_fence:
            buf.append(line)
            i += 1
            continue

        # --- 表格：连续的 | ... | 行 ---
        if _is_table_start(lines, i):
            if buf:
                emit("\n".join(buf), breadcrumb)
                buf = []
            rows: list[list[str]] = []
            while i < n and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                cells = _split_table_row(lines[i])
                # 分隔行不是数据，丢弃（但它的存在证明这是一张表）
                if not _TABLE_SEP_RE.match(lines[i]):
                    rows.append(cells)
                i += 1
            # 补齐列数：markdown 表格里允许省略尾部空单元格，
            # 补齐后表格才是规则矩形，下游按列取值不会越界。
            width = max((len(r) for r in rows), default=0)
            rows = [r + [""] * (width - len(r)) for r in rows]
            emit_table(rows, breadcrumb)
            continue

        # --- 标题：切段 + 维护面包屑 ---
        m = _HEADING_RE.match(line)
        if m:
            if buf:
                emit("\n".join(buf), breadcrumb)
                buf = []
            level = len(m.group(1))
            heading = _clean_heading(m.group(2))
            if heading:
                depth = level - 1
                if depth >= len(breadcrumb):
                    # 跳级标题（比如从 H1 直接到 H3）：用空位补齐层级，
                    # 否则面包屑会错位到错误的父节点上，溯源就是假的。
                    breadcrumb = breadcrumb + [""] * (depth - len(breadcrumb))
                breadcrumb = breadcrumb[:depth] + [heading]
            i += 1
            continue

        buf.append(line)
        i += 1

    if buf:
        emit("\n".join(buf), breadcrumb)

    return chunks


def _is_table_start(lines: list[str], i: int) -> bool:
    """判定 i 行是否为一张 markdown 表的表头。

    必须看下两行：孤立的一行 `| ... |` 更可能是正文里的排版，不是表。
    """
    if i + 1 >= len(lines):
        return False
    cur = lines[i].strip()
    if not (cur.startswith("|") and cur.endswith("|")):
        return False
    nxt = lines[i + 1].strip()
    if not (nxt.startswith("|") and nxt.endswith("|")):
        return False
    return bool(_TABLE_SEP_RE.match(nxt)) and "-" in nxt


def chunk_document_file(path: str | Path) -> list[KbChunk]:
    """从文件切分，doc_id 用文件名（不含扩展名）。

    单独提供这个入口，是为了让"doc_id 怎么来"只有一处定义 ——
    load_from_dir 和评估脚本必须用同一套规则，否则 expected_chunk_ids 对不上。
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    doc_id = p.stem
    # 文件首个 H1 作为标题；没有就退回文件名
    title = doc_id
    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m and len(m.group(1)) == 1:
            title = _clean_heading(m.group(2)) or doc_id
            break
    return chunk_document(doc_id=doc_id, title=title, markdown=text, source_path=str(p))
