"""两种 Embedder：离线 mock 与真实 bge-m3。

`MockEmbedder` 不是为了"占位"，而是为了**让检索质量可以被离线度量**：
它必须真的有语义区分度，否则测试里的"相关 > 不相关"断言会变成
自欺欺人的恒真式，RRF/精排的消融实验也全都不可信。

做法：拼音/词形层面上的"哈希词袋"——
  1. jieba 分词，得到实词；
  2. 每个词用 blake2b 稳定哈希到若干维（多哈希降低碰撞造成的伪相似）；
  3. 子词再补一层字符级 2-gram，缓解 jieba 对未登录词的切分歧义；
  4. L2 归一化，于是余弦相似度直接就是词袋的重叠度。

这个方案没有"理解"能力，但它是确定性的、可解释的、和 BM25 互补的：
同义改写的查询能靠字面重叠捞回来，真正的语义改写则要靠真实模型 ——
这正好也是面试里"mock 与真实模型的差距在哪"的诚实答案。
"""

from __future__ import annotations

import hashlib
import math

from ..config import Settings, get_settings
from .base import BackendUnavailableError, Embedder
from .chunking import tokenize

#: 每个词散列到几个维度。多哈希 = 更小的碰撞概率，代价是向量更稠密。
_N_HASHES = 3


def _stable_hash(token: str, seed: int) -> int:
    """确定性哈希。

    不能用内置 hash()：CPython 对 str 启用了 PYTHONHASHSEED 随机化，
    同一进程内稳定、跨进程不稳定 —— 而知识库是离线建索引、在线检索，
    跨进程不稳定等于每次重启向量空间都变了。
    """
    h = hashlib.blake2b(f"{seed}:{token}".encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big")


class MockEmbedder(Embedder):
    """确定性的哈希词袋向量化器。"""

    def __init__(self, dim: int = 1024, settings: Settings | None = None):
        s = settings or get_settings()
        self.settings = s
        # 维度以 settings 为准，但显式传参优先 —— 测试里会构造小维度向量的场景
        self.dim = int(dim or s.embedding_dim)

    # ------------------------------------------------------------------
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._encode_one(t or "") for t in texts]

    # ------------------------------------------------------------------
    def _encode_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        # 同一个词在短文里出现 2 次，不该让它的权重是长文的 2 倍 ——
        # 所以用 1+log(tf) 压一下词频（次线性 TF）。长 chunk 里反复出现
        # 的口头语（"订单""商品"）正是靠这一步被压下去的。
        tf: dict[str, int] = {}
        for tok in self._features(text):
            tf[tok] = tf.get(tok, 0) + 1

        for tok, cnt in tf.items():
            weight = 1.0 + math.log(cnt)
            base = _stable_hash(tok, 0)
            for h in range(_N_HASHES):
                idx = _stable_hash(tok, h) % self.dim
                # 符号哈希（signed hashing）：把冲突项的部分贡献抵消掉，
                # 减少"两个无关词撞到同一维却互相加分"的假相似。
                sign = 1.0 if (base >> h) & 1 else -1.0
                vec[idx] += sign * weight

        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= 1e-12:
            # 空文本 / 全是停用词：返回全零向量而不是抛错。
            # 下游拿到零向量时余弦相似度是 0（判定为"不相关"），这是正确语义。
            return vec
        return [v / norm for v in vec]

    # ------------------------------------------------------------------
    def _features(self, text: str) -> list[str]:
        toks = tokenize(text)
        feats: list[str] = list(toks)
        # 字符级 2-gram 兜底：jieba 对"退款时限"这类专有组合可能切错，
        # 字级 n-gram 能保住字面重叠这一层信号。
        for t in toks:
            if len(t) >= 3 and all("一" <= c <= "鿿" for c in t):
                feats += [t[i : i + 2] for i in range(len(t) - 1)]
        return feats


#: 真实模型清单 —— 只在这里出现一次，避免"两个文件写了两套模型名"。
_DEFAULT_BGE_MODEL = "BAAI/bge-m3"


class BgeEmbedder(Embedder):
    """真实 bge-m3 稠密向量。

    两个刻意的设计：

    1. **lazy import**：`torch` / `FlagEmbedding` 只在第一次 `encode` 时才导入。
       否则 `import mewhelp.retrieval` 这一个动作就会在没装 torch 的机器上
       炸掉整条链路 —— mock 路径必须完全不受可选依赖影响。
    2. **导入失败给出可执行的提示**：错误信息里带上 `pip install -e ".[bge]"`，
       而不是抛一个裸 ImportError 让使用者去猜要装什么。
    """

    def __init__(self, model_name: str | None = None, dim: int | None = None, settings: Settings | None = None):
        s = settings or get_settings()
        self.settings = s
        self.model_name = model_name or s.embedding_model or _DEFAULT_BGE_MODEL
        # bge-m3 固定 1024 维；参数只为测试可注入
        self.dim = int(dim or s.embedding_dim or 1024)
        self._model = None

    def _load(self):
        """首次调用时才加载模型。"""
        if self._model is not None:
            return self._model
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore
        except Exception as e:  # pragma: no cover - 依赖 torch，CI 上不会走到
            raise BackendUnavailableError(
                f"BgeEmbedder 需要 FlagEmbedding/torch（{type(e).__name__}: {e}）。"
                '请先安装：python -m pip install -e ".[bge]"'
            ) from e
        try:  # pragma: no cover - 同上
            self._model = BGEM3FlagModel(self.model_name, use_fp16=True)
        except Exception as e:  # pragma: no cover
            raise BackendUnavailableError(
                f"加载 {self.model_name} 失败（{type(e).__name__}: {e}）。"
                "模型没下载完或磁盘空间不足时会出现这种错误。"
            ) from e
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        model = self._load()
        out = model.encode(list(texts), batch_size=16, max_length=1024)
        dense = out["dense_vecs"]
        return [[float(x) for x in row] for row in dense]


def build_embedder(settings: Settings | None = None) -> Embedder:
    """按配置造一个 Embedder。

    放在这里而不是 pipeline 里，是为了让"配置 → 实现"的映射只有一处，
    API 层、评估脚本、测试都能拿到同一个对象。
    """
    s = settings or get_settings()
    if s.embedding_mode == "bge":
        return BgeEmbedder(settings=s)
    return MockEmbedder(settings=s)
