"""两种向量库：numpy 内存实现 + Milvus 实现。

`MemoryVectorStore` 是默认实现 —— 它让整条检索链路可以离线跑通、可断言、
可复现。别把它当成"临时方案"：单机知识库几万条 chunk 时，numpy 矩阵点积
（几毫秒）比一次带网络的向量库往返还快，工程上够用就是对的。

`MilvusVectorStore` 的价值在**一个 collection 同时扛稠密向量和稀疏向量**：
Milvus 2.5+ 内置 BM25 函数，插入时由服务端自动从原文生成稀疏向量，
省掉"再引一套 Elasticsearch 来跑 BM25"的运维 —— 这是本项目选 Milvus
而不是"FAISS + 自建倒排"的核心论点。
"""

from __future__ import annotations

import numpy as np

from ..config import Settings, get_settings
from ..schemas import KbChunk
from .base import BackendUnavailableError, RetrievalError, VectorStore


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    """按行 L2 归一化。

    统一在 store 内部归一化，而不是信任调用方 —— 余弦相似度的语义要求
    "两个向量都归一化"，只要有一边漏了，点积就不再是余弦，排序会静默地错。
    """
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0  # 零向量不做除法，保持零（相似度 0）
    return mat / norms


class MemoryVectorStore(VectorStore):
    """numpy 矩阵 + 余弦相似度。默认实现。"""

    def __init__(self, dim: int | None = None, settings: Settings | None = None):
        s = settings or get_settings()
        self.settings = s
        self.dim = int(dim or s.embedding_dim)
        self._ids: list[str] = []
        self._index: dict[str, int] = {}   # chunk_id -> 行号
        self._chunks: dict[str, KbChunk] = {}
        # 预分配 (0, dim) 而不是空的 (0,)：空矩阵做 vstack 时会退化成 (n,)，
        # 之后所有点积都会静默变成标量运算。
        self._mat: np.ndarray = np.zeros((0, self.dim), dtype=np.float32)

    # ------------------------------------------------------------------
    def upsert(self, chunks: list[KbChunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise RetrievalError(
                f"chunks({len(chunks)}) 与 vectors({len(vectors)}) 数量不一致，拒绝写入"
            )
        if not chunks:
            return 0

        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise RetrievalError(f"向量形状应为 (n, {self.dim})，实际 {arr.shape}")
        arr = _normalize_rows(arr)

        new_ids: list[str] = []
        new_rows: list[np.ndarray] = []
        for i, c in enumerate(chunks):
            cid = c.chunk_id
            if cid in self._index:
                # 覆盖：同 chunk_id 视为更新。知识库重建时文档内容会变，
                # 但 chunk_id 是按位置编号的，不覆盖就会留下旧版本的"幽灵证据"。
                self._mat[self._index[cid]] = arr[i]
                self._chunks[cid] = c
            else:
                self._index[cid] = -1  # 占位，下面统一补
                new_ids.append(cid)
                new_rows.append(arr[i])
                self._chunks[cid] = c

        if new_rows:
            start = self._mat.shape[0]
            self._mat = np.vstack([self._mat, np.vstack(new_rows)])
            for offset, cid in enumerate(new_ids):
                self._index[cid] = start + offset
            self._ids.extend(new_ids)

        return len(chunks)

    # ------------------------------------------------------------------
    def search(self, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        if top_k <= 0 or not self._ids:
            return []
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise RetrievalError(f"查询向量维度应为 {self.dim}，实际 {q.shape[0]}")
        n = float(np.linalg.norm(q))
        if n < 1e-12:
            return []  # 零查询向量：没有方向，任何相似度都无意义
        q = q / n

        sims = self._mat @ q  # 双方都已归一化，点积即余弦
        k = min(int(top_k), len(self._ids))
        # 用 argpartition 取前 k 再排序，避免全量排序 —— 知识库大了以后这是常数级差别
        idx = np.argpartition(-sims, k - 1)[:k] if k < len(sims) else np.arange(len(sims))
        order = sorted(idx.tolist(), key=lambda i: (-float(sims[i]), self._ids[i]))
        return [(self._ids[i], float(sims[i])) for i in order]

    # ------------------------------------------------------------------
    def delete_all(self) -> None:
        self._ids = []
        self._index = {}
        self._chunks = {}
        self._mat = np.zeros((0, self.dim), dtype=np.float32)

    def count(self) -> int:
        return len(self._ids)

    def get_chunk(self, chunk_id: str) -> KbChunk | None:
        """取回原文。融合阶段只有 id，要还原成 KbChunk 才能拼 prompt。"""
        return self._chunks.get(chunk_id)

    def all_chunks(self) -> list[KbChunk]:
        """按插入顺序返回全部 chunk —— BM25 索引重建时用。"""
        return [self._chunks[i] for i in self._ids]


# ======================================================================
# Milvus
# ======================================================================


class MilvusVectorStore(VectorStore):
    """Milvus 实现：一个 collection 同时承载 dense 与 sparse(BM25)。

    这里的所有真实网络调用都**不在构造期**发生（lazy 建连），理由是
    构造期抛错会污染依赖注入 —— 一个只想跑降级路径的进程不应该因为
    连不上 Milvus 而整个起不来。
    """

    #: 字段名常量：schema、插入、检索三处必须字面一致，抽出来避免手抖
    F_ID = "chunk_id"
    F_VEC = "vector"
    F_SPARSE = "sparse"
    F_TEXT = "text"
    F_TITLE = "title"

    def __init__(
        self,
        uri: str | None = None,
        collection: str | None = None,
        dim: int | None = None,
        settings: Settings | None = None,
        timeout_s: float = 5.0,
    ):
        s = settings or get_settings()
        self.settings = s
        self.uri = uri or s.milvus_uri
        self.collection = collection or s.milvus_collection
        self.dim = int(dim or s.embedding_dim)
        self.timeout_s = float(timeout_s)
        self._client = None
        self._ready = False

    # ------------------------------------------------------------------
    def _get_client(self):
        """惰性建连。连不上就抛 BackendUnavailableError（含可执行的修复提示）。"""
        if self._client is not None:
            return self._client
        try:
            from pymilvus import MilvusClient  # type: ignore
        except Exception as e:  # pragma: no cover - 依赖已装
            raise BackendUnavailableError(
                f"没有安装 pymilvus（{type(e).__name__}: {e}）。"
                "请先安装：python -m pip install pymilvus>=2.5"
            ) from e
        try:
            # 传 timeout：连不上时立刻失败，而不是卡在默认的长超时上。
            # 演示环境里"页面卡死 30 秒"和"立刻报错并降级"是完全不同的体验。
            self._client = MilvusClient(uri=self.uri, timeout=self.timeout_s)
        except Exception as e:
            raise BackendUnavailableError(
                f"连接 Milvus 失败（uri={self.uri}, {type(e).__name__}: {e}）。"
                "本地演示请用 MEWHELP_VECTOR_STORE_MODE=memory，"
                "或执行 docker compose up -d milvus 启动服务。"
            ) from e
        return self._client

    # ------------------------------------------------------------------
    def _ensure_collection(self):
        client = self._get_client()
        if self._ready:
            return client
        try:
            if not client.has_collection(self.collection):
                client.create_collection(
                    collection_name=self.collection,
                    schema=self._build_schema(),
                    index_params=self._build_index_params(),
                )
            else:
                client.load_collection(self.collection)
        except Exception as e:
            raise BackendUnavailableError(
                f"创建/加载 collection `{self.collection}` 失败（{type(e).__name__}: {e}）"
            ) from e
        self._ready = True
        return client

    def _build_schema(self):  # pragma: no cover - 需要 pymilvus 在场
        """dense + sparse 同表的结构。

        `text` 字段不只是"给人看的" —— BM25 Function 直接以它为输入，
        由服务端生成稀疏向量。所以原文必须完整落库，不能只存切片。
        """
        from pymilvus import DataType, Function, FunctionType

        schema = self._get_client().create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(self.F_ID, DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field(self.F_VEC, DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field(self.F_TEXT, DataType.VARCHAR, max_length=8192)
        schema.add_field(self.F_TITLE, DataType.VARCHAR, max_length=512)
        schema.add_field(self.F_SPARSE, DataType.SPARSE_FLOAT_VECTOR)
        # BM25 在服务端完成"分词+算权重"，客户端不需要再维护一份倒排
        schema.add_function(
            Function(
                name="bm25_text",
                function_type=FunctionType.BM25,
                input_field_names=[self.F_TEXT],
                output_field_names=[self.F_SPARSE],
            )
        )
        return schema

    def _build_index_params(self):  # pragma: no cover - 需要 pymilvus 在场
        params = self._get_client().prepare_index_params()
        # COSINE + 已归一化向量：与内存实现的分数量纲保持一致，
        # 否则换了 store 之后闸门阈值就得跟着改，那是隐形的行为漂移。
        params.add_index(self.F_VEC, index_type="AUTOINDEX", metric_type="COSINE")
        params.add_index(self.F_SPARSE, index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        return params

    # ------------------------------------------------------------------
    def upsert(self, chunks: list[KbChunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise RetrievalError("chunks 与 vectors 数量不一致，拒绝写入")
        if not chunks:
            return 0
        client = self._ensure_collection()
        rows = [
            {
                self.F_ID: c.chunk_id,
                self.F_VEC: [float(x) for x in v],
                self.F_TEXT: (c.text or "")[:8000],
                self.F_TITLE: (c.title or "")[:500],
            }
            for c, v in zip(chunks, vectors)
        ]
        try:
            client.upsert(collection_name=self.collection, data=rows)
        except Exception as e:
            raise RetrievalError(f"Milvus upsert 失败：{type(e).__name__}: {e}") from e
        return len(rows)

    # ------------------------------------------------------------------
    def search(self, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        client = self._ensure_collection()
        try:
            res = client.search(
                collection_name=self.collection,
                data=[[float(x) for x in vector]],
                anns_field=self.F_VEC,
                limit=int(top_k),
                output_fields=[self.F_ID],
            )
        except Exception as e:
            raise RetrievalError(f"Milvus 稠密检索失败：{type(e).__name__}: {e}") from e
        return _hits_to_pairs(res)

    def search_sparse(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """原生 BM25 稀疏检索 —— Milvus 服务端完成分词与打分。

        这个方法的存在本身就是卖点：同样的 20 条召回，走 Milvus 只需要一次
        网络往返，不需要在应用进程里再养一个 rank_bm25 索引和它的内存。
        """
        client = self._ensure_collection()
        try:
            res = client.search(
                collection_name=self.collection,
                data=[query],
                anns_field=self.F_SPARSE,
                limit=int(top_k),
                output_fields=[self.F_ID],
            )
        except Exception as e:
            raise RetrievalError(f"Milvus 稀疏检索失败：{type(e).__name__}: {e}") from e
        return _hits_to_pairs(res)

    def hybrid_search(self, query: str, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        """一次调用完成"稠密 + 稀疏 + RRF"，融合在服务端做。"""
        from pymilvus import AnnSearchRequest, RRFRanker

        client = self._ensure_collection()
        reqs = [
            AnnSearchRequest([float(x) for x in vector], self.F_VEC, {"metric_type": "COSINE"}, limit=int(top_k)),
            AnnSearchRequest([query], self.F_SPARSE, {"metric_type": "BM25"}, limit=int(top_k)),
        ]
        try:
            res = client.hybrid_search(
                collection_name=self.collection,
                reqs=reqs,
                ranker=RRFRanker(int(self.settings.rrf_k)),
                limit=int(top_k),
                output_fields=[self.F_ID],
            )
        except Exception as e:
            raise RetrievalError(f"Milvus 混合检索失败：{type(e).__name__}: {e}") from e
        return _hits_to_pairs(res)

    # ------------------------------------------------------------------
    def delete_all(self) -> None:
        client = self._ensure_collection()
        try:
            client.delete(collection_name=self.collection, filter=f'{self.F_ID} != ""')
        except Exception as e:
            raise RetrievalError(f"Milvus 清空失败：{type(e).__name__}: {e}") from e

    def count(self) -> int:
        client = self._ensure_collection()
        try:
            res = client.query(
                collection_name=self.collection, filter=f'{self.F_ID} != ""', output_fields=["count(*)"]
            )
        except Exception as e:
            raise RetrievalError(f"Milvus count 失败：{type(e).__name__}: {e}") from e
        if res and isinstance(res[0], dict):
            for key in ("count(*)", "count"):
                if key in res[0]:
                    return int(res[0][key])
        return 0


def _hits_to_pairs(res) -> list[tuple[str, float]]:
    """把 pymilvus 的嵌套 Hits 拍平成 [(id, score)]。

    不同版本的返回结构略有差异（dict / 对象），统一在这里吃掉差异，
    上层不需要知道 pymilvus 的内部形状。
    """
    out: list[tuple[str, float]] = []
    for hits in res or []:
        for h in hits or []:
            if isinstance(h, dict):
                cid = h.get("id") or h.get("chunk_id")
                score = h.get("distance", h.get("score", 0.0))
                entity = h.get("entity") or {}
                if isinstance(entity, dict) and entity.get("chunk_id"):
                    cid = entity["chunk_id"]
            else:  # pragma: no cover - 老版本对象式返回
                cid = getattr(h, "id", None)
                score = getattr(h, "distance", getattr(h, "score", 0.0))
                ent = getattr(h, "entity", None)
                if ent is not None:
                    cid = getattr(ent, "get", lambda *_: None)("chunk_id") or cid
            if cid is not None:
                out.append((str(cid), float(score)))
    return out


def build_store(dim: int | None = None, settings: Settings | None = None) -> VectorStore:
    """按配置造向量库。memory 是默认 —— 离线可跑优先。"""
    s = settings or get_settings()
    if s.vector_store_mode == "milvus":
        return MilvusVectorStore(dim=dim, settings=s)
    return MemoryVectorStore(dim=dim, settings=s)


__all__ = ["MemoryVectorStore", "MilvusVectorStore", "build_store"]
