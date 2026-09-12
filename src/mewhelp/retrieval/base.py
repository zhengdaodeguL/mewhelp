"""检索层的三个抽象接口。

这一层只定义"检索需要哪三种能力"，不定义任何实现细节：
  1. `Embedder`  —— 把文本映射成向量（稠密召回）
  2. `VectorStore` —— 存向量、按向量找最近邻（业务无关的存储）
  3. `Reranker`  —— 对召回结果做交叉编码精排

把三者拆开的理由：它们是三种完全独立的成本 —— 换 embedding 模型、换向量库、
开关精排，彼此之间不该互相牵制。消融实验要单独关掉某一路，也必须能只换一个对象。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import Evidence, KbChunk


class RetrievalError(RuntimeError):
    """检索层错误的基类 —— 让上层能一次性捕获整个模块的失败。"""


class BackendUnavailableError(RetrievalError):
    """真实后端（torch / Milvus 服务）不可用。

    刻意与"检索结果为空"区分：前者是环境问题（要装依赖/起服务），
    后者是知识问题（要补文档）。混成一个错误码，运维和运营就会互相甩锅。
    """


class Embedder(ABC):
    """文本 → 向量。

    约定：`encode` 返回的向量**必须 L2 归一化**（或者至少保证同一种度量的
    分值可比），因为融合和阈值判断都假设"分数量级一致"。
    """

    #: 向量维度，必须与向量库的 schema 一致
    dim: int

    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        """批量编码。顺序与入参一一对应，不能漏、不能重排。"""
        raise NotImplementedError

    def encode_one(self, text: str) -> list[float]:
        """单条编码的便捷入口 —— 查询侧几乎总是单条。"""
        out = self.encode([text])
        return out[0] if out else [0.0] * self.dim


class VectorStore(ABC):
    """向量库。只认 chunk_id 和向量，不认业务。"""

    @abstractmethod
    def upsert(self, chunks: list[KbChunk], vectors: list[list[float]]) -> int:
        """写入或覆盖（同 chunk_id 视为更新）。返回实际生效的条数。

        用 upsert 而不是 insert：知识库会反复重建（改了切分策略就要重灌），
        没有覆盖语义就只能每次 drop 整个 collection。
        """
        raise NotImplementedError

    @abstractmethod
    def search(self, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        """返回 [(chunk_id, score)]，按 score 降序。score 越大越相关。"""
        raise NotImplementedError

    @abstractmethod
    def delete_all(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError


class Reranker(ABC):
    """交叉编码精排：query 和候选**一起**过一遍模型。

    双塔（Embedder）是"各自编码、算个余弦"，快但粗；交叉编码能看见 query 和
    文档之间的词级交互，慢但准。工程上的标准姿势就是：双塔召回一大把，
    交叉编码只精排前几十条 —— 公平与成本的折中。
    """

    @abstractmethod
    def rerank(self, query: str, evidences: list[Evidence], top_k: int) -> list[Evidence]:
        """返回重排后的前 top_k 条，需写入 `rerank_score` 并保持 `score` 与之一致。"""
        raise NotImplementedError
