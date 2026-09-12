"""检索层：稠密 + 稀疏混合召回，RRF 融合，交叉编码精排。

对外只需要记住三个入口：
    from mewhelp.retrieval import build_pipeline            # 按配置装配
    from mewhelp.retrieval import chunk_document            # 结构化切分
    from mewhelp.retrieval import rrf_fuse                  # 融合算法
"""

from __future__ import annotations

from .base import (
    BackendUnavailableError,
    Embedder,
    Reranker,
    RetrievalError,
    VectorStore,
)
from .chunking import (
    DEFAULT_MAX_CHARS,
    STOPWORDS,
    chunk_document,
    chunk_document_file,
    extract_keywords,
    tokenize,
)
from .embedders import BgeEmbedder, MockEmbedder, build_embedder
from .fusion import rrf_fuse, weighted_fuse
from .pipeline import STRATEGIES, RetrievalPipeline, build_pipeline
from .rerankers import BgeReranker, MockReranker, build_reranker
from .stores import MemoryVectorStore, MilvusVectorStore, build_store

__all__ = [
    # 抽象
    "Embedder",
    "VectorStore",
    "Reranker",
    "RetrievalError",
    "BackendUnavailableError",
    # 切分
    "chunk_document",
    "chunk_document_file",
    "tokenize",
    "extract_keywords",
    "DEFAULT_MAX_CHARS",
    "STOPWORDS",
    # 融合
    "rrf_fuse",
    "weighted_fuse",
    # 实现
    "MockEmbedder",
    "BgeEmbedder",
    "build_embedder",
    "MemoryVectorStore",
    "MilvusVectorStore",
    "build_store",
    "MockReranker",
    "BgeReranker",
    "build_reranker",
    # 流水线
    "RetrievalPipeline",
    "build_pipeline",
    "STRATEGIES",
]
