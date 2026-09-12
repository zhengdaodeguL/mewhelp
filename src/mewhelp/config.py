"""集中配置。

每一个可调参数都放在这里，理由有两条：
  1. 流水线的每个取舍都能被看见、被改、被复现；
  2. 每个模块只依赖 Settings，不各自去读 os.environ。

默认值全部指向 mock —— 不装 Docker、不配 API Key 也能把整条链路跑通。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", ".env"),
        env_prefix="MEWHELP_",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------- LLM ----------------
    llm_mode: Literal["mock", "openai"] = "mock"
    llm_model: str = "gpt-5-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout_s: float = 60.0
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MEWHELP_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    openai_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MEWHELP_OPENAI_BASE_URL", "OPENAI_BASE_URL"),
    )

    # ---------------- Embedding ----------------
    embedding_mode: Literal["mock", "bge"] = "mock"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    # ---------------- Reranker ----------------
    reranker_mode: Literal["mock", "bge"] = "mock"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # ---------------- 存储 ----------------
    vector_store_mode: Literal["memory", "milvus"] = "memory"
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "mewhelp_kb"

    database_url: str = Field(
        default="sqlite:///./mewhelp.db",
        validation_alias=AliasChoices("MEWHELP_DATABASE_URL", "DATABASE_URL"),
    )

    # ---------------- 检索 ----------------
    dense_top_k: int = 20        # 稠密向量召回条数
    sparse_top_k: int = 20       # BM25 召回条数
    fusion_top_k: int = 10       # RRF 融合后保留条数
    rerank_top_k: int = 5        # 精排后真正喂给模型的条数
    rrf_k: int = 60              # RRF 平滑常数，60 是原论文默认值
    #: 引号里的内容原样保留，不当作可替换的指代词。
    #: 用户说「我说的是"那件事"」时，"那"是所指而非指代 —— 替换掉就改错了意思。
    quote_chars: tuple[str, ...] = ('"', "'", "「", "」", "『", "』", "《", "》")

    # ---------------- 置信度闸门 ----------------
    confidence_threshold: float = 0.35
    min_evidence_count: int = 1

    # ---------------- 会话上下文 ----------------
    window_turns: int = 3              # 近处滑窗保留的轮数（原文）
    summary_trigger_tokens: int = 2000  # 超过这个量就把更早的压成摘要

    # ---------------- 工具 ----------------
    tool_timeout_s: float = 5.0
    tool_max_retries: int = 2

    # ---------------- Agent ----------------
    agent_max_steps: int = 6     # ReAct 循环上限，防跑飞

    # ---------------- 可观测 ----------------
    langfuse_enabled: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3001"

    # ---------------- 数据 ----------------
    data_dir: Path = PROJECT_ROOT / "data"
    knowledge_dir: Path = PROJECT_ROOT / "data" / "knowledge"
    eval_dir: Path = PROJECT_ROOT / "data" / "eval"

    @property
    def is_fully_mocked(self) -> bool:
        """是否整套跑在 mock 上 —— 决定 README/接口要不要提示"当前是离线模式"。"""
        return (
            self.llm_mode == "mock"
            and self.embedding_mode == "mock"
            and self.vector_store_mode == "memory"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """测试用：改完环境变量后让配置重新加载。"""
    get_settings.cache_clear()
