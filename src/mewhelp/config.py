"""集中配置。

每一个可调参数都放在这里，理由有两条：
  1. 流水线的每个取舍都能被看见、被改、被复现；
  2. 每个模块只依赖 Settings，不各自去读 os.environ。

默认值全部指向 mock —— 不装 Docker、不配 API Key 也能把整条链路跑通。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
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
    llm_timeout_s: float = Field(default=30.0, gt=0)
    #: 连接阶段的超时，比分给读取的那一段短得多。
    #: 中转域名下面往往挂着好几个 IP，其中个别是黑洞 —— 连接也等满 60 秒的话，
    #: 第一次挑中坏 IP 就把整次预算烧光，重试压根没机会发生。
    llm_connect_timeout_s: float = Field(default=3.0, gt=0)
    #: 模型请求的重试次数。和工具的 `tool_max_retries` 分开：工具的失败要记进
    #: 业务轨迹（"这次查询重试过"），模型的重试只是链路抖动，不该混在一个开关里。
    llm_max_retries: int = Field(default=1, ge=0)
    #: 可规则兜底的消解、分类、槽位和记忆：短预算、不重试。
    #: 各步骤还受 llm_timeout_s 上限约束；调低总预算时不会被这里放大。
    llm_aux_timeout_s: float = Field(default=5.0, gt=0)
    llm_plan_timeout_s: float = Field(default=8.0, gt=0)
    llm_eval_timeout_s: float = Field(default=8.0, gt=0)
    #: 模型是否先输出思维链（`reasoning_content`）再给正文 —— DeepSeek-R 系、
    #: o 系这类推理模型都要打开。打开后输出预算改用下面那个更大的值，并且不发
    #: `response_format`（推理模型普遍不支持，个别中转直接 400）。
    llm_reasoning: bool = False
    #: 推理模式下的输出预算。思维链同样计入 `completion_tokens`，沿用 1024 的话
    #: 会出现"想完了、话没说出来"：`content` 为空且 `finish_reason=length`。
    llm_max_tokens_reasoning: int = 4096
    openai_api_key: str | None = None
    openai_base_url: str | None = None

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
    # ⚠️ 这两项目前**不在请求路径上**，改了不会改变问答行为。
    #
    # 它们服务的是 `memory/context.py` 的双层上下文（近处滑窗保原文、更早的
    # 压成摘要），那套实现和它的测试都在，但 `ChatService` 从来没有调过它 ——
    # 实际进模型的是会话库里保留的全部轮次（上限见
    # `api/service.py` 的 `DEFAULT_MAX_TURNS`）。
    #
    # 接上去不是一行的事，是个有代价的取舍，所以先把现状写明而不是让它假装生效：
    # 滑窗设为 3 轮意味着**超过 3 轮的对话几乎每轮都要多跑一次摘要模型调用**，
    # 而指代消解能扫到的原文只剩 3 轮（"它"指到 5 轮前的那一单就找不到了，
    # 只能靠摘要里恰好提到）。要接就得连着这两条一起定。
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

    @model_validator(mode="after")
    def _fallback_to_plain_openai_env(self) -> "Settings":
        """MEWHELP_ 前缀优先，只有它为空时才回落到不带前缀的 OPENAI_*。

        这里**不能**用 `AliasChoices("MEWHELP_OPENAI_API_KEY", "OPENAI_API_KEY")`：
        pydantic-settings 里环境变量的优先级高于 `.env` 文件，于是机器上任何历史
        遗留的 OPENAI_API_KEY 都会压过本项目 `.env` 里明确写的那一个 ——
        表现为"明明配了 key，跑起来用的却是别人家的"。显式回落才让 `.env` 说了算。
        """
        for field_name, env_name in (
            ("openai_api_key", "OPENAI_API_KEY"),
            ("openai_base_url", "OPENAI_BASE_URL"),
        ):
            if not getattr(self, field_name):
                setattr(self, field_name, os.environ.get(env_name) or None)
        return self

    @property
    def llm_output_budget(self) -> int:
        """不给单次调用指定 max_tokens 时的默认输出预算。"""
        return self.llm_max_tokens_reasoning if self.llm_reasoning else self.llm_max_tokens

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
