"""SQLAlchemy 表定义。

几张表的分工：

  KbChunkRow   知识块 —— 向量库的"影子副本"。向量库存向量，这里存原文，
               因为 Milvus 不适合做全文/模糊查询，而运营后台需要按标题搜。
  AuditLogRow  审计日志 —— 每个节点一步，一条请求能还原成调用树。
  SessionRow   会话 —— 摘要 + 滑窗原文，支撑多轮续接和中断恢复。
  UserMemoryRow 用户长期记忆 —— 跨会话保留的稳定事实（尺码、偏好、关注过的订单）。
  FlywheelRow  问题池 —— 数据飞轮的落地表。
  FeedbackRow  用户反馈 —— 点赞/点踩入口。

`SessionRow` 和 `UserMemoryRow` 刻意分成两张表，而不是把记忆也塞进会话：
**生命周期不一样**。会话是可以被用户删掉的（"清空历史"），而"这个用户穿 L 码"
不该因为删了一次对话就消失。混在一张表里，"删会话"就会顺手把用户档案删掉 ——
这类越界删除在代码审查里很难被发现，因为删的语句看起来只提到了 session。

一个刻意的选择：**审计日志按节点分步存**，不是一次请求存一行 JSON。
分步存才能在"某一步特别慢"或"某一步开始出错"时用 SQL 聚合查出来 ——
这才是可观测真正要回答的问题。
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class KbChunkRow(Base):
    """知识库片段（与 retrieval.KbChunk 一一对应）。"""

    __tablename__ = "kb_chunk"

    chunk_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(128), index=True)
    title: Mapped[str] = mapped_column(String(512), default="")
    breadcrumb_json: Mapped[str] = mapped_column(Text, default="[]")
    text: Mapped[str] = mapped_column(Text, default="")
    keywords_json: Mapped[str] = mapped_column(Text, default="[]")
    is_table: Mapped[bool] = mapped_column(Boolean, default=False)
    table_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    #: 是不是飞轮回灌进来的（区分"人写的"和"系统学到的"）
    from_flywheel: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class AuditLogRow(Base):
    """节点级审计日志 —— 调用树的数据源。"""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    #: 节点名（NodeName 的值）
    node: Mapped[str] = mapped_column(String(64), index=True)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    elapsed_ms: Mapped[float] = mapped_column(Float, default=0.0)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)

    __table_args__ = (
        # 按请求查整棵树 —— 这是最频繁的查询
        Index("ix_audit_request_node", "request_id", "node"),
    )


class SessionRow(Base):
    """会话状态：摘要 + 近处滑窗原文。

    把摘要和滑窗都落库，是为了让多轮续接和中断恢复能在进程重启后仍然成立 ——
    LangGraph 的 checkpointer 存的是图状态，这里存的是"人看得懂的那份"。
    """

    __tablename__ = "session"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    #: 会话标题。历史列表靠它一眼认出"这是哪次对话" —— 没有它，用户面对的
    #: 是一串 s_ab12cd34 这样的 id，历史功能等于没做。
    #: `server_default` 不是为了好看：老库里已经有 session 表了，补列走的是
    #: `ALTER TABLE ... ADD COLUMN`，SQLite/MySQL 都要求非空新列带默认值。
    title: Mapped[str] = mapped_column(String(128), default="", server_default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    turns_json: Mapped[str] = mapped_column(Text, default="[]")
    #: 被压进摘要的轮次数，用来核对"摘要有没有漏掉信息"
    summarized_turns: Mapped[int] = mapped_column(Integer, default=0)
    token_estimate: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


class UserMemoryRow(Base):
    """用户长期记忆 —— 跨会话保留的"关于这个人的稳定事实"。

    和 `SessionRow` 的分工见模块头。这里只强调两个约束：

      * `(user_id, mem_key)` 唯一：同一个人第二次说"我穿 L 码"是**更新**，
        不是新增一条。去重靠数据库唯一约束而不是"先查再写" —— 后者在并发下
        会漏，而漏出来的表现是记忆面板里堆着一串自相矛盾的旧值。
      * `hits` 记的是"这条记忆真的被用上了几次"（不是"被塞进 prompt 几次"）。
        两者差别很大：每条记忆每轮都会被塞进 prompt，那个数字只会随时间线性增长，
        回答不了"这条记忆有没有价值"。真正的口径见 `MemoryStore.mark_used`。
    """

    __tablename__ = "user_memory"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    #: 记忆类型：order(关注过的订单) / preference(偏好与备注) / manual(用户自己写的)
    kind: Mapped[str] = mapped_column(String(32), default="preference", index=True)
    #: 归一化后的去重键 —— 和 `MemoryStore` 用同一份规则算，不要在两处各写一遍
    mem_key: Mapped[str] = mapped_column(String(128))
    text: Mapped[str] = mapped_column(Text)
    #: 来源：auto(系统从对话里抽出来的) / manual(用户在记忆面板里自己写的)
    source: Mapped[str] = mapped_column(String(16), default="auto")
    #: 钉住的记忆永远进 prompt，且不被条数上限挤掉
    pinned: Mapped[bool] = mapped_column(Boolean, default=False)
    hits: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )

    __table_args__ = (
        UniqueConstraint("user_id", "mem_key", name="uq_user_memory_key"),
    )


class FlywheelRow(Base):
    """问题池。

    retrieval_snapshot_json 是这个表最值钱的列：没有它，人工审核时就没法
    判断"真缺知识"还是"知识有但没检索到" —— 一个补文档，一个调检索。
    """

    __tablename__ = "flywheel"

    record_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    question: Mapped[str] = mapped_column(Text)
    #: 归一化后的问题指纹，用来做"同题合并"
    question_key: Mapped[str] = mapped_column(String(128), index=True)
    resolved_question: Mapped[str] = mapped_column(Text, default="")
    intent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source: Mapped[str] = mapped_column(String(32), index=True)
    refusal_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    retrieval_snapshot_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: 被问频次 —— pending() 按它降序，先审被问得最多的
    ask_count: Mapped[int] = mapped_column(Integer, default=1, index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    reviewer: Mapped[str | None] = mapped_column(String(64), nullable=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        # 同题只留一条 —— 靠 question_key 唯一约束在数据库层兜底，
        # 这样即使并发写入也不会产生重复条目
        UniqueConstraint("question_key", name="uq_flywheel_question_key"),
    )


class FeedbackRow(Base):
    """用户反馈（点踩入口）。"""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    thumbs_down: Mapped[bool] = mapped_column(Boolean, default=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
