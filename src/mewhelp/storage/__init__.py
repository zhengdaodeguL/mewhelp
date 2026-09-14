"""存储层：业务数据、知识块、审计日志、会话、问题池都在这里落地。

设计取舍：默认 SQLite，配一行 `database_url` 就能换成 MySQL —— 同一套
SQLAlchemy 模型和代码路径。这样"离线跑得动"和"面试讲得了 MySQL"不冲突：
开发时不背 Docker 的包袱，演示时 `docker compose up -d mysql` 直接切过去。
"""

from .db import (
    ensure_utc,
    get_engine,
    get_session_factory,
    init_db,
    reset_engine_cache,
    session_scope,
)
from .models import (
    AuditLogRow,
    Base,
    FeedbackRow,
    FlywheelRow,
    KbChunkRow,
    SessionRow,
    UserMemoryRow,
)

__all__ = [
    "ensure_utc",
    "get_engine",
    "get_session_factory",
    "init_db",
    "reset_engine_cache",
    "session_scope",
    "Base",
    "KbChunkRow",
    "AuditLogRow",
    "SessionRow",
    "UserMemoryRow",
    "FlywheelRow",
    "FeedbackRow",
]
