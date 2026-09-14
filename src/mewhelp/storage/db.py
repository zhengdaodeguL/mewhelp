"""引擎与会话管理。

一个关键决定：**SQLite 和 MySQL 走完全相同的代码路径**，只换 URL。
不写"如果是 sqlite 就这样，否则那样"的分支 —— 那种分支一旦存在，
离线跑通的东西和线上跑的东西就是两个系统，测试的置信度直接归零。
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from ..config import PROJECT_ROOT, Settings, get_settings
from .models import Base

_ENGINE_CACHE: dict[str, Engine] = {}


def _normalize_sqlite_url(url: str) -> str:
    """把相对路径的 sqlite URL 锚到项目根。

    否则 `sqlite:///./mewhelp.db` 会跟着当前工作目录跑 —— 在项目根跑和在
    tests/ 下跑会连到两个不同的库，然后出现"测试里建的数据不见了"这种
    最难查的 bug。
    """
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return url
    raw = url[len(prefix) :]
    if raw in ("", ":memory:"):
        return "sqlite://"
    p = Path(raw)
    if not p.is_absolute():
        p = (PROJECT_ROOT / raw).resolve()
    return f"{prefix}{p.as_posix()}"


def get_engine(settings: Settings | None = None) -> Engine:
    settings = settings or get_settings()
    url = _normalize_sqlite_url(settings.database_url)
    if url in _ENGINE_CACHE:
        return _ENGINE_CACHE[url]

    kwargs: dict = {"future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        # check_same_thread=False: FastAPI 的线程池里会跨线程用同一个连接
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # MySQL 连接会闲置断掉，pre_ping 之外再给一个较短的 recycle
        kwargs["pool_recycle"] = 1800

    engine = create_engine(url, **kwargs)

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _record):  # pragma: no cover - 驱动回调
            cur = dbapi_conn.cursor()
            # 外键约束默认是关的，靠它兜住 FlywheelRow 的唯一约束之外的引用完整性
            cur.execute("PRAGMA foreign_keys=ON")
            # WAL 让读写不互相阻塞 —— 审计日志是高频写，会话是高频读
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

    _ENGINE_CACHE[url] = engine
    return engine


def get_session_factory(settings: Settings | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(settings), expire_on_commit=False, future=True)


def init_db(engine: Engine | None = None, settings: Settings | None = None) -> Engine:
    engine = engine or get_engine(settings)
    Base.metadata.create_all(engine)
    _add_missing_columns(engine)
    return engine


def _add_missing_columns(engine: Engine) -> list[str]:
    """给**已存在**的表补上新加的列，返回补了哪些。

    为什么需要这一步：`create_all` 只建"不存在的表"，表已经在了它一句话都不说。
    于是给 `SessionRow` 加一个 `title`，在空库上跑得好好的，一碰到开发机上那份
    演示数据（mewhelp.db 早就建过表了）就是 `no such column: session.title` ——
    而这类错误只在"已经有数据的环境"里出现，CI 从零建库永远复现不了。

    这里刻意不引 Alembic：整套 schema 只有一个进程、一个 SQLite 文件，
    为一次"加列"引入迁移框架和版本表，维护成本远大于它挡住的风险。
    真要改列名、改类型、拆表的时候再上。

    `ADD COLUMN` 只对"新列可空或带默认值"有效 —— 这是数据库的限制，不是这里的
    取舍，所以新增列时记得给 `server_default`（Python 侧的 `default=` 不进 DDL）。
    补列失败**不吞异常**：让调用方（`_init_db_safely`）决定要不要降级启动。
    """
    from sqlalchemy import inspect, text
    from sqlalchemy.schema import CreateColumn

    inspector = inspect(engine)
    added: list[str] = []
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                ddl = CreateColumn(column).compile(dialect=engine.dialect)
                conn.execute(text(f"ALTER TABLE {table.name} ADD COLUMN {ddl}"))
                added.append(f"{table.name}.{column.name}")
    return added


@contextmanager
def session_scope(
    factory: sessionmaker[Session] | None = None, settings: Settings | None = None
) -> Iterator[Session]:
    """事务边界。

    用 `with session_scope() as s:` 而不是裸 Session —— 否则异常路径上
    连接不归还，跑久了连接池会耗尽，而症状是"跑了几百次之后突然卡住"。
    """
    factory = factory or get_session_factory(settings)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine_cache() -> None:
    """测试用：换 database_url 之后让引擎重建。"""
    for engine in _ENGINE_CACHE.values():
        engine.dispose()
    _ENGINE_CACHE.clear()


def ensure_utc(value: datetime | None) -> datetime | None:
    """把从库里读出来的时间补成**带时区**的 UTC。

    读出来为什么会是朴素的：SQLite 的 DateTime 列不存时区，写进去的
    tz-aware datetime 只剩下了钟面时间。而一个没有时区标记的 isoformat
    字符串（"2026-09-14T08:34:41"）交给 JS 的 `new Date(...)` 会按**本地时区**
    解析 —— 界面上"3 分钟前"就变成了"8 小时前"。

    更阴的是它**有时对有时错**：刚写进内存的那条还是 tz-aware（序列化带 Z），
    从库里读回来的那条是朴素的 —— 同一个列表里两条时间用两把尺子量，
    看起来只是"排序有点怪"，查起来极难。

    只在**读路径**上补，写路径不动：库里继续存 UTC 的天真值，
    换 MySQL 或换驱动时不用重新对齐数据。
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
