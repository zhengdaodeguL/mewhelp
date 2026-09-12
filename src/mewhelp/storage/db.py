"""引擎与会话管理。

一个关键决定：**SQLite 和 MySQL 走完全相同的代码路径**，只换 URL。
不写"如果是 sqlite 就这样，否则那样"的分支 —— 那种分支一旦存在，
离线跑通的东西和线上跑的东西就是两个系统，测试的置信度直接归零。
"""

from __future__ import annotations

from contextlib import contextmanager
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
    return engine


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
