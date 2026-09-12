"""存储层测试：建表、往返、约束、SQLite/MySQL 同路径。"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from mewhelp.config import Settings
from mewhelp.storage import (
    AuditLogRow,
    FeedbackRow,
    FlywheelRow,
    KbChunkRow,
    SessionRow,
    get_engine,
    get_session_factory,
    init_db,
    session_scope,
)
from mewhelp.storage.db import reset_engine_cache


@pytest.fixture
def settings(tmp_path_factory):
    """一个测试模块共用一个库。

    不能用 tmp_path —— 它是**每个测试函数**一个目录，于是每个函数都拿到
    不同的 URL，建表只发生在第一个函数里，后面几个函数连到空库然后炸
    "no such table"。这类失败看起来像并发污染，其实是 fixture 作用域用错了。
    """
    d = tmp_path_factory.mktemp("storage")
    return Settings(database_url=f"sqlite:///{(d / 't.db').as_posix()}")


@pytest.fixture
def factory(settings):
    engine = init_db(get_engine(settings), settings)
    yield get_session_factory(settings)
    engine.dispose()
    reset_engine_cache()


@pytest.fixture(autouse=True)
def _clean_tables(settings, factory):
    """每个测试开始前清空数据 —— 隔离靠清表，不靠换库。"""
    from mewhelp.storage import Base

    yield
    with factory() as s:
        for table in reversed(Base.metadata.sorted_tables):
            s.execute(table.delete())
        s.commit()


def test_init_db_creates_all_tables(settings):
    engine = init_db(get_engine(settings), settings)
    from sqlalchemy import inspect

    names = set(inspect(engine).get_table_names())
    assert {"kb_chunk", "audit_log", "session", "flywheel", "feedback"} <= names
    engine.dispose()
    reset_engine_cache()


def test_kb_chunk_roundtrip(factory):
    with session_scope(factory) as s:
        s.add(
            KbChunkRow(
                chunk_id="refund#1",
                doc_id="refund",
                title="生鲜类",
                breadcrumb_json=json.dumps(["退款政策", "生鲜类"], ensure_ascii=False),
                text="生鲜类商品签收后 24 小时内可申请退款。",
                keywords_json=json.dumps(["生鲜", "退款"], ensure_ascii=False),
            )
        )
    with session_scope(factory) as s:
        row = s.get(KbChunkRow, "refund#1")
        assert row is not None
        assert json.loads(row.breadcrumb_json) == ["退款政策", "生鲜类"]
        assert row.is_table is False
        assert row.from_flywheel is False


def test_audit_log_is_stepwise_not_per_request(factory):
    """审计日志按节点分步存 —— 这样才能用 SQL 聚合出"哪一步慢"。

    如果一次请求只存一行，这个查询就写不出来，只能捞全量再在应用层算。
    """
    with session_scope(factory) as s:
        for node, ms in [("resolve", 12.0), ("retrieve", 480.0), ("generate", 900.0)]:
            s.add(
                AuditLogRow(
                    request_id="r1",
                    session_id="s1",
                    user_id="u1",
                    node=node,
                    elapsed_ms=ms,
                    payload_json="{}",
                )
            )
    with session_scope(factory) as s:
        rows = s.execute(select(AuditLogRow).where(AuditLogRow.request_id == "r1")).scalars().all()
        assert len(rows) == 3
        slowest = max(rows, key=lambda r: r.elapsed_ms)
        assert slowest.node == "generate"


def test_flywheel_question_key_is_unique(factory):
    """同题唯一约束必须在数据库层兜住 —— 应用层去重挡不住并发写入。"""
    from sqlalchemy.exc import IntegrityError

    with session_scope(factory) as s:
        s.add(
            FlywheelRow(
                record_id="f1",
                session_id="s1",
                question="能退吗",
                question_key="能退吗",
                source="refused",
            )
        )
    with pytest.raises(IntegrityError):
        with session_scope(factory) as s:
            s.add(
                FlywheelRow(
                    record_id="f2",
                    session_id="s1",
                    question="能退吗",
                    question_key="能退吗",
                    source="thumbs_down",
                )
            )


def test_flywheel_snapshot_survives_roundtrip(factory):
    """检索快照是这个表最值钱的列，必须能原样存取。"""
    snapshot = {
        "query": "这个能退吗",
        "evidences": [{"chunk": {"chunk_id": "c1", "doc_id": "d", "title": "退货", "text": "7 天"}, "score": 0.71}],
        "strategy": "hybrid_rerank",
    }
    with session_scope(factory) as s:
        s.add(
            FlywheelRow(
                record_id="f9",
                session_id="s1",
                question="这个能退吗",
                question_key="这个能退吗",
                source="refused",
                retrieval_snapshot_json=json.dumps(snapshot, ensure_ascii=False),
            )
        )
    with session_scope(factory) as s:
        row = s.get(FlywheelRow, "f9")
        back = json.loads(row.retrieval_snapshot_json)
        assert back["evidences"][0]["chunk"]["chunk_id"] == "c1"
        assert back["strategy"] == "hybrid_rerank"


def test_session_and_feedback(factory):
    with session_scope(factory) as s:
        s.add(
            SessionRow(
                session_id="s1",
                user_id="u1",
                summary="用户咨询过退款",
                turns_json=json.dumps([{"role": "user", "content": "我要退款"}], ensure_ascii=False),
                summarized_turns=2,
                token_estimate=120,
            )
        )
        s.add(
            FeedbackRow(
                session_id="s1", request_id="r1", thumbs_down=True, comment="答得不对"
            )
        )
    with session_scope(factory) as s:
        sess = s.get(SessionRow, "s1")
        assert sess.summarized_turns == 2
        assert sess.updated_at is not None
        fb = s.execute(select(FeedbackRow).where(FeedbackRow.request_id == "r1")).scalar_one()
        assert fb.thumbs_down is True


def test_relative_sqlite_url_anchors_to_project_root():
    """相对路径必须锚到项目根 —— 否则 cwd 不同会连到两个不同的库。"""
    from mewhelp.storage.db import _normalize_sqlite_url

    a = _normalize_sqlite_url("sqlite:///./mewhelp.db")
    b = _normalize_sqlite_url("sqlite:///mewhelp.db")
    assert a == b
    assert a.endswith("mewhelp.db")
    # 绝对路径原样保留
    assert _normalize_sqlite_url("sqlite:///:memory:").startswith("sqlite://")
