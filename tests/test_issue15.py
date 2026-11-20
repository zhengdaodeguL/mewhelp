"""Issue #15：先在 02277b7 上重放交互回归与真实 HTTP/SSE 时序。

只用固定证据、mock 模型、临时 SQLite 和回环 HTTP。延迟来自本文件，
不是对 8125/8126 或真实模型性能的估计。
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import threading
import time
from contextlib import contextmanager

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from mewhelp.api import create_app
from mewhelp.api.service import MAX_MESSAGE_CHARS, ChatService, SessionStore
from mewhelp.config import Settings
from mewhelp.generation import current_generation
from mewhelp.llm import LLMError, LLMResponse, MockLLMClient, OpenAILLMClient
from mewhelp.schemas import ConversationResult, Evidence, KbChunk, RetrievalResult, ToolCall
from mewhelp.storage import init_db
from mewhelp.tools import business
from mewhelp.workflow.graph import build_deps, build_graph


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None, llm_mode="mock", embedding_mode="mock",
        reranker_mode="mock", vector_store_mode="memory", langfuse_enabled=False,
        database_url=f"sqlite:///{tmp_path / 'issue15.db'}",
    )


class EvidenceRetriever:
    def __init__(self, *, rule=False):
        self.chunk = KbChunk(
            chunk_id="红薯#规格" if rule else "耳机#操作",
            doc_id="红薯" if rule else "耳机",
            title="红薯规格" if rule else "耳机操作",
            text="红薯重量规格有9斤、5斤、3斤、1斤。" if rule else "耳机双击按钮即可切换声音。",
        )

    def retrieve(self, query):
        return RetrievalResult(query=query, evidences=[Evidence(chunk=self.chunk, score=0.95)])


class TimedModel(MockLLMClient):
    """前处理 80ms，分片间隔 60ms，自评 400ms，记忆 450ms。"""
    def __init__(self, settings):
        super().__init__(settings)
        self.tasks = []
        self.marks = {}
        self.reject = False

    def chat(self, messages, **kwargs):
        task = kwargs.get("task")
        self.tasks.append(task)
        self.marks[f"{task}_start"] = time.perf_counter()
        if task == "slots":
            time.sleep(0.08)
        if task == "memory":
            time.sleep(0.45)
        if task == "self_eval":
            assert kwargs["temperature"] == 0.0
            time.sleep(0.40)
            self.marks["self_eval_end"] = time.perf_counter()
            return LLMResponse(text=json.dumps({
                "grounded": True, "answered": not self.reject, "coverage": 1.0,
            }))
        return super().chat(messages, **kwargs)

    def chat_stream(self, messages, *, on_token, **kwargs):
        assert kwargs["task"] == "generate" and kwargs["temperature"] == 0.0
        self.tasks.append("generate")
        on_token("耳机双击按钮")
        time.sleep(0.06)
        on_token("即可切换声音。")
        self.marks["last_model_token"] = time.perf_counter()
        return LLMResponse(text="耳机双击按钮即可切换声音。")


@contextmanager
def live_server(app):
    """真实套接字，避免 TestClient/ASGITransport 把整条 SSE 缓冲完才交付。"""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.perf_counter() + 5
        while not server.started:
            assert thread.is_alive() and time.perf_counter() < deadline, "mock HTTP 服务未启动"
            time.sleep(0.005)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
        assert not thread.is_alive()


async def read_sse(client, payload):
    start = time.perf_counter()
    events = []
    async with client.stream("POST", "/api/chat", json=payload) as response:
        assert response.status_code == 200
        name = ""
        async for line in response.aiter_lines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                events.append((name, json.loads(line[6:]), time.perf_counter() - start))
    return start, events, time.perf_counter() - start


async def saved_turn(client, session_id, *, status=None):
    deadline = time.perf_counter() + 4
    while time.perf_counter() < deadline:
        body = (await client.get(f"/api/sessions/{session_id}")).json()
        replies = [r for r in body.get("messages", []) if r["role"] == "assistant"]
        if replies and (status is None or replies[-1].get("status", "complete") == status):
            return body
        await asyncio.sleep(0.01)
    raise AssertionError(f"会话 {session_id} 未保存为 {status}")


@pytest.mark.parametrize("field,value", [(None, None), ("shipped_at", "2026-08-13T07:00:00+08:00"), ("shipped_on", "2026-08-13")])
def test_order_answer_reports_shipping_date_without_guessing(monkeypatch, field, value):
    order = dict(business.load_orders()["ORD-1007"])
    assert "shipped_at" not in order and "shipped_on" not in order
    if field:
        order[field] = value
    monkeypatch.setattr(business, "_find_order", lambda _id: order)
    result = business.get_order("ORD-1007")["data"]
    answer = result["note"]
    assert "下单日期：2026-08-12" in answer and "签收日期：2026-08-14" in answer
    assert "发货日期：" in answer
    if field:
        assert "发货日期：2026-08-13" in answer
    else:
        assert "发货日期：暂无记录" in answer
        assert "发货日期：2026-08-13" not in answer, "揽收时间不等于发货日期"


def test_default_model_budget_is_not_four_minutes():
    s = Settings(_env_file=None)
    assert s.llm_timeout_s <= 30
    assert s.llm_max_retries <= 1


@pytest.mark.parametrize("task", ["slots", "resolve", "classify", "plan_tool", "self_eval", "memory"])
def test_auxiliary_timeout_is_short_and_observable(settings, caplog, task):
    app = FastAPI()
    attempts = []
    slow = False

    @app.post("/v1/chat/completions")
    async def completion():
        attempts.append(time.perf_counter())
        if slow:
            await asyncio.sleep(0.35)
        return {"choices": [{"message": {"role": "assistant", "content": "{}"}, "finish_reason": "stop"}]}

    with live_server(app) as url:
        client = OpenAILLMClient(Settings(
            _env_file=None, openai_api_key="mock-key", openai_base_url=url + "/v1",
            llm_timeout_s=0.20, llm_connect_timeout_s=0.1, llm_max_retries=0,
            llm_aux_timeout_s=0.05, llm_eval_timeout_s=0.05, llm_plan_timeout_s=0.05,
        ))
        # SDK 首次调用会懒加载响应类型；这里只计网络等待与重试预算。
        client.chat([{"role": "user", "content": "预热"}], task=task)
        attempts.clear()
        slow = True
        start = time.perf_counter()
        with caplog.at_level(logging.WARNING), pytest.raises(LLMError):
            client.chat([{"role": "user", "content": "测试超时"}], task=task)
        elapsed = time.perf_counter() - start
    print(json.dumps({"case": "timeout", "task": task, "elapsed_ms": round(elapsed * 1000, 2)}, ensure_ascii=False))
    assert elapsed < 0.16, "可兜底步骤不应消耗生成步骤的超时预算"
    assert len(attempts) == 1
    assert any(
        getattr(r, "task", None) == task and getattr(r, "elapsed_ms", 0) >= 40
        and getattr(r, "timeout_s", None) == 0.05
        for r in caplog.records
    ), "超时日志必须带步骤、实际耗时和预算"


@pytest.mark.parametrize("rule", [False, True])
async def test_http_sse_first_token_done_and_eof_do_not_wait_for_tail(settings, monkeypatch, rule):
    model = TimedModel(settings)
    app = create_app(build_deps(settings=settings, llm=model, retriever=EvidenceRetriever(rule=rule)))
    service = app.state.service
    original = service._persist

    class StampDone:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            async def measured(message):
                if b"event: done" in message.get("body", b""):
                    model.marks["done_sent"] = time.perf_counter()
                await send(message)
            await self.app(scope, receive, measured)

    app.add_middleware(StampDone)

    def slow_persist(result):
        time.sleep(0.22)
        original(result)
        model.marks["persist_end"] = time.perf_counter()

    monkeypatch.setattr(service, "_persist", slow_persist)
    with live_server(app) as url:
        async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=5) as client:
            assert (await client.post("/api/auth/register", json={"email": "timing@example.org", "password": "a-test-password"})).status_code == 200
            question = "红薯有哪些重量规格？" if rule else "耳机如何切换声音？"
            start, events, eof = await read_sse(client, {"session_id": "timing", "message": question})
            tokens = [e for e in events if e[0] == "token"]
            done = next(e for e in events if e[0] == "done")
            metrics = {
                "case": "evidence_rule" if rule else "model", "first_token_ms": round(tokens[0][2] * 1000, 2),
                "last_token_ms": round(tokens[-1][2] * 1000, 2), "done_ms": round(done[2] * 1000, 2),
                "tail_ms": round((done[2] - tokens[-1][2]) * 1000, 2), "eof_ms": round(eof * 1000, 2),
                "tokens": len(tokens),
            }
            await saved_turn(client, "timing", status="complete")
            metrics["finalized_ms"] = round((model.marks["persist_end"] - start) * 1000, 2)
            print(json.dumps(metrics, ensure_ascii=False))
            assert events[0][0] == "meta" and events[-1][0] == "done"
            assert "".join(e[1]["text"] for e in tokens) == done[1]["answer"]
            assert tokens[0][2] < 0.25, metrics
            assert done[2] - tokens[-1][2] < 0.15, metrics
            assert eof - done[2] < 0.12, metrics
            assert model.marks["persist_end"] > start + done[2]
            # 后台工作须晚于服务端发送 done；客户端接收还可能稍晚一个调度周期。
            assert model.marks["memory_start"] > model.marks["done_sent"]
            if rule:
                assert "slots" not in model.tasks and "generate" not in model.tasks and "self_eval" not in model.tasks
            else:
                assert model.marks["self_eval_end"] > start + done[2]


async def test_background_answered_check_still_corrects_and_hands_off(settings):
    model = TimedModel(settings)
    model.reject = True
    app = create_app(build_deps(settings=settings, llm=model, retriever=EvidenceRetriever()))
    with live_server(app) as url:
        async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=5) as client:
            await client.post("/api/auth/register", json={"email": "audit@example.org", "password": "a-test-password"})
            _, events, _ = await read_sse(client, {"session_id": "audit", "message": "耳机如何切换声音？"})
            done = events[-1][1]
            assert done.get("status") == "verifying", "提前关闭正文流不能把未校验答复标成已完成"
            saved = await saved_turn(client, "audit", status="complete")
            answer = saved["messages"][-1]
            assert answer["handoff"] is True and "未能可靠回答" in answer["content"]
            assert model.tasks.count("self_eval") == 1


async def test_feedback_immediately_after_done_waits_for_saved_turn(settings):
    model = TimedModel(settings)
    app = create_app(build_deps(settings=settings, llm=model, retriever=EvidenceRetriever(rule=True)))
    with live_server(app) as url:
        async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=5) as client:
            await client.post("/api/auth/register", json={"email": "feedback@example.org", "password": "a-test-password"})
            await read_sse(client, {
                "session_id": "feedback", "request_id": "feedback-turn", "message": "红薯有哪些重量规格？",
            })
            response = await client.post("/api/feedback", json={
                "session_id": "feedback", "request_id": "feedback-turn", "thumbs_down": False,
            })
            assert response.status_code == 200 and response.json()["ok"]


@pytest.mark.parametrize("message", ["", "长" * (MAX_MESSAGE_CHARS + 1)], ids=["empty", "too_long"])
async def test_guarded_stream_keeps_client_request_id_for_history_sync(settings, message):
    init_db(settings=settings)
    service = ChatService(graph=build_graph(build_deps(settings=settings)), settings=settings)
    events = [json.loads(e["data"]) async for e in service.stream(
        user_id="u", session_id="guarded", request_id="client-turn", message=message,
    ) if e["event"] == "meta"]
    assert events[0]["request_id"] == "client-turn"
    history = SessionStore(settings).load("guarded", user_id="u")
    assert len(history) == 2
    assert all(turn.metadata["request_id"] == "client-turn" for turn in history)


@pytest.mark.parametrize("partial", ["", "已收到的部分回答"])
async def test_stop_saves_question_and_partial_once_and_retry_keeps_one_session(settings, partial):
    entered, closed, received = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class StopModel(MockLLMClient):
        def chat_stream(self, messages, *, on_token, **kwargs):
            async def request():
                try:
                    entered.set()
                    if partial:
                        on_token(partial)
                    await asyncio.Event().wait()
                finally:
                    closed.set()
            return current_generation.get().run(request())

    model = StopModel(settings)
    deps = build_deps(settings=settings, llm=model, retriever=EvidenceRetriever())
    init_db(settings=settings)
    service = ChatService(graph=build_graph(deps), settings=settings)
    events = service.stream(user_id="u", session_id="stop", message="耳机如何切换声音？")

    async def consume():
        async for event in events:
            if event["event"] == "token":
                received.set()

    consumer = asyncio.create_task(consume())
    await asyncio.wait_for(entered.wait(), 3)
    if partial:
        await asyncio.wait_for(received.wait(), 3)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    await events.aclose()
    await events.aclose()
    await asyncio.wait_for(closed.wait(), 3)
    if hasattr(service, "drain"):
        await service.drain()
    # 用新 Store 读 SQLite，不能只在内存里假装保存了。
    store = SessionStore(settings)
    history = store.load("stop", user_id="u")
    assert [t.role for t in history] == ["user", "assistant"]
    assert history[0].content == "耳机如何切换声音？"
    assert history[1].content == partial
    assert history[1].metadata.get("status") == "stopped"
    assert len(store.list_sessions("u")) == 1
    # 下一轮是新问答，不应覆盖已停止的那一轮，也不应新造会话。
    model.chat_stream = MockLLMClient(settings).chat_stream
    await asyncio.wait_for(anext(events, None), 1)
    retry = [e async for e in service.stream(user_id="u", session_id="stop", message="耳机如何切换声音？")]
    assert retry[-1]["event"] == "done"
    assert len(store.list_sessions("u")) == 1
    assert len(store.load("stop", user_id="u")) == 4


async def test_stop_before_chat_reaches_server_is_saved_and_late_chat_is_not_run(settings):
    """极快点击停止时，原 POST 可能尚未抵达；停止请求承担保存和去重。"""
    model = TimedModel(settings)
    app = create_app(build_deps(settings=settings, llm=model, retriever=EvidenceRetriever()))
    payload = {"session_id": "early-stop", "request_id": "same-turn", "message": "耳机如何切换声音？"}
    with live_server(app) as url:
        async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=5) as client:
            await client.post("/api/auth/register", json={"email": "early@example.org", "password": "a-test-password"})
            for _ in range(2):
                response = await client.post("/api/chat/stop", json=payload)
                assert response.status_code == 200 and response.json()["ok"]
            _, events, _ = await read_sse(client, payload)
            assert events[-1][1]["status"] == "stopped"
            saved = await saved_turn(client, "early-stop", status="stopped")
            assert len(saved["messages"]) == 2
            assert saved["messages"][0]["content"] == payload["message"]
            assert saved["messages"][1]["content"] == ""
            assert model.tasks == [], "已停止的晚到请求不能重新调用模型"
            assert (await client.get("/api/sessions")).json()["count"] == 1


async def test_next_turn_waits_for_early_stop_to_save_its_history(settings, monkeypatch):
    entered, release, invoked = threading.Event(), threading.Event(), threading.Event()
    seen = []

    class Graph:
        def invoke(self, state):
            seen.extend(state["history"])
            invoked.set()
            return {**state, "answer": "第二轮答复"}

    init_db(settings=settings)
    service = ChatService(graph=Graph(), settings=settings)
    original = service._persist

    def persist(result):
        if result.request_id == "first":
            entered.set()
            assert release.wait(3)
        original(result)

    monkeypatch.setattr(service, "_persist", persist)
    stopping = asyncio.create_task(service.stop(
        user_id="u", session_id="sequence", request_id="first", message="第一轮问题",
    ))

    async def followup():
        return [event async for event in service.stream(
            user_id="u", session_id="sequence", request_id="next", message="继续提问",
        )]

    following = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        following = asyncio.create_task(followup())
        await asyncio.to_thread(invoked.wait, 0.15)
    finally:
        release.set()
        await stopping
        if following is not None:
            await following
    assert [turn.content for turn in seen] == ["第一轮问题", ""]
    assert seen[-1].metadata["status"] == "stopped"
    history = SessionStore(settings).load("sequence", user_id="u")
    assert [turn.metadata["request_id"] for turn in history] == ["first", "first", "next", "next"]
    assert len(service.sessions.list_sessions("u")) == 1


async def test_stop_endpoint_cancels_backend_and_preserves_received_prefix(settings):
    closed = threading.Event()

    class PausedModel(MockLLMClient):
        def chat_stream(self, messages, *, on_token, **kwargs):
            async def request():
                try:
                    on_token("已经收到。")
                    await asyncio.Event().wait()
                finally:
                    closed.set()
            return current_generation.get().run(request())

    app = create_app(build_deps(settings=settings, llm=PausedModel(settings), retriever=EvidenceRetriever()))
    payload = {"session_id": "http-stop", "request_id": "received-turn", "message": "耳机如何切换声音？"}
    with live_server(app) as url:
        async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=5) as client:
            await client.post("/api/auth/register", json={"email": "stop@example.org", "password": "a-test-password"})
            received = asyncio.Event()

            async def consume():
                async with client.stream("POST", "/api/chat", json=payload) as response:
                    async for line in response.aiter_lines():
                        if line.startswith('data: {"text":'):
                            received.set()

            consumer = asyncio.create_task(consume())
            try:
                await asyncio.wait_for(received.wait(), 3)
                # 同一个请求重复停止，后端仍只保存一问一答。
                for _ in range(2):
                    response = await client.post("/api/chat/stop", json={**payload, "answer": "已经收到。"})
                    assert response.status_code == 200 and response.json()["ok"]
                assert closed.is_set()
                await asyncio.wait_for(consumer, 3)
                saved = await saved_turn(client, "http-stop", status="stopped")
                assert len(saved["messages"]) == 2
                assert saved["messages"][-1]["content"] == "已经收到。"
            finally:
                consumer.cancel()
                await asyncio.gather(consumer, return_exceptions=True)


async def test_delete_during_finalization_and_late_stop_do_not_resurrect_session(settings):
    model = TimedModel(settings)
    app = create_app(build_deps(settings=settings, llm=model, retriever=EvidenceRetriever()))
    payload = {"session_id": "deleted", "request_id": "deleted-turn", "message": "耳机如何切换声音？"}
    with live_server(app) as url:
        async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=5) as client:
            await client.post("/api/auth/register", json={"email": "delete@example.org", "password": "a-test-password"})
            await read_sse(client, payload)
            assert (await client.delete("/api/sessions/deleted")).json()["ok"]
            # 读档案等剩余收尾完成，删除后不应由后台再次创建历史。
            await client.get("/api/memories")
            assert (await client.get("/api/sessions")).json()["count"] == 0
            assert (await client.post("/api/chat/stop", json=payload)).json()["ok"] is False
            assert (await client.get("/api/sessions")).json()["count"] == 0


async def test_stop_request_cannot_write_another_accounts_session(settings):
    app = create_app(build_deps(settings=settings, retriever=EvidenceRetriever()))
    payload = {"session_id": "owned", "request_id": "owned-turn", "message": "耳机如何切换声音？"}
    with live_server(app) as url:
        async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=5) as owner, httpx.AsyncClient(base_url=url, trust_env=False, timeout=5) as other:
            await owner.post("/api/auth/register", json={"email": "owner@example.org", "password": "a-test-password"})
            assert (await owner.post("/api/chat/stop", json=payload)).json()["ok"]
            assert (await other.post("/api/chat/stop", json=payload)).status_code == 401
            await other.post("/api/auth/register", json={"email": "other@example.org", "password": "a-test-password"})
            assert (await other.post("/api/chat/stop", json=payload)).status_code == 404
            assert (await other.get("/api/sessions")).json()["count"] == 0
            saved = await saved_turn(owner, "owned", status="stopped")
            assert len(saved["messages"]) == 2


@pytest.mark.parametrize("status", ["stopped", "complete"])
@pytest.mark.parametrize("received", [None, "先收到。", "", "客户端不能编造答复"])
async def test_late_stop_confirmation_only_trims_stopped_prefix_and_preserves_cards(settings, status, received):
    init_db(settings=settings)
    service = ChatService(graph=build_graph(build_deps(settings=settings)), settings=settings)
    result = ConversationResult(
        user_id="u", session_id="prefix", request_id="prefix-turn", question="查阅资料",
        answer="先收到。后发送。", status=status,
        evidences=[Evidence(chunk=EvidenceRetriever().chunk, score=0.9)],
        tool_calls=[ToolCall(name="get_order", args={}, ok=True, result={"order_id": "ORD-TEST", "status_label": "已发货"})],
    )
    service._persist(result)
    before = service.sessions.load("prefix", user_id="u")[-1]
    for _ in range(2):
        await service.stop(
            user_id="u", session_id="prefix", request_id="prefix-turn", message="查阅资料", answer=received,
        )
    after = SessionStore(settings).load("prefix", user_id="u")
    expected = received if status == "stopped" and received is not None and result.answer.startswith(received) else result.answer
    assert len(after) == 2 and after[-1].content == expected
    assert after[-1].metadata == before.metadata
