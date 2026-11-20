"""中断 / 恢复测试 —— 确定性子流程最值钱的那个能力。

朴素做法是把"等用户补充"当成一次问答，下一轮重走整条流水线。代价：
重复检索、重复调模型，而且第二轮很容易丢掉第一轮的意图。

这里验证的是另一种做法：流程真的"停在原地"，用户补充后从断点继续。
"""

from __future__ import annotations

from pathlib import Path
import asyncio
import json

import pytest

from mewhelp.domain import Intent
from mewhelp.api.service import ChatService, SessionStore
from mewhelp.config import Settings
from mewhelp.generation import current_generation
from mewhelp.llm import LLMError, LLMResponse, MockLLMClient
from mewhelp.storage import init_db
from mewhelp.workflow.graph import build_deps, build_graph, make_sqlite_checkpointer
from mewhelp.workflow.state import new_state


@pytest.fixture
def streaming_service(tmp_path):
    """真正跑图，以可控的模型分片复现生成中的停顿。"""
    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path / 'stream.db'}")
    init_db(settings=settings)

    class StreamingLLM(MockLLMClient):
        def __init__(self):
            super().__init__(settings)
            self.next_piece = asyncio.Event()
            self.request_closed = asyncio.Event()
            self.fail = False
            self.handoff = False
            self.evaluated = False
            self.stream_calls = []

        def chat_stream(self, messages, *, on_token, **kwargs):
            self.stream_calls.append(kwargs["task"])

            async def generate():
                try:
                    on_token("第一片")
                    await self.next_piece.wait()
                    if self.fail:
                        raise LLMError("模拟流在正文中途失败")
                    on_token("第二片")
                    return LLMResponse(text="第一片第二片")
                finally:
                    self.request_closed.set()

            return current_generation.get().run(generate())

        def chat(self, messages, **kwargs):
            if kwargs.get("task") == "self_eval":
                self.evaluated = True
                if self.handoff:
                    return LLMResponse(text='{"grounded":false,"coverage":0}')
            assert kwargs.get("task") != "generate", "最终生成必须使用流式接口"
            return super().chat(messages, **kwargs)

    llm = StreamingLLM()
    deps = build_deps(settings=settings, llm=llm)
    deps.retriever.load_from_dir(Path("data/knowledge"))
    service = ChatService(
        graph=build_graph(deps), settings=settings, sessions=SessionStore(settings),
    )
    return service, llm


async def _first_token(events):
    opening = []
    async for event in events:
        opening.append(event)
        if event["event"] == "token":
            return opening
    raise AssertionError("未收到模型 token")


@pytest.mark.asyncio
@pytest.mark.parametrize("handoff", [False, True])
async def test_stream_tokens_arrive_before_generation_finishes(streaming_service, monkeypatch, handoff):
    service, llm = streaming_service
    llm.handoff = handoff

    async def no_sleep(*args, **kwargs):
        raise AssertionError("SSE 不应添加人为延迟")

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    events = service.stream(user_id="u", session_id="s", message="尺码怎么选")
    opening = await asyncio.wait_for(_first_token(events), 3)
    assert [e["event"] for e in opening] == ["meta", "evidence", "token"]
    assert json.loads(opening[-1]["data"])["text"] == "第一片"
    assert not llm.request_closed.is_set()
    assert not llm.evaluated
    assert service.sessions.load("s", user_id="u") == []
    llm.next_piece.set()
    remaining = [event async for event in events]
    assert remaining[0]["event"] == "token"
    assert json.loads(remaining[0]["data"])["text"] == "第二片"
    assert remaining[-1]["event"] == "done"
    answer = "".join(json.loads(e["data"])["text"] for e in opening + remaining if e["event"] == "token")
    assert json.loads(remaining[-1]["data"])["answer"] == answer
    assert json.loads(remaining[-1]["data"])["status"] == "verifying"
    assert json.loads(remaining[-1]["data"])["handoff"] is False
    saved = service.sessions.load("s", user_id="u")[-1]
    assert saved.content.startswith(answer)
    assert saved.metadata["handoff"] is handoff
    if handoff:
        assert "未能可靠回答" in saved.content
    assert llm.stream_calls == ["generate"]
    assert llm.evaluated


@pytest.mark.asyncio
async def test_stop_closes_request_saves_partial_and_retry_keeps_one_session(streaming_service):
    service, llm = streaming_service
    events = service.stream(user_id="u", session_id="s", message="尺码怎么选")
    await asyncio.wait_for(_first_token(events), 3)
    await events.aclose()
    await events.aclose()
    await asyncio.wait_for(llm.request_closed.wait(), 3)
    assert not llm.evaluated
    saved = service.sessions.load("s", user_id="u")
    assert [t.role for t in saved] == ["user", "assistant"]
    assert saved[-1].content == "第一片"
    assert saved[-1].metadata["status"] == "stopped"
    assert len(service.sessions.list_sessions("u")) == 1
    assert len(service.cache) == 0
    llm.next_piece.set()
    retry = [e async for e in service.stream(user_id="u", session_id="s", message="尺码怎么选")]
    assert retry[-1]["event"] == "done"
    assert len(service.sessions.list_sessions("u")) == 1
    assert [t.role for t in service.sessions.load("s", user_id="u")] == ["user", "assistant"] * 2
    assert len(service.cache) == 1


@pytest.mark.asyncio
async def test_broken_stream_saves_partial_as_incomplete(streaming_service):
    service, llm = streaming_service
    events = service.stream(user_id="u", session_id="s", message="尺码怎么选")
    await asyncio.wait_for(_first_token(events), 3)
    llm.fail = True
    llm.next_piece.set()
    with pytest.raises(LLMError, match="中途失败"):
        await anext(events)
    assert not llm.evaluated
    saved = service.sessions.load("s", user_id="u")
    assert len(saved) == 2 and saved[-1].content == "第一片"
    assert saved[-1].metadata["status"] == "interrupted"
    assert len(service.cache) == 0


@pytest.mark.asyncio
async def test_stop_after_final_token_saves_without_claiming_validation_passed(streaming_service):
    service, llm = streaming_service
    events = service.stream(user_id="u", session_id="s", message="尺码怎么选")
    await asyncio.wait_for(_first_token(events), 3)
    llm.next_piece.set()
    assert json.loads((await anext(events))["data"])["text"] == "第二片"
    await events.aclose()
    await events.aclose()
    saved = service.sessions.load("s", user_id="u")
    assert len(saved) == 2 and saved[-1].content == "第一片第二片"
    assert saved[-1].metadata["status"] == "stopped"
    assert len(service.cache) == 0


@pytest.mark.asyncio
async def test_stop_before_done_never_starts_background_memory_request(streaming_service):
    from mewhelp.memory.store import MemoryStore

    service, llm = streaming_service
    service.memory = MemoryStore(service.settings)
    service.llm = llm
    entered, closed = asyncio.Event(), asyncio.Event()
    original_chat = llm.chat

    def chat(messages, **kwargs):
        if kwargs.get("task") == "memory":
            async def extract():
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    closed.set()
            return current_generation.get().run(extract())
        return original_chat(messages, **kwargs)

    llm.chat = chat
    events = service.stream(user_id="u", session_id="s", message="尺码怎么选")
    await asyncio.wait_for(_first_token(events), 3)
    assert not entered.is_set()
    await events.aclose()
    await service.drain()
    await asyncio.wait_for(llm.request_closed.wait(), 3)
    assert not entered.is_set() and not closed.is_set()
    assert llm.stream_calls == ["generate"]
    assert service.sessions.load("s", user_id="u")[-1].metadata["status"] == "stopped"
    assert service.memory.list("u") == []


@pytest.fixture(scope="module")
def deps():
    d = build_deps()
    d.retriever.load_from_dir(Path("data/knowledge"))
    return d


@pytest.fixture
def checkpointer(tmp_path):
    return make_sqlite_checkpointer(str(tmp_path / "ckpt.db"))


def test_interrupt_suspends_and_resume_continues(deps, checkpointer):
    """缺订单号 -> 图挂起；补充订单号 -> 从断点继续，给出确定答复。"""
    from langgraph.types import Command

    graph = build_graph(deps, checkpointer=checkpointer)
    cfg = {"configurable": {"thread_id": "t-resume-1"}}

    state = new_state(
        request_id="r1", session_id="s1", user_id="u1", question="我要退款", history=[]
    )
    out = graph.invoke(state, config=cfg)

    # 挂起了：interrupt 的载荷能在结果里看到
    assert "__interrupt__" in out, f"没有挂起，实际返回: {list(out.keys())}"
    payload = out["__interrupt__"][0].value
    assert "prompt" in payload
    assert payload["missing"] == ["order_id"]
    # 挂起时绝不能有工具调用 —— 没凑齐前置条件就动手是这套流程要防的事
    assert not out.get("tool_calls")

    # 用户补充订单号 -> 从断点继续
    resumed = graph.invoke(Command(resume="ORD-1001"), config=cfg)
    assert resumed["answer"]
    assert not resumed.get("interrupt_prompt")


def test_resume_actually_continues_not_restarts(deps, checkpointer):
    """恢复必须"接着跑"，不是从头再来。

    LangGraph 恢复时会**重放节点**，但被重放节点的副作用不该重复发生 ——
    这正是 checkpointer 与 interrupt 配合的语义。这里用节点重放次数来验证：
    如果整条流水线被重启，classify 会再跑一次。
    """
    from langgraph.types import Command

    calls = {"classify": 0}
    original = deps.classifier.classify

    def counting_classify(text):
        calls["classify"] += 1
        return original(text)

    deps.classifier.classify = counting_classify
    try:
        graph = build_graph(deps, checkpointer=checkpointer)
        cfg = {"configurable": {"thread_id": "t-resume-2"}}
        graph.invoke(
            new_state(request_id="r2", session_id="s2", user_id="u1", question="我要退款", history=[]),
            config=cfg,
        )
        after_interrupt = calls["classify"]
        graph.invoke(Command(resume="ORD-1001"), config=cfg)
        after_resume = calls["classify"]
    finally:
        deps.classifier.classify = original

    assert after_interrupt == 1, "第一次调用应该只分类一次"
    # 恢复时重放的是 high_risk_flow 节点，classify 不该被再跑一遍
    assert after_resume == after_interrupt, "恢复时重跑了分类节点，说明流程被整体重启了"


def test_threads_are_isolated(deps, checkpointer):
    """不同会话的状态必须互不干扰 —— 否则 A 的订单号会串到 B 的退款里。"""
    from langgraph.types import Command

    graph = build_graph(deps, checkpointer=checkpointer)
    cfg_a = {"configurable": {"thread_id": "t-a"}}
    cfg_b = {"configurable": {"thread_id": "t-b"}}

    graph.invoke(
        new_state(request_id="ra", session_id="sa", user_id="ua", question="我要退款", history=[]),
        config=cfg_a,
    )
    graph.invoke(
        new_state(request_id="rb", session_id="sb", user_id="ub", question="我要退款", history=[]),
        config=cfg_b,
    )

    out_a = graph.invoke(Command(resume="ORD-1001"), config=cfg_a)
    out_b = graph.invoke(Command(resume="ORD-1007"), config=cfg_b)

    assert out_a["answer"] != out_b["answer"] or "1001" in out_a["answer"] or "1007" in out_b["answer"]
    # 两个线程各自拿到了自己的订单号对应的结果
    ids_a = {c.args.get("order_id") for c in out_a.get("tool_calls", [])}
    ids_b = {c.args.get("order_id") for c in out_b.get("tool_calls", [])}
    assert ids_a == {"ORD-1001"}, f"A 线程拿到了 {ids_a}"
    assert ids_b == {"ORD-1007"}, f"B 线程拿到了 {ids_b}"


def test_no_checkpointer_degrades_to_clarification(deps):
    """没挂 checkpointer 时，interrupt() 会抛异常把请求打成 500。

    所以这条路径必须显式降级：把追问当作答复返回。用户虽然没被满足，
    但清楚知道下一步给什么 —— 远好过一个 500。
    """
    graph = build_graph(deps, checkpointer=None)
    out = graph.invoke(
        new_state(request_id="r3", session_id="s3", user_id="u1", question="我要退款", history=[])
    )
    assert out["answer"], "必须给出追问话术"
    assert "订单号" in out["answer"]
    assert out["interrupt_prompt"]
    assert "__interrupt__" not in out, "没挂 checkpointer 不该产生挂起"


def test_checkpointer_persists_across_graph_instances(deps, checkpointer):
    """状态落盘后，换一个 graph 实例也能接着恢复。

    这条能力是把"多轮对话"做成可持久化服务的前提 ——
    进程重启不该丢掉等用户补充的那个流程。
    """
    from langgraph.types import Command

    graph1 = build_graph(deps, checkpointer=checkpointer)
    cfg = {"configurable": {"thread_id": "t-persist"}}
    graph1.invoke(
        new_state(request_id="r4", session_id="s4", user_id="u1", question="改地址", history=[]),
        config=cfg,
    )

    # 重新编译一个图（模拟进程重启后重新装配）
    graph2 = build_graph(deps, checkpointer=checkpointer)
    out = graph2.invoke(Command(resume="ORD-1001，换成上海市浦东新区"), config=cfg)
    assert out["answer"]
    assert "地址" in out["answer"] or "更新" in out["answer"]
