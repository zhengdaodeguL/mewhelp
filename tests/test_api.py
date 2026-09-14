"""服务层测试：FastAPI 路由 + SSE 事件流。

这一批测试盯的是"**契约**"，不是"实现"：

  * SSE 的事件名和顺序是前端的解析依据，改了就是破坏性变更；
  * `handoff` / `refusal_reason` 是"答不上来"这条业务链路的对外口径；
  * 空输入、超长输入这类脏数据必须被降级成一句人话，而不是 500。

所有用例都跑在临时 sqlite 上（`MEWHELP_DATABASE_URL` 指向 tmp_path），
不碰仓库里的 mewhelp.db —— 否则"跑一次测试，演示数据就脏了"。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mewhelp.api import create_app
from mewhelp.api.app import INDEX_HTML
from mewhelp.api.service import MAX_MESSAGE_CHARS, chunk_text, stream_events
from mewhelp.config import Settings
from mewhelp.schemas import ConversationResult
from mewhelp.workflow.graph import build_deps


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    """整个模块共用一个应用实例。

    建库 + 灌 70 条 chunk 的索引只做一次（`load_from_dir` 约 0.5s），
    所以用 module 作用域；代价是飞轮/会话状态在用例之间累积，
    需要干净状态的用例自己造数据（见 `test_flywheel_pending_lists_refused`）。
    """
    db_dir = tmp_path_factory.mktemp("api")
    settings = Settings(
        MEWHELP_DATABASE_URL=f"sqlite:///{(db_dir / 'test.db').as_posix()}"
    )
    return create_app(build_deps(settings=settings))


@pytest.fixture(scope="module")
def client(app):
    # 必须用 with：TestClient 只有在上下文管理器里才会跑 lifespan，
    # 而建表和灌知识库都在 lifespan 里 —— 不进去就会问什么都查不到。
    with TestClient(app) as c:
        yield c


# ============================================================
# 辅助：SSE 解析
# ============================================================
def parse_sse(text: str) -> list[tuple[str, dict]]:
    """把 SSE 响应体解析成 [(event, data), ...]。

    先统一换行：sse-starlette 默认用 `\\r\\n` 分行，而事件之间是空行，
    不归一化的话 `\\n\\n` 永远匹配不上（`\\r\\n\\r\\n` 里没有连续的 `\\n`），
    解析结果会是"一个事件都没收到"这种最容易误判成后端没推的假象。
    """
    events: list[tuple[str, dict]] = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        name, payload = None, ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                payload += line[len("data:") :].strip()
        if name and payload:
            events.append((name, json.loads(payload)))
    return events


def events_of(events, name):
    return [d for n, d in events if n == name]


def chat(client, message: str, session_id: str = "s-test", user_id: str = "u-test") -> dict:
    resp = client.post(
        "/api/chat/sync",
        json={"user_id": user_id, "session_id": session_id, "message": message},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# ============================================================
# 1. 健康检查
# ============================================================
def test_health_reports_component_modes(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "ok"
    components = body["components"]
    # 这三个模式是"当前是不是纯离线"的全部依据，一个都不能少
    for key in ("llm", "embedding", "vector_store"):
        assert key in components, f"健康检查缺少组件模式：{key}"
        assert isinstance(components[key], str) and components[key]
    assert components["llm"] == "mock"
    assert body["fully_mocked"] is True
    assert body["db_ready"] is True

    # 启动时灌进知识库了 —— 没有 chunk 的检索器会让所有知识类问题都被拒答
    assert body["kb_chunks"] > 0
    assert body["retriever"]["chunks"] == body["kb_chunks"]


# ============================================================
# 2. 非流式问答
# ============================================================
def test_chat_sync_returns_full_contract(client):
    body = chat(client, "生鲜类可以退款吗", session_id="s-sync-knowledge")

    for key in ("request_id", "session_id", "question", "answer", "intent", "exit", "path"):
        assert key in body, f"返回体缺少字段：{key}"
    assert body["intent"] == "policy_faq"
    assert body["exit"] == "knowledge"
    assert body["path"][:3] == ["resolve", "classify", "retrieve"]
    assert body["answer"].strip(), "答复不能是空的"
    assert body["handoff"] is False
    assert body["evidences"], "走知识出口的问答应该有检索证据"
    assert body["usage"]["total_tokens"] >= 0


def test_chat_sync_business_tool_goes_through_agent(client):
    """查订单走 Agent 分支，不是检索分支 —— 这是分流最直观的证据。"""
    body = chat(client, "帮我查下订单 ORD-1002 到哪了", session_id="s-sync-tool")
    assert body["intent"] == "order_query"
    assert body["exit"] == "business_tool"
    assert "agent" in body["path"]
    assert "retrieve" not in body["path"]
    assert body["tool_calls"], "业务工具出口必须留下工具调用记录"


# ============================================================
# 3. SSE 事件流
# ============================================================
def test_chat_sse_event_order_and_tokens(client):
    resp = client.post(
        "/api/chat",
        json={"user_id": "u-sse", "session_id": "s-sse", "message": "尺码怎么选"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(resp.text)
    names = [n for n, _ in events]

    # 顺序是硬约束：meta → evidence → tool* → token* → done
    assert names[0] == "meta", f"第一个事件必须是 meta，实际 {names[:3]}"
    assert names[-1] == "done", f"最后一个事件必须是 done，实际 {names[-3:]}"
    assert "error" not in names
    assert names.index("meta") < names.index("evidence") < names.index("done")
    assert names.count("meta") == 1 and names.count("done") == 1

    tokens = events_of(events, "token")
    assert len(tokens) >= 1, "至少要推一个 token —— 前端靠它逐字渲染"
    # 逐字拼回去必须和 done 里的完整答复完全一致（推流不能丢字或重复）
    done = events_of(events, "done")[0]
    assert "".join(t["text"] for t in tokens) == done["answer"]

    meta = events_of(events, "meta")[0]
    assert meta["request_id"] and meta["intent"] == "product_faq"
    assert meta["exit"] == "knowledge"
    assert meta["path"][0] == "resolve"
    assert meta["resolved_question"]

    evidence = events_of(events, "evidence")[0]
    assert evidence["chunks"], "知识出口的 SSE 必须带检索证据"
    for c in evidence["chunks"]:
        # snippet 是给用户看的原文摘录（工作区的"依据"卡片）；
        # chunk_id / score 是我们的口径，塞在事件里给开发者排查用，界面不展示
        assert set(c) == {"chunk_id", "title", "score", "snippet"}


def test_chat_sse_tool_events(client):
    """业务工具出口：tool 事件出现在 token 之前。"""
    resp = client.post(
        "/api/chat",
        json={"user_id": "u-sse2", "session_id": "s-sse2", "message": "帮我查下订单 ORD-1002 到哪了"},
    )
    events = parse_sse(resp.text)
    names = [n for n, _ in events]
    tools = events_of(events, "tool")
    assert tools, "应该至少有一次工具调用"
    # 键集就是契约。成功的调用多带一份 result —— 右侧工作区靠它渲染订单卡；
    # 失败的没有可展示的内容，就不推（否则工作区会多一张空卡片）。
    assert all(set(t) == {"name", "ok", "error", "result"} for t in tools if t["ok"])
    assert all(set(t) == {"name", "ok", "error"} for t in tools if not t["ok"])
    assert any(t["name"] == "track_logistics" for t in tools)
    assert names.index("tool") < names.index("token")


# ============================================================
# 4. 拒答路径（知识库外 → 转人工 + 落飞轮）
# ============================================================
def test_out_of_kb_question_hands_off(client):
    resp = client.post(
        "/api/chat",
        json={"user_id": "u-refuse", "session_id": "s-refuse", "message": "你们卖不卖火箭"},
    )
    events = parse_sse(resp.text)
    done = events_of(events, "done")[0]

    assert done["handoff"] is True, "知识库外的问题必须转人工"
    assert done["refusal_reason"], "转人工必须给出拒答原因（飞轮分诊要用）"
    assert done["answer"].strip()
    # 拒答走的是 retrieve → gate → handoff → flywheel，不是直接放弃
    names = [n for n, _ in events]
    assert names[-1] == "done"


def test_flywheel_pending_lists_refused_question(client):
    """拒答的问题必须落进问题池，且带着当时的检索快照。"""
    question = "你们卖不卖火箭"
    chat(client, question, session_id="s-refuse-2")

    resp = client.get("/api/flywheel/pending")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] >= 1

    hit = next((i for i in body["items"] if i["question"] == question), None)
    assert hit is not None, f"问题池里没找到拒答的题：{[i['question'] for i in body['items']]}"
    assert hit["status"] == "pending"
    assert hit["source"] in ("refused", "self_eval", "thumbs_down")
    assert hit["triage"], "必须给出分诊结论（该补文档还是该调检索）"
    assert hit["triage_label"]
    # 快照是飞轮最值钱的字段 —— 没有它，人工没法判断是真缺知识还是没检索到
    assert hit["snapshot"] is not None
    assert "top_score" in hit["snapshot"]


# ============================================================
# 5. 高风险中断路径
# ============================================================
def test_high_risk_missing_slot_asks_for_order_id(client):
    resp = client.post(
        "/api/chat",
        json={"user_id": "u-risk", "session_id": "s-risk", "message": "我要退款"},
    )
    events = parse_sse(resp.text)
    meta = events_of(events, "meta")[0]
    done = events_of(events, "done")[0]

    assert meta["intent"] == "refund_request"
    assert meta["exit"] == "high_risk"
    assert meta["path"] == ["resolve", "classify", "high_risk_flow", "generate"]
    # 缺订单号时必须追问，而不是编一个订单号继续（这条是业务铁律）
    assert "订单号" in done["answer"]
    assert "ORD-" in done["answer"]
    assert done["handoff"] is False, "追问是正常答复，不是转人工"


def test_high_risk_with_slot_actually_calls_tool(client):
    """槽位齐备时高风险子流程真的动手，并且答复里带着订单号。"""
    resp = client.post(
        "/api/chat/sync",
        json={"user_id": "u-risk2", "session_id": "s-risk2", "message": "帮我取消订单 ORD-1001"},
    )
    body = resp.json()
    assert body["intent"] == "cancel_order"
    assert body["exit"] == "high_risk"
    assert body["tool_calls"], "槽位齐备却没有调工具"
    assert body["tool_calls"][0]["name"] == "cancel_order"
    assert "ORD-1001" in body["answer"]


# ============================================================
# 6. 反馈
# ============================================================
def test_feedback_is_persisted(client):
    body = chat(client, "尺码怎么选", session_id="s-feedback")
    resp = client.post(
        "/api/feedback",
        json={
            "session_id": "s-feedback",
            "request_id": body["request_id"],
            "thumbs_down": True,
            "comment": "答得不对",
        },
    )
    assert resp.status_code == 200
    out = resp.json()
    assert out["ok"] is True
    assert out["feedback_id"] is not None
    # 点踩不只是记一笔：它是最强信号，应该带着当时的现场落进问题池
    assert out["flywheel"], "点踩应该落进问题池"


def test_feedback_on_unknown_request_does_not_fail(client):
    """拿不到当时现场时可以少落一条（不硬造），但接口本身不能失败。"""
    resp = client.post(
        "/api/feedback",
        json={"session_id": "s-none", "request_id": "req-不存在", "thumbs_down": True, "comment": ""},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


# ============================================================
# 7. 飞轮审核
# ============================================================
def test_flywheel_review_approves_and_prepares_backfill(client):
    chat(client, "你们卖不卖服务器机柜", session_id="s-review")
    items = client.get("/api/flywheel/pending").json()["items"]
    target = next(i for i in items if i["question"] == "你们卖不卖服务器机柜")

    resp = client.post(
        "/api/flywheel/review",
        json={
            "record_id": target["record_id"],
            "approved": True,
            "reviewer": "ops-1",
            "answer": "抱歉，平台暂不销售服务器机柜。",
        },
    )
    assert resp.status_code == 200
    out = resp.json()
    assert out["ok"] is True
    assert out["record"]["status"] == "approved"
    assert out["record"]["reviewer"] == "ops-1"
    # 回灌所需的 chunk 必须当场准备好（但不由这个接口直接写向量库）
    assert out["backfill_chunk_id"], "审核通过却没给出可回灌的 chunk"


def test_flywheel_review_unknown_record_does_not_500(client):
    resp = client.post(
        "/api/flywheel/review",
        json={"record_id": "fw_不存在", "approved": False, "reviewer": "ops"},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


# ============================================================
# 8. 脏输入
# ============================================================
@pytest.mark.parametrize(
    "message",
    [
        "",                       # 空消息
        "   ",                    # 只有空白
        "？" * 1,                 # 极短
        "很长的问题" * 500,        # 超过 2000 字上限
    ],
)
def test_bad_input_never_returns_500(client, message):
    resp = client.post(
        "/api/chat/sync",
        json={"user_id": "u-bad", "session_id": "s-bad", "message": message},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("answer", "").strip(), "任何输入都要给一句人话"


def test_bad_input_sse_still_streams_meta_and_done(client):
    resp = client.post(
        "/api/chat",
        json={"user_id": "u-bad", "session_id": "s-bad2", "message": ""},
    )
    assert resp.status_code == 200
    events = parse_sse(resp.text)
    names = [n for n, _ in events]
    assert names[0] == "meta" and names[-1] == "done"
    assert "error" not in names


def test_missing_fields_use_defaults(client):
    """少传 user_id / session_id 不该 422 —— 演示页和 curl 都不爱写全。"""
    resp = client.post("/api/chat/sync", json={"message": "尺码怎么选"})
    assert resp.status_code == 200


def test_oversized_message_is_capped_not_forwarded(client):
    """超长消息只回一句"太长了"，而不是把它整段丢进图（那才是真正的风险）。"""
    body = chat(client, "退款" * (MAX_MESSAGE_CHARS // 2 + 100), session_id="s-too-long")
    assert "太长" in body["answer"]
    assert body["path"] == [], "护栏路径不该伪造一条分流路径"


# ============================================================
# 9. trace / 演示页
# ============================================================
def test_trace_returns_call_tree(client):
    body = chat(client, "生鲜类可以退款吗", session_id="s-trace")
    resp = client.get(f"/api/trace/{body['request_id']}")
    assert resp.status_code == 200
    trace = resp.json()
    assert trace["count"] >= 4
    nodes = [s["node"] for s in trace["steps"]]
    assert nodes[0] == "resolve" and "classify" in nodes and "gate" in nodes


def test_trace_unknown_request_returns_empty(client):
    resp = client.get("/api/trace/req-根本不存在")
    assert resp.status_code == 200
    assert resp.json() == {"request_id": "req-根本不存在", "count": 0, "steps": []}


# ============================================================
# 9b. 订单快捷选择
# ============================================================
def test_orders_endpoint_powers_the_quick_picker(client):
    """订单快捷选择的数据源 —— 订单号是用户最记不住的东西，不该让他手打。"""
    resp = client.get("/api/orders")
    assert resp.status_code == 200
    body = resp.json()

    assert body["count"] > 0, "没有可选订单，选择器就是个摆设"
    assert body["demo"] is True, "没传 user_id 时必须标明这是演示数据"

    first = body["orders"][0]
    assert first["order_id"].startswith("ORD-")
    # 选择器只需要够用户认出"是哪一单"：状态 + 首件商品 + 金额
    assert first["status_label"]
    assert first["title"]
    assert first["total_amount"] > 0
    assert first["item_count"] >= 1


def test_orders_endpoint_scopes_to_the_given_user(client):
    """指定用户时只能看到他自己的单 —— 越权读单是这类接口最典型的洞。"""
    from mewhelp.tools.business import load_order_rows

    expected = {o["order_id"] for o in load_order_rows() if o["user_id"] == "U1002"}
    assert expected, "夹具里应该有 U1002 的订单"

    body = client.get("/api/orders", params={"user_id": "U1002"}).json()
    assert body["demo"] is False
    got = {o["order_id"] for o in body["orders"]}
    assert got and got <= expected, f"读到了别人的单：{got - expected}"

    # 查无此人时返回空列表，而不是回落到演示用户 —— 静默降级会让人以为
    # "这就是我的单"，比报错更危险
    assert client.get("/api/orders", params={"user_id": "U9999"}).json()["count"] == 0


def _asset_paths(html: str) -> list[str]:
    """从入口页里取出它引用的静态资源路径（React 构建产物）。"""
    return sorted(set(re.findall(r'(?:src|href)="(/assets/[^"]+)"', html)))


def _bundle(client) -> str:
    """把前端 JS 产物拼起来 —— 用于静态检查"页面里到底写了什么"。"""
    return "".join(p for p in (client.get(x).text for x in _asset_paths(client.get("/").text) if x.endswith(".js")))


def test_index_page_is_served_offline(client):
    """入口页 + 它引用的静态资源，都必须由服务端自己发得出来。

    前端换成 React 之后，「断网也能打开」这条约束没有变，反而更要盯紧：
    构建产物漏提交、或者忘了挂载 /assets，页面会返回 200 但白屏 ——
    而后端用例照样全绿。
    """
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")

    html = resp.text
    assert 'id="root"' in html, "React 挂载点丢了"

    assets = _asset_paths(html)
    assert any(a.endswith(".js") for a in assets), "入口页没有引用 JS 产物"
    assert any(a.endswith(".css") for a in assets), "入口页没有引用 CSS 产物"

    # 不引任何 CDN —— 断网也得能打开
    assert "http://cdn" not in html and "https://cdn" not in html
    assert "unpkg.com" not in html and "jsdelivr" not in html

    # 引用到的资源必须真的发得出来（漏挂载 /assets 是最常见的一种失误）
    for path in assets:
        asset = client.get(path)
        assert asset.status_code == 200, f"{path} 取不到，页面会白屏"
        assert asset.content, f"{path} 是空的"


def test_index_page_wires_to_real_api_routes(client):
    """前端调用到的每个接口，后端必须真的注册了。

    这条替代了旧版"页面 id 与 JS 选择器一一对应"的检查。React 里没有手写
    选择器了，但同类风险还在，甚至更隐蔽：前端写 `/api/chat` 而后端叫
    `/api/chat/sync`，页面照样渲染得漂漂亮亮，只在用户点发送时才静默失败。
    """
    bundle = _bundle(client)
    assert bundle, "没取到 JS 产物"

    # 产物里是 fetch(`${base}/api/chat`) 这种形态，所以直接扫 /api/ 开头的路径
    called = set(re.findall(r"/api/[a-zA-Z0-9/_-]+", bundle))
    assert called, "产物里没有任何 /api/ 调用，客户端是不是没被打进去？"

    registered = {getattr(r, "path", "") for r in client.app.routes}

    def is_served(path: str) -> bool:
        """这个路径在后端有没有对应的路由。

        带路径参数的调用在产物里长这样：`` `/api/sessions/${id}` `` ——
        扫描只能扫到 `/api/sessions/`（模板表达式不在字符集里）。所以
        **以斜杠结尾**的路径改为按前缀匹配，对应 FastAPI 的 `/api/sessions/{id}`。

        只在结尾是斜杠时放宽：字面量调用永远不会以斜杠结尾，所以
        "前端调了 /api/chat 而后端只有 /api/chat/sync"这种真错误依然会被抓住。
        """
        if path in registered:
            return True
        if not path.endswith("/"):
            return False
        stem = path.rstrip("/")
        return any(r.startswith(stem + "/") for r in registered)

    missing = sorted(p for p in called if not is_served(p))
    assert not missing, f"前端调了后端并不存在的接口：{missing}"


def test_frontend_does_not_render_internal_fields(client):
    """用户界面上不能出现内部信息。

    意图、节点路径、证据块、工具名、置信度这些都是**开发者视角**的东西。
    后端照常推它们（开发者走 /docs 和 /api/trace/{request_id} 看），
    但前端产物里不该有把它们渲染出去的字样 —— 这条就是那道界线本身。
    """
    bundle = _bundle(client)
    for leaked in ("refusal_reason", "latency_ms", "置信", "意图", "证据", "节点路径"):
        assert leaked not in bundle, f"前端产物里出现了内部字段：{leaked}"


# ============================================================
# 10. 会话历史（多轮）
# ============================================================
def test_session_history_is_persisted_and_reused(client):
    """第二轮要能看见第一轮 —— 历史没落库，多轮指代就永远是坏的。"""
    chat(client, "我要退款", session_id="s-multi")
    chat(client, "订单号是 ORD-1002", session_id="s-multi")

    # 从库里直接读回来核对（不信任内存态）
    store = client.app.state.service.sessions
    history = store.load("s-multi")
    assert [t.role for t in history] == ["user", "assistant", "user", "assistant"]
    assert history[0].content == "我要退款"
    assert history[1].metadata["intent"] == "refund_request"
    assert history[2].content == "订单号是 ORD-1002"


# ============================================================
# 11. 纯逻辑单测（不起 HTTP）
# ============================================================
def test_chunk_text_covers_answer_exactly():
    """切块必须不丢字、不重复，且块长在 2~4 之间。"""
    for text in ["", "短", "0123456789", "退款政策：生鲜类不适用七天无理由退款，请知悉。"]:
        pieces = list(chunk_text(text))
        assert "".join(pieces) == text
        assert all(1 <= len(p) <= 4 for p in pieces)
    assert list(chunk_text("")) == []


def test_chunk_text_is_deterministic():
    text = "同一句话每次切出来的块必须完全一致，否则前端快照和回放对不上。"
    assert list(chunk_text(text)) == list(chunk_text(text))


@pytest.mark.asyncio
async def test_stream_events_on_short_answer():
    """事件流的顺序与内容（sleep=0，不真的等）。"""
    result = ConversationResult(
        request_id="req-x",
        session_id="s-x",
        user_id="u-x",
        question="问",
        answer="答",
        path=["resolve", "classify"],
    )
    events = [e async for e in stream_events(result, sleep=0)]
    assert [e["event"] for e in events] == ["meta", "evidence", "token", "done"]
    assert json.loads(events[2]["data"]) == {"text": "答"}
    assert json.loads(events[3]["data"])["answer"] == "答"
    # 中文在流里保持可读（ensure_ascii=False），调试时肉眼能看懂
    assert "答" in events[3]["data"]


# ============================================================
# 11b. 工作区依赖的字段：工具结果 + 依据摘录
# ============================================================
def _sse(client, message: str, session_id: str = "s-ws") -> list[tuple[str, dict]]:
    resp = client.post(
        "/api/chat", json={"user_id": "u-ws", "session_id": session_id, "message": message}
    )
    assert resp.status_code == 200
    return parse_sse(resp.text)


def test_tool_event_carries_result_for_workspace(client):
    """工具结果要一并推给前端 —— 右侧工作区靠它渲染订单卡、物流轨迹。

    旧版只推 {name, ok, error}（那时界面只有一串对话）。工作区需要真实数据，
    这条钉住它别再被顺手删掉：删了界面不会报错，只会永远空着。
    """
    tools = events_of(_sse(client, "帮我查下订单 ORD-1001 的状态"), "tool")
    assert tools, "查订单应该产生 tool 事件"

    entry = next(t for t in tools if t["ok"])
    assert "result" in entry, "成功的结果没推给前端，工作区会是空的"
    assert entry["result"]["order_id"] == "ORD-1001"
    assert entry["result"]["items"], "订单明细丢了，订单卡渲染不出商品"


def test_failed_tool_does_not_push_a_result(client):
    """失败的结果没有可展示的内容，不该往工作区塞。

    界面上也做了同样的过滤，但契约层面先拦一道：两边各守一半，
    漏一边都只是"多一张空卡片"这种不痛不痒、极难发现的错。
    """
    tools = events_of(_sse(client, "帮我查下订单 ORD-9999 的状态"), "tool")
    assert tools, "查不存在的订单也该留下调用记录"
    assert all(not t["ok"] for t in tools)
    assert all("result" not in t for t in tools)


def test_evidence_event_carries_snippet(client):
    """依据要带原文摘录 —— 只给标题的话工作区那一栏等于没写。"""
    payload = events_of(_sse(client, "生鲜类可以退款吗"), "evidence")[0]
    chunks = payload["chunks"]
    assert chunks
    assert chunks[0]["snippet"].strip(), "依据卡片没有正文可展示"
    # 摘录要截断，不能把整篇塞进事件里
    assert len(chunks[0]["snippet"]) <= 141


def test_evidence_snippet_is_clean_markdown(client):
    """摘录必须是**给人看**的，不能把 Markdown 记号原样丢出去。

    知识库是 Markdown 写的：`**加粗**` 的星号、表格分隔行（`| - | - |`）
    都是写给渲染器的记号。整段截断搬上卡片，用户看到的就是一串符号。
    表格片段则只取表头当摘要 —— 把整张表拍在卡片上没人看得下去。
    """
    payload = events_of(_sse(client, "生鲜类可以退款吗"), "evidence")[0]
    for c in payload["chunks"]:
        snippet = c["snippet"]
        assert "**" not in snippet, f"加粗记号漏到界面上了：{snippet[:60]}"
        assert "|" not in snippet, f"表格行漏到界面上了：{snippet[:60]}"
        assert not snippet.strip().startswith("#")


# ============================================================
# 12. 前后端一致性：用 node 跑前端真实的 SSE 解析代码
# ============================================================
#: 前端解析器所在文件。它刻意只放纯函数（不碰 DOM、不碰 fetch），
#: 因此能被 node 直接加载 —— Node 24 原生支持擦除 TS 类型标注。
#: INDEX_HTML = <repo>/src/mewhelp/api/static/index.html，往上 4 层即仓库根。
_REPO_ROOT = INDEX_HTML.parents[4]
FRONTEND_SSE_PARSER = _REPO_ROOT / "web" / "src" / "api" / "sse.ts"
#: 工作区卡片的构建器（同样是纯函数，同样能被 node 直接跑）
FRONTEND_WORKSPACE = _REPO_ROOT / "web" / "src" / "workspace.ts"

_NODE_HARNESS = """
import { normalizeSse, drainEvents } from './sse.ts';
import fs from 'node:fs';

// 把真实的 SSE 响应体按很碎的字节边界切开再喂进去 —— 为的是逼出
// "一个事件被 TCP 切成几段"和"多字节字符被切开"这两种真实场景。
const raw = fs.readFileSync(process.argv[2]);
const size = Number(process.argv[3] || 7);
const dec = new TextDecoder('utf-8');
let buf = '', events = [];
for (let i = 0; i < raw.length; i += size) {
  buf = normalizeSse(buf + dec.decode(raw.subarray(i, i + size), { stream: true }));
  const d = drainEvents(buf);
  buf = d.rest;
  events.push(...d.events);
}
buf = buf + dec.decode();
console.log(JSON.stringify({ events, rest: buf }));
"""


def test_frontend_sse_parser_matches_server_bytes(client, tmp_path):
    """服务端推的字节 → 前端解析器 → 事件序列必须和 Python 侧一致。

    这条用例守的是**跨语言的契约**：事件名、事件顺序、data 结构一旦改动，
    后端测试全绿而页面悄悄白屏，是最容易漏掉的一类回归。

    跑的是 `web/src/api/sse.ts` 里那份**真实代码**，不是在 Python 里重写
    一遍解析逻辑 —— 重写的版本只能证明"我理解得对"，证明不了"前端写得对"。
    """
    node = shutil.which("node")
    if not node:
        pytest.skip("环境里没有 node，跳过前端解析器校验")
    if not FRONTEND_SSE_PARSER.is_file():
        pytest.fail(f"前端解析器不见了：{FRONTEND_SSE_PARSER}")

    # 拷到临时目录用相对路径 import —— Windows 上绝对路径（E:/…）会被
    # Node 当成 URL scheme 而报 ERR_UNSUPPORTED_ESM_URL_SCHEME
    (tmp_path / "sse.ts").write_text(
        FRONTEND_SSE_PARSER.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "harness.mjs").write_text(_NODE_HARNESS, encoding="utf-8")

    resp = client.post(
        "/api/chat",
        json={"user_id": "u-js", "session_id": "s-js", "message": "帮我查下订单 ORD-1002 到哪了"},
    )
    assert resp.status_code == 200
    raw_bytes = resp.content
    assert b"event: meta" in raw_bytes
    (tmp_path / "stream.bin").write_bytes(raw_bytes)

    proc = subprocess.run(
        [node, str(tmp_path / "harness.mjs"), str(tmp_path / "stream.bin"), "7"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)

    assert got["rest"] == "", "有半截事件没被消费掉 —— 前端会永远等下去"
    js_events = [(e["event"], e["data"]) for e in got["events"]]
    py_events = parse_sse(resp.text)
    assert [n for n, _ in js_events] == [n for n, _ in py_events]

    # 逐字拼回去也必须一致：这是"流式推送不丢字"的最终判据
    js_answer = "".join(d["text"] for n, d in js_events if n == "token")
    js_done = [d for n, d in js_events if n == "done"][0]
    assert js_answer == js_done["answer"]
    assert len([n for n, _ in js_events if n == "token"]) > 1
    # handoff 是前端唯一消费的 done 字段：它决定要不要显示"已转人工"提示条
    assert "handoff" in js_done


# ============================================================
# 13. 工作区卡片：用后端真实推的数据验证前端构建器
# ============================================================
_WORKSPACE_HARNESS = """
import { cardFromToolResult, cardsFromEvidence, mergeCards } from './workspace.ts';
import fs from 'node:fs';

const payload = JSON.parse(fs.readFileSync(process.argv[2], 'utf-8'));
const toolCards = payload.tools.map((t) => cardFromToolResult(t.name, t.result));
// 同一个订单问两遍，工作区不该堆出两张一样的卡
const merged = mergeCards(mergeCards([], toolCards.filter(Boolean)), toolCards.filter(Boolean));
console.log(JSON.stringify({
  toolCards,
  sourceCards: cardsFromEvidence(payload.chunks),
  mergedCount: merged.length,
}));
"""


def test_frontend_builds_workspace_cards_from_real_payloads(client, tmp_path):
    """后端真实推的数据 → 前端卡片构建器 → 必须认得出是什么。

    和 SSE 解析那条同理，这是**跨语言契约**。工作区最怕的不是报错，而是
    **静默为空**：后端改了个字段名，卡片构建器认不出来，界面不报错、
    只是右边那一栏永远空着 —— 这种错能活很久。
    """
    node = shutil.which("node")
    if not node:
        pytest.skip("环境里没有 node，跳过前端卡片构建校验")
    if not FRONTEND_WORKSPACE.is_file():
        pytest.fail(f"前端卡片构建器不见了：{FRONTEND_WORKSPACE}")

    (tmp_path / "workspace.ts").write_text(
        FRONTEND_WORKSPACE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "harness.mjs").write_text(_WORKSPACE_HARNESS, encoding="utf-8")

    # 一次问出「订单」和「物流」两种工具结果，再问一次知识库
    tools = events_of(_sse(client, "帮我查下订单 ORD-1001 的状态", "s-ws1"), "tool")
    tools += events_of(_sse(client, "帮我查下订单 ORD-1002 的物流", "s-ws2"), "tool")
    evidence = events_of(_sse(client, "生鲜类可以退款吗", "s-ws3"), "evidence")

    payload = {"tools": tools, "chunks": evidence[0]["chunks"] if evidence else []}
    (tmp_path / "payload.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    proc = subprocess.run(
        [node, str(tmp_path / "harness.mjs"), str(tmp_path / "payload.json")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)

    cards = [c for c in got["toolCards"] if c]
    kinds = [c["kind"] for c in cards]
    assert "order" in kinds, f"订单结果没被识别成订单卡：{got['toolCards']}"
    assert "logistics" in kinds, f"物流结果没被识别成物流卡：{got['toolCards']}"

    order = next(c for c in cards if c["kind"] == "order")
    assert order["orderId"] == "ORD-1001"
    assert order["items"], "订单卡里没有商品明细"
    assert order["statusLabel"] != "—", "订单卡没拿到可读的状态文案"

    logi = next(c for c in cards if c["kind"] == "logistics")
    assert logi["orderId"] == "ORD-1002"

    sources = got["sourceCards"]
    assert sources, "依据没被识别成卡片"
    assert sources[0]["title"] and sources[0]["snippet"]

    # 卡片 id 稳定 —— 否则同一单问两遍会堆出一摞重复卡片
    assert got["mergedCount"] == len(cards), "同一批卡片合并后不该变多或变少"


# ============================================================
# 14. 导出：纯前端逻辑，但错了用户拿到的就是一份残缺的记录
# ============================================================
FRONTEND_EXPORT = _REPO_ROOT / "web" / "src" / "session-export.ts"

_EXPORT_HARNESS = """
import { sessionToMarkdown, exportFileName } from './session-export.ts';
import fs from 'node:fs';

const input = JSON.parse(fs.readFileSync(process.argv[2], 'utf-8'));
console.log(JSON.stringify({
  md: sessionToMarkdown(input.messages, input.options),
  name: exportFileName(input.options.title, input.options.exportedAt),
}));
"""


def test_frontend_export_keeps_the_conversation_and_drops_state_notices(tmp_path):
    """导出的是**对话**，不是界面状态。

    `notice` 类消息是界面上的提示条（"回答中断了，请重试"），它根本不是
    客服说过的话。跟着导出会让人拿着一份记录去争论一句没人说过的话 ——
    这类错误没有任何自动检查会拦住，所以在这里钉一条。

    另外两件事一起验：标题要进文件、随附信息要能带上（工作区里的订单/物流）。
    """
    node = shutil.which("node")
    if not node:
        pytest.skip("环境里没有 node，跳过导出格式校验")
    if not FRONTEND_EXPORT.is_file():
        pytest.fail(f"导出模块不见了：{FRONTEND_EXPORT}")

    (tmp_path / "session-export.ts").write_text(
        FRONTEND_EXPORT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "harness.mjs").write_text(_EXPORT_HARNESS, encoding="utf-8")

    payload = {
        "messages": [
            {"id": "g", "role": "assistant", "kind": "normal", "content": "你好，我是客服助手"},
            {"id": "u1", "role": "user", "kind": "normal", "content": "帮我查下订单 ORD-1002"},
            {
                "id": "a1",
                "role": "assistant",
                "kind": "normal",
                "content": "订单 ORD-1002 当前状态：待发货。",
                "turnId": "req_abc",
            },
            {"id": "n1", "role": "assistant", "kind": "notice", "content": "回答中断了，请重试。"},
            {"id": "a2", "role": "assistant", "kind": "normal", "content": "   "},
        ],
        "options": {
            "title": "查订单 / 物流",
            "exportedAt": "2026-09-14T08:30:00+00:00",
            "notes": "订单 ORD-1002（待发货）\n  合计 ¥168",
        },
    }
    (tmp_path / "payload.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    proc = subprocess.run(
        [node, str(tmp_path / "harness.mjs"), str(tmp_path / "payload.json")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)
    md = got["md"]

    assert "# MewHelp 客服对话记录" in md
    assert "查订单 / 物流" in md
    assert "帮我查下订单 ORD-1002" in md
    assert "**客服：**" in md and "**我：**" in md
    assert "回答中断了" not in md, "界面提示条混进了对话记录"
    assert "本次涉及的信息" in md and "¥168" in md
    # 空内容不产生空段落
    assert "\n\n\n\n" not in md
    # 文件名要能直接落盘：标题里的斜杠不能变成目录分隔符
    assert "/" not in got["name"] and "\\" not in got["name"]
    assert got["name"].startswith("mewhelp-2026-09-14-") and got["name"].endswith(".md")
