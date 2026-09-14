"""长期记忆 + 会话历史：契约测试。

这一批盯的是三件事：

  * **记忆真的会让服务"更懂这个用户"**，而不只是往库里写一行字。
    判据是那个跨会话的场景：上一通对话问过 ORD-1002，新开一个会话直接问
    "我的订单到哪了"，系统要能自己把单号补上并真的去查。
  * **两张表的生命周期是分开的**：删掉会话不该顺手删掉用户档案。
  * **历史恢复回来的东西是用户视角的**：恢复对话可以，把意图/节点路径
    一起恢复回界面上不行。

和 `test_api.py` 一样，所有用例跑在临时 sqlite 上，不碰仓库里的 mewhelp.db。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text

from mewhelp.api import create_app
from mewhelp.config import Settings
from mewhelp.llm import MockLLMClient
from mewhelp.memory import (
    KIND_MANUAL,
    KIND_ORDER,
    MemoryStore,
    extract_by_rules,
    extract_candidates,
    memory_value,
    used_memory_ids,
)
from mewhelp.memory.extract import ORDER_MEMORY_LABEL
from mewhelp.nlu.coreference import CoreferenceResolver, _MEMORY_ORDER_PREFIX
from mewhelp.schemas import MemoryItem
from mewhelp.storage import get_engine, init_db, reset_engine_cache
from mewhelp.workflow.graph import build_deps


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    db_dir = tmp_path_factory.mktemp("memory")
    settings = Settings(
        MEWHELP_DATABASE_URL=f"sqlite:///{(db_dir / 'memory.db').as_posix()}"
    )
    return create_app(build_deps(settings=settings))


@pytest.fixture(scope="module")
def client(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def store(tmp_path):
    """独立的记忆库 —— 单测不共用 module 级那个（会互相看见对方的记忆）。"""
    settings = Settings(MEWHELP_DATABASE_URL=f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    reset_engine_cache()
    init_db(settings=settings)
    yield MemoryStore(settings)
    reset_engine_cache()


def _sse(client, message: str, session_id: str, user_id: str):
    resp = client.post(
        "/api/chat", json={"user_id": user_id, "session_id": session_id, "message": message}
    )
    assert resp.status_code == 200
    events = []
    for block in resp.text.replace("\r\n", "\n").split("\n\n"):
        name, payload = None, ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                payload += line[len("data:") :].strip()
        if name and payload:
            events.append((name, json.loads(payload)))
    return events


def _of(events, name):
    return [d for n, d in events if n == name]


# ============================================================
# 1. 规则抽取：什么该记、什么不该记
# ============================================================
def test_rules_keep_stable_facts():
    """稳定的东西要抓得住。"""
    size = extract_by_rules("我穿 L 码，帮我看看这件")
    assert [c.text for c in size] == ["尺码：L"]
    assert size[0].kind == "preference"

    order = extract_by_rules("帮我查下订单 ORD-1002 到哪了")
    assert order and order[0].kind == KIND_ORDER
    assert order[0].key == "last_order"
    assert order[0].text == f"{ORDER_MEMORY_LABEL}：ORD-1002"

    contact = extract_by_rules("以后不要打电话给我")
    assert any(c.text == "不喜欢被打电话" for c in contact)


def test_rules_do_not_record_questions_or_one_off_demands():
    """反过来更重要：**不该记的东西一条都不能进档案**。

    把"我要退款"记成用户偏好，或者把"我的地址是什么"记成"常用地址：是什么"，
    这类脏数据比没记忆更糟 —— 它会跟着用户一辈子，还会被注入每一次回答。
    """
    for text_ in (
        "我要退款",
        "帮我查下物流",
        "会员有什么优惠",
        "我的地址是什么？",
        "尺码怎么选",
        "我穿什么码好呢",
        "你好",
    ):
        assert extract_by_rules(text_) == [], f"这句不该被记住：{text_}"


def test_explicit_memo_is_always_kept():
    """用户说了"记住"，就不受"是不是提问句"的闸门限制。"""
    got = extract_by_rules("记住：我周六一般不在家")
    assert len(got) == 1
    assert got[0].kind == KIND_MANUAL
    assert got[0].text == "备忘：我周六一般不在家"
    # 键按内容取，两条不同的备忘不会互相覆盖
    assert got[0].key != extract_by_rules("记住：发票开公司抬头")[0].key


def test_mock_llm_uses_the_same_rules_as_the_rule_layer():
    """mock 通道抽出来的东西必须和规则层完全一致。

    两处各写一套的结果是"离线能记住、接上真模型就忘"（或者反过来）——
    这种偏差只有换模型那天才暴露，所以在这里钉死。
    """
    for text_ in ("我穿 L 码", "帮我查下订单 ORD-1007 到哪了", "我要退款"):
        resp = MockLLMClient().chat(
            [{"role": "user", "content": text_}], task="memory"
        )
        assert resp.json() == {
            "memories": [
                {"kind": c.kind, "key": c.key, "text": c.text}
                for c in extract_by_rules(text_)
            ]
        }


def test_extract_candidates_falls_back_to_rules_when_the_model_fails():
    """模型挂了，记忆功能降级但不会消失。"""

    class Broken:
        def chat(self, *a, **kw):  # noqa: ANN002, ANN003
            raise RuntimeError("model down")

    got = extract_candidates(Broken(), "我穿 L 码")
    assert [c.text for c in got] == ["尺码：L"]


def test_used_memory_ids_only_credits_real_help():
    """命中口径：只算"系统自己补进去的"，不算"用户自己说的"、也不算"塞进过 prompt"。"""
    mems = [
        MemoryItem(id=1, kind=KIND_ORDER, text="最近关注的订单：ORD-1002"),
        MemoryItem(id=2, kind="preference", text="尺码：L"),
    ]
    # 原话里没有单号、补全后有 —— 记忆帮上忙了
    assert used_memory_ids(mems, "我的订单到哪了", "订单 ORD-1002 到哪了") == [1]
    # 单号是用户自己说的 —— 不算记忆的功劳
    assert used_memory_ids(mems, "订单 ORD-1002 到哪了", "订单 ORD-1002 到哪了") == []
    # 一句话都没改（没调消解）时也不该记功
    assert used_memory_ids(mems, "退货政策", "退货政策") == []
    assert memory_value("尺码：L") == "L"


# ============================================================
# 2. MemoryStore：去重、钉住、上限
# ============================================================
def test_same_key_updates_instead_of_piling_up(store):
    """同一个人说两次尺码是**更新**，不是新增两条自相矛盾的记忆。"""
    store.upsert("u1", kind="preference", key="size", text="尺码：L")
    store.upsert("u1", kind="preference", key="size", text="尺码：XL")
    items = store.list("u1")
    assert [i.text for i in items] == ["尺码：XL"]
    assert "u1" not in {i.text for i in store.list("u2")}


def test_pin_and_source_survive_an_update(store):
    """覆盖内容不能抹掉"用户亲手钉住的"和"这是他自己写的"。"""
    store.upsert("u1", kind=KIND_MANUAL, key="memo:x", text="备忘：周六不在家", source="manual")
    item = store.list("u1")[0]
    assert store.set_pinned("u1", item.id, True)

    store.upsert("u1", kind="preference", key="memo:x", text="备忘：周日不在家")
    again = store.list("u1")[0]
    assert again.text == "备忘：周日不在家"
    assert again.pinned is True, "钉住的状态不能被一次内容更新抹掉"
    assert again.source == "manual", "用户自己写的不该被降级成系统猜的"


def test_delete_and_pin_require_the_owner(store):
    """删除和置顶都带 user_id —— 演示环境没有登录态，这是唯一的所有权校验。"""
    item = store.upsert("u1", kind="preference", key="size", text="尺码：L")
    assert store.delete("u2", item.id) is False
    assert store.set_pinned("u2", item.id, True) is False
    assert len(store.list("u1")) == 1
    assert store.delete("u1", item.id) is True
    assert store.list("u1") == []


def test_mark_used_counts_only_what_the_caller_says(store):
    """`mark_used` 的返回值是**影响的行数**，每次调用给每条记忆 +1。"""
    item = store.upsert("u1", kind=KIND_ORDER, key="last_order", text="最近关注的订单：ORD-1002")
    assert store.mark_used("u1", [item.id]) == 1
    assert store.mark_used("u1", [item.id]) == 1
    assert store.list("u1")[0].hits == 2
    # 不存在的 id 不该报错，只是什么都没更新
    assert store.mark_used("u1", [999999]) == 0
    assert store.mark_used("u1", []) == 0


def test_limit_evicts_the_stalest_unpinned_memory(store, tmp_path):
    """条数上限只管没被钉住的那些 —— 自动清理绝不能删掉用户亲手钉的东西。"""
    small = MemoryStore(store.settings, max_items=3)
    pinned = small.upsert("u1", kind="preference", key="keep", text="尺码：L")
    small.set_pinned("u1", pinned.id, True)
    for i in range(6):
        small.upsert("u1", kind="preference", key=f"k{i}", text=f"备忘：第 {i} 条")
    items = small.list("u1")
    assert len(items) == 3
    assert any(i.text == "尺码：L" for i in items), "钉住的被淘汰了"


def test_for_prompt_caps_what_goes_into_the_model(store):
    """进 prompt 的条数比面板展示的少 —— 记忆是背景，不能挤掉检索证据。"""
    for i in range(20):
        store.upsert("u1", kind="preference", key=f"k{i}", text=f"偏好：第 {i} 条")
    assert len(store.for_prompt("u1")) == store.prompt_items < len(store.list("u1"))


# ============================================================
# 3. 指代消解使用记忆
# ============================================================
def test_coreference_fills_the_order_from_memory():
    """新会话里说"我的订单"，靠记忆补出单号 —— 这是记忆最直观的价值。"""
    resolved = CoreferenceResolver().resolve(
        "我的订单到哪了？", [], memory=["最近关注的订单：ORD-1002"]
    )
    assert resolved.rewritten is True
    assert resolved.resolved == "订单 ORD-1002 到哪了？"


def test_coreference_ignores_memory_when_it_would_be_wrong():
    """三个条件缺一个就不许动手 —— 补错比不补更糟（用户看不出来）。"""
    resolver = CoreferenceResolver()
    memory = ["最近关注的订单：ORD-1002"]
    # ① 用户自己说了单号
    assert resolver.resolve("订单 ORD-1007 到哪了", [], memory=memory).rewritten is False
    # ② 没有"我的某一单"这种说法
    assert resolver.resolve("退货政策是什么", [], memory=memory).rewritten is False
    # ③ 记忆里没有单号（不能拿"常用地址：1001 号"里的数字凑）
    assert resolver.resolve("我的订单到哪了", [], memory=["常用地址：幸福路 1001 号"]).rewritten is False
    # 没传记忆时行为和以前完全一样
    assert resolver.resolve("我的订单到哪了", []).rewritten is False


def test_memory_prefix_is_the_same_on_both_sides():
    """记忆模块写进去的前缀，和消解模块认的前缀必须是同一个。

    这里刻意没做成 import：NLU 层不该依赖记忆层（评估脚本里没有记忆存储，
    消解要能单独跑）。代价是两处各有一个字面量 —— 那就用测试把它们钉在一起。
    """
    assert ORDER_MEMORY_LABEL + "：" == _MEMORY_ORDER_PREFIX


# ============================================================
# 4. 端到端：记忆 + 会话历史
# ============================================================
def test_memory_lets_a_fresh_session_answer_about_my_order(client):
    """**本批次的核心用例。**

    第一通对话问过 ORD-1002；换一个会话（= 用户新开一次对话）只说
    "我的订单到哪了"。没有记忆时这句话缺单号，只能反问"请问是哪一单"；
    有记忆时系统自己补上并真的去查了物流。
    """
    user = "u-memo-e2e"
    first = _of(_sse(client, "帮我查下订单 ORD-1002 的状态", "s-memo-1", user), "meta")
    assert first, "第一轮应当正常返回"

    events = _sse(client, "我的订单到哪了？", "s-memo-2", user)
    meta = _of(events, "meta")[0]
    assert "ORD-1002" in meta["resolved_question"], "记忆没有参与指代消解"

    tools = _of(events, "tool")
    assert tools, "补全出单号之后应该真的去查了"
    assert any(t["ok"] for t in tools)
    assert any(
        t.get("result", {}).get("order_id") == "ORD-1002" for t in tools if t["ok"]
    )

    # 这条记忆因为"真的帮上忙了"被记了一次命中
    items = client.get("/api/memories", params={"user_id": user}).json()["items"]
    order_mem = next(i for i in items if i["kind"] == "order")
    assert order_mem["hits"] >= 1, "被用上的记忆没有计数"
    # 记忆补上的单号必须被当成"已有槽位"：否则会被缺槽位的护栏拦回去反问
    assert "请把订单号发我" not in "".join(
        d["answer"] for d in _of(events, "done")
    ), "记忆明明知道是哪一单，却回了'请把订单号发我'"


def test_order_question_without_any_order_asks_instead_of_failing(client):
    """**没有**记忆可依的时候，缺订单号要追问，不能把 None 当参数去查。

    不拦这一道，用户点第一条快捷问题"我的订单到哪了？"收到的是
    "没有查到这个订单号，麻烦核对一下再发我" —— 他什么都没报错，
    只是我们没问。这类答复看起来像系统在责怪用户，是最伤信任的一种。

    这条和上面那条（有记忆时直接答）是一对：记忆补得上的时候必须放行，
    补不上的时候必须拦住。只测一边都会漏掉另一边的回归。
    """
    body = client.post(
        "/api/chat/sync",
        json={"user_id": "u-no-order", "session_id": "s-no-order", "message": "我的订单到哪了？"},
    ).json()
    assert body["tool_calls"] == [], "缺订单号时不该去调工具"
    assert "订单号" in body["answer"] and "ORD-" in body["answer"], body["answer"]
    # 追问是"把话说完"，不是转人工，也不是"我查不到"
    assert body["handoff"] is False
    assert "agent" in body["path"]


def test_quote_brings_an_early_message_back_into_context(client):
    """引用提问：被引用的那一句必须真的进这一轮的上下文。

    引用的都是**较早**的消息，而它们很可能已经滑出 `window_turns` 了 ——
    只靠会话历史，服务端根本看不到那一句。所以 `quote` 必须跟着这一次请求
    显式发出去，而不是"界面上的装饰"。

    判据用**指代消解的结果**：引用的那句里有单号，而这一轮的问题里没有，
    补全后的问题里应当出现它 —— 这条链路只有 quote 真的进了上下文才成立。
    """
    user = "u-quote"
    # 先铺垫几轮，把第一句挤出滑窗（window_turns = 3）
    client.post(
        "/api/chat/sync",
        json={"user_id": user, "session_id": "s-quote", "message": "帮我查下订单 ORD-1007 的状态"},
    )
    for i in range(4):
        client.post(
            "/api/chat/sync",
            json={"user_id": user, "session_id": "s-quote", "message": f"运费怎么算{i}"},
        )

    body = client.post(
        "/api/chat/sync",
        json={
            "user_id": user,
            "session_id": "s-quote",
            "message": "那它现在怎么样了",
            "quote": "帮我查下订单 ORD-1007 的状态",
        },
    ).json()
    assert "ORD-1007" in body["resolved_question"], (
        f"引用没有进上下文：{body['resolved_question']}"
    )
    assert any(t["name"] == "get_order" for t in body["tool_calls"]), "补出单号后应当去查这一单"

    # 引用不落库：被引用的消息本来就在会话历史里，存两遍会让它下次出现两次
    detail = client.get("/api/sessions/s-quote", params={"user_id": user}).json()
    contents = [m["content"] for m in detail["messages"]]
    assert contents.count("帮我查下订单 ORD-1007 的状态") == 1, "引用被写进会话历史了"


def test_memory_event_is_pushed_only_when_the_user_has_a_profile(client):
    """有档案才推 memory 事件 —— 前端因此可以把"收到事件"当成"面板该更新了"。"""
    user = "u-memo-event"
    empty = _sse(client, "退货政策是什么", "s-memo-e1", user)
    assert _of(empty, "memory") == [], "新用户不该收到 memory 事件"

    _sse(client, "我穿 L 码", "s-memo-e2", user)
    after = _sse(client, "退货政策是什么", "s-memo-e3", user)
    payload = _of(after, "memory")
    assert payload, "有了档案之后每一轮都该带上完整档案"
    items = payload[0]["items"]
    assert any(i["text"] == "尺码：L" for i in items)
    assert all("kind_label" in i for i in items), "文案由服务端给，前端不另写一套"
    # 顺序不变：memory 在 token 之前
    names = [n for n, _ in after]
    assert names.index("memory") < names.index("token")
    assert names[-1] == "done"


def test_memory_endpoints_crud(client):
    user = "u-memo-crud"
    created = client.post(
        "/api/memories", json={"user_id": user, "text": "发票开公司抬头"}
    ).json()
    assert created["ok"] is True
    item = created["item"]
    assert item["source"] == "manual" and item["pinned"] is True, "面板里写的默认钉住"

    listed = client.get("/api/memories", params={"user_id": user}).json()
    assert listed["count"] == 1
    assert listed["items"][0]["kind_label"]

    assert client.post("/api/memories", json={"user_id": user, "text": "  "}).json()["ok"] is False

    assert client.patch(
        f"/api/memories/{item['id']}", json={"user_id": user, "pinned": False}
    ).json()["ok"] is True
    assert client.get("/api/memories", params={"user_id": user}).json()["items"][0]["pinned"] is False

    # 别人的记忆删不掉
    assert client.delete(
        f"/api/memories/{item['id']}", params={"user_id": "u-other"}
    ).json()["ok"] is False
    assert client.delete(
        f"/api/memories/{item['id']}", params={"user_id": user}
    ).json()["ok"] is True
    assert client.get("/api/memories", params={"user_id": user}).json()["count"] == 0


def test_history_lists_renames_and_deletes_a_session(client):
    user = "u-hist"
    _sse(client, "帮我查下订单 ORD-1001 的状态", "s-hist-1", user)
    _sse(client, "生鲜类可以退款吗", "s-hist-2", user)

    body = client.get("/api/sessions", params={"user_id": user}).json()
    ids = [s["session_id"] for s in body["sessions"]]
    assert ids[0] == "s-hist-2", "最近更新的排前面"
    first = next(s for s in body["sessions"] if s["session_id"] == "s-hist-1")
    # 标题取自第一句用户原话 —— 用户自己那句话永远比一句概括更好认
    assert first["title"].startswith("帮我查下订单 ORD-1001")
    assert first["turns"] == 2

    renamed = client.patch(
        "/api/sessions/s-hist-1", json={"title": "那一单的物流"}, params={"user_id": user}
    ).json()
    assert renamed["ok"] is True
    detail = client.get("/api/sessions/s-hist-1", params={"user_id": user}).json()
    assert detail["title"] == "那一单的物流"

    assert client.delete("/api/sessions/s-hist-1", params={"user_id": user}).json()["ok"] is True
    assert client.get("/api/sessions/s-hist-1", params={"user_id": user}).json()["ok"] is False


def test_session_detail_is_user_facing_only(client):
    """恢复历史对话时，内部字段一个都不能跟着回来。

    它们在库里确实存着（`Turn.metadata`，给审计用）。过滤做在**契约模型**
    这一层而不是"返回前删几个 key"：后者在下次有人加字段时会静默漏出去。
    """
    user = "u-hist-view"
    _sse(client, "帮我查下订单 ORD-1001 的状态", "s-hist-view", user)
    detail = client.get("/api/sessions/s-hist-view", params={"user_id": user}).json()
    assert len(detail["messages"]) == 2
    for msg in detail["messages"]:
        assert set(msg) == {"role", "content", "handoff", "request_id"}
    assert detail["messages"][0]["role"] == "user"
    assert detail["messages"][1]["role"] == "assistant"
    # 句柄要回填，否则历史里的消息没法再被点赞/点踩（前端的收藏键也靠它）
    assert detail["messages"][1]["request_id"]


def test_sessions_are_scoped_to_the_user(client):
    _sse(client, "生鲜类可以退款吗", "s-scope", "u-scope-owner")
    mine = client.get("/api/sessions", params={"user_id": "u-scope-owner"}).json()
    assert any(s["session_id"] == "s-scope" for s in mine["sessions"])
    other = client.get("/api/sessions", params={"user_id": "u-scope-other"}).json()
    assert all(s["session_id"] != "s-scope" for s in other["sessions"])
    assert client.get("/api/sessions/s-scope", params={"user_id": "u-scope-other"}).json()["ok"] is False


def test_deleting_a_session_keeps_the_user_profile(client):
    """删的是聊天记录，不是"你认识我"。"""
    user = "u-keep"
    _sse(client, "帮我查下订单 ORD-1001 的状态", "s-keep", user)
    before = client.get("/api/memories", params={"user_id": user}).json()["count"]
    assert before >= 1

    client.delete("/api/sessions/s-keep", params={"user_id": user})
    after = client.get("/api/memories", params={"user_id": user}).json()
    assert after["count"] == before
    assert "ORD-1001" in json.dumps(after, ensure_ascii=False)


# ============================================================
# 5. 跨语言契约：前端能不能读懂后端推的档案
# ============================================================
#: 前端解析记忆的代码。它在 `web/src/api/client.ts` 里，靠 `./sse` 和
#: `../workspace` 两个同目录 / 上一级模块 —— 所以拷到临时目录时要保持目录形状。
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FRONTEND_API = _REPO_ROOT / "web" / "src" / "api" / "client.ts"

_MEMORY_HARNESS = """
import { memoryItems } from './api/client.ts';
import fs from 'node:fs';

const payload = JSON.parse(fs.readFileSync(process.argv[2], 'utf-8'));
// 事件里是 { items: [...] }，REST 接口也是。两种形态都喂一遍。
console.log(JSON.stringify({
  fromEvent: memoryItems(payload.items),
  fromRest: memoryItems(payload.rest),
}));
"""


def test_frontend_parses_the_real_memory_payload(client, tmp_path):
    """后端推的档案 → 前端解析器 → 面板能渲染的字段，必须对得上。

    和 SSE、工作区那两条同理，这是**跨语言契约**。记忆面板最怕的不是报错，
    而是**静默为空**：后端把 `kind_label` 改成 `kindLabel`，界面不报错、
    只是那一栏永远空着，或者永远显示默认的"记忆"两个字。

    跑的是 `web/src/api/client.ts` 里那份**真实代码**，不是在 Python 里
    重写一遍解析逻辑 —— 重写只能证明"我理解得对"。
    """
    node = shutil.which("node")
    if not node:
        pytest.skip("环境里没有 node，跳过前端解析器校验")
    if not _FRONTEND_API.is_file():
        pytest.fail(f"前端接口客户端不见了：{_FRONTEND_API}")

    (tmp_path / "api").mkdir()
    # 唯一的改动是给裸模块名补上 `.ts`：Node 的 ESM 解析器不做扩展名推断，
    # 而 Vite 会。不改这一处就只能把源码改成 `from './sse.ts'`，
    # 那是为了迁就测试去将就生产代码 —— 反过来了。
    source = _FRONTEND_API.read_text(encoding="utf-8")
    for bare, real in (("'./sse'", "'./sse.ts'"), ("'../workspace'", "'../workspace.ts'")):
        source = source.replace(bare, real)
    (tmp_path / "api" / "client.ts").write_text(source, encoding="utf-8")

    shutil.copy(_FRONTEND_API.parent / "sse.ts", tmp_path / "api" / "sse.ts")
    shutil.copy(_REPO_ROOT / "web" / "src" / "workspace.ts", tmp_path / "workspace.ts")
    (tmp_path / "harness.mjs").write_text(_MEMORY_HARNESS, encoding="utf-8")

    client.post("/api/memories", json={"user_id": "u-js", "text": "发票开公司抬头"})
    rest = client.get("/api/memories", params={"user_id": "u-js"}).json()
    events = _sse(client, "我穿 L 码", "s-js", "u-js")
    memory_events = _of(events, "memory")
    assert memory_events, "有档案的轮次必须推 memory 事件"

    (tmp_path / "payload.json").write_text(
        json.dumps({"items": memory_events[-1]["items"], "rest": rest["items"]}, ensure_ascii=False),
        encoding="utf-8",
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

    for source in ("fromEvent", "fromRest"):
        items = got[source]
        assert items, f"{source}: 前端一条记忆都没解析出来"
        first = items[0]
        assert first["id"] > 0, f"{source}: id 没解析出来（置顶/删除都要靠它）"
        assert first["text"], f"{source}: 正文丢了"
        assert first["kindLabel"] and first["kindLabel"] != "记忆", (
            f"{source}: 类型文案丢了，面板会一律显示成'记忆'"
        )
        assert isinstance(first["pinned"], bool)

    texts = " ".join(i["text"] for i in got["fromRest"])
    assert "发票开公司抬头" in texts
    assert any(i["kindLabel"] == "我的备忘" for i in got["fromRest"])
# ============================================================
# 6. 老库补列
# ============================================================
def test_init_db_adds_columns_to_a_legacy_table(tmp_path):
    """开发机上那份演示数据早就建过 session 表了。

    没有这一步轻量迁移，`create_all` 会一句话不说地跳过它，然后在第一次
    写会话时炸 `no such column: session.title` —— 而 CI 从零建库永远复现不了。
    """
    url = f"sqlite:///{(tmp_path / 'legacy.db').as_posix()}"
    legacy = create_engine(url, future=True)
    with legacy.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE session ("
                "session_id VARCHAR(64) PRIMARY KEY, user_id VARCHAR(64) NOT NULL, "
                "summary TEXT NOT NULL, turns_json TEXT NOT NULL, "
                "summarized_turns INTEGER NOT NULL, token_estimate INTEGER NOT NULL, "
                "updated_at DATETIME NOT NULL)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO session VALUES "
                "('s-old', 'u-old', '', '[]', 0, 0, '2026-01-01 00:00:00')"
            )
        )
    legacy.dispose()

    reset_engine_cache()
    settings = Settings(MEWHELP_DATABASE_URL=url)
    try:
        init_db(settings=settings)
        columns = {c["name"] for c in inspect(get_engine(settings)).get_columns("session")}
        assert "title" in columns, "老库没有补上 title 列"
        # 补列之后老数据仍在，且新列有默认值（非空列必须带 server_default 才加得上）
        with get_engine(settings).connect() as conn:
            row = conn.execute(text("SELECT session_id, title FROM session")).one()
        assert row[0] == "s-old" and row[1] == ""
    finally:
        reset_engine_cache()
