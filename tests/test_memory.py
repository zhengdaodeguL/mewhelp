"""会话上下文层的测试。

口径约定（全文件统一）：
  * 一轮 = 一个 user + 一个 assistant = 2 条消息；
  * `window_turns` 数的是**轮**，窗口长度 = 2 * window_turns 条消息。
    这点必须先钉死，否则"test 里算的轮数"和"实现里算的轮数"错位，
    测试看着通过、生产里窗口却只有一半。

测试的重点不是"函数能不能跑"，而是三件可被证伪的事：
  1. 触发时机对不对（不该摘的时候一次模型都不调）；
  2. 增量摘要是真的增量（第二次只说新增的，不把老内容重摘一遍）；
  3. 度量数字是不是真的（省了多少 token、丢了几轮信息，都能被复算出来）。
"""

from __future__ import annotations

import pytest

from mewhelp.config import Settings
from mewhelp.llm import LLMError, LLMResponse, MockLLMClient
from mewhelp.memory.context import (
    NEW_TURNS_HEADER,
    PREVIOUS_HEADER,
    REQUIREMENT_HEADER,
    ContextManager,
    context_of,
    estimate_tokens,
    turns_token_estimate,
)
from mewhelp.schemas import ContextBundle, Turn


# ============================================================
# 测试脚手架
# ============================================================


def make_settings(**overrides) -> Settings:
    """显式指定窗口和阈值，避免测试结果被默认值改动带着漂。"""
    base = {"window_turns": 3, "summary_trigger_tokens": 2000}
    base.update(overrides)
    return Settings(**base)


class FlakyLLM(MockLLMClient):
    """摘要调用必炸的模型 —— 用来验证降级路径。

    只炸 `summarize`，其余任务照常，这样才能证明"炸的是摘要、不是整条链路"。
    """

    def __init__(self, exc: Exception, settings: Settings | None = None) -> None:
        super().__init__(settings)
        self.exc = exc

    def chat(self, messages, **kw):  # type: ignore[override]
        if kw.get("task") == "summarize":
            self.calls.append({"task": "summarize", "messages": list(messages), "tools": None})
            raise self.exc
        return super().chat(messages, **kw)


class SilentLLM(MockLLMClient):
    """摘要返回空白 —— 模拟"模型什么都没压出来"。

    这种情况必须当成失败：空摘要会被下游误读成"早期内容已经压过了"。
    """

    def chat(self, messages, **kw):  # type: ignore[override]
        if kw.get("task") == "summarize":
            self.calls.append({"task": "summarize", "messages": list(messages), "tools": None})
            return LLMResponse(text="   ", model=self.model)
        return super().chat(messages, **kw)


def mk(role: str, content: str) -> Turn:
    return Turn(role=role, content=content)


def chat_turns(n: int, *, tag: str = "", body: str = "") -> list[Turn]:
    """n 轮对话（2n 条消息），内容长度中等且确定。

    `tag` 用来区分不同批次的轮次，方便断言"某一批有没有被重摘"。
    """
    out: list[Turn] = []
    for i in range(n):
        out.append(mk("user", f"[{tag}U{i}] 我想咨询一下订单和物流的情况，麻烦帮我看下{body}"))
        out.append(mk("assistant", f"[{tag}A{i}] 好的，我这就帮你查询，请稍等{body}"))
    return out


def summarize_calls(llm: MockLLMClient) -> list[dict]:
    return [c for c in llm.calls if c["task"] == "summarize"]


def prompt_of(call: dict) -> str:
    return call["messages"][0]["content"]


def new_section_lines(call: dict) -> list[str]:
    """取出某次摘要调用中"本次新增轮次"部分的转写行。

    行数 = 这次真正喂给模型的轮数，是判断增量的直接观测量。
    """
    content = prompt_of(call)
    assert NEW_TURNS_HEADER in content, "摘要 prompt 必须区分'旧摘要'和'新增轮次'"
    tail = content.split(NEW_TURNS_HEADER, 1)[1]
    return [
        ln
        for ln in tail.splitlines()
        if ln.strip() and (ln.startswith("用户") or ln.startswith("客服"))
    ]


# ============================================================
# 1. 轮次不超过窗口 —— 全部保原文，一次模型都不调
# ============================================================


def test_short_history_keeps_everything_in_window():
    llm = MockLLMClient()
    cm = ContextManager(llm, make_settings(window_turns=3))

    turns = chat_turns(2)  # 2 轮 ≤ 窗口 3 轮
    bundle = cm.build(turns)

    assert len(turns) == 4
    assert bundle.window == turns
    assert bundle.summary == ""
    assert bundle.summarized_turns == 0
    assert bundle.summary_generated is False
    assert summarize_calls(llm) == []
    assert bundle.token_estimate == turns_token_estimate(turns)


def test_window_is_measured_in_turns_not_messages():
    """窗口单位是"轮"：window_turns=3 要保住 6 条消息，不是 3 条。

    这条断言防的是最常见的实现错位 —— 按消息数切窗口，用户会发现
    "设了 3 轮，实际只记住了 1.5 轮"，指代消解立刻失效。
    """
    cm = ContextManager(MockLLMClient(), make_settings(window_turns=3))
    turns = chat_turns(5)  # 10 条

    bundle = cm.build(turns)
    assert len(bundle.window) == 6
    assert bundle.window == turns[-6:]
    assert bundle.window[0].role == "user", "窗口必须从一个完整的轮开始，不能从半截开始"


def test_history_within_window_never_calls_model_even_with_zero_threshold():
    llm = MockLLMClient()
    cm = ContextManager(llm, make_settings(window_turns=2, summary_trigger_tokens=0))

    turns = chat_turns(2)  # 恰好填满窗口
    bundle = cm.build(turns)

    assert bundle.window == turns
    assert bundle.summary == ""
    # 阈值低到 0 也不该触发：没有"更早的部分"，摘要无从谈起。
    # 这条防的是"无条件调模型"的实现 —— 阈值再低也得有料才摘。
    assert summarize_calls(llm) == []


# ============================================================
# 2. 滑出窗口但未超阈值 —— 原文丢掉，但不生成摘要
# ============================================================


def test_slides_out_without_summary_when_under_threshold():
    llm = MockLLMClient()
    cm = ContextManager(llm, make_settings(window_turns=2, summary_trigger_tokens=10**6))

    turns = chat_turns(4)  # 8 条，滑出 4 条
    bundle = cm.build(turns)

    assert bundle.window == turns[-4:]
    assert bundle.summary == ""
    assert bundle.summary_generated is False
    assert bundle.summarized_turns == 0
    # 模型一次都没被调用 —— 这是"不做过度设计"的可验证形式。
    assert summarize_calls(llm) == []
    # 但滑出的代价如实反映在度量里：token_estimate 只含滑窗。
    assert bundle.token_estimate == turns_token_estimate(turns[-4:])
    assert bundle.token_estimate < turns_token_estimate(turns)


def test_trigger_boundary_is_strictly_greater_than_threshold():
    """阈值边界：恰好等于阈值时不动模型（">" 而不是 ">="）。

    边界不写死的话，哪天阈值调到刚好等于历史长度，就会出现"每轮调一次模型"
    这种既烧钱又零收益的情况。
    """
    turns = chat_turns(3)  # 6 条，窗口 1 轮 → 早期部分 4 条
    older = turns[:-2]
    exact = turns_token_estimate(older)

    llm = MockLLMClient()
    at_boundary = ContextManager(llm, make_settings(window_turns=1, summary_trigger_tokens=exact))
    b1 = at_boundary.build(turns)
    assert summarize_calls(llm) == []
    assert b1.summary_generated is False
    assert b1.summarized_turns == 0

    below = ContextManager(llm, make_settings(window_turns=1, summary_trigger_tokens=exact - 1))
    b2 = below.build(turns)
    assert len(summarize_calls(llm)) == 1
    assert b2.summary_generated is True
    assert b2.summarized_turns == len(older)


# ============================================================
# 3. 超阈值 —— 生成摘要，轮数记账正确
# ============================================================


def test_summary_generated_when_threshold_crossed():
    llm = MockLLMClient()
    settings = make_settings(window_turns=1, summary_trigger_tokens=60)
    cm = ContextManager(llm, settings)

    turns = chat_turns(6)  # 12 条，滑出 10 条
    older = turns[:-2]
    assert turns_token_estimate(older) > settings.summary_trigger_tokens  # 前提成立

    bundle = cm.build(turns)

    assert bundle.summary != ""
    assert bundle.summary_generated is True
    assert bundle.window == turns[-2:]
    # 记账口径：summarized_turns 是"被折叠进摘要的轮次"，
    # 它与窗口互补，两者之和必须等于全部历史（否则就是悄悄丢了几轮）。
    assert bundle.summarized_turns == len(older)
    assert bundle.summarized_turns + len(bundle.window) == len(turns)
    assert len(summarize_calls(llm)) == 1

    msgs = bundle.as_messages()
    assert msgs[0]["role"] == "system"
    assert bundle.summary in msgs[0]["content"]
    assert [m["content"] for m in msgs[1:]] == [t.content for t in bundle.window]


def test_summary_prompt_carries_raw_excerpt_not_just_role_names():
    """摘要 prompt 里必须带原始内容 —— 否则压出来的摘要没有信息可留。"""
    llm = MockLLMClient()
    cm = ContextManager(llm, make_settings(window_turns=1, summary_trigger_tokens=10))
    turns = chat_turns(3)
    cm.build(turns)

    blob = prompt_of(summarize_calls(llm)[0])
    assert turns[0].content in blob
    assert turns[1].content in blob
    assert "必须保留" in blob, "prompt 里要明确要求保住订单号/金额/结论"


# ============================================================
# 4. 增量摘要 —— 不把已摘要的内容重复喂一遍
# ============================================================


def test_incremental_summary_does_not_resummarize_old_content():
    llm = MockLLMClient()
    cm = ContextManager(llm, make_settings(window_turns=2, summary_trigger_tokens=50))

    # --- 第一次：早期 4 条（2 轮）滑出，越过阈值 ---
    history = chat_turns(4, tag="A")  # 8 条
    first = cm.build(history)
    assert first.summary_generated is True
    assert first.summarized_turns == 4
    assert len(summarize_calls(llm)) == 1

    # --- 第二次：只追加对话，早期内容一字未变，且没有新轮次滑出 ---
    history = history + chat_turns(1, tag="B")
    second = cm.build(history, prev=first)

    assert len(summarize_calls(llm)) == 1, "没有新增滑出内容，却又调了一次模型"
    assert second.summary == first.summary
    assert second.summary_generated is False
    assert second.summarized_turns == 4, "已摘要的轮数不该被重复计入"

    # --- 第三次：又有新轮次滑出，此时才允许第二次调用 ---
    history = history + chat_turns(2, tag="C")
    third = cm.build(history, prev=second)

    assert len(summarize_calls(llm)) == 2
    second_call = summarize_calls(llm)[1]
    lines = new_section_lines(second_call)
    # 关键断言：第二次只喂新滑出的 3 轮（B 的 1 轮 + C 的 2 轮），
    # 绝不把 A 批次那 4 条重摘一遍。
    assert len(lines) == 6, f"增量输入规模不对：{lines}"
    assert "[AU0]" not in prompt_of(second_call).split(NEW_TURNS_HEADER, 1)[1]
    # 但旧摘要本身要带上，模型才能做合并压缩（否则早期信息被丢掉）
    assert first.summary in prompt_of(second_call)
    # 记账：累计覆盖 10 条 = 全部早期部分
    assert third.summarized_turns == len(history) - 4


def test_incremental_summary_merges_old_summary_and_new_turns():
    """增量不只是"省调用"，还要真的把旧摘要并进新摘要 —— 否则早期信息会断掉。"""
    llm = MockLLMClient()
    cm = ContextManager(llm, make_settings(window_turns=1, summary_trigger_tokens=10))

    history = [
        mk("user", "ORD-555001 我要退款"),
        mk("assistant", "已登记"),
        mk("user", "好的谢谢"),
        mk("assistant", "不客气"),
    ]
    first = cm.build(history)
    assert first.summary_generated is True
    assert "ORD-555001" in first.summary

    history = history + [mk("user", "ORD-555002 物流到哪了"), mk("assistant", "已发货")]
    second = cm.build(history, prev=first)

    history = history + [mk("user", "还有 ORD-555003 想退货"), mk("assistant", "好的")]
    third = cm.build(history, prev=second)
    assert third.summary_generated is True

    last_call = summarize_calls(llm)[-1]
    lines = new_section_lines(last_call)
    # ORD-555001 的原文早就滑出滑窗了，最后一次 prompt 的"新增轮次"里只有它那一轮
    assert len(lines) == 2
    assert "ORD-555001" not in "\n".join(lines)
    # 但摘要里它还在 —— 唯一的来源就是被当作基线传下去的旧摘要。
    assert "ORD-555002" in third.summary
    assert "ORD-555001" in third.summary
    # 旧摘要必须以"已有摘要"段的形式出现在 prompt 里，模型才能做合并压缩
    prompt = prompt_of(last_call)
    assert PREVIOUS_HEADER in prompt
    assert second.summary in prompt


def test_incremental_cost_per_round_is_constant():
    """长对话下每次调用的输入规模不随轮数增长 —— 这是增量的成本含义。"""
    llm = MockLLMClient()
    cm = ContextManager(llm, make_settings(window_turns=2, summary_trigger_tokens=50))

    history = chat_turns(4, tag="A")
    bundle = cm.build(history)
    for i in range(10):
        history = history + chat_turns(1, tag=f"N{i}")
        bundle = cm.build(history, prev=bundle)

    calls = summarize_calls(llm)
    assert len(calls) >= 2, "长对话下至少该发生两次增量摘要"
    lines_per_call = [len(new_section_lines(c)) for c in calls]
    # 每次最多喂"这两轮新滑出的"（4 条消息），与总轮数无关。
    assert max(lines_per_call) <= 4, f"单次摘要输入随轮数增长了：{lines_per_call}"


def test_stale_prev_after_history_rollback_is_rebuilt_not_skipped():
    """历史被截断/回滚时旧摘要不再可信 —— 必须重摘，不能装作没事。

    这是防信息静默丢失的护栏：覆盖下标越界还继续增量的话，实际没被摘要的
    轮次会被永久跳过。
    """
    llm = MockLLMClient()
    cm = ContextManager(llm, make_settings(window_turns=2, summary_trigger_tokens=10))

    long_run = chat_turns(6, tag="A")  # 12 条
    stale = cm.build(long_run)
    assert stale.summarized_turns == 8
    calls_before = len(summarize_calls(llm))

    rolled_back = long_run[:4] + chat_turns(2, tag="Z")  # 8 条，早期部分只剩 4 条
    fresh = cm.build(rolled_back, prev=stale)

    assert len(summarize_calls(llm)) == calls_before + 1, "旧摘要已越界，必须重建"
    assert fresh.summarized_turns == 4
    assert fresh.summarized_turns + len(fresh.window) == len(rolled_back)


def test_rehydrated_bundle_can_continue_incremental(monkeypatch):
    """跨进程恢复：storage 只存契约字段，用 link_prev 把增量状态接回来。"""
    from mewhelp.memory import link_prev

    llm = MockLLMClient()
    cm = ContextManager(llm, make_settings(window_turns=2, summary_trigger_tokens=50))
    history = chat_turns(4, tag="A")
    bundle = cm.build(history)
    assert len(summarize_calls(llm)) == 1

    # 假装 bundle 落库后又被读出来（私有增量状态丢失）
    restored = ContextBundle(
        summary=bundle.summary,
        window=list(bundle.window),
        summarized_turns=bundle.summarized_turns,
    )
    link_prev(restored, bundle)

    history = history + chat_turns(1, tag="B")
    again = cm.build(history, prev=restored)
    assert len(summarize_calls(llm)) == 1, "接回状态后不该再摘一次"
    assert again.summary == bundle.summary


# ============================================================
# 5. 信息不丢 —— 长对话后早期订单号仍可见
# ============================================================


def test_early_order_id_survives_long_conversation():
    llm = MockLLMClient()
    cm = ContextManager(llm, make_settings(window_turns=3, summary_trigger_tokens=30))

    # window_turns=3 是"3 轮"，即窗口长 6 条消息。所以最早的订单号必须再过两轮
    # 才会滑出窗口 —— 这条测试顺带把"窗口按轮计量"钉死。
    head = [
        mk("user", "我的订单是 ORD-778899，金额 199 元，现在想申请退款"),
        mk("assistant", "收到，ORD-778899 的退款我帮你走流程"),
    ]
    history = head + chat_turns(1, tag="B")  # 4 条：订单号还在滑窗里
    still_windowed = cm.build(history)
    assert still_windowed.summary == ""
    assert any("ORD-778899" in t.content for t in still_windowed.window)

    history = history + chat_turns(1, tag="C")  # 6 条：刚好填满窗口
    filled = cm.build(history, prev=still_windowed)
    assert filled.summary == ""

    history = history + chat_turns(1, tag="D")  # 8 条：订单号那轮终于滑出
    bundle = cm.build(history, prev=filled)
    assert bundle.summary_generated is True
    assert "ORD-778899" in bundle.summary  # 确实压进去了
    assert "退款" in bundle.summary

    # 接着聊几十轮，每轮都把上一轮的 bundle 续上（真实链路就是这么做的）
    for i in range(20):
        history = history + chat_turns(1, tag=f"N{i}")
        bundle = cm.build(history, prev=bundle)

    blob = "\n".join(m["content"] for m in bundle.as_messages())
    window_blob = "\n".join(t.content for t in bundle.window)

    # 关键断言：早期订单号早就不在滑窗里了，
    # 但仍必须能被模型看见 —— 这就是"双层"存在的唯一理由。
    assert "ORD-778899" not in window_blob, "订单号还在滑窗里，测试前提不成立"
    assert "ORD-778899" in blob
    assert "退款" in blob
    assert bundle.summarized_turns + len(bundle.window) == len(history)


def test_multiple_early_order_ids_all_retained():
    """多张订单号分散在早期轮次里，压完之后一张都不能丢。"""
    llm = MockLLMClient()
    cm = ContextManager(llm, make_settings(window_turns=1, summary_trigger_tokens=20))

    history = [
        mk("user", "ORD-100001 这单我要退款"),
        mk("assistant", "好的已登记 ORD-100001"),
        mk("user", "还有 ORD-100002 的物流到哪了"),
        mk("assistant", "ORD-100002 已发货"),
    ] + chat_turns(1, tag="B")
    bundle = cm.build(history)
    assert "ORD-100001" in bundle.summary

    for i in range(6):
        history = history + chat_turns(1, tag=f"M{i}")
        bundle = cm.build(history, prev=bundle)

    blob = "\n".join(m["content"] for m in bundle.as_messages())
    assert "ORD-100001" in blob
    assert "ORD-100002" in blob


# ============================================================
# 6. 度量 —— 真的省了 token
# ============================================================


def test_token_estimate_grows_with_turns_but_stays_cheaper_than_full_history():
    llm = MockLLMClient()
    cm = ContextManager(llm, make_settings(window_turns=3, summary_trigger_tokens=20))

    # 每轮带不同订单号：摘要里的实体随轮数累积，token_estimate 才会真的增长。
    # （重复寒暄会被 mock 去重压掉 —— 那是 mock 的正当行为，不适合用来测增长。）
    body = "，另外想确认一下退款要几天到账、运费怎么算，谢谢"
    history: list[Turn] = []
    bundle: ContextBundle | None = None
    snapshots: list[tuple[int, int, int]] = []  # (轮数, 双层估算, 全量估算)
    for i in range(15):
        history.append(mk("user", f"ORD-3000{i:02d} 帮我查一下这单的退款进度{body}"))
        history.append(mk("assistant", f"ORD-3000{i:02d} 的退款进度已经帮你查到了{body}"))
        bundle = cm.build(history, prev=bundle)
        snapshots.append((len(history), bundle.token_estimate, turns_token_estimate(history)))

    assert bundle is not None and bundle.summary != ""
    estimates = [s[1] for s in snapshots]
    fulls = [s[2] for s in snapshots]

    # 增长性：历史越长，需要承载的上下文越多（摘要里的实体在累积）
    assert estimates[-1] > estimates[0], f"token_estimate 没有随轮数增长: {estimates}"
    # 窗口填满之前（消息数 ≤ 2*window_turns）还没轮次滑出，双层等于全量，谈不上省；
    # 从窗口开始溢出的那一刻起，必须一直更省 —— 这两段分开断言才是诚实的度量。
    filled = next(i for i, s in enumerate(snapshots) if s[0] > 6)
    assert all(snapshots[i][1] == snapshots[i][2] for i in range(filled))
    assert all(
        e < f for e, f in zip(estimates[filled:], fulls[filled:])
    ), f"双层反而更贵: {estimates} vs {fulls}"
    # 长对话下省的幅度要显著（否则这个设计不值得存在）
    assert estimates[-1] * 2 < fulls[-1], f"没省下一半: 双层={estimates[-1]} 全量={fulls[-1]}"


def test_token_estimate_counts_summary_and_window_only():
    llm = MockLLMClient()
    cm = ContextManager(llm, make_settings(window_turns=1, summary_trigger_tokens=30))
    turns = chat_turns(5)
    bundle = cm.build(turns)
    assert bundle.summary != ""
    assert bundle.token_estimate == estimate_tokens(bundle.summary) + turns_token_estimate(
        bundle.window
    )
    # 被滑出且未进摘要的轮次不算在内 —— 它们确实没发出去
    assert bundle.token_estimate < turns_token_estimate(turns)


def test_estimate_tokens_calibration_matches_llm_mock():
    """口径必须和 mock 记账一致，否则"省了多少"这个数字没有可比性。"""
    from mewhelp.llm import _rough_tokens

    for text in ["", "你好", "帮我查一下订单 ORD-1001 的状态", "hello world", "退款 199 元"]:
        assert estimate_tokens(text) == _rough_tokens(text), text
    assert estimate_tokens("") == 0
    # 中文按 1 字 1 token 起算，再叠加固定的单条消息开销（+1，与 mock 记账同口径）
    assert estimate_tokens("退") == 2
    assert estimate_tokens("退退款") == 4


# ============================================================
# 7. 异常兜底
# ============================================================


@pytest.mark.parametrize(
    "exc",
    [LLMError("上游 500"), ValueError("响应格式不对"), RuntimeError("连接被重置")],
)
def test_llm_failure_degrades_to_empty_summary(exc: Exception):
    llm = FlakyLLM(exc)
    cm = ContextManager(llm, make_settings(window_turns=2, summary_trigger_tokens=10))

    turns = chat_turns(4)
    bundle = cm.build(turns)  # 不许抛

    assert bundle.summary == ""
    assert bundle.summary_generated is False
    assert bundle.summarized_turns == 0
    # 降级不等于失能：滑窗原文照常保留，请求还能继续跑。
    assert bundle.window == turns[-4:]
    assert bundle.token_estimate == turns_token_estimate(turns[-4:])
    assert all(m["role"] != "system" for m in bundle.as_messages())


def test_empty_summary_response_is_treated_as_failure():
    """模型返回空串 = 什么都没压出来，不能记成"已摘要"。"""
    llm = SilentLLM()
    cm = ContextManager(llm, make_settings(window_turns=1, summary_trigger_tokens=10))
    bundle = cm.build(chat_turns(3))

    assert bundle.summary == ""
    assert bundle.summary_generated is False
    assert bundle.summarized_turns == 0


def test_degraded_bundle_allows_retry_next_round():
    """降级之后不能把"摘要覆盖范围"记死，否则会永久跳过早期轮次。"""
    flaky = FlakyLLM(LLMError("boom"))
    cm_bad = ContextManager(flaky, make_settings(window_turns=2, summary_trigger_tokens=10))
    turns = chat_turns(3)
    degraded = cm_bad.build(turns)

    assert degraded.summary == ""
    assert context_of(degraded)["trusted"] is False

    # 下一轮换了健康的模型：早期轮次必须被重新压进摘要
    healthy = MockLLMClient()
    cm_ok = ContextManager(healthy, make_settings(window_turns=2, summary_trigger_tokens=10))
    turns = turns + chat_turns(1, tag="B")
    recovered = cm_ok.build(turns, prev=degraded)

    assert recovered.summary != ""
    assert recovered.summary_generated is True
    assert recovered.summarized_turns == len(turns) - 4


def test_zero_window_compresses_everything():
    """window_turns=0 是允许的（极省 token 的档位）：全部轮次都走摘要。"""
    llm = MockLLMClient()
    cm = ContextManager(llm, make_settings(window_turns=0, summary_trigger_tokens=10))

    turns = chat_turns(3)
    bundle = cm.build(turns)

    assert bundle.window == []
    assert bundle.summary != ""
    assert bundle.summary_generated is True
    assert bundle.summarized_turns == len(turns)
    assert bundle.token_estimate == estimate_tokens(bundle.summary)
    assert bundle.token_estimate < turns_token_estimate(turns)


def test_window_boundary_ignores_non_user_messages():
    """工具消息/system 提示不能把窗口切歪 —— 否则指代消解会从半截开始。"""
    llm = MockLLMClient()
    cm = ContextManager(llm, make_settings(window_turns=2, summary_trigger_tokens=10**6))

    turns = [
        mk("user", "第一轮问题"),
        mk("assistant", "第一轮回答"),
        mk("user", "第二轮问题"),
        mk("assistant", "第二轮回答"),
        mk("tool", "查到的订单：ORD-1234"),
        mk("assistant", "根据订单，答案是……"),
        mk("user", "第三轮问题"),
        mk("assistant", "第三轮回答"),
    ]
    bundle = cm.build(turns)

    # 最近 2 轮 = 从"第二轮问题"起到末尾（含中间的工具消息），共 6 条
    assert len(bundle.window) == 6
    assert bundle.window[0].content == "第二轮问题"
    assert bundle.window[-1].content == "第三轮回答"
    assert any(t.role == "tool" for t in bundle.window), "工具消息属于那一轮，必须一起保留"
    # 窗口之前的 2 条被滑出（阈值拉高所以没有摘要）
    assert bundle.summary == ""


def test_empty_history_is_safe():
    cm = ContextManager(MockLLMClient(), make_settings())
    bundle = cm.build([])
    assert bundle.window == []
    assert bundle.summary == ""
    assert bundle.token_estimate == 0
    assert bundle.as_messages() == []


def test_build_does_not_mutate_input_turns():
    """build 只读不写 —— 节点手里同一份 history 可能还在被别的分支读。"""
    cm = ContextManager(MockLLMClient(), make_settings(window_turns=1, summary_trigger_tokens=10))
    turns = chat_turns(3)
    before = [(t.role, t.content) for t in turns]
    cm.build(turns)
    assert [(t.role, t.content) for t in turns] == before


# ============================================================
# 8. append
# ============================================================


def test_append_is_pure_and_returns_new_list():
    cm = ContextManager(MockLLMClient(), make_settings())
    session = [mk("user", "你好")]
    snapshot = list(session)

    out = cm.append(session, mk("assistant", "在的"))

    assert session == snapshot, "append 不能改动调用方手里的 list"
    assert out is not session
    assert len(out) == 2
    assert out[-1].content == "在的"


def test_append_caps_history_growth():
    cm = ContextManager(MockLLMClient(), make_settings(window_turns=3))
    session: list[Turn] = []
    for i in range(200):
        session = cm.append(session, mk("user", f"第{i}轮"))

    # 上限必须存在：无上限的会话列表就是内存泄漏，
    # 而滑窗/摘要只需要最近若干轮，留下远大于窗口的余量即可。
    assert len(session) == 60
    assert session[-1].content == "第199轮"
    assert session[0].content == "第140轮"


def test_append_then_build_keeps_metadata():
    from mewhelp.domain import Intent

    cm = ContextManager(MockLLMClient(), make_settings())
    session = cm.append([], Turn(role="user", content="我要退款", intent=Intent.REFUND_REQUEST))
    session = cm.append(session, mk("assistant", "好的"))

    bundle = cm.build(session)
    assert bundle.window[0].intent is Intent.REFUND_REQUEST
    assert bundle.as_messages()[0] == {"role": "user", "content": "我要退款"}
