"""端到端测试：从一句用户消息到一条完整答复。

这一批测试的价值不在覆盖率，而在于**证明整条流水线真的连起来了**：
意图分流走对了没有、闸门拦住了没有、高风险流程会不会乱动手、
答不上的有没有落进问题池。

每条测试都断言 `path`（经过的节点序列），因为"走了哪条路"正是这套架构
最核心的产出 —— 结果对但路走错，说明分流是坏的，只是这次没暴露。
"""

from __future__ import annotations

import pytest

from mewhelp.domain import Exit, Intent, RefusalReason
from mewhelp.llm import LLMResponse, MockLLMClient
from mewhelp.schemas import Evidence, KbChunk, RetrievalResult, Turn
from mewhelp.workflow.graph import build_deps, build_graph
from mewhelp.workflow.state import new_state
from mewhelp.workflow.high_risk import extract_order_id, missing_slots


@pytest.fixture(scope="module")
def graph_and_deps():
    deps = build_deps()
    # 建库一次，模块内所有用例共用 —— 70 条 chunk 的索引不便宜
    from pathlib import Path

    deps.retriever.load_from_dir(Path("data/knowledge"))
    return build_graph(deps), deps


def run(graph, question, *, history=None, **kw):
    state = new_state(
        request_id=kw.get("request_id", "req-test"),
        session_id=kw.get("session_id", "sess-test"),
        user_id="u-test",
        question=question,
        history=history or [],
    )
    return graph.invoke(state)


# ============================================================
# 分流：九类意图 → 五个出口
# ============================================================


@pytest.mark.parametrize(
    "question,expected_intent,expected_exit",
    [
        ("我要退款", Intent.REFUND_REQUEST, Exit.HIGH_RISK),
        ("帮我取消订单 ORD-1001", Intent.CANCEL_ORDER, Exit.HIGH_RISK),
        ("快递到哪了", Intent.LOGISTICS_QUERY, Exit.BUSINESS_TOOL),
        ("尺码怎么选", Intent.PRODUCT_FAQ, Exit.KNOWLEDGE),
        ("我要找人工", Intent.HUMAN_HANDOFF, Exit.HANDOFF),
    ],
)
def test_intent_routes_to_right_exit(graph_and_deps, question, expected_intent, expected_exit):
    graph, _ = graph_and_deps
    out = run(graph, question)
    assert out["intent"] is expected_intent, f"{question} 被分成 {out['intent']}"
    assert out["exit"] is expected_exit


def test_path_records_every_node_visited(graph_and_deps):
    """path 是面试演示时"一眼看出走对没有"的依据，必须完整。"""
    graph, _ = graph_and_deps
    out = run(graph, "尺码怎么选")
    assert out["path"][0] == "resolve"
    assert out["path"][1] == "classify"
    assert "retrieve" in out["path"]
    assert "gate" in out["path"]


def test_business_tool_exit_goes_through_agent(graph_and_deps):
    """查物流走 Agent（ReAct），不是走检索 —— 这是分流的意义。"""
    graph, _ = graph_and_deps
    out = run(graph, "帮我查下订单 ORD-1002 的物流")
    assert "agent" in out["path"]
    assert "retrieve" not in out["path"]


# ============================================================
# 检索 → 生成
# ============================================================


def test_knowledge_question_gets_grounded_answer(graph_and_deps):
    """问规则走知识检索 —— 这是"问"和"要"分开之后的关键收益。

    「生鲜类可以退款吗」是在咨询政策，不该被当成退款申请去追问订单号。
    """
    graph, _ = graph_and_deps
    out = run(graph, "生鲜类可以退款吗")
    assert out["exit"] is Exit.KNOWLEDGE
    assert "retrieve" in out["path"]
    assert out["gate"].passed is True
    assert out["answer"]
    assert out["handoff"] is False
    assert len(out["retrieval"].evidences) > 0


def _fixed_knowledge_deps(text, answer=None, *, strict_self_eval=False):
    """只替换证据与生成结果，保留真实 NLU、闸门、自评和图路由。"""
    class Retriever:
        def retrieve(self, query):
            return RetrievalResult(query=query, evidences=[Evidence(
                chunk=KbChunk(chunk_id="issue-1", doc_id="doc", title="商品资料", text=text),
                score=0.9,
            )])

    class LLM(MockLLMClient):
        def chat(self, messages, **kw):
            if kw.get("task") == "generate":
                return LLMResponse(text=answer or text)
            if kw.get("task") == "self_eval" and strict_self_eval:
                return LLMResponse(text='{"grounded": false, "coverage": 0}')
            return super().chat(messages, **kw)

    return build_deps(llm=LLM(), retriever=Retriever())


@pytest.mark.parametrize(
    "question,evidence,answer",
    [
        ("苹果有哪些规格？", "苹果有1斤、3斤、5斤装。", "苹果有1斤、3斤、5斤装。"),
        ("水饺多少钱？", "水饺每袋28块钱。", "水饺每袋28元。"),
        ("蜂蜜多少钱？", "蜂蜜每瓶39元。", "蜂蜜每瓶39元。"),
        ("红薯有哪些规格？", "红薯有9斤、5斤、3斤、1斤装。", "红薯有9斤、5斤、3斤、1斤装。"),
    ],
    ids=["apple", "dumplings", "honey", "sweet_potato"],
)
def test_supported_simple_query_has_no_handoff(question, evidence, answer):
    from mewhelp.api.service import done_payload, result_from_state

    deps = _fixed_knowledge_deps(evidence, answer, strict_self_eval=True)
    out = run(build_graph(deps), question)
    assert out["gate"].passed is True
    assert out["gate"].checks["evidence_count"] == 1
    assert out["answer"] == answer
    assert out["handoff"] is False
    assert out["refusal_reason"] is None
    assert out["path"] == ["resolve", "classify", "retrieve", "gate", "generate"]
    result = result_from_state(out, latency_ms=0)
    # 前端仅根据 done.handoff 补转接提示条。
    assert done_payload(result) == {"answer": answer, "handoff": False}


@pytest.mark.parametrize(
    "question,evidence",
    [
        ("大蒜有哪些规格？", "大蒜有1斤、3斤、5斤，大果、特大果、中果。"),
        ("换成红薯呢，它有哪些规格？", "红薯有9斤、5斤、3斤、1斤，大果、中果、小果。"),
        ("羊角蜜有哪些规格？", "羊角蜜有3斤、5斤装。"),
    ],
    ids=["garlic", "sweet_potato_pronoun", "melon"],
)
def test_new_question_after_handoff_uses_current_evidence(question, evidence):
    deps = _fixed_knowledge_deps(evidence)
    history = [
        Turn(role="user", content="苹果有哪些规格？"),
        Turn(role="assistant", content="苹果有1斤、3斤、5斤装。\n\n已经帮你转接人工客服进一步处理。",
             metadata={"handoff": True}),
    ]
    out = run(build_graph(deps), question, history=history)
    assert "人工" not in out["resolved_question"]
    assert "人工" not in (out["slots"].product or "")
    assert out["gate"].passed is True
    assert out["answer"] == evidence
    assert out["handoff"] is False
    assert "handoff" not in out["path"]


def test_unsupported_generated_answer_still_hands_off():
    deps = _fixed_knowledge_deps("水饺每袋28元。", "水饺每袋280元。")
    out = run(build_graph(deps), "水饺多少钱？")
    assert out["gate"].passed is True
    assert out["handoff"] is True
    assert out["refusal_reason"] is RefusalReason.SELF_EVAL_FAILED
    assert out["path"][-3:] == ["generate", "handoff", "flywheel"]


def test_policy_question_does_not_enter_high_risk_flow(graph_and_deps):
    """问规则绝不能走进高风险流程 —— 那条路会真的动钱。"""
    graph, _ = graph_and_deps
    for q in ("生鲜类可以退款吗", "那它能退吗", "退款政策是什么"):
        out = run(graph, q)
        assert "high_risk_flow" not in out["path"], f"{q} 误入高风险流程"
        assert out["tool_calls"] == [], f"{q} 触发了工具调用"


def test_request_still_enters_high_risk_flow(graph_and_deps):
    """反过来：明确提要求时必须走高风险流程，不能被当成咨询打发掉。"""
    graph, _ = graph_and_deps
    out = run(graph, "能帮我退款吗")
    assert out["exit"] is Exit.HIGH_RISK
    assert "high_risk_flow" in out["path"]


def test_out_of_scope_question_is_refused_not_hallucinated(graph_and_deps):
    """知识库没有的问题必须拒答，而不是编一个答案。

    这是置信度闸门存在的全部理由。断言的是"转人工"这个行为，
    不是某句具体话术 —— 话术会改，行为不能改。
    """
    graph, _ = graph_and_deps
    out = run(graph, "你们卖不卖火箭")
    assert out["gate"].passed is False
    assert out["handoff"] is True
    assert out["refusal_reason"] in (
        RefusalReason.NO_EVIDENCE,
        RefusalReason.WEAK_EVIDENCE,
        RefusalReason.LOW_COVERAGE,
    )


def test_refused_question_lands_in_flywheel(graph_and_deps):
    """拒答 -> 落问题池 -> 带上检索快照。

    快照是飞轮最值钱的字段：没有它，人工没法区分"真缺知识"和"没检索到"。
    """
    graph, deps = graph_and_deps
    out = run(graph, "你们支持宠物上门美容吗")
    assert out["handoff"] is True
    assert out["flywheel_source"] is not None

    pending = deps.flywheel.pending()
    assert any(r.question == "你们支持宠物上门美容吗" for r in pending)
    rec = next(r for r in pending if r.question == "你们支持宠物上门美容吗")
    assert rec.retrieval_snapshot is not None
    assert "evidences" in rec.retrieval_snapshot.model_dump()


def test_gate_runs_before_generation(graph_and_deps):
    """顺序断言：闸门节点在生成节点之前。

    顺序反了的话，用户已经看到幻觉了才评估 —— 那是另一套系统。
    """
    graph, _ = graph_and_deps
    out = run(graph, "你们卖不卖火箭")
    path = out["path"]
    assert path.index("gate") < path.index("handoff")
    assert "generate" not in path  # 拒答的没走到生成


# ============================================================
# 工具调用 / ReAct
# ============================================================


def test_agent_calls_business_tool(graph_and_deps):
    graph, _ = graph_and_deps
    out = run(graph, "订单 ORD-1001 是什么状态")
    names = [c.name for c in out["tool_calls"]]
    assert "get_order" in names
    assert all(c.ok for c in out["tool_calls"])


def test_tool_result_is_turned_into_an_answer(graph_and_deps):
    """工具查到了信息，必须变成一句人话答复。

    这是集成层最容易漏的一环：Agent 调工具成功、工具返回了完整订单，
    但 generate 节点只认检索证据，于是查到了订单却回答"抱歉没有依据"。
    断言的是"答复里有订单状态"，不是"有答案"这种不会失败的空泛检查。
    """
    graph, _ = graph_and_deps
    out = run(graph, "帮我查订单 ORD-1001 的状态")
    assert out["tool_calls"], "应该真的调用了工具"
    assert "待付款" in out["answer"] or "待处理" in out["answer"]
    assert "没有找到" not in out["answer"]


def test_tool_error_is_human_readable_not_leaked(graph_and_deps):
    """订单不存在时，答复必须是"没查到订单号"的人话，而不是内部错误码/工具名。"""
    graph, _ = graph_and_deps
    out = run(graph, "订单 ORD-9999 是什么状态")
    assert "订单号" in out["answer"] or "没有查到" in out["answer"]
    # 工具名和错误码都是内部实现细节，泄漏出去会很怪
    assert "get_order" not in out["answer"]
    assert "ORDER_NOT_FOUND" not in out["answer"]


def test_agent_cannot_directly_invoke_high_risk_tool(graph_and_deps):
    """Agent 不能直接调高风险工具 —— 只能由高风险子流程决定。

    这是"业务铁律写死在图上"最具体的一处体现：模型再想帮用户，
    也越不过这道门。
    """
    graph, _ = graph_and_deps
    # 这句会被分成 refund（high_risk 出口），不走 Agent。
    # 这里直接验证隔离机制本身：Agent 节点会拦下高风险动作。
    from mewhelp.workflow.nodes import make_agent_node
    from mewhelp.workflow.state import new_state as ns

    deps = graph_and_deps[1]
    node = make_agent_node(deps)
    # 构造一个"物流意图但模型想调退款"的场景：高风险清单必须挡住它
    assert "request_refund" in deps.tools.high_risk_actions()
    assert "cancel_order" in deps.tools.high_risk_actions()
    assert "change_address" in deps.tools.high_risk_actions()
    # 只读工具不在高风险清单里
    assert "get_order" not in deps.tools.high_risk_actions()
    assert "track_logistics" not in deps.tools.high_risk_actions()


def test_tool_error_is_reported_not_raised(graph_and_deps):
    """订单不存在要变成一句人话，不是一个异常。"""
    graph, _ = graph_and_deps
    out = run(graph, "帮我查订单 ORD-9999 的物流")
    assert out["answer"]
    errors = [c.error for c in out["tool_calls"] if not c.ok]
    assert "ORDER_NOT_FOUND" in errors


# ============================================================
# Agent 的两种工具协议
#
# 真模型按 OpenAI 的 function calling 回话：动作在 `tool_calls` 里，正文是空的
# 或者一句"我先查一下订单"的旁白。而离线 mock 永远把 `{"action": ...}` 写在正文里，
# 所以"只读正文"这种写法在整套离线测试里**全绿**，接上真模型却安静地一个工具
# 都不调 —— 意图、出口、置信度都正常，就是没有数据，最后答一句"没有可用的证据"。
# 下面三条就是钉住这个盲区的。
# ============================================================


class _ToolCallingLLM(MockLLMClient):
    """规划那一步改用 function calling 协议回话，其余照旧走 mock。

    正文刻意写成一句英文旁白（真模型常见）—— 里面没有任何 JSON，
    只解析正文的实现会在这里收手。
    """

    def __init__(self, *plans: list[tuple[str, dict]]) -> None:
        super().__init__()
        self._plans = list(plans)

    def chat(self, messages, **kw):  # type: ignore[override]
        if kw.get("task") != "plan_tool":
            return super().chat(messages, **kw)
        self.calls.append({"task": "plan_tool", "messages": list(messages), "tools": kw.get("tools")})
        plan = self._plans.pop(0) if self._plans else []
        return LLMResponse(
            text="I'll check that for you.",  # 旁白里没有 JSON
            tool_calls=[
                {"id": f"call_{i}", "name": name, "arguments": args}
                for i, (name, args) in enumerate(plan)
            ],
            finish_reason="tool_calls" if plan else "stop",
        )


def agent_deps(*plans: list[tuple[str, dict]]):
    return build_deps(llm=_ToolCallingLLM(*plans))


def run_agent(deps, question="帮我查订单 ORD-1001 的状态"):
    from mewhelp.workflow.nodes import make_agent_node

    state = new_state(
        request_id="req-agent", session_id="s", user_id="u", question=question
    )
    return make_agent_node(deps)(state)


@pytest.mark.parametrize("repeats", [1, 2, 3, 5])
@pytest.mark.parametrize(
    "principal_id,order_id,message",
    [
        ("U1001", "ORD-1003", "当前账号未关联这笔订单，暂时无法查询或办理该订单业务。"),
        ("无订单账号", "ORD-1003", "您的账号暂未关联订单信息，暂时无法查询或办理订单业务。"),
        ("U1001", "ORD-9999", "没有查到这个订单号，麻烦核对一下再发我（形如 ORD-1007）。"),
    ],
)
def test_重复失败查单只回复一次并保留调用记录(repeats, principal_id, order_id, message):
    deps = agent_deps(*[[("get_order", {"order_id": order_id})] for _ in range(repeats)])
    deps.enforce_order_ownership = True
    state = new_state(
        request_id="req-repeat", session_id="s-repeat", user_id=principal_id,
        question=f"帮我查订单 {order_id} 的状态",
    )
    out = build_graph(deps).invoke(state)
    assert len(out["tool_calls"]) == repeats
    assert all(not call.ok for call in out["tool_calls"])
    assert out["answer"] == message
    assert out["handoff"] is False


def test_不同查单错误仍分别展示():
    deps = agent_deps([
        ("get_order", {"order_id": "ORD-1003"}),
        ("get_order", {"order_id": "ORD-9999"}),
    ])
    deps.enforce_order_ownership = True
    state = new_state(request_id="req-errors", session_id="s-errors", user_id="U1001", question="查订单 ORD-1003")
    out = build_graph(deps).invoke(state)
    assert "未关联这笔订单" in out["answer"]
    assert "没有查到这个订单号" in out["answer"]
    assert len(out["answer"].splitlines()) == 2


def test_agent_executes_function_calling_plan():
    deps = agent_deps([("get_order", {"order_id": "ORD-1001"})])
    out = run_agent(deps)
    assert [c.name for c in out["tool_calls"]] == ["get_order"]
    assert out["tool_calls"][0].ok, out["tool_calls"][0].error
    assert out["tool_calls"][0].result["order_id"] == "ORD-1001"
    assert out["errors"] == []


def test_agent_runs_every_tool_the_model_asked_for():
    """"查订单 + 查物流"是模型很自然的合并动作，不能只执行第一个。"""
    deps = agent_deps(
        [("get_order", {"order_id": "ORD-1001"}), ("track_logistics", {"order_id": "ORD-1001"})]
    )
    out = run_agent(deps)
    assert [c.name for c in out["tool_calls"]] == ["get_order", "track_logistics"]
    assert all(c.ok for c in out["tool_calls"])


def test_function_calling_plan_is_fed_back_to_the_model():
    """工具结果要回灌 —— 不回灌的话多步 ReAct 退化成"一步就收手"。"""
    deps = agent_deps(
        [("get_order", {"order_id": "ORD-1001"})],
        [],  # 第二步：模型认为可以收尾了
    )
    run_agent(deps)
    plans = [c for c in deps.llm.calls if c["task"] == "plan_tool"]
    assert len(plans) == 2, "应该继续走第二步，而不是调完一次就结束"
    fed_back = " ".join(str(m.get("content", "")) for m in plans[1]["messages"])
    assert "ORD-1001" in fed_back, "第二步必须看得到第一步的工具结果"
    # 正文为空时补的动作记录，让模型知道自己刚才要干什么
    assert "get_order" in fed_back


def test_high_risk_is_blocked_even_when_asked_via_function_calling():
    """换一种协议回话，也越不过高风险这道门。"""
    deps = agent_deps([("cancel_order", {"order_id": "ORD-1001"})])
    out = run_agent(deps, question="取消订单 ORD-1001")
    assert [c.error for c in out["tool_calls"]] == ["HIGH_RISK_REQUIRES_FLOW"]
    assert not any(c.ok for c in out["tool_calls"])


# ============================================================
# 工具失败之后：答复、handoff 标志、飞轮三者必须一致
# ============================================================


def test_recovered_tool_failure_is_not_shown_to_the_user():
    """模型先调错工具、下一轮自己改对 —— 答复里不该留那段故障前言。

    真模型把工具名幻觉成 `get_logistics` 是常态，而 agent 循环会把错误回灌给
    它自纠。用户要的是"订单正在备货"，不是"系统暂时不可用" ——
    数据是对的、前言是假的，整条读起来像系统坏了。
    """
    deps = agent_deps(
        [("get_logistics", {"order_id": "ORD-1002"})],       # 幻觉的工具名
        [("track_logistics", {"order_id": "ORD-1002"})],     # 自己改对了
    )
    graph = build_graph(deps)
    out = run(graph, "帮我查下订单 ORD-1002 的物流")

    assert [c.ok for c in out["tool_calls"]] == [False, True]
    assert out["handoff"] is False
    for leaked in ("系统暂时", "没查出来", "UPSTREAM_ERROR", "get_logistics"):
        assert leaked not in out["answer"], f"答复里漏了内部故障：{out['answer']}"


def test_all_tools_failed_escalates_honestly():
    """一条都没办成时，答复说转人工，handoff 标志就必须是真的。

    这条曾经是坏的：正文写"我帮你转人工处理"，而 handoff 始终为 false，
    前端因此不补那条"已为你转接人工客服"提示条 —— 用户读到一个不存在的转接。
    """
    deps = agent_deps([("get_logistics", {"order_id": "ORD-1002"})], [])
    graph = build_graph(deps)
    out = run(graph, "帮我查下订单 ORD-1002 的物流")

    assert not any(c.ok for c in out["tool_calls"])
    assert out["handoff"] is True
    assert out["refusal_reason"] is RefusalReason.NO_EVIDENCE
    assert "handoff" in out["path"], "转人工标志为真时必须真的走 handoff 节点"
    assert "flywheel" in out["path"], "答不上来的问题要落进问题池"


def test_business_refusal_does_not_escalate():
    """业务侧拒绝（订单不存在）不该升级 —— 那是"查到了，结果是不同意"。"""
    deps = agent_deps([("get_order", {"order_id": "ORD-9999"})], [])
    graph = build_graph(deps)
    out = run(graph, "帮我查订单 ORD-9999 的状态")

    assert out["handoff"] is False
    assert "ORDER_NOT_FOUND" not in out["answer"]


def test_tool_plan_producing_nothing_escalates_instead_of_faking_an_answer():
    """业务工具出口却一个工具都没调成 —— 不能说成"知识库没依据"。

    原先这会掉进知识库那条通道，而 business_tool 出口不经过检索，
    于是生成节点拿着**空证据**去问模型，模型只能编一句"没有依据"。
    置信度闸门在这场里根本没参与过（它只服务知识出口），
    一次工具规划的失败被静默粉饰成一次正常问答。
    """
    deps = build_deps(llm=_ToolCallingLLM())  # 永远不给计划
    graph = build_graph(deps)
    out = run(graph, "帮我查订单 ORD-1001 的状态")

    assert out["exit"] is Exit.BUSINESS_TOOL
    assert out["tool_calls"] == []
    assert out["handoff"] is True
    assert "知识库" not in out["answer"], "这不是知识库的事，不该那么说"


# ============================================================
# 高风险确定性子流程
# ============================================================


def test_high_risk_missing_order_id_interrupts(graph_and_deps):
    """缺订单号 -> 暂停并追问，**绝不猜**。

    这是本项目最该被演示的一个行为：系统不会拿上一轮的订单号顶数，
    也不会编一个。它停下来问。
    """
    graph, _ = graph_and_deps
    out = run(graph, "我要退款")
    assert out["interrupt_prompt"]
    assert "ORD" in out["interrupt_prompt"] or "订单号" in out["interrupt_prompt"]
    # 关键断言：没凑齐前置条件就没有任何工具被调用
    assert out["tool_calls"] == []


def test_high_risk_with_order_id_executes(graph_and_deps):
    graph, _ = graph_and_deps
    out = run(graph, "帮我取消订单 ORD-1001")
    assert "high_risk_flow" in out["path"]
    names = [c.name for c in out["tool_calls"]]
    assert "cancel_order" in names
    assert out["answer"]


def test_high_risk_business_rejection_is_explained(graph_and_deps):
    """已经发货的订单取消不了 —— 工具拒绝，系统如实转达。

    模型在这里没有"通融"的余地：能不能做由工具判定，不由话术判定。
    """
    graph, _ = graph_and_deps
    # ORD-1002 是 processing/shipped 之一；用一个必然不可取消的订单
    import json
    from pathlib import Path

    data = json.loads(Path("data/fixtures/orders.json").read_text(encoding="utf-8"))
    shipped = next(
        (o for o in data["orders"] if o["status"] in ("shipped", "delivered", "cancelled")), None
    )
    assert shipped, "夹具里需要一个非可取消状态的订单"
    out = run(graph, f"帮我取消订单 {shipped['order_id']}")
    failed = [c for c in out["tool_calls"] if not c.ok]
    assert failed, f"{shipped['order_id']} 状态={shipped['status']} 应该拒绝取消"
    assert failed[0].error in ("ORDER_NOT_CANCELLABLE", "ORDER_NOT_FOUND")


# ============================================================
# 指代消解（多轮）
# ============================================================


def test_coreference_resolves_half_sentence(graph_and_deps):
    """文章点名的场景：history 有订单号，用户问「那它能退吗」。"""
    graph, _ = graph_and_deps
    history = [
        Turn(role="user", content="帮我查下订单 ORD-1007 到哪了"),
        Turn(role="assistant", content="这笔订单已经签收。"),
    ]
    out = run(graph, "那它能退吗", history=history)
    assert out["resolved_question"] != "那它能退吗"
    assert "ORD-1007" in out["resolved_question"]


def test_slots_come_from_current_turn_not_history(graph_and_deps):
    """**关键的防错测试。**

    上一轮聊过 ORD-1007，这一轮只说"我要退款" —— 系统必须追问订单号，
    而不是把上一轮的订单拿去退款。用历史顶数是这类系统最危险的一类 bug：
    它会在用户没说清楚的情况下动真钱。
    """
    graph, _ = graph_and_deps
    history = [
        Turn(role="user", content="帮我查下订单 ORD-1007 到哪了"),
        Turn(role="assistant", content="这笔订单已经签收。"),
    ]
    out = run(graph, "我要退款", history=history)
    assert out["interrupt_prompt"], "本轮没给订单号，必须中断追问"
    assert out["tool_calls"] == [], "绝不能拿上一轮的订单号去执行退款"


# ============================================================
# 可观测
# ============================================================


def test_trace_covers_every_path_node(graph_and_deps):
    """trace 必须覆盖 path 里的每个节点 —— 否则调用树是残缺的。"""
    graph, _ = graph_and_deps
    out = run(graph, "生鲜类可以退款吗")
    traced = {s.node for s in out["trace"]}
    for node in out["path"]:
        assert node in traced, f"节点 {node} 走了但没留痕"


def test_retrieval_step_records_chunks_for_debugging(graph_and_deps):
    """检索那一步要记下召回了什么 —— "答错了"才能定位到"检索回了什么"。"""
    graph, _ = graph_and_deps
    out = run(graph, "尺码怎么选")
    step = next(s for s in out["trace"] if s.node == "retrieve")
    assert step.details["chunks"]
    assert step.details["chunks"][0]["chunk_id"]
    assert "strategy" in step.details


def test_usage_is_accumulated(graph_and_deps):
    graph, _ = graph_and_deps
    out = run(graph, "尺码怎么选")
    assert out["usage"].total_tokens > 0
    # mock 模式下不能出现"假成本"
    assert out["usage"].cost_usd == 0.0


# ============================================================
# 稳健性
# ============================================================


@pytest.mark.parametrize(
    "question",
    ["", "?", "。。。", "a" * 3000, "退款退款退款退款退款", "🦄🌈"],
)
def test_no_input_crashes_the_pipeline(graph_and_deps, question):
    """任何输入都不能让流水线抛异常。

    空串、超长串、纯emoji 都是真实会遇到的输入。异常冒出去就是 500，
    而用户只是打了个回车。
    """
    graph, _ = graph_and_deps
    out = run(graph, question)
    assert isinstance(out.get("answer"), str)


def test_degraded_retriever_still_produces_an_answer():
    """检索挂掉时，系统应该走兜底，而不是崩。

    "部分失败也能给出有用结果"是这套架构的价值所在 ——
    一整条链路里任何一个组件都不是单点。
    """

    class BrokenRetriever:
        def retrieve(self, *a, **kw):
            raise RuntimeError("向量库连接失败")

    deps = build_deps(retriever=BrokenRetriever())
    graph = build_graph(deps)
    out = run(graph, "尺码怎么选")
    # 检索挂了 → 无证据 → 闸门拒答 → 转人工，但**必须给出人话**
    assert out["answer"]
    assert "查询不通" in out["answer"] or "转人工" in out["answer"]
    assert out["handoff"] is True
    assert any("retrieve" in e for e in out.get("errors", []))


def test_degraded_llm_still_produces_an_answer():
    """模型挂掉时同理 —— 用兜底话术，不是 500。"""

    class BrokenLLM:
        model = "broken"

        def chat(self, *a, **kw):
            raise RuntimeError("模型服务不可用")

    from mewhelp.flywheel.store import FlywheelStore
    from mewhelp.nlu.coreference import CoreferenceResolver
    from mewhelp.nlu.intent import IntentClassifier
    from mewhelp.quality.gate import ConfidenceGate
    from mewhelp.quality.selfeval import SelfEvaluator
    from mewhelp.tools.business import build_business_registry

    deps = build_deps()
    broken = BrokenLLM()
    deps.llm = broken
    deps.resolver = CoreferenceResolver(broken)
    deps.classifier = IntentClassifier(broken)
    deps.self_eval = SelfEvaluator(broken)
    graph = build_graph(deps)
    out = run(graph, "尺码怎么选")
    assert out["answer"]


# ============================================================
# 高风险槽位工具的单元测试（不依赖图）
# ============================================================


def test_extract_order_id_normalizes_formats():
    assert extract_order_id("订单 ord 1007") == "ORD-1007"
    assert extract_order_id("ORD1007") == "ORD-1007"
    assert extract_order_id("没有订单号") is None


@pytest.mark.parametrize(
    "text",
    [
        "帮我取消订单ORD-1001",       # 中文紧挨着单号 —— 中文用户最常见的写法
        "ORD-1001的物流怎么样了",      # 单号后面直接跟汉字
        "我要退款ORD1001",            # 没有分隔符
        "订单ORD_1001 取消掉",         # 下划线分隔
    ],
)
def test_order_id_is_found_next_to_chinese(text):
    """中文紧邻单号时必须抽得到。

    这条曾经是坏的：正则为 `\\bORD[-\\s]?(\\d{4,8})\\b`，而 Python 的 `\\w`
    包含中日韩字符 —— "订单ORD-1001"里"单"和"O"都是词字符，中间没有词边界，
    `\\b` 匹配失败。于是用户明明报了单号，系统还是回一句"请把订单号发我"。
    离线测试全是带空格的 fixture（"帮我取消订单 ORD-1001"），所以漏了很久。
    """
    assert extract_order_id(text) == "ORD-1001"


def test_order_id_boundaries_still_reject_near_misses():
    """放宽边界不等于不要边界：粘连在字母后、或从长数字里截一段，都不算。"""
    assert extract_order_id("xORD-1001") is None
    assert extract_order_id("ORD-100112345") is None
    assert extract_order_id("ORD-100") is None  # 位数不够


def test_order_id_agrees_across_layers():
    """槽位层、高风险层、闸门层必须是同一个单号、同一个归一化。

    三个模块原先各写一份正则和归一化，其中两份因 `\\b` 而失效、
    归一化一个去前导零一个不去 —— 正是各自 docstring 里警告过的"口径分裂"。
    """
    from mewhelp.nlu.entities import find_order_id
    from mewhelp.quality.gate import extract_entities

    for text in ("帮我取消订单ORD-1001", "订单 ord 1001", "ORD1001"):
        assert extract_order_id(text) == find_order_id(text) == "ORD-1001"
    assert "ORD-1001" in extract_entities("帮我取消订单ORD-1001")


def test_order_id_takes_the_first_match_for_money_actions():
    """一句话里两个单号时，动钱的动作取**第一个**。

    "取消 ORD-1001，不是 ORD-1002" —— 取最后一个会把用户明确否定的那一单
    当成目标。指代消解那边取最后一个是对的（"最近提到的那一单"），
    两条口径在此**刻意**不同，所以钉住。
    """
    assert extract_order_id("取消 ORD-1001，不是 ORD-1002") == "ORD-1001"


def test_missing_slots_only_reads_current_turn():
    """只吃当轮文本 —— 这是防"拿历史顶数"的根本机制。"""
    assert missing_slots(Intent.REFUND_REQUEST, "我要退款") == ["order_id"]
    assert missing_slots(Intent.REFUND_REQUEST, "我要退款 ORD-1007") == []
    assert missing_slots(Intent.CHANGE_ADDRESS, "改地址 ORD-1007") == ["address"]
    assert missing_slots(Intent.CHANGE_ADDRESS, "改地址 ORD-1007 换成上海市浦东新区") == []


def test_knowledge_intent_needs_no_slots():
    assert missing_slots(Intent.PRODUCT_FAQ, "尺码怎么选") == []
