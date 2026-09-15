"""可观测层测试。

覆盖两条主线：
  * 关闭态（离线用户的默认状态）—— 必须零成本、零 import、功能可用；
  * 开启态 —— Langfuse 挂载失败/连不上时，降级不能反过来伤到业务。
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from mewhelp.config import Settings
from mewhelp.domain import NodeName, RefusalReason
from mewhelp.observability import Tracer, build_tracer, reset_tracer
from mewhelp.observability import tracing as tracing_mod
from mewhelp.schemas import (
    ConversationResult,
    Evidence,
    GateDecision,
    KbChunk,
    RetrievalResult,
    TokenUsage,
    ToolCall,
    TraceStep,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# 工具
# ============================================================
def offline_settings(**over) -> Settings:
    """离线配置：观测关闭、不碰网络。"""
    base = {"langfuse_enabled": False, "llm_mode": "mock"}
    base.update(over)
    return Settings(**base)


def enabled_settings(**over) -> Settings:
    """开启了观测、但指向一个必然连不上的 host 的配置。"""
    base = {
        "langfuse_enabled": True,
        "langfuse_public_key": "pk-test",
        "langfuse_secret_key": "sk-test",
        "langfuse_host": "http://127.0.0.1:1",
        "llm_mode": "mock",
    }
    base.update(over)
    return Settings(**base)


class FakeSpan:
    def __init__(self, box: dict, name: str) -> None:
        self.box = box
        self.name = name

    def update(self, **kw):
        self.box["updates"].append(kw)


class FakeSpanCM:
    def __init__(self, box: dict, name: str, explode: bool = False) -> None:
        self.box = box
        self.name = name
        self.explode = explode

    def __enter__(self):
        if self.explode:
            raise RuntimeError("span 开不起来")
        self.box["spans"].append(self.name)
        return FakeSpan(self.box, self.name)

    def __exit__(self, *exc_info):
        return False


def install_fake_langfuse(monkeypatch, client_factory) -> dict:
    """把假 langfuse 塞进 sys.modules —— 延迟导入会拿到它。

    这样能在不联网的前提下，精确模拟"SDK 正常 / SDK 报错 / SDK 连不上"三条路。
    """
    box = {"spans": [], "observed": [], "updates": [], "flushes": 0, "shutdowns": 0}
    mod = types.ModuleType("langfuse")
    mod.Langfuse = client_factory(box)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", mod)
    return box


def ok_client(box: dict):
    """正常的假 SDK：每个 step 对应一个 span，flush/shutdown 正常计数。"""

    class _Client:
        def __init__(self, **kwargs):
            box["init_kwargs"] = kwargs

        def start_as_current_observation(self, name, as_type=None, input=None, metadata=None):
            box["observed"].append(
                {"name": name, "as_type": as_type, "input": input, "metadata": metadata}
            )
            return FakeSpanCM(box, name)

        def flush(self):
            box["flushes"] += 1

        def shutdown(self):
            box["shutdowns"] += 1

    return _Client


def exploding_client(*, on_init: bool = False, on_span: bool = True):
    """坏掉的假 SDK —— 用来验证降级。"""

    class _Client:
        def __init__(self, **kwargs):
            if on_init:
                raise ConnectionError("langfuse 后端不可达")

        def start_as_current_observation(self, *a, **kw):
            if on_span:
                raise RuntimeError("SDK 版本不兼容")
            return FakeSpanCM({}, "x")

        def flush(self):
            raise RuntimeError("连不上")

        def shutdown(self):
            raise RuntimeError("关不掉")

    return _Client


# ============================================================
# 1. 关闭态：零 import / 零网络 / 完全可用
# ============================================================
def test_disabled_does_not_import_langfuse_subprocess():
    """在干净解释器里验证：关闭态的构造 + 使用全程不 import langfuse。

    必须在子进程里断言 `sys.modules`：同进程里别的测试可能已经把 langfuse
    导进来了，那种"脏"状态下断言不了这条最关键的约束。
    """
    code = (
        "import sys, json\n"
        "from mewhelp.config import Settings\n"
        "from mewhelp.observability import build_tracer\n"
        "t = build_tracer(Settings(langfuse_enabled=False))\n"
        "with t.step('retrieve', query='q') as st:\n"
        "    st.set_output('o')\n"
        "t.flush()\n"
        "print(json.dumps({\n"
        "  'langfuse': sorted(m for m in sys.modules if m.split('.')[0] == 'langfuse'),\n"
        "  'otel': sorted(m for m in sys.modules if m.split('.')[0] == 'opentelemetry'),\n"
        "  'httpx': 'httpx' in sys.modules,\n"
        "  'enabled': t.enabled,\n"
        "  'has_bridge': t._bridge is not None,\n"
        "  'steps': len(t.steps),\n"
        "}))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT),
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["langfuse"] == [], f"关闭态竟然导入了 langfuse: {out['langfuse']}"
    assert out["otel"] == [], f"关闭态竟然导入了 opentelemetry: {out['otel']}"
    assert out["enabled"] is False
    assert out["has_bridge"] is False  # 桥接对象都没建，自然没有网络和后台线程
    assert out["steps"] == 1


def test_disabled_in_process_keeps_langfuse_out_of_sys_modules():
    """同进程版的同一条约束（用前后快照对比，避免污染其它测试）。"""
    before = {m for m in sys.modules if m.split(".")[0] == "langfuse"}
    tracer = Tracer(settings=offline_settings(), enabled=False)
    with tracer.step(NodeName.RETRIEVE, query="运费怎么算"):
        pass
    tracer.flush()
    after = {m for m in sys.modules if m.split(".")[0] == "langfuse"}
    assert after - before == set(), f"关闭态新增了 import: {after - before}"


def test_disabled_step_and_flush_are_safe_noops():
    tracer = build_tracer(offline_settings())
    assert tracer.enabled is False
    with tracer.step("gate") as st:
        st.note(top_score=0.9)
    tracer.flush()  # 不报错
    tracer.flush()
    tracer.close()
    assert len(tracer.steps) == 1
    assert tracer.steps[0].details["top_score"] == 0.9


def test_disabled_still_collects_trace_in_memory():
    """关闭观测 ≠ 不留痕：本地 Tracer（`trace()`）还得有数据可返回。

    注意 `trace()` 是**进程内**的诊断能力，走的是 CLI / 评估脚本，
    不是暴露给前端的 HTTP 接口 —— 后者的 `/api/trace/{id}` 已关闭
    （见 `test_dev_facing_endpoints_are_not_exposed`）。
    """
    tracer = build_tracer(offline_settings())
    with tracer.bind("req-1"):
        with tracer.step(NodeName.RESOLVE):
            pass
        with tracer.step(NodeName.CLASSIFY):
            pass
    with tracer.bind("req-2"):
        with tracer.step(NodeName.RETRIEVE):
            pass

    assert [s.node for s in tracer.trace("req-1")] == ["resolve", "classify"]
    assert [s.node for s in tracer.trace("req-2")] == ["retrieve"]
    assert tracer.trace("不存在") == []
    assert tracer.current_request_id is None  # bind 退出后复原


# ============================================================
# 2. 上下文管理器语义
# ============================================================
def test_step_records_elapsed_ms():
    tracer = build_tracer(offline_settings())
    with tracer.step(NodeName.GENERATE, input="问题", output="回答") as st:
        time.sleep(0.02)
    assert st.elapsed_ms > 0, f"elapsed_ms 没被填上: {st.elapsed_ms}"
    assert st.elapsed_ms >= 15, f"sleep 20ms 却只记了 {st.elapsed_ms}ms"
    # step() 返回的就是最终被收集的那个对象（不是副本）
    assert tracer.steps[-1] is st


def test_step_without_as_binding_still_fills_fields():
    """即使不 `as st`，退出时也要把耗时写回；否则 trace 里全是 0。"""
    tracer = build_tracer(offline_settings())
    with tracer.step(NodeName.GATE, passed=True):
        time.sleep(0.005)
    s = tracer.steps[-1]
    assert s.elapsed_ms > 0
    assert s.details["passed"] is True


def test_step_accepts_nodename_and_str():
    tracer = build_tracer(offline_settings())
    with tracer.step(NodeName.HIGH_RISK_FLOW):
        pass
    with tracer.step("custom_node"):
        pass
    assert [s.node for s in tracer.steps] == ["high_risk_flow", "custom_node"]


def test_input_output_and_tokens():
    tracer = build_tracer(offline_settings())
    with tracer.step(
        NodeName.GENERATE,
        input="用户问运费",
        output="运费规则：满 99 包邮",
        tokens_in=12,
        tokens_out=7,
    ):
        pass
    s = tracer.steps[-1]
    assert s.input_digest == "用户问运费"
    assert s.output_digest.startswith("运费规则")
    assert (s.tokens_in, s.tokens_out) == (12, 7)
    # 保留参数不该漏进 details
    assert "input" not in s.details and "tokens_in" not in s.details


def test_set_output_after_the_fact():
    tracer = build_tracer(offline_settings())
    with tracer.step(NodeName.RETRIEVE) as st:
        st.details["query"] = "q"
        st.set_output({"n": 3})
    assert tracer.steps[-1].output_digest == '{"n": 3}'


def test_nested_steps():
    """真实流水线里节点会套节点（agent 里调工具），不能互相串味。"""
    tracer = build_tracer(offline_settings())
    with tracer.step(NodeName.AGENT) as outer:
        with tracer.step(NodeName.TOOL, name="get_order") as inner:
            time.sleep(0.005)
        assert inner.node == "tool"
        assert outer.node == "agent"
    assert [s.node for s in tracer.steps] == ["tool", "agent"]
    assert tracer.steps[0].elapsed_ms > 0 and tracer.steps[1].elapsed_ms > 0


# ============================================================
# 3. 字段完整性 / JSON 可序列化
# ============================================================
def test_tracestep_fields_and_json_serializable():
    tracer = build_tracer(offline_settings())
    ev = Evidence(chunk=KbChunk(chunk_id="c1", doc_id="d1", title="运费政策", text="满99包邮"))
    retrieval = RetrievalResult(
        query="运费怎么算",
        evidences=[ev],
        dense_hits=5,
        sparse_hits=4,
        fused_hits=6,
        reranked_hits=1,
        strategy="hybrid_rerank",
        latency_ms=12.5,
    )
    with tracer.step(
        NodeName.RETRIEVE,
        input="运费怎么算",
        retrieval=retrieval,
        tool=ToolCall(name="get_order", args={"order_id": "ORD-1001"}, ok=False, error="TIMEOUT"),
        usage=TokenUsage(input_tokens=10, output_tokens=4, total_tokens=14),
        gate=GateDecision.reject(0.1, RefusalReason.WEAK_EVIDENCE, {"top_score": 0.1}),
    ) as st:
        time.sleep(0.005)

    s = tracer.steps[-1]
    # 契约要求的字段一个都不少
    for field in (
        "node",
        "started_at",
        "elapsed_ms",
        "input_digest",
        "output_digest",
        "details",
        "tokens_in",
        "tokens_out",
        "error",
    ):
        assert field in s.model_dump(), f"TraceStep 少了字段 {field}"
    assert s.node == "retrieve"
    assert s.elapsed_ms > 0
    assert s.error is None

    # "检索回了哪些 chunk" —— chunk_id + 标题 + 分数都要在
    assert s.details["retrieval"]["chunks"] == [
        {"chunk_id": "c1", "title": "运费政策", "score": 0.0}
    ]
    assert s.details["retrieval"]["dense_hits"] == 5
    assert s.details["retrieval"]["strategy"] == "hybrid_rerank"
    # "调了哪个工具" —— 失败也要记
    assert s.details["tool"]["name"] == "get_order"
    assert s.details["tool"]["ok"] is False
    assert s.details["tool"]["error"] == "TIMEOUT"
    assert s.details["tool"]["args"] == {"order_id": "ORD-1001"}
    # "烧了多少 token" —— 既在 details 也在一等字段里
    assert s.details["usage"]["total_tokens"] == 14
    # 闸门判据逐条可解释
    assert s.details["gate"]["reason"] == "weak_evidence"
    assert s.details["gate"]["checks"] == {"top_score": 0.1}

    dumped = s.model_dump(mode="json")
    json.dumps(dumped, ensure_ascii=False)  # 不抛 = 可序列化
    assert json.loads(json.dumps(dumped))["node"] == "retrieve"


def test_arbitrary_details_are_coerced_to_jsonable():
    """details 里塞了奇怪对象也不能让序列化炸掉 —— 前端拿的是 JSON。"""

    class Weird:
        def __repr__(self):
            return "<weird>"

    tracer = build_tracer(offline_settings())
    with tracer.step(
        NodeName.TOOL,
        enum=NodeName.TOOL,
        obj=Weird(),
        tup=(1, 2),
        st={1, 2},
        nested={"a": {"b": {"c": {"d": [NodeName.GATE]}}}},
    ):
        pass
    s = tracer.steps[-1]
    json.dumps(s.model_dump(mode="json"), ensure_ascii=False)  # 不抛即可
    assert s.details["enum"] == "tool"  # Enum 取 value
    assert s.details["tup"] == [1, 2]  # tuple → list
    assert isinstance(s.details["obj"], str)  # 未知对象 → 字符串摘要
    assert s.details["st"] == [1, 2] or s.details["st"] == [2, 1]  # set → 确定化列表
    assert s.details["nested"]["a"]["b"]["c"]["d"] == ["gate"]  # 深层 Enum 也被拍平


def test_long_values_are_truncated_not_stored_whole():
    """不存全量证据 —— 日志爆炸的代价是没人再看日志。"""
    tracer = build_tracer(offline_settings())
    with tracer.step(NodeName.GENERATE, input="很长的输入" * 500, output="很长的输出" * 500):
        pass
    s = tracer.steps[-1]
    assert 0 < len(s.input_digest) < 400
    assert 0 < len(s.output_digest) < 400
    assert "…(+" in s.input_digest  # 明确标注被截断了


def test_details_total_size_is_capped_deterministically():
    tracer = build_tracer(offline_settings())
    huge = {f"k{i:02d}": "x" * 1000 for i in range(20)}

    def run() -> dict:
        t = build_tracer(offline_settings())
        with t.step(NodeName.TOOL, **huge):
            pass
        return t.steps[-1].details

    d1, d2 = run(), run()
    assert d1 == d2, "裁剪结果必须确定（丢键顺序不能随机）"
    assert len(json.dumps(d1, ensure_ascii=False, default=str)) <= 4500
    assert "_dropped" in d1
    assert len(d1["_dropped"]) == 20 - (len(d1) - 1)  # 丢了多少个键可解释


# ============================================================
# 4. 异常注入
# ============================================================
def test_exception_propagates_and_tracer_survives():
    tracer = build_tracer(offline_settings())
    with pytest.raises(ValueError, match="业务炸了"):
        with tracer.step(NodeName.TOOL, name="get_order"):
            raise ValueError("业务炸了")

    s = tracer.steps[-1]
    assert s.error is not None and "ValueError" in s.error and "业务炸了" in s.error
    assert s.elapsed_ms > 0  # 异常路径也要记耗时
    assert s.details["error_type"] == "ValueError"

    # Tracer 没坏：后续步骤照常
    with tracer.step(NodeName.GENERATE) as st2:
        st2.set_output("ok")
    assert tracer.steps[-1].output_digest == "ok"
    tracer.flush()
    assert len(tracer.steps) == 2


def test_base_exception_style_errors_also_propagate():
    """KeyboardInterrupt/SystemExit 这种 BaseException 更不能被观测层吞掉。"""
    tracer = build_tracer(offline_settings())
    with pytest.raises(KeyboardInterrupt):
        with tracer.step(NodeName.AGENT):
            raise KeyboardInterrupt
    assert "KeyboardInterrupt" in (tracer.steps[-1].error or "")
    tracer.flush()  # 收尾依旧安全


def test_enabled_exception_still_propagates(monkeypatch):
    """开启观测时也一样：span 关得掉，异常照样冒出去。"""
    box = install_fake_langfuse(monkeypatch, ok_client)
    tracer = build_tracer(enabled_settings())
    assert tracer.enabled is True

    with pytest.raises(RuntimeError, match="boom"):
        with tracer.step(NodeName.GENERATE):
            raise RuntimeError("boom")
    assert "RuntimeError: boom" in (tracer.steps[-1].error or "")
    assert box["observed"][0]["name"] == "generate"


# ============================================================
# 5. Langfuse 挂载 / 降级
# ============================================================
def test_every_step_maps_to_one_span(monkeypatch):
    box = install_fake_langfuse(monkeypatch, ok_client)
    tracer = build_tracer(enabled_settings())
    assert tracer._bridge is not None
    assert box["init_kwargs"]["host"] == "http://127.0.0.1:1"

    with tracer.bind("req-9"):
        for node in (NodeName.RESOLVE, NodeName.CLASSIFY, NodeName.RETRIEVE):
            with tracer.step(node, input=f"{node} 的输入") as st:
                st.set_output(f"{node} 的输出")

    assert [o["name"] for o in box["observed"]] == ["resolve", "classify", "retrieve"]
    assert all(o["as_type"] == "span" for o in box["observed"])
    # span 上带 request_id，Langfuse 里能按请求过滤
    assert box["observed"][0]["metadata"]["request_id"] == "req-9"
    assert box["observed"][0]["input"] == "resolve 的输入"
    # 收尾：output / 耗时回填到 span 上
    assert box["updates"][0]["output"] == "resolve 的输出"
    assert box["updates"][0]["metadata"]["elapsed_ms"] > 0
    assert box["updates"][0]["level"] == "DEFAULT"

    tracer.flush()
    tracer.close()
    # flush 幂等：重复调用安全，close 内部也会 flush 一次
    assert box["flushes"] == 2
    assert box["shutdowns"] == 1


def test_error_step_marks_span_error_level(monkeypatch):
    box = install_fake_langfuse(monkeypatch, ok_client)
    tracer = build_tracer(enabled_settings())
    with pytest.raises(ValueError):
        with tracer.step(NodeName.TOOL):
            raise ValueError("工具炸了")
    assert box["updates"][0]["level"] == "ERROR"
    assert "工具炸了" in box["updates"][0]["status_message"]


def test_missing_keys_degrades_with_warning(monkeypatch, caplog):
    """开了开关却没配 key：降级而不是抛异常。"""
    install_fake_langfuse(monkeypatch, ok_client)
    with caplog.at_level(logging.WARNING, logger="mewhelp.observability"):
        tracer = Tracer(settings=enabled_settings(langfuse_public_key=None, langfuse_secret_key=None))
    assert tracer.enabled is True  # 开关本身没被改
    assert tracer._bridge is None  # 但桥接没建起来 → 不联网
    assert any("降级" in r.getMessage() for r in caplog.records)

    # 业务照常
    with tracer.step(NodeName.GATE) as st:
        st.set_output("ok")
    tracer.flush()
    assert len(tracer.steps) == 1


def test_span_open_failure_does_not_break_business(monkeypatch, caplog):
    """SDK 的 API 与预期不符（span 开不出来）时，业务必须照常跑。"""
    install_fake_langfuse(monkeypatch, lambda box: exploding_client(on_span=True))

    tracer = Tracer(settings=enabled_settings())
    assert tracer._bridge is not None  # 桥建起来了，坏在调用时

    with caplog.at_level(logging.WARNING, logger="mewhelp.observability"):
        with tracer.step(NodeName.RETRIEVE, query="q") as st:
            st.set_output("结果还在")
        tracer.flush()  # 不抛
        tracer.close()  # 不抛

    assert tracer.steps[-1].output_digest == "结果还在"
    assert tracer.steps[-1].elapsed_ms > 0
    messages = [r.getMessage() for r in caplog.records]
    assert any("span 开启失败" in m for m in messages)
    assert any("flush 失败" in m for m in messages)


def test_client_init_failure_degrades(monkeypatch, caplog):
    """连 Langfuse 客户端都建不起来（比如后端不可达）也不能崩。"""
    install_fake_langfuse(monkeypatch, lambda box: exploding_client(on_init=True))

    with caplog.at_level(logging.WARNING, logger="mewhelp.observability"):
        tracer = Tracer(settings=enabled_settings())
    assert tracer._bridge is None
    assert any("初始化失败" in r.getMessage() for r in caplog.records)

    with tracer.step(NodeName.HANDOFF):
        pass
    tracer.flush()
    assert [s.node for s in tracer.steps] == ["handoff"]


def test_warning_is_emitted_only_once(monkeypatch, caplog):
    """每步刷一条 warning 会把日志淹掉 —— 同类故障只报一次。"""
    install_fake_langfuse(monkeypatch, lambda box: exploding_client(on_span=True))
    tracer = Tracer(settings=enabled_settings())
    with caplog.at_level(logging.WARNING, logger="mewhelp.observability"):
        for _ in range(5):
            with tracer.step(NodeName.RETRIEVE):
                pass
        tracer.flush()
    assert len([r for r in caplog.records if "span 开启失败" in r.getMessage()]) == 1
    assert len(tracer.steps) == 5  # 本地 trace 一条没丢


def test_real_langfuse_points_at_dead_host():
    """真 SDK + 指向一个必然连不上的端口：业务不崩，flush 不抛。

    假 SDK 测的是我们的 try/except；真 SDK 测的是"后端真的死了会怎样"。
    """
    pytest.importorskip("langfuse")
    tracer = Tracer(settings=enabled_settings(langfuse_host="http://127.0.0.1:1"))
    if tracer._bridge is None:  # 环境里 langfuse 建不起来 —— 降级路径本身就通过
        pytest.skip("环境无法初始化 langfuse，降级路径已由其它用例覆盖")

    with tracer.step(NodeName.RETRIEVE, query="运费怎么算") as st:
        time.sleep(0.005)
        st.set_output("满 99 包邮")
    tracer.flush()  # 不抛
    tracer.close()  # 不抛
    assert len(tracer.steps) == 1
    assert tracer.steps[-1].output_digest == "满 99 包邮"


def test_enabled_false_overrides_settings_flag(monkeypatch):
    """显式 enabled 覆盖配置：测试和演练时不必改环境变量。"""
    install_fake_langfuse(monkeypatch, ok_client)
    tracer = Tracer(settings=enabled_settings(), enabled=False)
    assert tracer.enabled is False
    assert tracer._bridge is None
    with tracer.step(NodeName.RETRIEVE):
        pass
    tracer.flush()


# ============================================================
# 6. 零侵入挂载：wrap / record / 与 state.trace 衔接
# ============================================================
def test_wrap_node_function():
    tracer = build_tracer(offline_settings())

    def retrieve(self, state):
        """检索节点。"""
        time.sleep(0.005)
        return {"retrieval": "r"}

    wrapped = tracer.wrap(retrieve, NodeName.RETRIEVE)
    out = wrapped(None, {"question": "运费"})
    assert out == {"retrieval": "r"}
    s = tracer.steps[-1]
    assert s.node == "retrieve"
    assert s.elapsed_ms > 0
    assert "运费" in s.input_digest
    assert s.output_digest  # 节点返回值自动成了输出摘要

    # functools.wraps：包装后 LangGraph 里注册的仍是被包装的函数身份
    assert wrapped.__name__ == "retrieve"
    assert wrapped.__doc__ == retrieve.__doc__


def test_wrap_defaults_to_function_name():
    tracer = build_tracer(offline_settings())

    def classify(self, state):
        return "ok"

    tracer.wrap(classify)(None, {})
    assert tracer.steps[-1].node == "classify"


def test_wrap_records_error_and_rethrows():
    tracer = build_tracer(offline_settings())

    def boom(self, state):
        raise KeyError("order_id")

    with pytest.raises(KeyError):
        tracer.wrap(boom, NodeName.TOOL)(None, {})
    assert "KeyError" in (tracer.steps[-1].error or "")


def test_record_external_tracesteps_for_api():
    """编排层自己往 state["trace"] 追加 step，最后一次性交给 Tracer。"""
    tracer = build_tracer(offline_settings())
    steps = [
        TraceStep(node="resolve", elapsed_ms=3.0, details={"request_id": "req-x"}),
        TraceStep(node="classify", elapsed_ms=5.0, details={"request_id": "req-x"}),
    ]
    for s in steps:
        tracer.record(s)
    assert [s.node for s in tracer.trace("req-x")] == ["resolve", "classify"]
    assert len(tracer.steps) == 2


def test_bind_nesting_and_restore():
    tracer = build_tracer(offline_settings())
    with tracer.bind("outer"):
        with tracer.bind("inner"):
            with tracer.step(NodeName.RETRIEVE):
                pass
        with tracer.step(NodeName.GENERATE):
            pass
    assert [s.node for s in tracer.trace("inner")] == ["retrieve"]
    assert [s.node for s in tracer.trace("outer")] == ["generate"]


def test_step_request_id_kwarg_wins_over_binding():
    tracer = build_tracer(offline_settings())
    with tracer.bind("outer"):
        with tracer.step(NodeName.RETRIEVE, request_id="explicit"):
            pass
    assert [s.node for s in tracer.trace("explicit")] == ["retrieve"]
    assert tracer.trace("outer") == []


def test_memory_is_bounded():
    """服务长跑不能无限攒 trace / step。"""
    tracer = build_tracer(offline_settings())
    for i in range(tracing_mod._MAX_TRACES + 60):
        tracer.record(TraceStep(node="n", details={"request_id": f"r{i}"}))
    assert len(tracer.request_ids()) == tracing_mod._MAX_TRACES
    assert tracer.request_ids()[0] == "r60"  # 最老的被淘汰，新的还在
    assert len(tracer.steps) == tracing_mod._MAX_TRACES + 60
    tracer.clear()
    assert tracer.steps == [] and tracer.request_ids() == []


# ============================================================
# 7. 进程级实例 / 确定性
# ============================================================
def test_get_tracer_is_process_singleton(monkeypatch):
    monkeypatch.setattr(tracing_mod, "_DEFAULT_TRACER", None)
    try:
        a = tracing_mod.get_tracer(offline_settings())
        b = tracing_mod.get_tracer(offline_settings())
        assert a is b
        reset_tracer()
        c = tracing_mod.get_tracer(offline_settings())
        assert c is not a
    finally:
        reset_tracer()


def test_deterministic_same_input_same_output():
    """同输入同输出 —— node / digest / details 不允许有随机成分。"""

    def run() -> dict:
        tracer = build_tracer(offline_settings())
        ev = Evidence(chunk=KbChunk(chunk_id="c1", doc_id="d1", title="t", text="x"))
        with tracer.step(
            NodeName.RETRIEVE,
            query="q",
            retrieval=RetrievalResult(query="q", evidences=[ev], dense_hits=2),
            usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        ) as st:
            st.set_output("o")
        d = tracer.steps[0].model_dump(mode="json")
        d.pop("started_at")  # 时间戳天生会变，契约里它也不是判据
        d.pop("elapsed_ms")
        return d

    assert run() == run()


# ============================================================
# 8. 与 API 层的接缝
# ============================================================
def test_trace_steps_fit_conversation_result_contract():
    """一条真实请求走完，trace 要能原样塞进 ConversationResult.trace。

    这是"一条请求还原成一棵调用树"的最末端：前面记的字段再全，
    在这里序列化不过就等于没记。前端拿到的就是这个 JSON。
    """
    ev = Evidence(
        chunk=KbChunk(chunk_id="kb-1", doc_id="d1", title="运费政策", text="满 99 包邮"),
        score=0.87,
    )
    retrieval = RetrievalResult(query="运费怎么算", evidences=[ev], dense_hits=20, reranked_hits=1)

    tracer = build_tracer(offline_settings())
    with tracer.bind("req-abc"):
        with tracer.step(NodeName.RETRIEVE, input="运费怎么算", retrieval=retrieval) as st:
            time.sleep(0.005)
        with tracer.step(
            NodeName.TOOL,
            tool=ToolCall(name="get_order", args={"order_id": "ORD-1001"}, ok=True, elapsed_ms=3.2),
        ):
            pass
        with tracer.step(NodeName.GENERATE, tokens_in=412, tokens_out=88) as st:
            st.set_output("运费满 99 包邮")
    tracer.flush()

    result = ConversationResult(
        request_id="req-abc",
        session_id="s-1",
        user_id="u-1",
        question="运费怎么算",
        answer="运费满 99 包邮",
        path=["retrieve", "generate"],
        trace=tracer.trace("req-abc"),
        usage=TokenUsage(input_tokens=412, output_tokens=88, total_tokens=500),
    )
    assert [s.node for s in result.trace] == ["retrieve", "tool", "generate"]
    assert result.trace[0].details["retrieval"]["chunks"][0]["chunk_id"] == "kb-1"
    assert result.trace[1].details["tool"]["name"] == "get_order"
    assert result.trace[2].output_digest == "运费满 99 包邮"
    assert result.trace[2].tokens_in == 412

    blob = result.model_dump_json()  # GET /api/chat/sync 的返回体
    assert json.loads(blob)["trace"][0]["node"] == "retrieve"
    assert len(blob) < 8000, "trace 没做摘要，返回体被证据全文撑爆了"

