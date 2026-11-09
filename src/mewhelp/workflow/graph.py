"""LangGraph 状态图。

图的形状（也是这份代码最该被看懂的地方）：

    START → resolve → classify → route ─┬→ retrieve → gate ─┬→ generate → END
                                        │                    └→ handoff → flywheel → END
                                        ├→ agent(ReAct) ────────→ generate → END
                                        ├→ high_risk_flow ─┬→ generate → END
                                        │                  └→ handoff → flywheel → END
                                        └→ handoff → flywheel → END

三个刻意的设计：

1. **route 是一道条件边，不是 if/else。** 九类意图在 `INTENT_TO_EXIT` 里
   收敛成五个出口，图因此只有五条边。以后加一类意图，只要它落进已有出口，
   这张图一行都不用改。

2. **handoff 汇到 flywheel。** 答不上的问题不会白白溜走 —— 这是数据飞轮
   的入口，也是这套系统和"调 API 的 demo"的分界线。

3. **所有路径最终汇合到 generate。** 主干只有一条出口，答复的格式、
   埋点、流式推送都只需要实现一次。

关于 checkpointer：**默认不启用**。启用后 `interrupt()` 才能真正挂起流程
（高风险子流程缺订单号时），代价是每个线程要有独立的 thread_id，且状态
会落盘。演示中断/恢复时传入 `SqliteSaver` 即可。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..config import Settings, get_settings
from ..llm import LLMClient, build_llm
from .nodes import (
    gate_route,
    high_risk_route,
    make_agent_node,
    make_classify_node,
    make_flywheel_node,
    make_gate_node,
    make_generate_node,
    make_handoff_node,
    make_high_risk_node,
    make_resolve_node,
    make_retrieve_node,
    route_by_exit,
)
from .state import AgentState


@dataclass
class GraphDeps:
    """依赖容器。

    显式注入而不是在节点里 `build_xxx()`，有两个直接好处：
      * 测试时把任何一个换成假实现，只测那一环；
      * 评估时换掉 retriever 就能跑消融，图本身不用动。
    """

    llm: LLMClient
    resolver: Any
    classifier: Any
    retriever: Any
    gate: Any
    self_eval: Any
    tools: Any
    flywheel: Any
    settings: Settings = field(default_factory=get_settings)
    max_agent_steps: int = 6
    #: HTTP account sessions have no connection to the bundled sample order file.
    enforce_order_ownership: bool = False
    #: 有没有挂 checkpointer。高风险节点靠它决定"缺槽位时"是挂起还是收尾 ——
    #: 显式传进来，比在节点里回想"我当时编译时挂了没"可靠得多。
    checkpointer: Any | None = None


def build_deps(
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    retriever: Any | None = None,
    tools: Any | None = None,
    flywheel: Any | None = None,
) -> GraphDeps:
    """按配置装配一套默认依赖。

    每个组件都允许外部覆盖 —— 这正是上面那个容器的意义。
    """
    from ..flywheel.store import FlywheelStore
    from ..nlu.coreference import CoreferenceResolver
    from ..nlu.intent import IntentClassifier
    from ..quality.gate import ConfidenceGate
    from ..quality.selfeval import SelfEvaluator
    from ..retrieval.pipeline import build_pipeline
    from ..tools.business import build_business_registry

    s = settings or get_settings()
    llm = llm or build_llm(s)

    return GraphDeps(
        llm=llm,
        resolver=CoreferenceResolver(llm, s),
        classifier=IntentClassifier(llm, s),
        retriever=retriever if retriever is not None else build_pipeline(s),
        gate=ConfidenceGate(s),
        self_eval=SelfEvaluator(llm, s),
        tools=tools if tools is not None else build_business_registry(),
        flywheel=flywheel if flywheel is not None else FlywheelStore(settings=s),
        settings=s,
        max_agent_steps=s.agent_max_steps,
    )


def build_graph(deps: GraphDeps, checkpointer: Any | None = None) -> Any:
    """编译状态图。"""
    # 让高风险节点知道这次编译有没有持久化 —— 决定它缺槽位时"挂起"还是"追问收尾"
    deps.checkpointer = checkpointer
    g = StateGraph(AgentState)

    g.add_node("resolve", make_resolve_node(deps))
    g.add_node("classify", make_classify_node(deps))
    g.add_node("retrieve", make_retrieve_node(deps))
    g.add_node("gate", make_gate_node(deps))
    g.add_node("agent", make_agent_node(deps))
    g.add_node("high_risk_flow", make_high_risk_node(deps))
    g.add_node("generate", make_generate_node(deps))
    g.add_node("handoff", make_handoff_node(deps))
    g.add_node("flywheel", make_flywheel_node(deps))

    g.add_edge(START, "resolve")
    g.add_edge("resolve", "classify")
    # 全流程最重要的岔口：九类意图在这里收敛成五条边
    g.add_conditional_edges(
        "classify",
        route_by_exit,
        {
            "retrieve": "retrieve",
            "agent": "agent",
            "high_risk_flow": "high_risk_flow",
            "handoff": "handoff",
        },
    )

    g.add_edge("retrieve", "gate")
    # 闸门的分支：证据够就生成，不够就转人工（+落问题池）
    g.add_conditional_edges("gate", gate_route, {"generate": "generate", "handoff": "handoff"})

    g.add_edge("agent", "generate")
    # 高风险流程：跑完了就出答复；需要人工的转 handoff
    g.add_conditional_edges(
        "high_risk_flow",
        high_risk_route,
        {"handoff": "handoff", "__end__": "generate"},
    )

    # 生成之后要**再看一眼 handoff**，不能直接进 END。
    #
    # generate 有三条会置 handoff=True 的出口：生成失败兜底、自评没过（判成编造）、
    # 工具全没调成。原先这三条都直连 END，后果有两个：
    #   1. 答复正文里写着"帮你转人工"，而 handoff 标志是假的 —— 前端只在
    #      handoff=true 时才补那条"已为你转接人工客服"提示条，用户看到的
    #      是一句没有兑现的承诺；
    #   2. 绕过了 flywheel，`FlywheelSource.SELF_EVAL` 这个入口在生产路径上
    #      永远是 0 —— 答不上来的问题一条都进不了问题池，而那正是这套
    #      "数据飞轮"存在的意义。
    # handoff 节点对"已经有答复"的情况是**追加**一句转接提示、不覆盖正文，
    # 所以走这一趟不会把已经生成好的答复弄丢。
    g.add_conditional_edges(
        "generate",
        lambda s: "handoff" if s.get("handoff") else END,
        {"handoff": "handoff", END: END},
    )
    g.add_edge("handoff", "flywheel")
    g.add_edge("flywheel", END)

    return g.compile(checkpointer=checkpointer)


def build_default_graph(
    *, settings: Settings | None = None, checkpointer: Any | None = None
) -> tuple[Any, GraphDeps]:
    """开箱即用：装配依赖 + 编译图。

    返回 (graph, deps)，因为调用方通常还需要 deps（比如 API 层要拿
    deps.flywheel 做审核、拿 deps.retriever 做检索预览）。
    """
    deps = build_deps(settings=settings)
    return build_graph(deps, checkpointer=checkpointer), deps


def make_sqlite_checkpointer(path: str = "mewhelp_checkpoints.db") -> Any:
    """中断/恢复演示用的 checkpointer。

    单独抽一个函数，是为了让"启用持久化"这件事在调用处一眼可见 ——
    它改变的是行为语义（线程隔离、状态落盘），不该藏在默认参数里。
    """
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn)
