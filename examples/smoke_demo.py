"""离线冒烟：建库 → 跑图 → 打印分流路径。

这是 README 里那条"3 分钟跑起来"的命令。它只用 mock 依赖，不连 Docker、
不要 API Key、不需要网络；同输入同输出，可以当回归基线用：

    python examples/smoke_demo.py

为什么单独留一个脚本而不是只给 `python -c`：一行式命令没法被 review，
而这段代码恰恰是新同学理解"一条消息怎么流过整张图"最短的入口 ——
读它比读 README 的架构图更快，也比读 workflow/nodes.py 更短。
"""

from __future__ import annotations

from mewhelp.config import get_settings
from mewhelp.workflow.graph import build_default_graph
from mewhelp.workflow.state import new_state

#: 四道题刻意覆盖四条不同的边：知识检索、表格/多条件、高风险中断、业务拒绝。
#: 少一条就看不出分流是否真的生效 —— 只问一句"运费怎么算"是测不出这张图的。
DEMO_QUESTIONS: tuple[str, ...] = (
    "运费怎么算",          # knowledge 出口：检索 → 闸门 → 生成
    "生鲜类可以退款吗",     # 带实体的知识问题：走实体覆盖率判据
    "我要退款",            # high_risk 出口：缺订单号 → 追问（启用 checkpointer 时是真中断）
    "帮我取消 ORD-1008",   # high_risk 出口：工具业务拒绝，不猜、不通融
)


def main() -> int:
    settings = get_settings()
    graph, deps = build_default_graph()
    # 建库是必须显式做的一步：build_default_graph 只装配检索器，不替你灌数据。
    # 不做这一步的话每条知识问题都会因为"零证据"被闸门拒答 —— 那不是 bug，
    # 是没建索引。
    indexed = deps.retriever.load_from_dir(settings.knowledge_dir)
    print(f"chunks indexed: {indexed}")

    for question in DEMO_QUESTIONS:
        out = graph.invoke(
            new_state(request_id="req-demo", session_id="s-demo", user_id="u-demo", question=question)
        )
        print(f"\nQ: {question}")
        print("  intent :", out["intent"], out["intent_confidence"], "->", out["exit"])
        print("  path   :", " -> ".join(out["path"]))
        print("  answer :", (out["answer"] or "")[:60].replace("\n", " "))
        if out.get("refusal_reason"):
            print("  refusal:", out["refusal_reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
