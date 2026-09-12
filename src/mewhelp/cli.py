"""命令行入口（`mewhelp ...`）。

五个子命令对应五种"打开这套系统的方式"：

    serve    起服务，配演示页 —— 给人看
    chat     单轮问答，终端打印分流路径和证据 —— 自己试
    index    重建知识库索引并打印 chunk 统计 —— 看清检索吃进去的是什么
    flywheel 打印问题池统计和待审列表 —— 飞轮转起来没有
    demo     跑一组预设问题，把分流路径和答复并排打出来 —— 面试演示用

设计上有两点刻意：

1. **`demo` 与 `chat` 共用同一条调用路径**（都走 `ChatService`），而不是各自
   拼一遍。演示脚本和真实入口走不同的路，是"演示能跑、线上不能跑"的经典成因。
2. **中文输出前重设 stdout 编码**。Windows 控制台默认 GBK，中文答复会直接
   抛 UnicodeEncodeError —— 那是环境问题，不该让用户以为程序坏了。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .api.service import ChatService, SessionStore
from .config import Settings, get_settings
from .domain import INTENT_LABELS, INTENT_SPECS, INTENT_TO_EXIT
from .flywheel.feedback import triage_hint, triage_label
from .observability.tracing import build_tracer
from .workflow.graph import GraphDeps, build_deps, build_graph

#: 演示用的预设问题。**每条都刻意落在不同的出口上** —— 演示要展示的是
#: "九类意图怎么收敛到五个出口"，四个问题走同一条路就白演了。
DEMO_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("业务工具", "帮我查下订单 ORD-1002 到哪了"),
    ("知识检索", "生鲜类可以退款吗"),
    ("高风险（缺槽位会追问）", "我要退款"),
    ("高风险（槽位齐备）", "帮我取消订单 ORD-1001"),
    ("知识库外（应拒答转人工）", "你们卖不卖火箭"),
    ("转人工", "我要找人工"),
)


# ============================================================
# 输出辅助
# ============================================================
def _setup_stdout() -> None:
    """把 stdout 切成 UTF-8。

    Windows 控制台是 GBK，打印中文答复会抛 UnicodeEncodeError；
    这不是代码 bug，但用户看到的是一串报错，所以在这里兜住。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover - 非 tty / 老 Python
            pass


def _rule(char: str = "─", width: int = 68) -> str:
    return char * width


def _print_path(path: list[str]) -> None:
    print("  路径：" + (" → ".join(path) if path else "（未走流水线）"))


def _print_evidences(evidences: list[Any], limit: int = 5) -> None:
    if not evidences:
        print("  证据：无")
        return
    print(f"  证据（{len(evidences)} 条）：")
    for i, ev in enumerate(evidences[:limit], 1):
        print(f"    [{i}] {ev.score:.3f}  {ev.chunk.title}  ({ev.chunk_id})")


def _print_tools(tool_calls: list[Any]) -> None:
    if not tool_calls:
        return
    print("  工具：")
    for tc in tool_calls:
        mark = "OK " if tc.ok else "ERR"
        print(f"    [{mark}] {tc.name}  {tc.error or ''}")


def _build_service(settings: Settings, *, index_kb: bool = True) -> tuple[ChatService, GraphDeps]:
    """装配一套完整的 deps + 图 + 服务，需要时把知识库灌进去。

    每一步都是**一次**装配：CLI 反复调 `_build_service` 会让每次问答都重建
    索引（几十毫秒到几秒），所以调用方拿到之后要复用。
    """
    deps = build_deps(settings=settings)
    if index_kb:
        kb_dir = Path(getattr(settings, "knowledge_dir", "") or "data/knowledge")
        if not kb_dir.is_dir():
            kb_dir = Path("data/knowledge")
        deps.retriever.load_from_dir(kb_dir)
    graph = build_graph(deps)
    service = ChatService(
        graph=graph,
        sessions=SessionStore(settings),
        tracer=build_tracer(settings),
        settings=settings,
    )
    return service, deps


def _print_result(result: Any) -> None:
    print(_rule())
    print(f"  问题：{result.question}")
    intent = result.intent.value if result.intent else "—"
    label = INTENT_LABELS.get(result.intent, "") if result.intent else ""
    exit_ = result.exit.value if result.exit else "—"
    print(f"  意图：{intent} {('（' + label + '）') if label else ''}   出口：{exit_}"
          f"   置信度：{result.intent_confidence:.2f}")
    _print_path(result.path)
    _print_tools(result.tool_calls)
    _print_evidences(result.evidences)
    flags = []
    if result.handoff:
        flags.append("转人工")
    if result.interrupted:
        flags.append("流程中断（等用户补充）")
    if result.refusal_reason:
        flags.append(f"拒答原因={result.refusal_reason.value}")
    print(f"  状态：{'、'.join(flags) if flags else '正常答复'}"
          f"   耗时 {result.latency_ms} ms"
          f"   tokens {result.usage.input_tokens}/{result.usage.output_tokens}")
    print("  答复：")
    for line in (result.answer or "").splitlines() or [""]:
        print("    " + line)
    print(_rule())


# ============================================================
# 子命令
# ============================================================
def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    print(f"启动 MewHelp 服务：http://{args.host}:{args.port}")
    print("演示页：GET /    接口文档：/docs")
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    service, _ = _build_service(get_settings())
    # 手工把历史接上：CLI 单轮调用也走同一套 SessionStore，
    # 这样 --session 相同的两次 `mewhelp chat` 就是一次真正的多轮对话。
    history = service.sessions.load(args.session)
    result = service.ask(user_id=args.user, session_id=args.session, message=args.question)
    _print_result(result)
    if history:
        print(f"  （本次会话此前已有 {len(history)} 轮历史）")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    """重建索引并打印 chunk 统计。

    统计来自**重新切分**而不是读检索器内部结构：切分是唯一的口径，
    从内部结构反推出来的数字迟早和实际入库的对不上。
    """
    from .retrieval.chunking import chunk_document_file

    settings = get_settings()
    kb_dir = Path(getattr(settings, "knowledge_dir", "") or "data/knowledge")
    if not kb_dir.is_dir():
        kb_dir = Path("data/knowledge")
    if not kb_dir.is_dir():
        print(f"知识库目录不存在：{kb_dir}")
        return 1

    docs: list[tuple[str, list[Any]]] = []
    for md in sorted(kb_dir.glob("*.md"), key=lambda p: p.name):
        docs.append((md.name, chunk_document_file(md)))

    total = sum(len(c) for _, c in docs)
    tables = sum(1 for _, cs in docs for c in cs if c.is_table)
    chars = [c.char_len for _, cs in docs for c in cs]

    print(f"知识库目录：{kb_dir}")
    print(_rule())
    print(f"  {'文件':<22}{'chunk':>7}{'表格':>6}{'平均字数':>10}")
    for name, cs in docs:
        avg = int(sum(c.char_len for c in cs) / len(cs)) if cs else 0
        print(f"  {name:<22}{len(cs):>7}{sum(1 for c in cs if c.is_table):>6}{avg:>10}")
    print(_rule())
    if chars:
        print(f"  合计：{total} 个 chunk（其中表格 {tables} 个）"
              f"，平均 {int(sum(chars) / len(chars))} 字，最长 {max(chars)} 字")

    deps = build_deps(settings=settings)
    written = deps.retriever.load_from_dir(kb_dir)
    print(f"  索引完成：写入 {written} 条，检索器现有 {deps.retriever.count()} 条")
    stats = deps.retriever.stats()
    print(f"  store={stats.get('store')} count={stats.get('store_count')} "
          f"embedder={stats.get('embedder')} reranker={stats.get('reranker')}")
    if stats.get("last_error"):
        print(f"  注意：{stats['last_error']}")
    return 0


def cmd_flywheel(args: argparse.Namespace) -> int:
    _, deps = _build_service(get_settings(), index_kb=False)
    store = deps.flywheel
    stats = store.stats()
    print("问题池统计".center(60, "─"))
    for key in ("total", "pending", "approved", "rejected", "backfilled"):
        print(f"  {key:<12}{stats.get(key, 0)}")
    print(f"  by_source   {stats.get('by_source')}")
    print(f"  by_reason   {stats.get('by_refusal_reason')}")
    print(f"  triage      {stats.get('triage')}")

    rows = store.pending(limit=args.limit)
    print()
    print(f"待审列表（按被问频次降序，共 {len(rows)} 条）".center(60, "─"))
    if not rows:
        print("  问题池是空的 —— 先跑 `mewhelp demo` 或 `mewhelp chat` 制造几条拒答。")
        return 0
    for i, rec in enumerate(rows, 1):
        hint = triage_hint(rec, deps.settings)
        snap = rec.retrieval_snapshot
        ev = f"{len(snap.evidences)} 条证据 / top={snap.top_score:.3f}" if snap else "无快照"
        print(f"  {i}. [{rec.status}] 被问 {rec.ask_count} 次 · {rec.source.value}"
              f" · {rec.intent.value if rec.intent else '—'}")
        print(f"     {rec.question}")
        print(f"     分诊：{hint} —— {triage_label(hint)}")
        print(f"     现场：{ev}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """把一组预设问题跑一遍，并排打印分流路径和答复。"""
    service, _ = _build_service(get_settings())
    print("MewHelp 分流演示".center(60, "═"))
    print("九类意图 → 五个出口（knowledge / business_tool / high_risk / handoff / fallback）")
    print()
    print("  意图映射表：")
    for spec in INTENT_SPECS:
        print(f"    {spec.intent.value:<16}{spec.label:<8}→ {INTENT_TO_EXIT[spec.intent].value}")
    print()

    questions = DEMO_QUESTIONS[: args.limit] if args.limit else DEMO_QUESTIONS
    for idx, (tag, question) in enumerate(questions):
        # 每条用一个独立会话：演示要展示的是"同一句话会走到哪个出口"，
        # 共享会话会让上一轮的指代消解改写后面的问题，把演示搞乱。
        # 会话 id 用序号而不是 hash(question)：Python 的 str 哈希带进程级随机盐，
        # 用它会让同一份演示每次跑出不同的会话 id。
        session = f"demo-{idx}"
        result = service.ask(user_id="demo", session_id=session, message=question)
        print(f"【{tag}】{question}")
        print(f"  意图 {result.intent.value if result.intent else '—'}"
              f" → 出口 {result.exit.value if result.exit else '—'}")
        _print_path(result.path)
        _print_tools(result.tool_calls)
        _print_evidences(result.evidences, limit=3)
        tail = "转人工" if result.handoff else "已答复"
        if result.refusal_reason:
            tail += f"（{result.refusal_reason.value}）"
        print(f"  答复（{tail}）：")
        for line in (result.answer or "").splitlines() or [""]:
            print("    " + line)
        print()
    return 0


# ============================================================
# 入口
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mewhelp",
        description="MewHelp —— 电商 AI 客服（Workflow + Agent 混合架构），全 mock 离线可跑",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="启动 FastAPI 服务（含演示页）")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--log-level", default="info")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("chat", help="单轮问答，打印分流路径和证据")
    p.add_argument("question", help="要问的问题（带引号）")
    p.add_argument("--session", default="cli-default", help="会话 id（同 id 即多轮）")
    p.add_argument("--user", default="cli-user")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("index", help="重建知识库索引并打印 chunk 统计")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("flywheel", help="打印问题池统计和待审列表")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_flywheel)

    p = sub.add_parser("demo", help="跑一组预设问题，演示分流路径")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0 = 全部）")
    p.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    _setup_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
