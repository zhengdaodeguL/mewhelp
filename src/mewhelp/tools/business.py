"""业务工具：订单 / 物流 / FAQ / 三个高风险动作。

这一层的价值不在于"读了个 JSON"，而在于**业务铁律写在哪里**。
规则写在工具里，而不是写在 prompt 里让模型记住 —— 因为：
  * prompt 里的规则会被模型用"帮用户一次"的善意绕过去，工具里的不会；
  * 规则写在代码里能被单测钉死，写在 prompt 里只能靠抽样评测。
面试追问"如果模型答应给用户退款了怎么办"的答案是：它只能*提议*，
真正生效要过 request_refund 这道门，门后面的规则是确定的。

关于"今天"：
退款时效、生鲜短窗口这些规则都依赖当前日期。直接用 `date.today()`
会让"正好第 30 天可退"这条用例明天失效，CI 变成薛定谔的红。
所以日期一律从外面注入（三个高风险工具都接受 `today` 关键字参数），
没注入时才退化成真实当天 —— 生产要实时，测试要可复现，两者不冲突。

返回契约（业务工具通用）：
    {"ok": bool, "data": Any, "error": <ToolError 值>|None, "code": str, "message": str}
`ok=False` 表示**业务拒绝**（预期内），不是异常。调用方（ToolRegistry）
会把它翻成 `ToolCall(ok=False, error=<error>)`。
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..config import get_settings
from ..domain import ToolError
from ..schemas import ToolSpec
from .registry import ToolRegistry

#: 退货/退款时效（自然日）。写成常量而不是散落在分支里的 30，
#: 是因为它同时出现在规则判断和给用户的解释话术里，两处必须一致。
REFUND_WINDOW_DAYS = 30

#: 生鲜类的"短窗口"：签收后 24 小时。
#: 生鲜不是不能退，而是窗口短得多 —— 把它单独拎出来，
#: 是为了防止用户听到一句干巴巴的"超过 30 天不能退"。
FRESH_WINDOW_HOURS = 24

#: 工具入参统一用 ISO 日期字符串而不是 python 对象。
#: function-calling 传过来的一定是 JSON，让类型在门口就收敛，
#: 内部解析失败也有明确错误码，而不是让 date 对象渗进 schema。
_DATE_FMT = "%Y-%m-%d"

_DEFAULT_ADDRESS_MARKERS = ("默认地址", "原地址", "原来的地址")

#: 今天的兜底口径 —— 固定用东八区。
#: 订单时间戳都是 +08:00，如果用系统本地时区（CI 容器常是 UTC），
#: 每天会有 8 小时窗口把"昨天签收"算成"今天签收"。
CN_TZ = timezone(timedelta(hours=8))


# ======================================================================
# 夹具加载
# ======================================================================


def fixtures_dir() -> Path:
    """夹具目录。走 settings 而不是自己拼相对路径 —— cwd 不同就找不到数据。"""
    return Path(get_settings().data_dir) / "fixtures"


@lru_cache(maxsize=8)
def _load_json(path_str: str, _mtime: float) -> dict[str, Any]:
    """读夹具。mtime 参与缓存键：改了夹具不用重启进程就能生效。

    缓存的意义不只是省 IO：夹具内容在**一次请求内必须稳定**，
    否则同一次回答里"订单状态"前后不一致，是最难解释的那类 bug。
    """
    with open(path_str, encoding="utf-8") as fh:
        return json.load(fh)


def _read_fixture(filename: str) -> dict[str, Any]:
    path = fixtures_dir() / filename
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    try:
        return _load_json(str(path), mtime)
    except (json.JSONDecodeError, OSError):
        # 夹具坏了不该让整条链路挂掉：返回空集合，业务工具会正常给出
        # ORDER_NOT_FOUND —— 对用户来说"查不到单"是能理解的降级，
        # 比 500 好得多。
        return {}


def load_orders() -> dict[str, dict[str, Any]]:
    """订单表，按 order_id 建索引。索引在这里建一次，调用点就不用 O(n) 扫。"""
    doc = _read_fixture("orders.json")
    return {o["order_id"]: o for o in doc.get("orders", []) if o.get("order_id")}


def load_order_rows() -> list[dict[str, Any]]:
    """按原始顺序返回订单 —— 测试和评估要遍历全量时用。"""
    return list(_read_fixture("orders.json").get("orders", []))


def load_products() -> dict[str, dict[str, Any]]:
    """商品表，按 sku 索引（与订单 items[].sku 对齐）。"""
    doc = _read_fixture("products.json")
    return {p["sku"]: p for p in doc.get("products", []) if p.get("sku")}


def fixture_reference_date() -> date | None:
    """夹具自带的基准日期。

    夹具里的订单签收日期是写死的，如果"今天"是系统当天，那么随着真实
    时间流逝，同一批用例会从"可退"漂移成"不可退"。夹具声明了 as_of，
    就说明这套数据是**围绕那一天设计的**；业务工具在没收到显式 today 时
    用它兜底，保证离线跑永远看到同样的结果。生产实现要去掉这个兜底。
    """
    raw = _read_fixture("orders.json").get("as_of")
    if not raw:
        return None
    try:
        return datetime.strptime(raw, _DATE_FMT).date()
    except ValueError:
        return None


def reset_fixture_cache() -> None:
    """测试用：换掉夹具目录后清缓存。"""
    _load_json.cache_clear()


# ======================================================================
# 公共辅助
# ======================================================================


def _ok(data: Any, message: str = "") -> dict[str, Any]:
    # 人类可读的结论放进 data 里（而不是信封套上）——上层拿到 result 就能
    # 直接把它当回答的素材，不用再回到封套层去取第二样东西。
    if message and isinstance(data, dict) and "note" not in data:
        data = {**data, "note": message}
    return {"ok": True, "data": data, "error": None, "code": "OK", "message": message}


def _fail(code: str, message: str, *, error: str | None = None, **extra: Any) -> dict[str, Any]:
    out = {
        "ok": False,
        "data": None,
        # `error` 是给 ToolCall.error 用的标准错误码；`code` 是更细的场景说明，
        # 比如同样是 UPSTREAM_ERROR，"缺 address 参数"和"下游炸了"要分开。
        "error": error or code,
        "code": code,
        "message": message,
        "order_id": None,
        "retryable": False,
    }
    out.update(extra)
    return out


def _parse_date(value: str | date | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value).strip(), _DATE_FMT).date()
    except ValueError:
        return None


def _resolve_today(today: str | date | None) -> date:
    """把注入的日期归一成 date；没注入就用夹具基准日，再没有才用真实当天。"""
    parsed = _parse_date(today)
    if parsed is not None:
        return parsed
    return fixture_reference_date() or datetime.now(CN_TZ).date()


def _find_order(order_id: str | None) -> dict[str, Any] | None:
    if not order_id:
        return None
    # 订单号对大小写和空格敏感度低：用户在微信里手打的 "ord 1001" 也该能查到，
    # 但查不到时必须**明确报 ORDER_NOT_FOUND**，不能靠模糊匹配猜一个近似的单
    # —— 猜错的代价是把别人的订单信息念给当前用户听。
    key = str(order_id).strip().upper().replace(" ", "-")
    orders = load_orders()
    if key in orders:
        return orders[key]
    compact = key.replace("-", "")
    for oid, order in orders.items():
        if oid.replace("-", "") == compact:
            return order
    return None


def order_summary(order: dict[str, Any], *, today: date | None = None) -> dict[str, Any]:
    """订单的对外视图 —— 刻意不直接返回夹具原始 dict。

    直接返回原始 dict 会让新增内部字段（成本价、风控标记）在不经意间
    暴露给模型和用户。走一次显式投影，暴露面就是可审计的。

    `today` 决定"签收几天了" —— 这个数字直接决定用户能不能退，
    所以它必须按调用时的口径（而不是执行时的真实时间）算出来，
    否则注入日期做测试时，订单视图和退款判断会给出互相矛盾的两个答案。
    """
    items = order.get("items") or []
    amount = round(sum(float(i["price"]) * int(i["qty"]) for i in items), 2)
    delivered_on = _parse_date(order.get("delivered_on"))
    ref = today or _resolve_today(None)
    return {
        "order_id": order["order_id"],
        "user_id": order.get("user_id"),
        "status": order.get("status"),
        "status_label": {
            "refunded": "已退款", "refunding": "退款中",
        }.get(order.get("refund_status")) or STATUS_LABELS.get(order.get("status", ""), order.get("status")),
        "shipping_address": order.get("shipping_address"),
        "delivered_on": order.get("delivered_on"),
        "days_since_delivery": (ref - delivered_on).days if delivered_on else None,
        "refund_status": order.get("refund_status", "none"),
        "created_at": order.get("created_at"),
        "items": [
            {
                "sku": i["sku"],
                "name": i["name"],
                "category": i["category"],
                "price": i["price"],
                "qty": i["qty"],
                # 多件商品最常被追问"这个多少钱"，先把小计算好，
                # 免得模型自己乘错（乘法恰好是它最容易出错的算术）。
                "subtotal": round(float(i["price"]) * int(i["qty"]), 2),
            }
            for i in items
        ],
        "total_amount": amount,
        "currency": "CNY",
    }


def list_orders(
    user_id: str | None = None, *, today: date | None = None, limit: int = 8
) -> list[dict[str, Any]]:
    """某个用户名下的订单摘要，按创建时间倒序。

    用途是**订单快捷选择**：用户问"我那单到哪了"时不必手打 ORD-1007，
    点一下就行。这里刻意返回精简摘要（订单号 / 状态 / 首件商品 / 金额），
    而不是整单 —— 一个选择器只需要够用户认出"是哪一单"。

    `user_id` 为空时回落到夹具里的第一个用户。真实系统里这个参数必须来自
    登录态，而不是让调用方随便传 —— 否则这就是个越权读单的洞。
    """
    rows = load_order_rows()
    target = user_id or (rows[0].get("user_id") if rows else None)
    mine = [o for o in rows if o.get("user_id") == target]
    mine.sort(key=lambda o: str(o.get("created_at") or ""), reverse=True)

    out: list[dict[str, Any]] = []
    for order in mine[:limit]:
        summary = order_summary(order, today=today)
        items = summary["items"]
        out.append(
            {
                "order_id": summary["order_id"],
                "status": summary["status"],
                "status_label": summary["status_label"],
                "total_amount": summary["total_amount"],
                "currency": summary["currency"],
                "item_count": sum(int(i["qty"]) for i in items),
                "title": items[0]["name"] if items else "",
                "created_at": summary["created_at"],
            }
        )
    return out


#: 状态 → 中文。模型面对 "shipped" 容易自造成"已发货"以外的说法，
#: 给出标准译名能让它保持口径一致，也让前端不用自己维护一张表。
STATUS_LABELS: dict[str, str] = {
    "pending": "待付款/待处理",
    "processing": "备货中",
    "shipped": "已发货",
    "delivered": "已签收",
    "cancelled": "已取消",
}

_STATUS_LABELS_CN: dict[str, str] = {
    "pending": "待处理",
    "processing": "备货中",
    "shipped": "已发货",
    "delivered": "已签收",
    "cancelled": "已取消",
}


# ======================================================================
# 只读工具
# ======================================================================


def get_order(order_id: str) -> dict[str, Any]:
    """按订单号查订单详情。"""
    order = _find_order(order_id)
    if order is None:
        return _fail(
            ToolError.ORDER_NOT_FOUND.value,
            f"没有找到订单 {order_id or '(未提供订单号)'}，请确认订单号是否正确，"
            "或提供下单时使用的账号。",
            order_id=order_id,
        )
    summary = order_summary(order)
    details = [f"订单 {summary['order_id']} 当前状态：{summary['status_label']}。"]
    if summary.get("created_at"):
        details.append(f"下单日期：{str(summary['created_at'])[:10]}。")
    if summary.get("delivered_on"):
        details.append(f"签收日期：{summary['delivered_on']}。")
    elif (order.get("logistics") or {}).get("events"):
        times = [event.get("time") for event in order["logistics"]["events"] if event.get("time")]
        if times:
            details.append(f"最新物流时间：{max(times)}。")
    if summary["items"]:
        goods = "、".join(f"{item['name']} ×{item['qty']}" for item in summary["items"])
        details.append(f"商品：{goods}。")
    return _ok(summary, "".join(details))


def _fmt_day(d: date) -> str:
    return f"{d.month} 月 {d.day} 日"


def track_logistics(order_id: str) -> dict[str, Any]:
    """查物流轨迹。"""
    order = _find_order(order_id)
    if order is None:
        return _fail(
            ToolError.ORDER_NOT_FOUND.value,
            f"没有找到订单 {order_id or '(未提供订单号)'}，无法查询物流。",
            order_id=order_id,
        )

    status = order.get("status")
    logi = order.get("logistics") or {}
    base = {
        "order_id": order["order_id"],
        "status": status,
        "status_label": _STATUS_LABELS_CN.get(status, status),
        "carrier": logi.get("carrier"),
        "tracking_no": logi.get("tracking_no"),
        "eta": None,
        "events": [],
        "latest": None,
        "days_since_delivery": None,
    }

    if status == "cancelled":
        base["note"] = "该订单已取消，没有产生物流记录。"
        return _ok(base, base["note"])

    if not logi:
        # 未发货不是错误 —— 用户问"发货了吗"，答案是"还没发"，
        # 把它做成错误码会逼上层把常态当异常处理。
        base["note"] = (
            "订单已下单，正在等待仓库出库。" if status == "pending"
            else "订单正在备货，预计 24 小时内发出。"
        )
        return _ok(base, base["note"])

    events = list(logi.get("events") or [])
    normalised: list[dict[str, Any]] = []
    for ev in events:
        ts = str(ev.get("time", ""))
        parsed = None
        try:
            parsed = datetime.fromisoformat(ts)
        except ValueError:
            parsed = None
        normalised.append(
            {
                "time": ts,
                "date": parsed.date().isoformat() if parsed else None,
                "status": ev.get("status"),
                "desc": ev.get("desc"),
            }
        )
    normalised.sort(key=lambda e: (e["date"] or "", e["time"]))

    base["events"] = normalised
    base["latest"] = normalised[-1] if normalised else None
    base["eta"] = logi.get("eta")

    if status == "shipped":
        eta = _parse_date(logi.get("eta"))
        base["note"] = f"包裹在途，承运商预计 {_fmt_day(eta)} 送达。" if eta else "包裹在途。"
    elif status == "delivered":
        delivered_on = _parse_date(order.get("delivered_on"))
        # 签收后多少天是退款判断的另一半依据，放在物流里，
        # 是为了让"能不能退"这个问题可以从一次调用里得到全部输入。
        base["days_since_delivery"] = (
            (_resolve_today(None) - delivered_on).days if delivered_on else None
        )
        base["note"] = "包裹已签收。"
    else:
        base["note"] = "物流信息已更新。"

    return _ok(base, base["note"])


# 搜索词里的"停用噪声"：这些词几乎出现在每个问题里，
# 留着只会让所有 FAQ 得分一样高，等于没排。
_FAQ_STOPWORDS = frozenset(
    "的了吗呢吧啊请问一下我的能不能可不可以有没有怎么怎么办如何想要帮我查个这那个是"
)


def _tokenize(query: str) -> list[str]:
    """极简分词：中文按字、英文/数字按词。

    不引 jieba 是有意的 —— 这个函数只用来给 FAQ 排序一个廉价的初筛信号，
    真正的语义匹配在 RAG 那一路。让它依赖一个模型级分词器，
    会把"提示走 RAG"这件轻活变成重活。
    """
    tokens: list[str] = []
    for chunk in re.findall(r"[a-zA-Z0-9]+|[一-鿿]+", query or ""):
        if chunk.isascii():
            tokens.append(chunk.lower())
        else:
            tokens.extend(ch for ch in chunk if ch not in _FAQ_STOPWORDS)
    return tokens


@lru_cache(maxsize=1)
def _faq_dir() -> Path:
    """FAQ 目录。优先用知识库目录 —— FAQ 本来就是知识库的一部分，
    在这里另开一份数据源会产生"两处口径不一致"的隐患。"""
    settings = get_settings()
    kb = Path(settings.knowledge_dir)
    if kb.is_dir() and any(kb.glob("*.md")):
        return kb
    faq = Path(settings.data_dir) / "faq"
    return faq


def _faq_snippets(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """从知识库 Markdown 里挑最相关的几段。

    返回空列表是**正常结果**（知识库为空、话题在库里没有），
    不是错误 —— 上层据此走 RAG 或转人工即可。
    """
    directory = _faq_dir()
    if not directory.is_dir():
        return []
    tokens = _tokenize(query)
    if not tokens:
        return []

    scored: list[tuple[float, str, str, str]] = []
    for path in sorted(directory.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for heading, body in _split_markdown(text):
            body_flat = body.strip()
            if not body_flat:
                continue
            score = _score(tokens, f"{heading} {body_flat}")
            if score > 0:
                scored.append((score, path.name, heading, body_flat))

    scored.sort(key=lambda row: (-row[0], row[1], row[2]))
    out: list[dict[str, Any]] = []
    for score, source, heading, body in scored[:top_k]:
        out.append(
            {
                "title": heading,
                "snippet": body[:200].replace("\n", " "),
                "score": round(score, 4),
                "source": source,
            }
        )
    return out


def _split_markdown(text: str) -> list[tuple[str, str]]:
    """按标题切段。契约里知识库就是"按标题层级 + 表格"写的，照这个结构切。"""
    sections: list[tuple[str, str]] = []
    heading = ""
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            if buf:
                sections.append((heading, "\n".join(buf)))
                buf = []
            heading = line.lstrip("#").strip()
            continue
        buf.append(line)
    if buf:
        sections.append((heading, "\n".join(buf)))
    return sections


def _score(tokens: list[str], text: str) -> float:
    """命中覆盖率：命中的不同 token 占查询 token 的比例。

    用覆盖率而不是命中次数，是因为长段落天然会命中更多次 ——
    那会让"哪段最长哪段最相关"，恰恰是最没用的排序。
    """
    if not text:
        return 0.0
    hits = {t for t in set(tokens) if t in text}
    return len(hits) / max(1, len(set(tokens)))


def search_faq(query: str) -> dict[str, Any]:
    """FAQ 薄封装：给一个粗排信号，真正的语义匹配交给 RAG。"""
    query = (query or "").strip()
    if not query:
        return _fail(
            "MISSING_QUERY",
            "请提供要查询的问题文本。",
            error=ToolError.UPSTREAM_ERROR.value,
        )
    hits = _faq_snippets(query)
    if hits:
        message = (
            "已匹配到相关知识库片段，建议走知识检索（RAG）流程生成有依据的回答，"
            "不要直接复述这里的片段。"
        )
    else:
        # 没有命中时不能沿用"已匹配到"的话术 —— 模型看到一句"已匹配"，
        # 很容易顺着编一个答案出来。空结果就明说空结果。
        message = (
            "未在知识库中匹配到相关片段，请改用知识检索（RAG）流程，"
            "若仍无结果则按规则拒答或转人工，不要凭猜测作答。"
        )
    return _ok(
        {
            "query": query,
            "hits": hits,
            "hit_count": len(hits),
        },
        # 把"别用我来答题"写进 message 而不是注释里：这个工具最容易被
        # 误当成知识问答用，而它的粗排结果不足以支撑有依据的回答。
        message,
    )


# ======================================================================
# 高风险动作（high_risk=True，Agent 不能直接调用）
# ======================================================================


def request_refund(order_id: str, today: str | date | None = None) -> dict[str, Any]:
    """申请退款。只有已签收且在时效内的订单可以退。

    规则顺序是有讲究的：先判存在 → 再判状态 → 再判是否已退过 → 再判时效。
    如果先判时效，一个未签收的订单会得到"超过 30 天"这种明显错误的解释，
    用户会以为系统坏了。**错误信息本身也是产品**。
    """
    order = _find_order(order_id)
    if order is None:
        return _fail(
            ToolError.ORDER_NOT_FOUND.value,
            f"没有找到订单 {order_id or '(未提供订单号)'}，无法申请退款。",
            order_id=order_id,
        )

    status = order.get("status")
    refund_status = order.get("refund_status", "none")
    snapshot = {
        "order_id": order["order_id"],
        "status": status,
        "refund_status": refund_status,
        "data": order_summary(order, today=_resolve_today(today)),
    }

    if status != "delivered":
        if status == "cancelled":
            msg = "该订单已取消，无需申请退款；如已付款会自动原路退回。"
        elif status in ("pending", "processing"):
            msg = f"订单当前为「{_STATUS_LABELS_CN.get(status)}」，尚未发货，请改用取消订单。"
        elif status == "shipped":
            msg = "订单已发货但还没签收，请等签收后再申请退款。"
        else:
            msg = f"订单当前状态为「{status}」，不满足退款条件。"
        return _fail("REFUND_STATUS_NOT_DELIVERED", msg, error=ToolError.REFUND_NOT_ELIGIBLE.value, **snapshot)

    if refund_status in ("refunded", "refunding"):
        done = refund_status == "refunded"
        return _fail(
            "REFUND_ALREADY_APPLIED",
            "该订单已完成退款，请留意到账通知。" if done else "该订单的退款申请正在处理中，请勿重复提交。",
            error=ToolError.REFUND_NOT_ELIGIBLE.value,
            **snapshot,
        )

    delivered_on = _parse_date(order.get("delivered_on"))
    if delivered_on is None:
        # 数据缺 delivered_on 是脏数据，不能当成"符合条件"放过 ——
        # 宁可与人工核对，也不能自动把钱退出去。
        return _fail(
            "DELIVERY_DATE_UNKNOWN",
            "该订单缺少签收时间记录，需要人工核实后才能处理退款。",
            error=ToolError.REFUND_NOT_ELIGIBLE.value,
            **snapshot,
        )

    ref = _resolve_today(today)
    age = (ref - delivered_on).days
    deadline = delivered_on + timedelta(days=REFUND_WINDOW_DAYS)
    window = {
        "delivered_on": delivered_on.isoformat(),
        "evaluated_on": ref.isoformat(),
        "days_since_delivery": age,
        "window_days": REFUND_WINDOW_DAYS,
        "deadline": deadline.isoformat(),
    }

    if age > REFUND_WINDOW_DAYS:
        # 生鲜要单独说话：它的时效短得多，用户听到"超过 30 天"会觉得
        # 系统在念一条跟自己无关的规则。先把生鲜的真实窗口告诉他。
        fresh = _fresh_items(order)
        if fresh:
            return _fail(
                "REFUND_WINDOW_EXPIRED",
                f"订单签收已 {age} 天，超过 {REFUND_WINDOW_DAYS} 天退款时效。"
                f"其中「{fresh[0]['name']}」属于生鲜类，签收后仅 {FRESH_WINDOW_HOURS} 小时内受理品质问题，"
                "目前已超期；如仍想反馈，可转人工进一步核实。",
                error=ToolError.REFUND_NOT_ELIGIBLE.value,
                fresh_items=[f["name"] for f in fresh],
                **snapshot,
                **window,
            )
        return _fail(
            "REFUND_WINDOW_EXPIRED",
            f"订单签收已 {age} 天，超过 {REFUND_WINDOW_DAYS} 天退款时效，无法在线申请退款。"
            "如果商品存在质量问题，可以转人工进一步处理。",
            error=ToolError.REFUND_NOT_ELIGIBLE.value,
            **snapshot,
            **window,
        )

    fresh = _fresh_items(order)
    data = {
        **snapshot,
        **window,
        "refundable": True,
        "remaining_days": REFUND_WINDOW_DAYS - age,
        "amount": sum(
            float(i["price"]) * int(i["qty"]) for i in (order.get("items") or [])
        ),
        "blocked_items": [f["name"] for f in fresh],
        "requires_human_approval": True,
    }
    msg = f"订单已签收 {age} 天，在 {REFUND_WINDOW_DAYS} 天时效内，可以申请退款。"
    if fresh:
        msg += (
            f"注意：「{'、'.join(f['name'] for f in fresh)}」属于生鲜类，"
            "仅对签收后 24 小时内的品质问题受理，需提供照片。"
        )
    msg += "本次申请需要人工审核后才会生效。"
    return _ok(data, msg)


def cancel_order(order_id: str, today: str | date | None = None) -> dict[str, Any]:
    """取消订单。只允许未发货（pending / processing）。"""
    order = _find_order(order_id)
    if order is None:
        return _fail(
            ToolError.ORDER_NOT_FOUND.value,
            f"没有找到订单 {order_id or '(未提供订单号)'}，无法取消。",
            order_id=order_id,
        )

    status = order.get("status")
    snapshot = {
        "order_id": order["order_id"],
        "status": status,
        "status_label": _STATUS_LABELS_CN.get(status, status),
        "data": order_summary(order),
    }

    if status in ("pending", "processing"):
        data = {
            **snapshot,
            "cancellable": True,
            # 只给"预计结果"不给"已取消"：这个动作要过人工审核才生效，
            # 工具越权先把状态写死，就是"模型一句话就把单删了"。
            "resulting_status": "cancelled",
            "refund_note": (
                "订单未付款，无需退款。"
                if status == "pending"
                else "如已付款，已支付金额将在取消审核通过后原路退回，通常 1~3 个工作日到账。"
            ),
            "requires_human_approval": True,
        }
        return _ok(data, f"订单当前为「{data['status_label']}」，可以取消。本次操作需人工审核后生效。")

    if status == "shipped":
        msg = "订单已发货，无法直接取消。建议签收后申请退款，或使用「拒收」处理。"
    elif status == "delivered":
        msg = "订单已签收，无法取消。如不需要，请申请退款。"
    elif status == "cancelled":
        msg = "该订单此前已取消，无需重复操作。"
    else:
        msg = f"订单当前状态为「{status}」，不支持取消。"

    return _fail(ToolError.ORDER_NOT_CANCELLABLE.value, msg, **snapshot)


def change_address(
    order_id: str, address: str, today: str | date | None = None
) -> dict[str, Any]:
    """修改收货地址。只允许未发货（pending / processing）。"""
    order = _find_order(order_id)
    if order is None:
        return _fail(
            ToolError.ORDER_NOT_FOUND.value,
            f"没有找到订单 {order_id or '(未提供订单号)'}，无法修改地址。",
            order_id=order_id,
        )

    status = order.get("status")
    current = order.get("shipping_address")
    snapshot = {"order_id": order["order_id"], "status": status, "current_address": current}

    if status not in ("pending", "processing"):
        if status == "shipped":
            msg = (
                "订单已发货，收货地址已锁定，无法修改。"
                "可以联系承运商改派，或签收后退货。"
            )
        elif status == "delivered":
            # 别把 delivered 写成 shipped 的复读：签收后锁定的理由不同
            # （货已经在用户手上了），但"锁定"这个结论必须一致地传达。
            msg = "订单已签收，收货地址已锁定，无法修改。如商品需要退回，请申请退货。"
        elif status == "cancelled":
            msg = "订单已取消，地址不再需要修改。"
        else:
            msg = f"订单当前状态为「{status}」，不支持修改地址。"
        return _fail(
            ToolError.ORDER_ADDRESS_LOCKED.value, msg, **snapshot
        )

    new_address = (address or "").strip()
    if not new_address:
        # 地址缺失是参数问题不是业务拒绝：重试或追问补全都能解决，
        # 归到业务错误码会让上层误判成"这个订单不能改地址"。
        return _fail(
            "MISSING_ADDRESS",
            "请提供要修改成的新收货地址。",
            error=ToolError.UPSTREAM_ERROR.value,
            **snapshot,
        )

    if new_address in _DEFAULT_ADDRESS_MARKERS or new_address == current:
        # 幂等：用户说"改回原来的"在语义上已经成立，不必再走一次审核。
        # 这也是防止人工审核队列被无效单挤满。
        return _ok(
            {
                **snapshot,
                "new_address": current,
                "changed": False,
                "requires_human_approval": False,
            },
            "新的收货地址与原地址一致，无需修改。",
        )

    data = {
        **snapshot,
        "new_address": new_address,
        "changed": True,
        "resulting_status": status,
        "requires_human_approval": True,
    }
    return _ok(
        data,
        f"收货地址将修改为：{new_address}。订单当前为「{_STATUS_LABELS_CN.get(status, status)}」，"
        "尚未发货，可以修改；本次修改需人工审核后生效。",
    )


def _fresh_items(order: dict[str, Any]) -> list[dict[str, Any]]:
    return [i for i in (order.get("items") or []) if i.get("category") == "生鲜"]


# ======================================================================
# 规格与注册
# ======================================================================

_ORDER_ID_PROP = {
    "type": "string",
    "description": "订单号，形如 ORD-1001。必须来自用户提供或订单上下文，不要自行编造。",
}

TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="get_order",
        description="按订单号查询订单详情（状态、商品明细、金额、收货地址、签收时间、退款状态）。",
        parameters={
            "type": "object",
            "properties": {"order_id": _ORDER_ID_PROP},
            "required": ["order_id"],
            "additionalProperties": False,
        },
        high_risk=False,
    ),
    ToolSpec(
        name="track_logistics",
        description="按订单号查询物流轨迹（承运商、运单号、最新节点、预计送达时间）。",
        parameters={
            "type": "object",
            "properties": {"order_id": _ORDER_ID_PROP},
            "required": ["order_id"],
            "additionalProperties": False,
        },
        high_risk=False,
    ),
    ToolSpec(
        name="search_faq",
        description=(
            "在知识库里做一次关键词粗筛，返回相关片段的位置提示。"
            "这不是问答工具：它只提供检索信号，正式回答必须走知识检索流程。"
        ),        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要检索的问题或关键词。"}
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        high_risk=False,
    ),
    ToolSpec(
        name="request_refund",
        description=(
            "为已签收订单申请退款。仅当订单状态为已签收、签收未超过 30 天、且未申请过退款时才符合条件。"
            "生鲜类商品仅受理签收后 24 小时内的品质问题。"
            "这是高风险动作：只能向用户提议，必须经人工/用户确认后才可提交。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "order_id": _ORDER_ID_PROP,
                "today": {
                    "type": "string",
                    "description": "（可选）用于计算退款时效的当前日期，ISO 格式 YYYY-MM-DD。测试用，正常调用请省略。",
                },
            },
            "required": ["order_id"],
            "additionalProperties": False,
        },
        high_risk=True,
    ),
    ToolSpec(
        name="cancel_order",
        description=(
            "取消订单。仅当订单处于待处理或备货中（未发货）时才允许。"
            "高风险动作：只能提议，需审核后生效。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "order_id": _ORDER_ID_PROP,
                "today": {
                    "type": "string",
                    "description": "（可选）当前日期，ISO 格式 YYYY-MM-DD。测试用，正常调用请省略。",
                },
            },
            "required": ["order_id"],
            "additionalProperties": False,
        },
        high_risk=True,
    ),
    ToolSpec(
        name="change_address",
        description=(
            "修改订单收货地址。仅当订单未发货（待处理或备货中）时才允许；已发货后地址锁定。"
            "高风险动作：只能提议，需审核后生效。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "order_id": _ORDER_ID_PROP,
                "address": {"type": "string", "description": "修改后的完整收货地址。"},
                "today": {
                    "type": "string",
                    "description": "（可选）当前日期，ISO 格式 YYYY-MM-DD。测试用，正常调用请省略。",
                },
            },
            "required": ["order_id", "address"],
            "additionalProperties": False,
        },
        high_risk=True,
    ),
)

_BUSINESS_FNS = {
    "get_order": get_order,
    "track_logistics": track_logistics,
    "search_faq": search_faq,
    "request_refund": request_refund,
    "cancel_order": cancel_order,
    "change_address": change_address,
}


def register_business_tools(
    registry: ToolRegistry | None = None, *, today: str | date | None = None
) -> ToolRegistry:
    """把六个业务工具装进注册表（或一个新建的）。

    `today` 只用于覆盖注册表本身的默认值；真正的时效判断仍然支持
    逐次传参覆盖 —— 同一个 registry 可以先在 2026-08-20 上调一次、
    再在 2026-08-21 上调一次，验证"今天刚好过期"。
    """
    registry = registry or ToolRegistry(today=today)
    if today is not None and registry.today is None:
        registry.today = today
    for spec in TOOL_SPECS:
        registry.register(spec, _BUSINESS_FNS[spec.name])
    return registry


def build_business_registry(today: str | date | None = None) -> ToolRegistry:
    """开箱即用的业务工具注册表。"""
    return register_business_tools(ToolRegistry(today=today), today=today)
