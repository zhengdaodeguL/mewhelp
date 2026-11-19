"""工具层测试：数据夹具的自洽性、注册表的超时/重试/审计、业务铁律的边界。

测试的价值全在"边界"上，所以这里的用例是按**业务规则的分支**组织的，
不是按函数组织的：每条规则都有"成立"和"不成立"两侧，退款时效还钉死在
第 30 天和第 31 天这两个临界点上。

关于日期：所有依赖"今天"的用例都显式传入 today，不依赖系统时间 ——
否则这些用例会在某一天突然变红，而那天没人碰过这些代码。
"""

from __future__ import annotations

import json
import socket
import time
from datetime import timedelta

import pytest

from mewhelp.config import Settings
from mewhelp.domain import ToolError
from mewhelp.schemas import ToolSpec
from mewhelp.tools import ToolRegistry, build_business_registry, register_business_tools
from mewhelp.tools import business
from mewhelp.tools.business import (
    FRESH_WINDOW_HOURS,
    REFUND_WINDOW_DAYS,
    change_address,
    get_order,
    load_order_rows,
    load_orders,
    load_products,
    request_refund,
    search_faq,
    track_logistics,
)

#: 夹具自带的基准日。所有时效用例都从它出发计算，
#: 于是"30 天整"就是 2026-08-20，"第 31 天"就是 2026-08-21 看到的 2026-07-21 那单。
BASE = "2026-08-20"


@pytest.fixture
def registry() -> ToolRegistry:
    """默认注册表 —— today 固定在夹具基准日，行为完全可复现。"""
    return build_business_registry(today=BASE)


def payload(call) -> dict:
    """把 ToolCall.result 取回成业务信封，断言里少写一层嵌套。"""
    assert isinstance(call.result, dict)
    return call.result


@pytest.mark.parametrize(
    "name,order_id,principal_id,extra",
    [
        ("get_order", "ORD-1001", "U1001", {}),
        ("track_logistics", "ORD-1003", "U1002", {}),
        ("request_refund", "ORD-1004", "U1002", {}),
        ("cancel_order", "ORD-1001", "U1001", {}),
        ("change_address", "ORD-1001", "U1001", {"address": "北京市朝阳区测试路 8 号"}),
    ],
)
def test_五种订单工具允许本人订单且保留匿名行为(registry, name, order_id, principal_id, extra):
    args = {"order_id": order_id, **extra}
    anonymous = registry.call(name, args, principal_id=None)
    owned = registry.call(name, args, principal_id=principal_id)
    assert owned.ok and anonymous.ok
    assert owned.result == anonymous.result
    assert owned.attempts == anonymous.attempts == 1


@pytest.mark.parametrize("name", ["get_order", "track_logistics", "request_refund", "cancel_order", "change_address"])
@pytest.mark.parametrize("principal_id", ["U1001", "无订单账号"])
def test_五种订单工具拒绝跨账号且不泄露数据(registry, name, principal_id):
    args = {"order_id": "ORD-1003"}
    if name == "change_address":
        args["address"] = "北京市朝阳区测试路 8 号"
    call = registry.call(name, args, principal_id=principal_id)
    assert not call.ok and call.error == "ORDER_ACCOUNT_UNLINKED"
    assert call.attempts == 0
    assert registry.drain_audit() == [call]
    message = (
        "当前账号未关联这笔订单，暂时无法查询或办理该订单业务。"
        if principal_id == "U1001"
        else "您的账号暂未关联订单信息，暂时无法查询或办理订单业务。"
    )
    assert call.result == {"code": "ORDER_ACCOUNT_UNLINKED", "message": message}


@pytest.mark.parametrize("name", ["get_order", "track_logistics", "request_refund", "cancel_order", "change_address"])
@pytest.mark.parametrize("principal_id", ["U1001", "无订单账号", None])
def test_五种订单工具区分订单不存在(registry, name, principal_id):
    args = {"order_id": "ORD-9999"}
    if name == "change_address":
        args["address"] = "北京市朝阳区测试路 8 号"
    call = registry.call(name, args, principal_id=principal_id)
    assert not call.ok and call.error == "ORDER_NOT_FOUND"
    assert "没有找到订单" in payload(call)["message"]
    assert "未关联" not in payload(call)["message"]


@pytest.mark.parametrize("order_id", ["ord 1001", " ord-1001 ", "ORD1001"])
def test_归属检查复用业务层订单号归一化(registry, order_id):
    owned = registry.call("get_order", {"order_id": order_id}, principal_id="U1001")
    assert owned.ok and payload(owned)["order_id"] == "ORD-1001"
    other = registry.call("get_order", {"order_id": order_id}, principal_id="U1002")
    assert not other.ok and other.error == "ORDER_ACCOUNT_UNLINKED"


def test_本人订单仍受原有业务规则限制(registry):
    call = registry.call("cancel_order", {"order_id": "ORD-1003"}, principal_id="U1002")
    assert not call.ok and call.error == "ORDER_NOT_CANCELLABLE"


# ======================================================================
# 1. 夹具自洽性 —— 数据错了，上层所有测试都是在验证错误的东西
# ======================================================================


VALID_STATUS = {"pending", "processing", "shipped", "delivered", "cancelled"}
VALID_REFUND = {"none", "refunding", "refunded"}
VALID_CATEGORIES = {"生鲜", "数码", "服饰", "家居"}
REQUIRED_ORDER_FIELDS = (
    "order_id",
    "user_id",
    "status",
    "shipping_address",
    "delivered_on",
    "refund_status",
    "created_at",
    "items",
)


def test_orders_fixture_shape_is_valid():
    rows = load_order_rows()
    assert len(rows) >= 10, "契约要求至少 8 单；这里刻意多留几单覆盖长尾状态"
    seen_ids = set()
    for row in rows:
        for field in REQUIRED_ORDER_FIELDS:
            assert field in row, f"{row.get('order_id')} 缺字段 {field}"
        assert row["order_id"] not in seen_ids, f"订单号重复：{row['order_id']}"
        seen_ids.add(row["order_id"])
        assert row["status"] in VALID_STATUS
        assert row["refund_status"] in VALID_REFUND
        assert row["items"], "订单必须有商品行，否则金额和品类判断都没有依据"
        for item in row["items"]:
            assert item["category"] in VALID_CATEGORIES
            assert item["price"] > 0 and item["qty"] >= 1


def test_all_statuses_are_covered():
    """五个状态每种都要有 —— 缺一个就有业务分支永远测不到。"""
    assert {r["status"] for r in load_order_rows()} == VALID_STATUS


def test_delivered_orders_have_past_delivered_on():
    """已签收必须有签收日期，且必须早于基准日。

    未来日期会让"签收几天了"算出负数，退款规则会静默放行 —— 这是
    夹具最容易出的、又最难在测试里发现的错。
    """
    ref = business.fixture_reference_date()
    assert ref is not None, "夹具必须声明 as_of，否则时效用例不可复现"
    for row in load_order_rows():
        if row["status"] == "delivered":
            assert row["delivered_on"], f"{row['order_id']} 已签收却无签收日期"
            assert business._parse_date(row["delivered_on"]) < ref


def test_non_delivered_orders_have_no_delivered_on():
    for row in load_order_rows():
        if row["status"] != "delivered":
            assert row["delivered_on"] is None, f"{row['order_id']} 未签收却有签收日期"


def test_fixture_contains_both_refundable_and_non_refundable_orders():
    """夹具必须自带"能退"和"不能退"两类样本。

    否则 30 天边界一改，测试可能全部落在同一侧，看起来全绿其实什么都没验证。
    """
    ref = business.fixture_reference_date()
    refundable, expired, fresh, cancelled = [], [], [], []
    for row in load_order_rows():
        if row["status"] == "cancelled":
            cancelled.append(row["order_id"])
        if row["status"] != "delivered":
            continue
        age = (ref - business._parse_date(row["delivered_on"])).days
        if row["refund_status"] != "none":
            continue
        if age > REFUND_WINDOW_DAYS:
            expired.append(row["order_id"])
        else:
            refundable.append(row["order_id"])
        if any(i["category"] == "生鲜" for i in row["items"]):
            fresh.append(row["order_id"])
    assert refundable, "缺少可退样本（签收 <= 30 天）"
    assert expired, "缺少不可退样本（签收 > 30 天）"
    assert fresh, "缺少生鲜样本"
    assert cancelled, "缺少已取消样本"


def test_products_fixture_shape_and_sku_alignment():
    """商品 SKU 必须覆盖订单里出现的每一个 —— 否则商品咨询会漏答。"""
    products = load_products()
    assert len(products) >= 8
    for sku, prod in products.items():
        assert prod["name"] and prod["category"] in VALID_CATEGORIES
        # 尺码/材质/保修是 product_faq 最常被问的三件事，字段必须存在
        assert "sizes" in prod and "material" in prod and "warranty" in prod
        assert prod["price"] > 0
    ordered_skus = {i["sku"] for row in load_order_rows() for i in row["items"]}
    missing = ordered_skus - set(products)
    assert not missing, f"订单引用了不存在的 SKU：{sorted(missing)}"


def test_fixture_loading_is_deterministic():
    """同输入同输出：两次加载必须完全一致（缓存不能引入漂移）。"""
    assert json.dumps(load_orders(), sort_keys=True, ensure_ascii=False) == json.dumps(
        load_orders(), sort_keys=True, ensure_ascii=False
    )


def test_missing_fixture_degrades_to_empty(tmp_path, monkeypatch):
    """夹具丢了只该降级成"查不到"，不该抛异常 —— 数据面故障不能变成 500。"""
    monkeypatch.setattr(business, "fixtures_dir", lambda: tmp_path)
    business.reset_fixture_cache()
    try:
        assert load_orders() == {}
        assert business.fixture_reference_date() is None
    finally:
        monkeypatch.undo()
        business.reset_fixture_cache()


# ======================================================================
# 2. 只读工具：成功路径 + ORDER_NOT_FOUND
# ======================================================================


def test_get_order_success(registry):
    call = registry.call("get_order", {"order_id": "ORD-1004"})
    assert call.ok and call.error is None
    data = call.result
    assert data["order_id"] == "ORD-1004"
    assert data["status"] == "delivered"
    assert data["delivered_on"] == "2026-08-10"
    assert len(data["items"]) == 2
    # 金额要算对：599*1 + 129*2 = 857
    assert data["total_amount"] == pytest.approx(857.0)
    assert data["items"][1]["subtotal"] == pytest.approx(258.0)
    assert data["days_since_delivery"] == 10
    assert data["note"].startswith("订单 ORD-1004")


def test_get_order_tolerates_user_typed_order_id(registry):
    """用户在聊天里手打的 "ord 1004" 也该查得到 —— 但绝不能靠模糊匹配猜单。"""
    assert registry.call("get_order", {"order_id": "ord 1004"}).ok
    assert registry.call("get_order", {"order_id": "ORD1004"}).ok


def test_get_order_not_found(registry):
    call = registry.call("get_order", {"order_id": "ORD-9999"})
    assert call.ok is False
    assert call.error == ToolError.ORDER_NOT_FOUND.value
    assert call.result["order_id"] == "ORD-9999"
    assert "没有找到订单" in call.result["message"]


def test_get_order_missing_param_is_not_an_exception(registry):
    """漏传订单号不能炸，也不能被误判成"这单不存在"。

    这两种情况对上层是完全不同的信号：一个是"模型参数给错了，
    让它重试或追问用户"，另一个是"这单真的不在系统里，别再试了"。
    混成一个错误码，Agent 会对着一单真实存在的订单反复说"查无此单"。
    """
    call = registry.call("get_order", {})
    assert call.ok is False
    assert call.error == ToolError.UPSTREAM_ERROR.value
    assert call.result["code"] == "MISSING_ARGUMENTS"
    assert "order_id" in call.result["message"]
    assert call.attempts == 1, "参数错了不该重试"


def test_get_order_missing_param_is_distinct_from_not_found(registry):
    """两个错误码必须能被区分开 —— 这是上面那条用例的存在意义。"""
    missing = registry.call("get_order", {})
    not_found = registry.call("get_order", {"order_id": "ORD-9999"})
    assert missing.error != not_found.error
    assert missing.result["code"] != not_found.result["code"]


def test_track_logistics_shipped_has_trace(registry):
    call = registry.call("track_logistics", {"order_id": "ORD-1003"})
    assert call.ok
    data = call.result
    assert data["carrier"] == "顺丰速运"
    assert data["tracking_no"] == "SF1234567890123"
    assert data["eta"] == "2026-08-22"
    assert len(data["events"]) == 3
    # 轨迹必须按时间正序：倒序的轨迹让人无从判断"现在到哪了"
    dates = [e["date"] for e in data["events"]]
    assert dates == sorted(dates)
    assert data["latest"]["status"] == "派送中"


def test_track_logistics_delivered_reports_days_since_delivery(registry):
    """签收天数放在物流里，是为了让"能不能退"能从一次调用里拿全输入。"""
    call = registry.call("track_logistics", {"order_id": "ORD-1004"})
    assert call.ok
    assert call.result["days_since_delivery"] == 10
    assert call.result["latest"]["status"] == "已签收"


def test_track_logistics_pending_is_not_an_error(registry):
    """"还没发货"是常态而不是错误。

    如果把它做成错误码，上层会把每一次正常的催单都当成异常处理，
    用户在 UI 上看到的也会是红色报错，而不是一句"仓库备货中"。
    """
    call = registry.call("track_logistics", {"order_id": "ORD-1001"})
    assert call.ok, "未发货必须是成功返回"
    assert call.result["carrier"] is None
    assert "等待仓库出库" in call.result["note"]


def test_track_logistics_cancelled(registry):
    call = registry.call("track_logistics", {"order_id": "ORD-1009"})
    assert call.ok
    assert "已取消" in call.result["note"]


def test_track_logistics_not_found(registry):
    call = registry.call("track_logistics", {"order_id": "ORD-0000"})
    assert call.ok is False
    assert call.error == ToolError.ORDER_NOT_FOUND.value


def test_search_faq_points_to_rag(registry):
    """薄封装的契约：命中也好、不命中也好，都必须提示走 RAG。"""
    call = registry.call("search_faq", {"query": "生鲜退款政策"})
    assert call.ok
    assert call.result["query"] == "生鲜退款政策"
    assert isinstance(call.result["hits"], list)
    assert "RAG" in call.result["note"]


def test_search_faq_empty_query_is_rejected(registry):
    call = registry.call("search_faq", {"query": "   "})
    assert call.ok is False
    assert call.result["code"] == "MISSING_QUERY"


def test_search_faq_scores_are_deterministic():
    a = search_faq("生鲜 退款")
    b = search_faq("生鲜 退款")
    assert json.dumps(a, sort_keys=True, ensure_ascii=False) == json.dumps(
        b, sort_keys=True, ensure_ascii=False
    )


def test_search_faq_picks_up_knowledge_files(tmp_path, monkeypatch):
    """知识库目录存在时，FAQ 应该真的读到内容，而不是永远返回空。"""
    (tmp_path / "refund.md").write_text(
        "# 退款政策\n\n生鲜类商品签收后 24 小时内可申请退款，需提供照片。\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(business, "_faq_dir", lambda: tmp_path)
    try:
        result = search_faq("生鲜退款")
        assert result["ok"]
        assert result["data"]["hit_count"] >= 1
        assert result["data"]["hits"][0]["title"] == "退款政策"
        assert result["data"]["hits"][0]["source"] == "refund.md"
    finally:
        monkeypatch.undo()


# ======================================================================
# 3. 业务铁律：cancel_order
# ======================================================================


def test_cancel_order_allowed_when_pending(registry):
    call = registry.call("cancel_order", {"order_id": "ORD-1001"})
    assert call.ok
    assert call.result["cancellable"] is True
    assert call.result["resulting_status"] == "cancelled"
    # 高风险动作不能自称已完成：必须先过审核
    assert call.result["requires_human_approval"] is True


def test_cancel_order_allowed_when_processing(registry):
    call = registry.call("cancel_order", {"order_id": "ORD-1002"})
    assert call.ok
    assert call.result["status"] == "processing"


@pytest.mark.parametrize(
    "order_id,status",
    [("ORD-1003", "shipped"), ("ORD-1004", "delivered"), ("ORD-1009", "cancelled")],
)
def test_cancel_order_rejected_after_shipping(registry, order_id, status):
    call = registry.call("cancel_order", {"order_id": order_id})
    assert call.ok is False
    assert call.error == ToolError.ORDER_NOT_CANCELLABLE.value
    assert call.result["status"] == status


def test_cancel_order_not_found(registry):
    call = registry.call("cancel_order", {"order_id": "ORD-8888"})
    assert call.ok is False
    assert call.error == ToolError.ORDER_NOT_FOUND.value


def test_cancel_order_does_not_mutate_fixture(registry):
    """工具只能"提议"，绝不能真的改掉数据源。

    如果这里改写了夹具，"审核没通过但订单已经没了"就成了真的。
    """
    before = json.dumps(load_orders(), sort_keys=True, ensure_ascii=False)
    registry.call("cancel_order", {"order_id": "ORD-1001"})
    assert before == json.dumps(load_orders(), sort_keys=True, ensure_ascii=False)


# ======================================================================
# 4. 业务铁律：change_address
# ======================================================================


def test_change_address_allowed_when_pending(registry):
    call = registry.call(
        "change_address",
        {"order_id": "ORD-1001", "address": "浙江省杭州市余杭区文一西路 969 号"},
    )
    assert call.ok
    assert call.result["changed"] is True
    assert call.result["new_address"] == "浙江省杭州市余杭区文一西路 969 号"
    assert call.result["current_address"] != call.result["new_address"]
    assert call.result["requires_human_approval"] is True


def test_change_address_allowed_when_processing(registry):
    call = registry.call(
        "change_address", {"order_id": "ORD-1002", "address": "江苏省南京市江宁区将军大道 100 号"}
    )
    assert call.ok
    assert call.result["status"] == "processing"


def test_change_address_identical_is_idempotent(registry):
    """改成同一个地址不该消耗一次人工审核 —— 审核队列是最稀缺的资源。"""
    current = load_orders()["ORD-1001"]["shipping_address"]
    call = registry.call("change_address", {"order_id": "ORD-1001", "address": current})
    assert call.ok
    assert call.result["changed"] is False
    assert call.result["requires_human_approval"] is False


@pytest.mark.parametrize("order_id", ["ORD-1003", "ORD-1004"])
def test_change_address_locked_after_shipping(registry, order_id):
    """已发货/已签收 → 地址锁定。这是快递业的现实，不是可以商量的规则。"""
    call = registry.call("change_address", {"order_id": order_id, "address": "任意新地址"})
    assert call.ok is False
    assert call.error == ToolError.ORDER_ADDRESS_LOCKED.value
    assert "锁定" in call.result["message"]


def test_change_address_cancelled_order(registry):
    call = registry.call("change_address", {"order_id": "ORD-1009", "address": "任意新地址"})
    assert call.ok is False
    assert call.error == ToolError.ORDER_ADDRESS_LOCKED.value


def test_change_address_requires_address(registry):
    """缺地址是参数问题，不是业务拒绝 —— 两者对上层意味着完全不同的下一步。"""
    call = registry.call("change_address", {"order_id": "ORD-1001", "address": "  "})
    assert call.ok is False
    assert call.result["code"] == "MISSING_ADDRESS"
    assert call.error == ToolError.UPSTREAM_ERROR.value


def test_change_address_not_found(registry):
    call = registry.call("change_address", {"order_id": "ORD-7777", "address": "x"})
    assert call.ok is False
    assert call.error == ToolError.ORDER_NOT_FOUND.value


def test_change_address_missing_order_id_is_not_found_after_lookup(registry):
    """缺订单号时 change_address 走的是入参校验，不是业务拒绝。"""
    call = registry.call("change_address", {"address": "x"})
    assert call.ok is False
    assert call.result["code"] == "MISSING_ARGUMENTS"


# ======================================================================
# 5. 业务铁律：request_refund —— 含 30 天边界
# ======================================================================


def test_refund_success_within_window():
    # ORD-1004 签收 2026-08-10，基准日 2026-08-20 → 第 10 天
    call = build_business_registry(today=BASE).call("request_refund", {"order_id": "ORD-1004"})
    assert call.ok
    assert call.result["days_since_delivery"] == 10
    assert call.result["remaining_days"] == 20
    assert call.result["deadline"] == "2026-09-09"
    assert call.result["requires_human_approval"] is True


def test_refund_allowed_exactly_on_day_30():
    """边界：正好第 30 天必须可退。

    "第 30 天"到底是可退还是不可退，是这条规则唯一有歧义的地方，
    必须由测试钉死，不能靠读代码猜。
    """
    call = build_business_registry(today=BASE).call("request_refund", {"order_id": "ORD-1005"})
    assert call.ok is True, "签收正好 30 天应当仍然可退"
    assert call.result["days_since_delivery"] == REFUND_WINDOW_DAYS == 30
    assert call.result["remaining_days"] == 0


def test_refund_rejected_on_day_31():
    """边界：第 31 天必须不可退 —— 与上一条配对，构成完整的闭环。

    只测一侧的边界测试是假边界测试：如果阈值被误改成 29，
    只有这条会红；被误改成 31，只有上一条会红。两条一起才锁得住。
    """
    call = build_business_registry(today=BASE).call("request_refund", {"order_id": "ORD-1005"})
    assert call.ok
    # 同一单，把"今天"往后推一天 → 立刻越界
    late = build_business_registry(today="2026-08-21").call(
        "request_refund", {"order_id": "ORD-1005"}
    )
    assert late.ok is False
    assert late.error == ToolError.REFUND_NOT_ELIGIBLE.value
    assert late.result["code"] == "REFUND_WINDOW_EXPIRED"
    assert late.result["days_since_delivery"] == 31


def test_refund_rejected_when_long_expired():
    # ORD-1006 签收 2026-07-20 → 第 31 天
    call = build_business_registry(today=BASE).call("request_refund", {"order_id": "ORD-1006"})
    assert call.ok is False
    assert call.error == ToolError.REFUND_NOT_ELIGIBLE.value
    assert call.result["code"] == "REFUND_WINDOW_EXPIRED"
    assert call.result["days_since_delivery"] == 31


def test_refund_expired_message_explains_short_fresh_window():
    """生鲜超期的解释必须提到 24 小时窗口。

    只说"超过 30 天"对生鲜用户是没有信息量的 —— 他本来就不该按 30 天算，
    系统却拿一条跟他无关的规则打发他。
    """
    call = business.request_refund("ORD-1007", today="2026-09-20")
    assert call["ok"] is False
    assert call["code"] == "REFUND_WINDOW_EXPIRED"
    assert call["fresh_items"] == ["智利车厘子 2kg 礼盒装"]
    assert f"{FRESH_WINDOW_HOURS} 小时" in call["message"]


def test_refund_fresh_order_within_window_flags_short_rule():
    """生鲜在 30 天窗口内也不能按普通商品讲 —— 24 小时窗口必须一起说明。"""
    call = build_business_registry(today=BASE).call("request_refund", {"order_id": "ORD-1007"})
    assert call.ok
    assert call.result["blocked_items"] == ["智利车厘子 2kg 礼盒装"]
    assert "生鲜" in call.result["note"]


@pytest.mark.parametrize(
    "order_id,status",
    [("ORD-1001", "pending"), ("ORD-1002", "processing"), ("ORD-1003", "shipped")],
)
def test_refund_rejected_before_delivery(registry, order_id, status):
    call = registry.call("request_refund", {"order_id": order_id})
    assert call.ok is False
    assert call.error == ToolError.REFUND_NOT_ELIGIBLE.value
    # 错误码是 REFUND_NOT_ELIGIBLE，场景码要能区分"还没签收"
    assert call.result["code"] == "REFUND_STATUS_NOT_DELIVERED"
    assert call.result["status"] == status
    if status in ("pending", "processing"):
        # 未发货时要引导用户去取消，而不是只说一句"不满足条件"
        assert "取消订单" in call.result["message"]


def test_refund_rejected_when_already_refunded(registry):
    call = registry.call("request_refund", {"order_id": "ORD-1010"})
    assert call.ok is False
    assert call.error == ToolError.REFUND_NOT_ELIGIBLE.value
    assert call.result["code"] == "REFUND_ALREADY_APPLIED"


def test_refund_rejected_when_refund_in_progress(registry):
    """重复提交退款请求必须被挡住 —— 否则会重复出款。"""
    call = registry.call("request_refund", {"order_id": "ORD-1011"})
    assert call.ok is False
    assert call.result["code"] == "REFUND_ALREADY_APPLIED"
    assert "处理中" in call.result["message"]


def test_refund_rejected_for_cancelled_order(registry):
    call = registry.call("request_refund", {"order_id": "ORD-1009"})
    assert call.ok is False
    assert call.error == ToolError.REFUND_NOT_ELIGIBLE.value


def test_refund_not_found(registry):
    call = registry.call("request_refund", {"order_id": "ORD-6666"})
    assert call.ok is False
    assert call.error == ToolError.ORDER_NOT_FOUND.value


def test_refund_missing_delivered_on_is_not_silently_approved(monkeypatch):
    """脏数据（已签收但缺签收日期）必须挡住，不能默认可退。

    放行的代价是真金白银退出去一笔不该退的款；挡住的代价只是一次人工核实。
    """
    rows = [dict(r) for r in load_order_rows()]
    for row in rows:
        if row["order_id"] == "ORD-1004":
            row["delivered_on"] = None
    monkeypatch.setattr(
        business, "_read_fixture", lambda name: {"orders": rows, "as_of": BASE}
    )
    business.reset_fixture_cache()
    try:
        call = business.request_refund("ORD-1004", today=BASE)
        assert call["ok"] is False
        assert call["code"] == "DELIVERY_DATE_UNKNOWN"
    finally:
        monkeypatch.undo()
        business.reset_fixture_cache()


def test_refund_window_constant_is_the_single_source_of_truth(registry):
    """30 这个数字只允许有一处定义 —— 规则和话术必须同源。

    如果实现里手写一个 30、常量又改成 15，用户会听到"超过 30 天不能退"
    而系统其实按 15 天拒绝 —— 这是最难被用户举证、也最伤信任的一类 bug。
    """
    # 边界内侧：话术和字段里出现的窗口天数必须等于常量
    inside = registry.call("request_refund", {"order_id": "ORD-1004"})
    assert inside.ok
    assert inside.result["window_days"] == REFUND_WINDOW_DAYS
    assert str(REFUND_WINDOW_DAYS) in inside.result["note"]
    # 边界外侧同样要一致
    outside = registry.call("request_refund", {"order_id": "ORD-1006"})
    assert outside.ok is False
    assert outside.result["window_days"] == REFUND_WINDOW_DAYS
    assert str(REFUND_WINDOW_DAYS) in outside.result["message"]
    # 规则字段算出来的边界日必须真的等于签收日 + 常量
    delivered = business._parse_date(outside.result["delivered_on"])
    assert (delivered + timedelta(days=REFUND_WINDOW_DAYS)).isoformat() == outside.result["deadline"]


def test_refund_window_boundary_is_derived_from_constant():
    """时效边界必须由常量推出来，而不是碰巧写成 30。

    改小常量，边界必须跟着移动：夹具里签收 30 天的那单应当变成不可退。
    这条用例防的是"常量改了但判断里还写着 30"。
    """
    ref = business.fixture_reference_date()
    order = load_orders()["ORD-1005"]
    delivered = business._parse_date(order["delivered_on"])
    # 夹具确实把这一单设计在离边界最近的位置上
    assert (ref - delivered).days == REFUND_WINDOW_DAYS
    # 用常量自己算出来的边界日，正好就是基准日
    assert delivered + timedelta(days=REFUND_WINDOW_DAYS) == ref


# ======================================================================
# 6. 注册表机制：schema / 审计 / 大时间观 / 重试 / 超时
# ======================================================================


def test_specs_is_valid_openai_function_calling_schema(registry):
    specs = registry.specs()
    assert len(specs) == 6
    # 必须能被 json.dumps —— 它要被塞进 HTTP 请求体发给模型
    assert json.loads(json.dumps(specs, ensure_ascii=False)) == specs
    for entry in specs:
        assert set(entry) == {"type", "function"}
        assert entry["type"] == "function"
        fn = entry["function"]
        assert fn["name"] and fn["description"]
        assert isinstance(fn["high_risk"], bool)
        params = fn["parameters"]
        assert params["type"] == "object"
        assert isinstance(params["properties"], dict)
        assert all(p["type"] in {"string", "number", "integer", "boolean"} for p in params["properties"].values())
        # required 必须是 properties 的子集：否则模型会照着一个不存在的参数名去填
        assert set(params.get("required", [])) <= set(params["properties"])
        for name in params.get("required", []):
            assert "description" in params["properties"][name], "必填参数必须有描述"


def test_high_risk_flag_reaches_schema(registry):
    """高风险标记必须一路传到 schema —— 上层靠它决定"能不能直接执行"。"""
    flags = {e["function"]["name"]: e["function"]["high_risk"] for e in registry.specs()}
    assert flags == {
        "get_order": False,
        "track_logistics": False,
        "search_faq": False,
        "request_refund": True,
        "cancel_order": True,
        "change_address": True,
    }
    assert registry.high_risk_actions() == frozenset(
        {"request_refund", "cancel_order", "change_address"}
    )


def test_describe_lists_tools_and_warns_on_high_risk(registry):
    text = registry.describe()
    for name in registry.names():
        assert name in text
    assert "高风险" in text
    # 这条警告必须真的提醒模型"只能提议" —— 它是拦截模型越权的第一道软防线
    assert "不能直接执行" in text


def test_describe_on_empty_registry():
    assert "没有可用工具" in ToolRegistry().describe()


def test_schema_matches_implementation_signature(registry):
    """注册期自检必须干净 —— schema 声明的参数，实现要真的收得到。"""
    assert registry.binding_problems == []


def test_unknown_tool_returns_error_not_exception(registry):
    call = registry.call("no_such_tool", {"x": 1})
    assert call.ok is False
    assert call.error == ToolError.UPSTREAM_ERROR.value
    assert "no_such_tool" in call.result["message"]
    # 把可用工具名回给上层，模型下一轮才能自己纠正
    assert call.result["available"] == registry.names()


def test_bad_param_name_fails_fast_without_retrying(registry):
    """参数名对不上时不该重试：重试再多次也不会让 schema 变对，只会让用户等更久。"""
    call = registry.call("get_order", {"oid": "ORD-1004"})
    assert call.ok is False
    assert call.attempts == 1
    assert call.result["code"] == "MISSING_ARGUMENTS"


def test_unknown_param_name_is_reported_as_such(registry):
    """必填参数给齐了、但多给了一个不认识的参数 → 要说清是多给了什么。

    只说"缺少 order_id"会让模型以为参数不够，于是把同一个错误参数再传一次。
    """
    call = registry.call("get_order", {"order_id": "ORD-1004", "oid": "x"})
    assert call.ok is False
    assert call.result["code"] == "UNKNOWN_ARGUMENTS"
    assert "oid" in call.result["message"]
    assert call.attempts == 1


def test_optional_param_can_be_omitted(registry):
    """schema 里有默认值的可选参数，不给也必须能跑。"""
    call = registry.call("request_refund", {"order_id": "ORD-1004"})
    assert call.ok, "today 是可选参数，省略时应当走注册表默认值"


def test_audit_records_both_success_and_failure(registry):
    registry.call("get_order", {"order_id": "ORD-1004"})
    registry.call("get_order", {"order_id": "ORD-9999"})
    audit = registry.audit
    assert len(audit) == 2
    assert all(tc.name == "get_order" for tc in audit)
    assert audit[0].ok is True and audit[0].result["order_id"] == "ORD-1004"
    assert audit[1].ok is False and audit[1].error == ToolError.ORDER_NOT_FOUND.value
    assert all(tc.elapsed_ms >= 0 for tc in audit)
    assert all(tc.attempts >= 1 for tc in audit)


def test_drain_audit_clears(registry):
    registry.call("get_order", {"order_id": "ORD-1004"})
    drained = registry.drain_audit()
    assert len(drained) == 1
    assert registry.audit == []


def test_tool_exception_is_contained():
    """工具内部抛异常 → 收敛成 ToolCall(ok=False)，绝不外泄。"""
    registry = ToolRegistry()

    def boom(order_id: str) -> dict:
        raise RuntimeError("数据库连接断了")

    registry.register(
        ToolSpec(name="boom", description="故意炸", parameters={"type": "object", "properties": {"order_id": {"type": "string"}}}),
        boom,
    )
    call = registry.call("boom", {"order_id": "ORD-1004"})
    assert call.ok is False
    assert call.error == ToolError.UPSTREAM_ERROR.value
    assert "RuntimeError" in call.result["exception"]
    assert "数据库连接断了" in call.result["exception"]
    # 异常也必须进审计，否则"为什么这单没查出来"就查无实据
    assert registry.audit[-1].ok is False


def test_non_dict_return_is_passed_through():
    """工具可以返回非 dict —— 简单工具不该被迫套一层信封。"""
    registry = ToolRegistry()
    registry.register(
        ToolSpec(name="ping", description="返回 pong", parameters={"type": "object", "properties": {}}),
        lambda: "pong",
    )
    call = registry.call("ping", {})
    assert call.ok and call.result == "pong"


def test_retry_succeeds_on_second_attempt_and_records_it():
    """flaky 工具重试 1 次后成功，attempts 必须如实写 2。

    审计里最怕"看起来成功、其实是第三次才成功" —— 这决定了要不要去修下游。
    """
    registry = ToolRegistry()
    state = {"n": 0}

    def flaky(order_id: str) -> dict:
        state["n"] += 1
        if state["n"] == 1:
            raise ConnectionError("第一次抖动")
        return {"ok": True, "data": {"order_id": order_id}}

    registry.register(
        ToolSpec(name="flaky", description="抖动", parameters={"type": "object", "properties": {"order_id": {"type": "string"}}}),
        flaky,
    )
    call = registry.call("flaky", {"order_id": "ORD-1004"})
    assert call.ok is True
    assert call.attempts == 2, "重试次数必须如实记录"
    assert call.result == {"order_id": "ORD-1004"}


def test_all_retries_exhausted_reports_attempts():
    registry = ToolRegistry()
    counter = {"n": 0}

    def always_fail(order_id: str) -> dict:
        counter["n"] += 1
        raise RuntimeError("一直炸")

    registry.register(
        ToolSpec(name="bad", description="坏", parameters={"type": "object", "properties": {"order_id": {"type": "string"}}}),
        always_fail,
    )
    call = registry.call("bad", {"order_id": "ORD-1004"})
    assert call.ok is False
    assert call.attempts == 2, "tool_max_retries 默认 2，应该尝试两次"
    assert counter["n"] == 2


def test_timeout_is_caught_as_tool_timeout():
    """超时必须变成 ToolCall(ok=False, error="TIMEOUT")，不能把整个请求拖死。"""
    registry = ToolRegistry()
    registry._timeout_override = 0.2  # 显式开关，测试意图自解释

    def hang() -> dict:
        time.sleep(5)
        return {"ok": True, "data": {}}

    registry.register(
        ToolSpec(name="hang", description="卡住", parameters={"type": "object", "properties": {}}),
        hang,
    )
    t0 = time.perf_counter()
    call = registry.call("hang", {})
    elapsed = time.perf_counter() - t0

    assert call.ok is False
    assert call.error == ToolError.TIMEOUT.value
    assert call.error == "TIMEOUT", "错误码字面量也是契约的一部分"
    assert call.attempts == 1
    # 关键断言：真的按时返回了，没有被 sleep(5) 拖住
    assert elapsed < 2.0, f"超时保护失效，实际耗时 {elapsed:.2f}s"


def test_configured_timeout_comes_from_settings():
    """超时阈值必须来自 settings，而不是散落在代码里的魔法数字。"""
    registry = ToolRegistry(Settings(tool_timeout_s=0.15))

    def hang() -> dict:
        time.sleep(3)
        return {}

    registry.register(
        ToolSpec(name="hang", description="卡住", parameters={"type": "object", "properties": {}}),
        hang,
    )
    t0 = time.perf_counter()
    call = registry.call("hang", {})
    assert call.error == ToolError.TIMEOUT.value
    assert time.perf_counter() - t0 < 2.0


def test_retry_attempts_are_capped_by_total_budget():
    """重试不能把总耗时放大成 retries × timeout。

    每个 attempt 都吃同一个总预算：下游彻底挂了的时候，
    用户等 5 秒和等 15 秒是完全不同的体验。
    """
    registry = ToolRegistry(Settings(tool_timeout_s=0.3, tool_max_retries=3))

    def hang() -> dict:
        time.sleep(3)
        return {}

    registry.register(
        ToolSpec(name="hang", description="卡住", parameters={"type": "object", "properties": {}}),
        hang,
    )
    t0 = time.perf_counter()
    call = registry.call("hang", {})
    elapsed = time.perf_counter() - t0
    assert call.error == ToolError.TIMEOUT.value
    assert elapsed < 1.5, f"总预算没兜住，实际耗时 {elapsed:.2f}s"


def test_zero_timeout_still_records_one_attempt():
    """配置成 0 秒这种极端值也要给出能解释的记录，不能 attempts=0。"""
    registry = ToolRegistry(Settings(tool_timeout_s=0.0))
    registry.register(
        ToolSpec(name="any", description="d", parameters={"type": "object", "properties": {}}),
        lambda: "ok",
    )
    call = registry.call("any", {})
    assert call.ok is False
    assert call.error == ToolError.TIMEOUT.value
    assert call.attempts == 1


# ======================================================================
# 7. 注入与确定性
# ======================================================================


def test_today_can_be_injected_per_call():
    """同一个 registry 上推一天，"刚好过期"这件事就能被演示出来。"""
    registry = build_business_registry(today=BASE)
    assert registry.call("request_refund", {"order_id": "ORD-1005"}).ok is True
    assert (
        registry.call("request_refund", {"order_id": "ORD-1005", "today": "2026-08-21"}).ok
        is False
    )


def test_explicit_arg_beats_registry_default():
    registry = build_business_registry(today="2026-08-21")
    call = registry.call("request_refund", {"order_id": "ORD-1005", "today": BASE})
    assert call.ok is True, "调用方显式传的日期优先级必须高于注册表默认值"


def test_today_is_not_injected_into_tools_that_do_not_accept_it():
    """只读工具不接受 today —— 注入逻辑不能把参数塞崩。"""
    registry = build_business_registry(today=BASE)
    assert registry.call("get_order", {"order_id": "ORD-1004"}).ok is True
    assert registry.call("search_faq", {"query": "退款"}).ok is True
    assert registry.call("track_logistics", {"order_id": "ORD-1003"}).ok is True


def test_default_registry_is_offline_and_deterministic():
    """默认配置必须纯 mock 可跑：不发网络请求、不读 API Key。"""
    registry = build_business_registry()
    before = socket.socket
    socket.socket = lambda *a, **k: pytest.fail("工具层不允许发起网络请求")
    try:
        first = registry.call("get_order", {"order_id": "ORD-1004"})
        second = registry.call("get_order", {"order_id": "ORD-1004"})
    finally:
        socket.socket = before
    assert first.ok and second.ok
    # 同输入同输出：除耗时外必须逐字节相同
    assert first.model_dump(exclude={"elapsed_ms"}) == second.model_dump(exclude={"elapsed_ms"})


def test_repeated_refund_calls_are_stable():
    """重复调用不能因为内部状态累积而给出不同答案。"""
    results = [business.request_refund("ORD-1004", today=BASE) for _ in range(3)]
    assert all(r["ok"] for r in results)
    assert json.dumps(results[0], sort_keys=True, ensure_ascii=False) == json.dumps(
        results[2], sort_keys=True, ensure_ascii=False
    )


def test_register_business_tools_into_existing_registry():
    """契约签名：register_business_tools 要能往已有注册表里加工具。"""
    registry = ToolRegistry(today=BASE)
    assert register_business_tools(registry) is registry
    assert len(registry.specs()) == 6
    assert registry.call("get_order", {"order_id": "ORD-1004"}).ok


def test_duplicate_register_last_wins():
    registry = ToolRegistry()
    spec = ToolSpec(name="dup", description="v1", parameters={"type": "object", "properties": {}})
    registry.register(spec, lambda: "v1")
    registry.register(spec.model_copy(update={"description": "v2"}), lambda: "v2")
    assert registry.call("dup", {}).result == "v2"
    assert len(registry.specs()) == 1


def test_register_rejects_empty_name_and_non_callable():
    registry = ToolRegistry()
    with pytest.raises(ValueError):
        registry.register(ToolSpec(name="", description="x", parameters={}), lambda: 1)
    with pytest.raises(TypeError):
        registry.register(
            ToolSpec(name="x", description="x", parameters={}), "not callable"  # type: ignore[arg-type]
        )


def test_malformed_args_do_not_crash(registry):
    """模型偶尔把 action_input 生造成数组/字符串 —— 不能因此抛异常。

    真实项目里这类输入是常态：function-calling 的 arguments 是模型生成的
    JSON，越界的情况总会发生。工具层必须在门口收敛掉。
    """
    for bad in ([], "not json", 42, {"a": 1}, "{\"order_id\": \"ORD-1004\"}"):
        call = registry.call("get_order", bad)  # type: ignore[arg-type]
        assert isinstance(call.ok, bool), f"输入 {bad!r} 没能收敛成 ToolCall"
    # 合法 JSON 字符串应当被真的解析出来，而不是一律当空参数丢掉
    assert registry.call("get_order", '{"order_id": "ORD-1004"}').ok


def test_unknown_tool_attempts_zero_reflects_no_downstream_call():
    """未知工具 attempts=0：一次都没真的执行过，审计不能凑数。"""
    call = ToolRegistry().call("ghost", {})
    assert call.ok is False
    assert call.attempts == 0


def test_elapsed_ms_is_measured_not_faked():
    """elapsed_ms 要反映真实耗时 —— 假数据会让"哪一步慢"这个查询失去意义。"""
    registry = ToolRegistry()

    def slow() -> dict:
        time.sleep(0.05)
        return {"ok": True, "data": {}}

    registry.register(
        ToolSpec(name="slow", description="慢", parameters={"type": "object", "properties": {}}),
        slow,
    )
    call = registry.call("slow", {})
    assert call.elapsed_ms >= 50.0
