"""高风险确定性子流程。

退款、退货、取消订单、改地址这四件事的共同点：**做错了要赔钱**。
所以它们不走 ReAct —— 模型在这条路上没有自由发挥的空间，只能沿着预定的
状态机走，每一步的准入条件都在工具里（见 tools/business.py）。

为什么需要"中断"这件事：

    用户：我要退款
    系统：需要订单号才能帮你查      ← 流程在这里暂停，而不是编一个订单号继续
    用户：ORD-1007
    系统：（从暂停的位置继续）这笔订单可以退，条件是……

朴素做法是把"等用户补充"存成一次问答再重新走一遍全流程，代价是：
重复检索、重复调模型，而且第二轮很容易丢失第一轮的意图。LangGraph 的
`interrupt()` + checkpointer 让流程真的"停在原地"，恢复时状态原样还在。

**槽位必须来自用户当轮明说的内容。** 这里有个真实的风险：如果用正则去
历史里"扫"订单号，上一轮讨论过的订单会被当成这一轮的 —— 用户说
"我要退款"（本轮无订单号）时，系统会拿上轮的订单直接退款。所以
`slots_for_refund` 这类函数一律只吃当轮文本。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..domain import Intent, ToolError
from ..nlu.entities import ORDER_RE
from ..nlu.entities import normalize_order_id as normalize_entity_order_id
from ..schemas import SlotExtraction, ToolCall

#: 订单号口径来自 NLU 的实体层，这里**不再自己写一份**。
#:
#: 原先这里自带一条 `\bORD[-\s]?(\d{4,8})\b`，`quality/gate.py` 里还有一条
#: 一模一样的副本。两条都因为 `\b` 在中文里不成立而漏抽（见 entities.ORDER_RE
#: 的注释），于是"帮我取消订单ORD-1001"会被判成缺订单号，反过来问用户要号码 ——
#: 而他刚刚才给过。两个文件的注释都写着"全项目只定义一次"，实际有四份。
ORDER_ID_RE = ORDER_RE


def normalize_order_id(raw: str) -> str:
    """把 `ord 1007` / `ORD-1007` / `ORD1007` 统一成 `ORD-1007`。

    和 `entities.normalize_order_id` 同一口径（含去前导零），
    否则同一个单号在槽位层和这里会得出两个不同的字符串。
    """
    m = ORDER_ID_RE.search(raw)
    return normalize_entity_order_id(m.group(1)) if m else raw.strip()


def extract_order_id(text: str) -> str | None:
    """从文本里抽订单号。只认当轮文本，不扫历史。

    取**第一个**匹配，和 `entities.find_order_id`（取最后一个）不同，这是刻意的：
    那边服务的是指代消解，"最近提到的那一单"才是用户说的"它"；
    这里服务的是取消/退款这种会动钱的指令，一句话里出现两个单号时
    （"取消 ORD-1001，不是 ORD-1002"）第一个才是他要动的那个 ——
    取最后一个会把一个明确的否定项当成目标。
    """
    m = ORDER_ID_RE.search(text or "")
    return normalize_entity_order_id(m.group(1)) if m else None


#: 哪些意图需要订单号才能进入子流程
NEEDS_ORDER_ID: frozenset[Intent] = frozenset(
    {
        Intent.REFUND_REQUEST,
        Intent.RETURN_REQUEST,
        Intent.CANCEL_ORDER,
        Intent.CHANGE_ADDRESS,
    }
)

#: 哪些意图除了订单号还要一个新地址
NEEDS_ADDRESS: frozenset[Intent] = frozenset({Intent.CHANGE_ADDRESS})

#: 缺槽位时给用户的追问话术。写清"要什么、为什么"，
#: 比一句"请提供更多信息"的转化率高得多。
_PROMPTS: dict[tuple[Intent, str], str] = {
    # 查订单 / 查物流也在这张表里，虽然它们走的是 Agent 而不是高风险子流程。
    # 理由是一样的：缺订单号时该**追问**，不该把 None 当参数传给工具 ——
    # 那样用户收到的是"没有查到这个订单号"，听起来像他报错了，
    # 实际上他什么都没报错，只是系统没问。
    (Intent.ORDER_QUERY, "order_id"): "请把订单号发我（形如 ORD-1007），我帮你查这笔订单的状态。",
    (Intent.LOGISTICS_QUERY, "order_id"): "请把订单号发我（形如 ORD-1007），我看下这个包裹到哪了。",
    (Intent.REFUND_REQUEST, "order_id"): "可以帮你申请退款。请把订单号发我（形如 ORD-1007），我先确认一下这笔订单的状态和时效。",
    (Intent.RETURN_REQUEST, "order_id"): "可以帮你走退货。请把订单号发我（形如 ORD-1007），我确认下是否在退货窗口内。",
    (Intent.CANCEL_ORDER, "order_id"): "可以帮你取消。请把订单号发我（形如 ORD-1007），我需要确认它还没发货。",
    (Intent.CHANGE_ADDRESS, "order_id"): "可以改地址。请把订单号发我（形如 ORD-1007），我确认下还没发货。",
    (Intent.CHANGE_ADDRESS, "address"): "请把完整的新收货地址发我（省市 + 详细地址），我帮你更新。",
}


def missing_slots(
    intent: Intent, question: str, slots: SlotExtraction | None = None
) -> list[str]:
    """判断这个意图在当前这一轮还缺什么。

    **只信当轮文本。** `slots` 是给调用方显式传入"确定性槽位"用的口子，
    默认 None 表示从当轮重抽 —— 绝不从 history 里捞。
    """
    need: list[str] = []
    if intent in NEEDS_ORDER_ID:
        oid = (slots.order_id if slots and slots.order_id else None) or extract_order_id(question)
        if not oid:
            need.append("order_id")
    if intent in NEEDS_ADDRESS:
        # 地址有两个来源：分类器抽出来的结构化槽位，和当轮原句里的正则。
        # 只看后者会漏掉"换成公司地址"这种写法 —— 那正是要用户重说一遍的
        # 无谓来回；只看前者则可能采信到没清洗干净的残留值，所以两个都过校验。
        addr = _address_from(slots.address if slots else None, question)
        if not addr:
            need.append("address")
    return need


def _address_from(slot_address: str | None, question: str) -> str | None:
    """确认这一轮到底有没有给出一个可用的新地址。

    两个来源的性质完全不同，必须分开处理：

      * `slot_address` 是分类器**结构化抽出来**的值，可以整体采信
        （但仍要挡掉"那个地址"这类指代性说法）；
      * `question` 是用户的**整句原话**，只能靠正则从中精确切出地址片段。

    早期版本对原句也做了"够长就当地址"的兜底，结果"改地址 ORD-1007"
    整句都被当成地址写进订单 —— 这种错误会把货发到不存在的地方，
    代价远高于多问用户一句。所以原句这条路只认正则，不认长度。
    """
    if slot_address:
        stripped = slot_address.strip()
        if len(stripped) >= 4 and not re.fullmatch(r"(那个|这个|原来的|之前的)?地址", stripped):
            return stripped
    return _extract_address(question)


def interrupt_prompt(intent: Intent, missing: list[str]) -> str:
    """把缺失的槽位翻成一句人话。"""
    for key in missing:
        text = _PROMPTS.get((intent, key))
        if text:
            return text
    return "还需要你补充一点信息才能继续。"


@dataclass
class FlowOutcome:
    """子流程跑完的结果。

    `interrupted=True` 表示流程停在半路等用户补充，这时候 `tool_calls`
    一定是空的 —— 没凑齐前置条件就动手，正是这条流程要防的事。
    """

    interrupted: bool
    missing: list[str]
    tool_calls: list[ToolCall]
    interrupt_prompt: str = ""
    answer: str = ""
    #: 触发拒答/转人工的原因（业务拒绝时填）
    error: str | None = None


def run_high_risk_flow(
    intent: Intent,
    question: str,
    registry: Any,
    *,
    slots: SlotExtraction | None = None,
    principal_id: str | None = None,
) -> FlowOutcome:
    """执行高风险子流程。

    流程本身是线性的，没有模型参与的余地：
        1. 检查前置槽位 -> 缺就中断（不猜、不编、不拿历史顶数）
        2. 调用对应的业务工具 -> 规则在工具里判定
        3. 工具拒绝（业务不允许）-> 如实转达，不做"通融"

    第 3 步是这套设计的价值所在：模型可以在旁边写话术，但"能不能退"
    这个判断永远来自工具，不来自模型的善意。
    """
    missing = missing_slots(intent, question, slots)
    if missing:
        return FlowOutcome(
            interrupted=True,
            missing=missing,
            tool_calls=[],
            interrupt_prompt=interrupt_prompt(intent, missing),
        )

    order_id = (slots.order_id if slots and slots.order_id else None) or extract_order_id(question)

    plan: list[tuple[str, dict[str, Any]]] = []
    if intent is Intent.REFUND_REQUEST:
        plan.append(("request_refund", {"order_id": order_id}))
    elif intent is Intent.RETURN_REQUEST:
        # 退货先查订单状态，再判断能不能进退货流程 —— 两步是有顺序的，
        # 因为"这单还没发货"和"已经签收 40 天"要给的答复完全不同。
        plan.append(("get_order", {"order_id": order_id}))
        plan.append(("request_refund", {"order_id": order_id}))
    elif intent is Intent.CANCEL_ORDER:
        plan.append(("cancel_order", {"order_id": order_id}))
    elif intent is Intent.CHANGE_ADDRESS:
        address = _address_from(slots.address if slots else None, question)
        if not address:
            return FlowOutcome(
                interrupted=True,
                missing=["address"],
                tool_calls=[],
                interrupt_prompt=interrupt_prompt(intent, ["address"]),
            )
        plan.append(("change_address", {"order_id": order_id, "address": address}))

    calls: list[ToolCall] = []
    error: str | None = None
    for name, args in plan:
        call = (
            registry.call(name, args, principal_id=principal_id)
            if principal_id is not None
            else registry.call(name, args)
        )
        calls.append(call)
        if not call.ok:
            # 业务拒绝是预期内的结果，不是故障 —— 记下来如实转达即可。
            # 第一个失败就停：后续步骤的前置条件已经不成立了。
            error = call.error or ToolError.UPSTREAM_ERROR.value
            break

    return FlowOutcome(
        interrupted=False,
        missing=[],
        tool_calls=calls,
        answer=_render_outcome(intent, calls, error),
        error=error,
    )


def _extract_address(text: str) -> str | None:
    """从当轮文本里抽新地址。

    地址没有强格式，这里只做保守抽取：明确说了"改成/换成/寄到"之后的
    片段才算，否则宁可不认（认错了会把货发到错误的地方，代价远高于多问一句）。
    """
    m = re.search(
        r"(?:改成|换成|寄到|送到|改到|收货地址是|地址是)[：:，,\s]*([^\n。；;？?！!]{4,60})",
        text or "",
    )
    if not m:
        return None
    addr = m.group(1).strip()
    # 指代性说法不是地址值 —— 拿到"那个地址"就当没拿到
    if re.fullmatch(r"(那个|这个|原来的|之前的|原来的那个)?地址", addr):
        return None
    return addr


def _render_outcome(intent: Intent, calls: list[ToolCall], error: str | None) -> str:
    """把工具结果翻成给用户的答复。

    刻意不交给模型写这段：这是"做了什么"的事实陈述，模型写只会引入
    与工具结果不一致的风险。回复的措辞可以润色，事实不能。
    """
    if not calls:
        return "没有可以执行的操作。"

    last = calls[-1]
    data = last.result if isinstance(last.result, dict) else {}
    order_id = data.get("order_id") or last.args.get("order_id") or ""
    prefix = f"订单 {order_id}：" if order_id else ""
    if error:
        # 账号未关联时没有查到订单，沿用账号提示，不能伪装成该单的业务结论。
        if error == "ORDER_ACCOUNT_UNLINKED":
            return data.get("message") or _REJECTION_TEXT["ORDER_ACCOUNT_UNLINKED"]
        if intent is Intent.CHANGE_ADDRESS:
            return _REJECTION_TEXT.get(error, f"这笔操作没有完成（{error}）。")
        return prefix + (data.get("message") or _REJECTION_TEXT.get(error, f"这笔操作没有完成（{error}）。"))

    if intent is Intent.CANCEL_ORDER:
        return prefix + str(data.get("note") or "取消申请需人工审核后生效。") + str(data.get("refund_note") or "")
    if intent is Intent.CHANGE_ADDRESS:
        return f"已把订单 {data.get('order_id', '')} 的收货地址更新为：{data.get('shipping_address') or data.get('address', '')}。"
    if intent in {Intent.REFUND_REQUEST, Intent.RETURN_REQUEST}:
        # 工具只核验申请条件；保留时效、生鲜限制和审核要求，不宣称已经提交。
        return prefix + str(data.get("note") or "退货退款申请需人工审核后生效。")
    return "操作已完成。"


_REJECTION_TEXT: dict[str, str] = {
    "ORDER_ACCOUNT_UNLINKED": "当前账号未关联这笔订单，暂时无法查询或办理该订单业务。",
    ToolError.ORDER_NOT_FOUND.value: "没有查到这个订单号，麻烦你核对一下再发我（形如 ORD-1007）。",
    ToolError.ORDER_NOT_CANCELLABLE.value: "这笔订单已经发货了，没法直接取消。你可以等收到后走退货流程，我可以帮你登记。",
    ToolError.ORDER_ADDRESS_LOCKED.value: "订单已经发出，收货地址改不了了。可以联系承运方改派，或者收到后走退货。",
    ToolError.REFUND_NOT_ELIGIBLE.value: "这笔订单不在退款条件内（可能已超过时效，或属于不支持无理由退款的品类）。我可以帮你转人工复核。",
    ToolError.TIMEOUT.value: "系统查询超时了，请稍后再试，或我帮你转人工。",
    ToolError.UPSTREAM_ERROR.value: "系统暂时不可用，我帮你转人工处理。",
}


#: 业务拒绝里，哪些应该进一步转人工而不是就此结束
ESCALATE_TO_HUMAN: frozenset[str] = frozenset(
    {
        ToolError.REFUND_NOT_ELIGIBLE.value,
        ToolError.TIMEOUT.value,
        ToolError.UPSTREAM_ERROR.value,
    }
)
