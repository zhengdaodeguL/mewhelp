"""领域常量：意图、出口、错误码、拒答原因。

这个模块是纯数据、零依赖 —— 检索层、NLU 层、LLM mock、评估脚本都从
这里取同一份定义，避免"同一个意图在两个文件里拼写不一样"这类事故。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Intent(StrEnum):
    """九类细分意图。"""

    ORDER_QUERY = "order_query"          # 查订单
    LOGISTICS_QUERY = "logistics_query"  # 查物流
    REFUND_REQUEST = "refund_request"    # 申请退款
    RETURN_REQUEST = "return_request"    # 申请退货
    CANCEL_ORDER = "cancel_order"        # 取消订单
    CHANGE_ADDRESS = "change_address"    # 修改收货地址
    PRODUCT_FAQ = "product_faq"          # 商品咨询
    POLICY_FAQ = "policy_faq"            # 规则/政策咨询
    HUMAN_HANDOFF = "human_handoff"      # 转人工


class Exit(StrEnum):
    """五个处理出口 —— 九类意图收敛到这里。

    收敛的意义有两个：一是每类问题的人力/模型成本可以按出口统计，
    二是相同的业务流程只开发一次（四条高风险意图共用一条子流程）。
    """

    KNOWLEDGE = "knowledge"          # 知识检索出口（RAG）
    BUSINESS_TOOL = "business_tool"  # 业务工具出口（Function Calling）
    HIGH_RISK = "high_risk"          # 高风险确定性子流程（可中断/可恢复）
    HANDOFF = "handoff"              # 直接转人工
    FALLBACK = "fallback"            # 兜底出口（低置信/未知意图）—— 第 5 个出口


#: 九类意图 → 五个出口。这是全流程最重要的一张表。
INTENT_TO_EXIT: dict[Intent, Exit] = {
    Intent.ORDER_QUERY: Exit.BUSINESS_TOOL,
    Intent.LOGISTICS_QUERY: Exit.BUSINESS_TOOL,
    Intent.PRODUCT_FAQ: Exit.KNOWLEDGE,
    Intent.POLICY_FAQ: Exit.KNOWLEDGE,
    Intent.REFUND_REQUEST: Exit.HIGH_RISK,
    Intent.RETURN_REQUEST: Exit.HIGH_RISK,
    Intent.CANCEL_ORDER: Exit.HIGH_RISK,
    Intent.CHANGE_ADDRESS: Exit.HIGH_RISK,
    Intent.HUMAN_HANDOFF: Exit.HANDOFF,
}

#: 走高风险子流程的意图 —— 这几类业务铁律写死在图上，不交给 Agent 自由发挥。
HIGH_RISK_INTENTS: frozenset[Intent] = frozenset(
    i for i, e in INTENT_TO_EXIT.items() if e is Exit.HIGH_RISK
)

#: 需要业务工具（订单/物流）的意图。
TOOL_INTENTS: frozenset[Intent] = frozenset(
    i for i, e in INTENT_TO_EXIT.items() if e is Exit.BUSINESS_TOOL
)


@dataclass(frozen=True)
class IntentSpec:
    """一个意图的完整定义：中文名、典型说法、关键词。

    `keywords` 同时被三处复用：
      - mock LLM 的意图分类器
      - 规则高速通道（高置信时跳过模型调用）
      - 评估集的自动校验
    """

    intent: Intent
    label: str
    keywords: tuple[str, ...]
    examples: tuple[str, ...]


INTENT_SPECS: tuple[IntentSpec, ...] = (
    IntentSpec(
        Intent.ORDER_QUERY,
        "查订单",
        ("订单", "下单", "买了", "订单号", "order", "查单", "我的单"),
        ("帮我查一下订单", "我上周买的东西到哪了", "订单 ORD-1001 是什么状态"),
    ),
    IntentSpec(
        Intent.LOGISTICS_QUERY,
        "查物流",
        ("物流", "快递", "发货", "到哪", "运单", "签收", "配送", "几天到"),
        ("快递到哪了", "怎么还没发货", "什么时候能送到"),
    ),
    IntentSpec(
        Intent.REFUND_REQUEST,
        "申请退款",
        ("退款", "退钱", "返钱", "退回", "退我", "refund"),
        ("我要退款", "东西坏了能退钱吗", "申请退款"),
    ),
    IntentSpec(
        Intent.RETURN_REQUEST,
        "申请退货",
        ("退货", "退回货", "寄回", "不要了", "退掉", "七天无理由"),
        ("我要退货", "这个能退吗", "七天无理由怎么退"),
    ),
    IntentSpec(
        Intent.CANCEL_ORDER,
        "取消订单",
        ("取消", "不想要了", "别发了", "撤销订单", "cancel"),
        ("帮我取消订单", "我不想买了", "订单能撤吗"),
    ),
    IntentSpec(
        Intent.CHANGE_ADDRESS,
        "修改收货地址",
        ("改地址", "修改地址", "换地址", "收货地址", "地址写错", "改收货"),
        ("地址写错了帮我改一下", "能改收货地址吗", "换成公司地址"),
    ),
    IntentSpec(
        Intent.PRODUCT_FAQ,
        "商品咨询",
        ("尺码", "材质", "颜色", "保修", "参数", "适合", "能不能用", "多大"),
        ("这个尺码怎么选", "材质是什么", "保修多久"),
    ),
    IntentSpec(
        Intent.POLICY_FAQ,
        "规则政策咨询",
        ("政策", "规则", "几天", "包邮", "运费", "发票", "会员", "优惠券"),
        ("运费怎么算", "能开发票吗", "包邮吗"),
    ),
    IntentSpec(
        Intent.HUMAN_HANDOFF,
        "转人工",
        ("人工", "客服", "找人", "投诉", "转接", "真人"),
        ("我要找人工", "转人工", "我要投诉"),
    ),
)

#: 意图 → 关键词，供规则通道和 mock 分类器直接查表。
INTENT_KEYWORDS: dict[Intent, tuple[str, ...]] = {
    s.intent: s.keywords for s in INTENT_SPECS
}

INTENT_LABELS: dict[Intent, str] = {s.intent: s.label for s in INTENT_SPECS}


class RefusalReason(StrEnum):
    """置信度闸门的拒答原因 —— 也是数据飞轮的入口分类依据。"""

    NO_EVIDENCE = "no_evidence"              # 一条证据都没召回
    WEAK_EVIDENCE = "weak_evidence"          # 召回了但分数撑不起回答
    LOW_COVERAGE = "low_coverage"            # 证据覆盖不了问题的关键实体
    SELF_EVAL_FAILED = "self_eval_failed"    # 生成前自评没通过
    MISSING_SLOT = "missing_slot"            # 关键槽位缺失（缺订单号等）
    HIGH_RISK_UNAPPROVED = "high_risk_unapproved"  # 高风险动作未通过人工


class ToolError(StrEnum):
    """业务工具的标准错误码。"""

    ORDER_NOT_FOUND = "ORDER_NOT_FOUND"
    ORDER_NOT_CANCELLABLE = "ORDER_NOT_CANCELLABLE"
    ORDER_ADDRESS_LOCKED = "ORDER_ADDRESS_LOCKED"
    REFUND_NOT_ELIGIBLE = "REFUND_NOT_ELIGIBLE"
    TIMEOUT = "TIMEOUT"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"


class FlywheelSource(StrEnum):
    """数据飞轮的三个入口。"""

    REFUSED = "refused"        # 检索证据弱被拒答
    SELF_EVAL = "self_eval"    # 生成前自评没过
    THUMBS_DOWN = "thumbs_down"  # 用户点踩


class NodeName(StrEnum):
    """LangGraph 节点名 —— 同时用作审计日志和 Langfuse span 名。"""

    RESOLVE = "resolve"          # 指代消解
    CLASSIFY = "classify"        # 意图识别
    ROUTE = "route"              # 分流（图上的条件边）
    RETRIEVE = "retrieve"        # 混合检索
    RERANK = "rerank"            # 精排
    GATE = "gate"                # 置信度闸门
    TOOL = "tool"                # 工具调用
    HIGH_RISK_FLOW = "high_risk_flow"  # 高风险子流程
    AGENT = "agent"              # 主力 Agent（ReAct）
    GENERATE = "generate"        # 生成回答
    HANDOFF = "handoff"          # 转人工
    FLYWHEEL = "flywheel"        # 落问题池
