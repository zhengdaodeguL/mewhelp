"""跨模块数据契约（Pydantic 模型）。

这是整个项目的"接缝"。每个模块只依赖这里的模型，不依赖彼此的内部结构，
所以检索层、NLU 层、编排层、API 层可以各自替换实现而不影响别人。

命名约定：
  *Chunk   —— 知识库里的一个片段（切分后的原子单位）
  Evidence —— 检索回来的一条证据（带分数和来源）
  *Result  —— 某个节点/模块的产出
  Trace*   —— 可观测层记录的一步
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from .domain import Exit, FlywheelSource, Intent, RefusalReason


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================
# 知识库 / 检索
# ============================================================


class KbChunk(BaseModel):
    """知识库中一个可检索的片段。

    按标题层级 + 表格结构切分，而不是定长滑窗 —— 目的是避免一条完整规则
    被从中间劈开，导致模型只看到半个条件就给出错误答案。
    """

    chunk_id: str
    doc_id: str
    title: str                      # 所属最小标题（面包屑，如 "退款政策 > 生鲜类"）
    breadcrumb: list[str] = Field(default_factory=list)
    text: str
    keywords: list[str] = Field(default_factory=list)
    #: 表格类片段额外保留结构，生成时按行渲染，避免丢列
    table: list[list[str]] | None = None
    is_table: bool = False
    char_len: int = 0
    source_path: str | None = None

    def prompt_text(self) -> str:
        """渲染成喂给模型的形式（表格补回表头）。"""
        if self.is_table and self.table:
            head, *rows = self.table
            lines = [" | ".join(head), " | ".join("-" * len(head))]
            lines += [" | ".join(r) for r in rows]
            return f"[{self.title}]\n" + "\n".join(lines)
        return f"[{self.title}]\n{self.text}"


class Evidence(BaseModel):
    """一条检索证据。分数口径在每一路都不同，融合后才统一。"""

    chunk: KbChunk
    #: 该证据在最终排序里的分数（RRF 融合分或精排分，取决于所处阶段）
    score: float = 0.0
    #: 来源标记，用于消融实验（"dense" / "sparse" / "hybrid" / "rerank"）
    retriever: str = "unknown"
    #: 各路原始排名，实验时用来看"谁把这条捞上来的"
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    def short(self, n: int = 60) -> str:
        return self.chunk.text[:n].replace("\n", " ")


class RetrievalResult(BaseModel):
    """一次检索的完整结果 —— 含中间态，供评估和 Langfuse 还原调用树。"""

    query: str
    #: 最终交给下游的证据（已精排、已截断到 rerank_top_k）
    evidences: list[Evidence] = Field(default_factory=list)
    #: 各阶段条数，用于回答"为什么没检索到"
    dense_hits: int = 0
    sparse_hits: int = 0
    fused_hits: int = 0
    reranked_hits: int = 0
    #: 召回的分桶 top-k，评估时算 Recall@k / MRR 用
    retrieved_ids: list[str] = Field(default_factory=list)
    #: 消融实验标记：dense | sparse | hybrid | hybrid_rerank
    strategy: str = "hybrid_rerank"
    latency_ms: float = 0.0

    @property
    def top_score(self) -> float:
        return max((e.score for e in self.evidences), default=0.0)

    def to_context(self, max_chars: int = 4000) -> str:
        """拼成喂给模型的证据块，带编号方便模型引用和自评。"""
        parts: list[str] = []
        used = 0
        for i, ev in enumerate(self.evidences, 1):
            block = f"[{i}] {ev.chunk.prompt_text()}"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)


# ============================================================
# NLU：指代消解 + 意图识别
# ============================================================


class ResolveResult(BaseModel):
    """指代消解产出。

    "那它能退吗" 这种半截话，要结合最近几轮补成一句独立能看懂的完整问题。
    """

    original: str
    resolved: str
    #: 是否真的做了改写（没改写时下游可以省一次模型调用）
    rewritten: bool = False
    #: 被替换掉的指代词 → 补全后的实体，审计时能一眼看出改了什么
    substitutions: dict[str, str] = Field(default_factory=dict)
    #: 补全时引用了第几轮历史（从 1 开始，负数表示"上一轮"）
    referenced_turns: list[int] = Field(default_factory=list)


class ClassifyResult(BaseModel):
    intent: Intent
    confidence: float
    label: str = ""
    #: 打分依据（命中的关键词 / few-shot 例子），用于解释和调试
    matched: list[str] = Field(default_factory=list)
    #: 命中哪条通道：rule | mock | llm | demoted | fallback
    #: demoted = 模型报了高风险动作意图、但这句话是在问规则，被按业务铁律改判成政策咨询
    method: str = "mock"


class SlotExtraction(BaseModel):
    """从对话里抠出来的结构化槽位。"""

    order_id: str | None = None
    address: str | None = None
    reason: str | None = None
    product: str | None = None

    def missing_for(self, intent: Intent) -> list[str]:
        """这个意图还缺哪些必填槽位 —— 子流程靠它决定要不要中断。"""
        need: list[str] = []
        if intent in _NEEDS_ORDER_ID and not self.order_id:
            need.append("order_id")
        if intent is Intent.CHANGE_ADDRESS and not self.address:
            need.append("address")
        return need


#: 哪些意图必须有订单号才能继续
_NEEDS_ORDER_ID: frozenset[Intent] = frozenset(
    {
        Intent.ORDER_QUERY,
        Intent.LOGISTICS_QUERY,
        Intent.REFUND_REQUEST,
        Intent.RETURN_REQUEST,
        Intent.CANCEL_ORDER,
        Intent.CHANGE_ADDRESS,
    }
)


# ============================================================
# 闸门 / 兜底
# ============================================================


class GateDecision(BaseModel):
    """置信度闸门的结论。

    注意这道检查卡在**模型开口之前**，而不是等它答完再评估 —— 顺序不同，
    代价差一个数量级：答完再判，用户已经看到幻觉了。
    """

    passed: bool
    confidence: float
    reason: RefusalReason | None = None
    #: 每个判据的得分，"为什么判它不过"要能逐条讲清楚
    checks: dict[str, float] = Field(default_factory=dict)
    #: 面向前端/人工的解释话术
    message: str = ""

    @classmethod
    def ok(cls, confidence: float, checks: dict[str, float]) -> GateDecision:
        return cls(passed=True, confidence=confidence, checks=checks)

    @classmethod
    def reject(
        cls, confidence: float, reason: RefusalReason, checks: dict[str, float], message: str = ""
    ) -> GateDecision:
        return cls(
            passed=False,
            confidence=confidence,
            reason=reason,
            checks=checks,
            message=message or _DEFAULT_REFUSAL_TEXT.get(reason, "抱歉，这个问题我需要转人工帮你确认。"),
        )


_DEFAULT_REFUSAL_TEXT: dict[RefusalReason, str] = {
    RefusalReason.NO_EVIDENCE: "抱歉，我在知识库里没有找到相关依据，帮你转人工核实一下。",
    RefusalReason.WEAK_EVIDENCE: "抱歉，我找到的资料不足以给出确定答复，帮你转人工确认。",
    RefusalReason.LOW_COVERAGE: "抱歉，我能确认的信息不完整，帮你转人工补充确认。",
    RefusalReason.SELF_EVAL_FAILED: "抱歉，我的回答没有通过事实校验，帮你转人工处理。",
    RefusalReason.MISSING_SLOT: "需要你提供订单号才能继续帮你处理。",
    RefusalReason.HIGH_RISK_UNAPPROVED: "这笔操作需要人工审核后才会生效。",
}


class SelfEvalResult(BaseModel):
    """生成前自评：回答是否每一句都被证据支撑。"""

    grounded: bool
    coverage: float = 0.0
    unsupported_claims: list[str] = Field(default_factory=list)
    note: str = ""


# ============================================================
# 工具调用
# ============================================================


class ToolCall(BaseModel):
    """一次工具调用的记录（成功和失败都记，审计要能还原）。"""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    result: Any = None
    error: str | None = None
    elapsed_ms: float = 0.0
    attempts: int = 1


class ToolSpec(BaseModel):
    """工具的对外描述 —— 用来生成 function-calling schema。"""

    name: str
    description: str
    parameters: dict[str, Any]
    #: 高风险工具需要人工审核，Agent 不能直接调用
    high_risk: bool = False


# ============================================================
# 会话上下文
# ============================================================


class Turn(BaseModel):
    role: Literal["user", "assistant", "tool", "system"]
    content: str
    ts: datetime = Field(default_factory=_now)
    intent: Intent | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextBundle(BaseModel):
    """双层上下文：近处滑窗保原文，更早的压成摘要。

    单独抽成一个模型，是为了让"省了多少 token、丢了什么信息"可以被度量，
    而不是藏在一段拼字符串的代码里。
    """

    summary: str = ""
    #: 滑窗内的原文轮次
    window: list[Turn] = Field(default_factory=list)
    #: 被压进摘要的轮次数
    summarized_turns: int = 0
    token_estimate: int = 0
    #: 摘要生成时用掉的模型调用
    summary_generated: bool = False

    def as_messages(self) -> list[dict[str, str]]:
        """渲染成 chat messages。摘要是 system 的一部分，滑窗是原始对话。"""
        msgs: list[dict[str, str]] = []
        if self.summary:
            msgs.append(
                {
                    "role": "system",
                    "content": f"以下是本次会话更早内容的摘要，供你理解上下文：\n{self.summary}",
                }
            )
        msgs += [{"role": t.role, "content": t.content} for t in self.window]
        return msgs


# ============================================================
# 长期记忆
# ============================================================


class MemoryItem(BaseModel):
    """一条长期记忆 —— 跨会话保留的"关于这个用户的稳定事实"。

    存在的理由很具体：用户抱怨的"聊到第十轮它就不认识我了"，根因是上下文
    会被滑窗和摘要丢掉。而"他穿 L 码""他关注的是 ORD-1007 那一单"这类事实
    **本来就不该随时间衰减** —— 它们不是对话内容，是用户档案。

    字段刻意都取"用户视角"的名字：`kind` / `text` / `source` 是给人看的，
    没有 `score`、`mem_key`、`embedding` 这类内部口径 —— 记忆面板会把它们
    原样展示出来，往里塞内部字段迟早会漏到界面上。
    """

    id: int
    kind: str
    text: str
    #: auto(系统从对话里抽的) / manual(用户自己写的)。界面据此提示"这条是我
    #: 自己加的，系统不会改它"。
    source: str = "auto"
    pinned: bool = False
    #: 这条记忆真正帮上忙的次数（补全了用户没说清的话），不是"被塞进 prompt 的次数"
    hits: int = 0
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


#: 记忆类型的中文名。放在契约层而不是界面层：它既是界面文案，也是
#: "系统能记住哪几类事"的完整清单，两处各写一份必然会分叉。
MEMORY_KIND_LABELS: dict[str, str] = {
    "order": "关注的订单",
    "preference": "偏好",
    "manual": "我的备忘",
}


class SessionSummary(BaseModel):
    """历史列表里的一行。

    只有"用户认得出这是哪次对话"所需的最小信息。刻意**不带**摘要、路径、
    意图这些内部字段 —— 历史列表和对话本身一样，是用户视角的东西。
    """

    session_id: str
    title: str
    #: 消息条数（一问一答算两条）
    turns: int
    preview: str = ""
    updated_at: datetime = Field(default_factory=_now)


class SessionMessage(BaseModel):
    """历史里的一条消息 —— 已经按用户视角过滤过。

    刻意**没有** `intent` / `path` / `exit` 这些字段：它们在库里确实存着
    （`Turn.metadata`，给审计用），但恢复历史对话时不该跟着回到界面上。
    过滤做在**类型这一层**，而不是在返回之前手动删几个 key —— 手动删的话，
    下次有人加字段就会顺手漏出去，而且不会有任何测试失败。
    """

    role: Literal["user", "assistant"]
    content: str
    #: 当时是不是转人工了（前端据此把"已为你转接人工"提示条也画回来）
    handoff: bool = False
    #: 不透明句柄：反馈与收藏的键，界面不展示
    request_id: str = ""


class SessionDetail(BaseModel):
    """恢复一次历史对话所需的全部内容。"""

    session_id: str
    title: str
    updated_at: datetime = Field(default_factory=_now)
    messages: list[SessionMessage] = Field(default_factory=list)


# ============================================================
# 可观测
# ============================================================


class TraceStep(BaseModel):
    """调用树上的一步。一条请求还原成一棵树，靠的就是这些 step。"""

    node: str
    started_at: datetime = Field(default_factory=_now)
    elapsed_ms: float = 0.0
    #: 这一步的输入摘要（不存全量，避免日志爆炸）
    input_digest: str = ""
    output_digest: str = ""
    #: 检索回了什么 / 调了哪个工具 / 烧了多少 token
    details: dict[str, Any] = Field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    error: str | None = None


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    #: 折算成本，mock 模式下恒为 0 —— 离线跑不能有"假成本"
    cost_usd: float = 0.0

    def add(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


# ============================================================
# 端到端
# ============================================================


class ConversationResult(BaseModel):
    """一次完整问答的产出 —— API 层的返回体就是它。"""

    request_id: str
    session_id: str
    user_id: str
    question: str
    resolved_question: str = ""

    intent: Intent | None = None
    intent_confidence: float = 0.0
    exit: Exit | None = None

    answer: str = ""
    #: 走的是哪条路径，前端/面试演示时一眼看出分流对不对
    path: list[str] = Field(default_factory=list)
    refusal_reason: RefusalReason | None = None
    handoff: bool = False

    evidences: list[Evidence] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    #: 本轮之后该用户的全部长期记忆。放"全量"而不是"本轮新增"，
    #: 是因为前端拿它直接替换面板状态即可 —— 增量合并一旦漏掉一次事件，
    #: 面板就会和真实记忆永久对不上，而且没人会发现。
    memories: list[MemoryItem] = Field(default_factory=list)

    #: 子流程中断时，这里带着恢复所需的载荷
    interrupted: bool = False
    interrupt_prompt: str | None = None

    trace: list[TraceStep] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=_now)


class FlywheelRecord(BaseModel):
    """落进问题池的一条记录。

    关键设计：落池时**连同当时的检索快照一起存**。没有快照，人工审核时
    就没法判断"是真缺知识"还是"知识有但没检索到" —— 这两种情况的处理
    方式完全不同（一个补文档，一个调检索）。
    """

    record_id: str
    session_id: str
    question: str
    resolved_question: str = ""
    intent: Intent | None = None
    source: FlywheelSource
    refusal_reason: RefusalReason | None = None
    #: 当时的检索快照 —— 飞轮最值钱的字段
    retrieval_snapshot: RetrievalResult | None = None
    #: 同题被问了多少次，人工按频次优先审核
    ask_count: int = 1
    status: Literal["pending", "approved", "rejected", "merged"] = "pending"
    created_at: datetime = Field(default_factory=_now)
    reviewed_at: datetime | None = None
    reviewer: str | None = None
    #: 审核通过后补的答案，回灌知识库用
    answer: str | None = None
