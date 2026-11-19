"""LLM 抽象层。

整个项目只通过 `LLMClient` 说话。这一层存在的理由：

  * 默认的 `MockLLMClient` 是**确定性**的 —— 同一个输入永远给同一个输出。
    没有它，测试不可复现、CI 要烧钱、离线演示没法做。
  * `OpenAILLMClient` 走任何 OpenAI 兼容端点，两者接口完全一致。
  * 换模型不改一行业务代码。这是"什么时候才轮到微调"这个判断的前提：
    先能廉价地换 Prompt、换模型、换检索，才谈得上微调。

Mock 不是"返回固定字符串"，而是按任务类型做**规则化推理**：意图分类真的读
关键词表，指代消解真的做指代替换，ReAct 真的按规则决定调哪个工具。所以
换成真模型时，图的形状、状态、节点行为都不变，变的只是"脑子的实现"。
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .config import Settings, get_settings
from .schemas import TokenUsage


class LLMError(RuntimeError):
    """模型调用失败 —— 上层据此决定重试还是兜底。"""


@dataclass
class LLMResponse:
    text: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    #: 工具调用（OpenAI function-calling 协议）
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    raw: Any = None
    #: 推理模型在正文之前写的那段思维链。只用于排查，**不进任何 prompt、更不进
    #: 答复** —— 它常常包含模型对用户的臆测（"用户可能在试探"），漏出去就是事故。
    reasoning: str = ""

    def json(self) -> Any:
        """把 text 当 JSON 解析，容忍 ```json 围栏和前后废话。"""
        return extract_json(self.text)


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def extract_json(text: str) -> Any:
    """从模型输出里挖出 JSON。

    真模型经常在 JSON 外面包一层解释或 ```json 围栏，解析前先剥掉 ——
    这是工程里最容易被忽略、又最容易让整条链路崩掉的一步。
    """
    text = (text or "").strip()
    if not text:
        raise LLMError("empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 退而求其次：抓第一个平衡的 {...} 或 [...]
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise LLMError(f"cannot parse JSON from response: {text[:200]}")


def as_bool(value: Any) -> bool:
    """把模型给的"布尔"读成布尔。

    真模型经常把 `true` 写成字符串 `"true"` / `"是"` / `"通过"`，反向也一样。
    直接拿原值当条件判断的后果不是报错，而是**静默取反**：
    字符串 `"false"` 是真值，`if payload.get("fallback")` 会判成"模型说它拿不准"，
    于是一次高置信的判断被丢掉、换成规则层的弱猜测 —— 而日志里一切正常。

    所以凡是读模型给的布尔字段，都必须过这一道，不许直接 `if data.get(...)`。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "是", "对", "通过", "有"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


class LLMClient(ABC):
    """所有 LLM 实现的共同接口。"""

    model: str

    @abstractmethod
    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        task: str = "generic",
    ) -> LLMResponse:
        """一次同步对话。

        `task` 是给 Mock 实现用的语义标签（"classify" / "resolve" / ...），
        真实实现可以忽略它。这让 mock 能在不改变接口的前提下做"分任务推理"。
        """

    def chat_json(self, messages, **kw) -> Any:
        resp = self.chat(messages, json_mode=True, **kw)
        return resp.json()


# ============================================================
# Mock 实现
# ============================================================


class MockLLMClient(LLMClient):
    """确定性规则实现 —— 分任务模拟模型行为。

    它不是"假的"，它是"用规则复刻模型在每一步该做的判断"。这样：
      - 测试可复现、零成本、离线可跑；
      - 换成真模型时，图的结构和状态流转完全一致；
      - mock 与真模型可以跑同一套评估集做对照。
    """

    model = "mock-deterministic"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        #: 记录收到的 prompt，测试里断言"prompt 里确实带了证据"
        self.calls: list[dict[str, Any]] = []

    # ---------------------------------------------------------
    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        task: str = "generic",
    ) -> LLMResponse:
        t0 = time.perf_counter()
        self.calls.append({"task": task, "messages": list(messages), "tools": tools})

        handler = _MOCK_HANDLERS.get(task, _mock_generic)
        text = handler(self, messages, tools)

        tokens_in = sum(_rough_tokens(_msg_text(m)) for m in messages)
        tokens_out = _rough_tokens(text)
        return LLMResponse(
            text=text,
            usage=TokenUsage(
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                total_tokens=tokens_in + tokens_out,
                cost_usd=0.0,  # 离线跑不该有"假成本"
            ),
            model=self.model,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


def _msg_text(m: Any) -> str:
    if isinstance(m, dict):
        c = m.get("content", "")
        return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
    return str(m)


def _rough_tokens(text: str) -> int:
    """粗略 token 估算：中文约 1 字 1 token，英文约 4 字符 1 token。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    other = len(text) - cjk
    return cjk + max(1, other // 4)


# ---------------- 分任务 mock 处理器 ----------------


def _last_user_text(messages: Sequence[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            return _msg_text(m)
    return _msg_text(messages[-1]) if messages else ""


def _mock_resolve(client: MockLLMClient, messages, tools) -> str:
    """指代消解：把"那它能退吗"补成完整问题。

    规则：如果当前问句里有指代词，且历史里出现过具体实体，就把实体填进去。
    """
    from .domain import INTENT_KEYWORDS  # 局部导入，避免循环依赖

    user = _last_user_text(messages)
    history_text = " ".join(_msg_text(m) for m in messages if m.get("role") == "user")[:-len(user)]
    original = user
    question = user

    # 历史的实体候选：订单号、商品名
    order_ids = re.findall(r"ORD[- ]?\d{4,8}", history_text, re.I)
    products = re.findall(r"(?:那|这)([款个件台双])([^\s，,。？?！!]{2,8})", history_text)

    subs: dict[str, str] = {}
    # 指代词 → 需要补全的位置
    if order_ids:
        for pron in ("它", "这个", "那个", "这单", "那单"):
            if pron in question and not re.search(r"ORD[- ]?\d{4,8}", question):
                # 只在问题明显是在问订单相关的事时才补
                if any(kw in question for kw in ("退", "取消", "到哪", "发货", "地址", "单")):
                    question = question.replace(pron, f"订单 {order_ids[-1]}")
                    subs[pron] = f"订单 {order_ids[-1]}"
                    break
    if not subs and products:
        for pron in ("它", "这个", "那个"):
            if pron in question:
                question = question.replace(pron, f"{products[-1][0]}{products[-1][1]}")
                subs[pron] = products[-1][0] + products[-1][1]
                break

    return json.dumps(
        {
            "resolved": question,
            "rewritten": question != original,
            "substitutions": subs,
        },
        ensure_ascii=False,
    )


def _mock_classify(client: MockLLMClient, messages, tools) -> str:
    """意图识别：按关键词打分，得分最高的意图胜出。

    与规则层保持同一个业务判断：**问规则和提请求是两回事**。
    「那它能退吗」是在问退货条件，「我要退款」是在发起退款 —— 后者会真的
    动钱，前者不该被当成退款申请去追问订单号。判断逻辑和 rules 层共享
    同一套标记（见 nlu/intent.py 的 _is_question），这里只做等价复刻，
    免得 mock 通道和规则通道给出互相矛盾的答案。
    """
    from .domain import INTENT_LABELS, INTENT_SPECS  # noqa: F401

    text = _last_user_text(messages)
    is_question = _mock_looks_like_question(text)

    best_intent = "human_handoff"
    best_score = 0.0
    best_matched: list[str] = []

    for spec in INTENT_SPECS:
        matched = [kw for kw in spec.keywords if kw in text]
        if not matched:
            continue
        # 长关键词权重更高（"退款" 比 "退" 更有信息量）
        score = sum(len(kw) for kw in matched) + len(matched) * 0.5
        # 咨询句里的"动作类"意图降权：让"政策咨询"有机会胜出
        if is_question and spec.intent.value in _MOCK_ACTION_INTENTS:
            score *= 0.35
        if score > best_score:
            best_score = score
            best_intent = spec.intent.value
            best_matched = matched

    # 咨询句没有明确动作 -> 归到规则政策咨询，由知识库回答
    if is_question and best_intent in _MOCK_ACTION_INTENTS:
        return json.dumps(
            {"intent": "policy_faq", "confidence": 0.72, "matched": best_matched, "fallback": False},
            ensure_ascii=False,
        )

    # 冷启动：一个关键词都没命中 → 走政策咨询（最容易由知识库兜住）
    if best_score == 0.0:
        return json.dumps(
            {"intent": "policy_faq", "confidence": 0.30, "matched": [], "fallback": True},
            ensure_ascii=False,
        )

    total = sum(
        len(kw) for spec in INTENT_SPECS for kw in spec.keywords if kw in text
    )
    # 归一化到 0.5~0.98：命中的关键词占全部命中的比例，加一个下限
    share = (best_score / total) if total else 1.0
    confidence = round(min(0.98, max(0.50, 0.50 + share * 0.48)), 3)

    return json.dumps(
        {
            "intent": best_intent,
            "confidence": confidence,
            "matched": best_matched,
            "fallback": False,
        },
        ensure_ascii=False,
    )


#: 会"动钱"的意图 —— 咨询句里它们必须让位给规则咨询
_MOCK_ACTION_INTENTS = frozenset(
    {"refund_request", "return_request", "cancel_order", "change_address"}
)

_MOCK_IMPERATIVE_RE = re.compile(r"帮我|给我|麻烦|替我|请帮|我要|我想|烦请")
_MOCK_OWN_ORDER_RE = re.compile(r"我的|我这|这单|那一单|那单|这订单|订单能|订单可以")
_MOCK_MODAL_RE = re.compile(r"可以|能否|是否|能退|能用|能开|能补|行不行")
_MOCK_DEFINITION_RE = re.compile(r"是什么|有哪些|怎么算|什么规则|什么政策|什么条件|如何计算")


def _mock_looks_like_question(text: str) -> bool:
    """问规则，还是提请求？与 nlu/intent.py 的 _is_question 同一口径。

    两个通道必须给出一致答案，否则 mock 通道测过的行为在接真模型后就不成立，
    接真模型时暴露的行为在 mock 下又复现不了。
    """
    if _MOCK_IMPERATIVE_RE.search(text):
        return False
    if _MOCK_OWN_ORDER_RE.search(text):
        return False
    return bool(_MOCK_DEFINITION_RE.search(text) or _MOCK_MODAL_RE.search(text))


def _mock_slots(client: MockLLMClient, messages, tools) -> str:
    text = " ".join(_msg_text(m) for m in messages)
    order = re.search(r"ORD[- ]?(\d{4,8})", text, re.I)
    addr = re.search(r"(?:地址|改成|换到|寄到)[：:\s]*([^\n，,。？?！!]{4,40})", text)
    return json.dumps(
        {
            "order_id": f"ORD-{order.group(1)}" if order else None,
            "address": addr.group(1).strip() if addr else None,
            "reason": None,
            "product": None,
        },
        ensure_ascii=False,
    )


def _mock_self_eval(client: MockLLMClient, messages, tools) -> str:
    """生成前自评：回答里的每个数字/结论是否都能在证据里找到。"""
    blob = " ".join(_msg_text(m) for m in messages)
    # 抓回答部分（约定 prompt 里用 <answer> 包起来）
    ans = re.search(r"<answer>(.*?)</answer>", blob, re.S)
    answer = ans.group(1) if ans else blob
    ctx = blob[: ans.start()] if ans else ""

    # 检查回答里的数字（金额/天数）是否在证据里出现
    claims: list[str] = []
    for num in re.findall(r"\d+(?:\.\d+)?\s*(?:天|元|小时|个工作日|%|折)", answer):
        if num.replace(" ", "") not in ctx.replace(" ", ""):
            claims.append(num)

    coverage = 1.0 if not claims else max(0.0, 1.0 - len(claims) * 0.34)
    return json.dumps(
        {
            "grounded": not claims,
            "coverage": round(coverage, 3),
            "unsupported_claims": claims,
            "note": "所有数字均在证据中出现" if not claims else f"有 {len(claims)} 处数字未在证据中找到",
        },
        ensure_ascii=False,
    )


#: Markdown 表格的分隔行，如 `- | - | -`；只表达结构，没有信息量
_TABLE_SEP = re.compile(r"^\s*[-:|][-:|\s]*$")


def _is_table_sep(line: str) -> bool:
    return "|" in line and bool(_TABLE_SEP.match(line))


def _looks_like_table(lines: list[str]) -> bool:
    """判断一组行是不是表格。

    注意**不能**用"行首是不是 |"来判：知识库里的表格首行是
    `商品状态 | 申请时间窗 | …`（没有首尾竖线），按行首判会整个漏掉，
    最后表格被压成一行竖线塞进答复里。
    """
    if sum(1 for ln in lines if "|" in ln) >= 2:
        return True
    return any(_is_table_sep(ln) for ln in lines)


def _render_evidence_block(title: str, body: str) -> str:
    """把一条证据渲染成**人读得下去**的一段。

    关键是别把换行全压掉：表格逐行才有结构，压成一行之后用户看到的
    就是一串竖线。表格保持成块，散文才压成一行。
    """
    kept = [ln.rstrip() for ln in body.strip().splitlines() if ln.strip()]
    if _looks_like_table(kept):
        # 去掉 |---|---| 分隔行：表头和数据行已经足够表达结构
        rows = [ln for ln in kept if not _is_table_sep(ln)]
        return f"{title}：\n" + "\n".join(rows)
    return f"{title}：" + " ".join(" ".join(kept).split())


def _mock_generate(client: MockLLMClient, messages, tools) -> str:
    """生成回答：把证据块抽出来，按模板组织成有依据的答复。

    输出刻意用**空行分块** —— 前端按块渲染成段落与表格，而不是一面文字墙。
    这也是真模型该遵守的形态：让模型直接吐一段没有结构的长文，
    界面上就没法把它排好看。
    """
    blob = " ".join(_msg_text(m) for m in messages)
    blocks = re.findall(r"\[(\d+)\]\s*\[([^\]]+)\]\n(.+?)(?=\n\n\[\d+\]|\Z)", blob, re.S)
    if blocks:
        parts = [_render_evidence_block(title, body) for _, title, body in blocks[:3]]
        return "根据平台规则：\n\n" + "\n\n".join(parts)
    return "抱歉，我暂时没有找到可以回答这个问题的依据。"


def _mock_plan_tool(client: MockLLMClient, messages, tools) -> str:
    """ReAct 的 Thought/Action 一步：按问题里的线索决定调哪个工具。

    注意**先看工具可用性再选动作**：工具清单是运行时传进来的，规则实现必须
    以它为准，不然接真模型和跑 mock 会走出两条不同的路径，mock 测过的分支
    在真实环境里根本不会触发。
    """
    text = _last_user_text(messages).lower()
    available = {t["function"]["name"] for t in (tools or []) if "function" in t}
    order = re.search(r"ord[- ]?\d{4,8}", text, re.I)
    order_id = order.group(0).upper().replace(" ", "-") if order else None

    def pick(name: str) -> str:
        return json.dumps(
            {
                "thought": f"用户想了解 {name} 相关信息，需要调用 {name} 工具并带上订单号。",
                "action": name,
                "action_input": {"order_id": order_id},
            },
            ensure_ascii=False,
        )

    if any(k in text for k in ("物流", "快递", "到哪", "发货", "运单", "签收")):
        # 只有在工具清单里同时有两者时才需要挑：物流问题优先用更专用的那个，
        # 但拿不到就退而求其次用 get_order —— 这是"工具不可用时的降级"，
        # 不是"猜一个工具"。
        if "track_logistics" in available:
            return pick("track_logistics")
        if "get_order" in available:
            return pick("get_order")

    if any(k in text for k in ("订单", "买了", "下单", "状态", "单号")) and "get_order" in available:
        return pick("get_order")

    if any(k in text for k in ("运费", "包邮", "发票", "政策", "规则")) and "search_faq" in available:
        return json.dumps(
            {"thought": "这是规则类问题，应该走知识库检索。", "action": "search_faq",
             "action_input": {"query": _last_user_text(messages)}},
            ensure_ascii=False,
        )

    return json.dumps(
        {"thought": "现有工具无法处理这个问题。", "action": "finish", "action_input": {}},
        ensure_ascii=False,
    )


def _mock_memory(client: MockLLMClient, messages, tools) -> str:
    """长期记忆抽取：从用户这句话里挑出"下次对话还用得上"的事实。

    直接复用 `memory.extract.extract_by_rules`，而不是在这里再写一套规则 ——
    规则层本来就是这个任务的确定性实现，真模型只是它之上多认得一些自由表述。
    两处各写一份的话，会出现"mock 抽得出、接上真模型抽不出"（或者反过来）
    这种只有换模型那天才暴露的偏差。
    """
    from .memory.extract import extract_by_rules  # 局部导入：避免 llm ←→ memory 成环

    candidates = extract_by_rules(_last_user_text(messages))
    return json.dumps(
        {
            "memories": [
                {"kind": c.kind, "key": c.key, "text": c.text} for c in candidates
            ]
        },
        ensure_ascii=False,
    )


def _mock_summarize(client: MockLLMClient, messages, tools) -> str:
    """把更早的对话压成摘要 —— 保留意图、实体、结论，丢掉寒暄。"""
    blob = "\n".join(_msg_text(m) for m in messages)
    orders = sorted(set(re.findall(r"ORD[- ]?\d{4,8}", blob, re.I)))
    intents: list[str] = []
    for kw, label in (
        ("退款", "退款"), ("退货", "退货"), ("取消", "取消订单"),
        ("地址", "改地址"), ("物流", "查物流"), ("订单", "查订单"),
    ):
        if kw in blob:
            intents.append(label)
    bits = []
    if intents:
        bits.append("用户先后咨询了：" + "、".join(dict.fromkeys(intents)))
    if orders:
        bits.append("涉及的订单号：" + "、".join(orders))
    return "。".join(bits) + "。" if bits else "用户进行了若干轮寒暄，没有涉及具体业务诉求。"


def _mock_generic(client: MockLLMClient, messages, tools) -> str:
    return "（mock）已收到请求。"


_MOCK_HANDLERS = {
    "resolve": _mock_resolve,
    "classify": _mock_classify,
    "slots": _mock_slots,
    "self_eval": _mock_self_eval,
    "generate": _mock_generate,
    "plan_tool": _mock_plan_tool,
    "summarize": _mock_summarize,
    "memory": _mock_memory,
}

#: Mock 能处理的 task 集合 —— 测试用来确认没有任务悄悄退化成 _mock_generic
MOCK_TASKS = frozenset(_MOCK_HANDLERS)


# ============================================================
# 真实实现
# ============================================================


class OpenAILLMClient(LLMClient):
    """任何 OpenAI 兼容端点（官方 / 中转 / vLLM / Ollama 都行）。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.openai_api_key:
            raise LLMError(
                "MEWHELP_LLM_MODE=openai 但没找到 API Key。"
                "请设置 MEWHELP_OPENAI_API_KEY 或 OPENAI_API_KEY。"
            )
        from openai import OpenAI

        self._client = OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            timeout=self._timeout(),
            max_retries=self.settings.llm_max_retries,
        )
        self.model = self.settings.llm_model

    def _timeout(self) -> Any:
        """连接超时单独调短，读写沿用 `llm_timeout_s`。

        中转域名下常挂着多个 IP，挑中黑洞 IP 时连接会一直挂着 —— 如果连接阶段
        也等满 `llm_timeout_s`，第一次就把整次调用的预算烧完了，`max_retries`
        形同虚设。压短之后坏 IP 快速失败，重试换一个 IP 就通了。
        """
        import httpx

        return httpx.Timeout(
            self.settings.llm_timeout_s,
            connect=self.settings.llm_connect_timeout_s,
        )

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        task: str = "generic",
    ) -> LLMResponse:
        t0 = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.settings.llm_temperature if temperature is None else temperature,
        }
        if max_tokens or self.settings.llm_output_budget:
            kwargs["max_tokens"] = max_tokens or self.settings.llm_output_budget
        if json_mode and not self.settings.llm_reasoning:
            # 推理模型普遍不认 `response_format`：发了要么被忽略、要么个别中转直接
            # 400。提示词里本来就写着"只输出 JSON"，`extract_json` 也容忍围栏和
            # 前后废话，所以关掉它并不降低解析成功率。
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = list(tools)

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 — 统一收敛成 LLMError
            raise LLMError(f"LLM 调用失败: {exc}") from exc

        choice = resp.choices[0]
        message = choice.message
        usage = getattr(resp, "usage", None)
        tool_calls: list[dict[str, Any]] = []
        for tc in getattr(message, "tool_calls", None) or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})

        text = (message.content or "").strip()
        reasoning = (getattr(message, "reasoning_content", None) or "").strip()
        finish_reason = choice.finish_reason or "stop"

        # 空正文是**失败**，不是"答完了什么都不用说"。判空必须放在这里：下游拿到
        # 空字符串只会原样渲染成一条空气泡（界面上就是"转人工"），而日志里没有
        # 任何异常 —— 排查时只会看到"模型没报错但也没说话"。
        # 唯一合法的空正文是"这一轮要调工具"，所以 tool_calls 非空时放过。
        if not text and not tool_calls:
            detail = "，思维链已耗尽输出预算" if reasoning else ""
            raise LLMError(
                f"模型返回空正文（finish_reason={finish_reason}{detail}，"
                f"task={task}，max_tokens={kwargs.get('max_tokens')}）"
            )

        # 被预算截断的正文同样是**失败** —— 它比空正文更危险：空正文看起来像
        # "模型没说话"，而半句话看起来像一句完整的答复。用户会一字一字收到
        # "款项将在 1 到 3 个工"然后流就结束了，SSE 的 done 还是正常收尾，
        # trace 里 self_eval 也过（半句话里的数字往往仍来自证据），
        # 事后没有任何线索能看出这一轮是断的。
        #
        # 宁可让调用方走兜底：生成节点会回一句"没能组织出可靠的答复，帮你转人工"，
        # 摘要/记忆这类任务会退回规则层，规划任务会收手 —— 每一个都比发半句话好。
        if finish_reason == "length" and not tool_calls:
            raise LLMError(
                f"模型正文被输出预算截断（finish_reason=length，task={task}，"
                f"max_tokens={kwargs.get('max_tokens')}，"
                f"正文 {len(text)} 字，思维链 {len(reasoning)} 字）。"
                "推理模型要把思维链和正文的预算分开算。"
            )

        return LLMResponse(
            text=text,
            usage=TokenUsage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
            ),
            model=self.model,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            latency_ms=(time.perf_counter() - t0) * 1000,
            raw=resp,
            reasoning=reasoning,
        )


# ============================================================
# 工厂
# ============================================================


def build_llm(settings: Settings | None = None) -> LLMClient:
    """按配置返回 LLM 实现。全项目只在这里判断一次模式。"""
    settings = settings or get_settings()
    if settings.llm_mode == "openai":
        return OpenAILLMClient(settings)
    return MockLLMClient(settings)
