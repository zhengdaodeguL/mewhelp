"""从一轮对话里抽出**值得长期记住**的事实。

这一层要回答的不是"这句话说了什么"，而是"这句话在下一通对话里还有没有用"。
判据就一条：**它会不会随对话结束而失效**。

  * "我穿 L 码"            → 记。下个月再来买还是 L 码。
  * "别打电话给我"          → 记。这是人的偏好，不是这一单的偏好。
  * "我要退款"              → 不记。这是一次性诉求，办完就没了。
  * "帮我查下订单 ORD-1002" → 单号本身记（用户之后说"我那单"时要用），
                              "帮我查下"这部分不记。

两条实现纪律：

1. **抽取走 LLM 抽象层**（`task="memory"`），mock 用规则复刻同一件事。
   如果 mock 走规则、真模型走另一条路，就会出现"离线演示有记忆、接上真模型
   反而没有"这种最打击信心的断裂 —— 项目里其它任务（消解、分类、自评）
   都是这么统一的，这一层不例外。

2. **解析失败一律返回空**，绝不抛。记忆是锦上添花，它挂掉不该让用户
   连这一轮的答复都拿不到。模型给的 JSON 再离谱，最坏结果也只是"这次没记住"。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..nlu.entities import find_order_id

logger = logging.getLogger(__name__)

#: 一轮对话最多记几条。多了会把记忆面板淹掉，也会让 prompt 里的"已知信息"
#: 长到开始干扰生成 —— 用户一句话里交代两件事是常态，三件以上就该怀疑
#: 是不是规则抓错了。
MAX_PER_TURN = 3

#: 单条记忆文本的长度上限。记忆是"档案条目"不是"聊天记录"，
#: 超过这个长度的八成是把整句话抄下来了。
MAX_TEXT_CHARS = 60

#: 记忆条目里"值"与"标签"的分隔符。`mark_used` 靠它把值抠出来做命中判断，
#: 所以这是**格式约定**，不是装饰 —— 改模板时必须保持"标签：值"的形状。
LABEL_SEP = "："

#: 记忆类型
KIND_ORDER = "order"
KIND_PREFERENCE = "preference"
KIND_MANUAL = "manual"

#: "关注过的订单"这条记忆的固定前缀。指代消解靠它认单号（见 coreference）。
ORDER_MEMORY_LABEL = "最近关注的订单"


@dataclass(frozen=True)
class MemoryCandidate:
    """一条待写入的记忆。

    `key` 是**去重键**：同一个人第二次说"我穿 L 码"时，键相同就应该覆盖而不是
    新增。键取语义名（"size"）而不是内容哈希 —— 内容哈希会让"我穿L码"和
    "尺码是 L"变成两条互相矛盾的记忆。
    """

    kind: str
    key: str
    text: str


# ============================================================
# 规则层（mock 通道，以及真模型不可用时的兜底）
# ============================================================

#: "这句话在索取信息，不是在陈述自己的情况"。
#:
#: 刻意**不复用** `nlu.intent._is_question`：那个函数判的是"问规则还是提请求"，
#: 它的 `_OWN_ORDER_RE` 会把"我的地址是什么"判成**请求**（因为带"我的"），
#: 而这里需要的判断恰好相反 —— 那句话明显在问，绝不能记成"常用地址：是什么"。
#: 两个函数看着像，回答的是不同问题，合并只会让两边都错。
_ASKING_RE = re.compile(
    r"吗|呢|什么|怎么|如何|多少|几时|几天|哪个|哪里|哪家|是否|能不能|可不可以|[?？]"
)

#: 尺码。带上动词前缀才不会把"尺码怎么选"里出现的 S/M/L 抓下来。
_SIZE_RE = re.compile(r"我\s*(?:穿|要|买|拿|选|用)\s*([SMLX]{1,3}|\d{2,3})\s*(?:码|号)")

#: 联系方式偏好。"不要打电话"是长期偏好，值得记。
_CONTACT_RE = re.compile(
    r"(?:别|不要|不用|不想|请勿)\s*(打电话|发短信|加微信|发语音|发邮件|联系我)"
)

#: 会员等级。必须带第一人称，否则知识库问题（"会员有什么优惠"）会被记成事实。
_MEMBER_RE = re.compile(
    r"我\s*(?:是|有|开了|办的是|用的是)\s*(?:个)?\s*"
    r"(黑卡|金卡|银卡|PLUS|plus|Plus|VIP|vip|会员)"
)

#: 称呼。"我叫小明""我姓王"。
_NAME_RE = re.compile(r"我\s*(?:叫|姓)\s*([一-龥A-Za-z]{1,8})")

#: 用户明确要求记住的东西 —— 不套用"是不是在提问"的闸门，他都说"记住"了。
_MEMO_RE = re.compile(r"(?:记住|记一下|记下|备注)\s*[：:，,]?\s*(.{2,40})")


def _is_asking(text: str) -> bool:
    return bool(_ASKING_RE.search(text))


def _memo_key(text: str) -> str:
    """备忘类记忆的键 —— 按内容哈希，因为"用户自己写的备注"每一条都是独立的，
    不能像"尺码"那样互相覆盖（两条备忘不是同一个事实的新旧版本）。
    """
    return "memo:" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def extract_by_rules(question: str) -> list[MemoryCandidate]:
    """规则抽取 —— mock 通道的实现，也是真模型抽不出东西时的兜底。

    只有**用户说的话**会被抽取。助手说过什么不重要：记忆是"关于这个人的事实"，
    不是"这场对话的内容"。

    同一个键命中多次时保留**先命中的**（长关键词优先）：用户说
    "我穿 L 码，S 码太紧"，`_SIZE_RE` 只会命中第一个 L。
    """
    text = (question or "").strip()
    if not text:
        return []

    found: dict[str, MemoryCandidate] = {}

    def keep(cand: MemoryCandidate) -> None:
        found.setdefault(cand.key, cand)

    # 订单号：疑问句里也算数 —— "我的订单 ORD-1002 到哪了"里的单号是事实。
    order_id = find_order_id(text)
    if order_id:
        keep(
            MemoryCandidate(
                kind=KIND_ORDER,
                key="last_order",
                text=f"{ORDER_MEMORY_LABEL}{LABEL_SEP}{order_id}",
            )
        )

    # 下面这些都是"陈述自己的情况"，提问句一律不记。
    if not _is_asking(text):
        if m := _SIZE_RE.search(text):
            keep(MemoryCandidate(KIND_PREFERENCE, "size", f"尺码{LABEL_SEP}{m.group(1)}"))
        if m := _CONTACT_RE.search(text):
            keep(
                MemoryCandidate(
                    KIND_PREFERENCE, "contact_pref", f"不喜欢被{m.group(1)}"
                )
            )
        if m := _MEMBER_RE.search(text):
            keep(MemoryCandidate(KIND_PREFERENCE, "member", f"会员{LABEL_SEP}{m.group(1)}"))
        if m := _NAME_RE.search(text):
            keep(MemoryCandidate(KIND_PREFERENCE, "name", f"称呼{LABEL_SEP}{m.group(1)}"))

    if m := _MEMO_RE.search(text):
        note = m.group(1).strip(" 。，,.")
        if note:
            keep(MemoryCandidate(KIND_MANUAL, _memo_key(note), f"备忘{LABEL_SEP}{note}"))

    return list(found.values())[:MAX_PER_TURN]


# ============================================================
# 模型层
# ============================================================

#: 抽取 prompt。写清"记什么、不记什么"比写 JSON 格式更重要 —— 格式错了
#: 解析层能兜住，判据错了会往用户档案里堆一堆一次性的废话。
_SYSTEM_PROMPT = (
    "你负责维护客服系统里的用户档案。从用户这句话里挑出**下次对话还用得上**"
    "的稳定事实，例如尺码、称呼、联系方式偏好、会员等级、常用收货地址、"
    "他明确让你记住的事，以及他提到的订单号。\n"
    "不要记：一次性的诉求（我要退款、帮我查物流）、寒暄、以及任何提问本身。\n"
    '只输出 JSON：{"memories": [{"kind": "order|preference|manual", '
    '"key": "稳定的英文短键", "text": "简短的中文描述"}]}\n'
    "没有可记的就输出 {\"memories\": []}。"
)


def extract_candidates(llm: Any, question: str) -> list[MemoryCandidate]:
    """调用模型抽取候选记忆，失败则退回规则层。

    退回规则而不是返回空，是因为规则层恰好覆盖了最高频的那几类；
    真模型只是在这之上多认得出"我住杭州，一般周末在家"这种自由表述。
    换句话说：模型挂了，记忆功能降级但**不会消失**。
    """
    text = (question or "").strip()
    if not text:
        return []

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    try:
        resp = llm.chat(messages, temperature=0.0, json_mode=True, task="memory")
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 —— 抽取失败不影响这一轮答复
        logger.warning("记忆抽取失败，改用规则层：%s", exc)
        return extract_by_rules(text)

    parsed = parse_candidates(data)
    if parsed:
        return parsed

    # 一条都没抽到，要和"形状不认识"分开 —— 这两件事的处理方式完全相反。
    #
    #   模型明确说"没有"（`{"memories": []}`）→ 正常结论，返回空。
    #   形状我们不认识（裸数组、多套一层 data、条目缺 key）→ **功能静默失效**。
    #
    # 后者原先直接返回空，且不打任何日志：用户档案在生产上会整体消失
    # （他说"我穿 L 码"、面板空空如也、下一轮系统也不记得），而事后排查
    # 一条线索都没有。这里退回规则层并留一条 warning，和异常那条通道对齐。
    if not _says_nothing(data):
        logger.warning("记忆抽取：模型返回的形状不认识，改用规则层。payload=%s", str(data)[:200])
        return extract_by_rules(text)
    return parsed


def _says_nothing(data: Any) -> bool:
    """模型是否**明确**表达了"这句话没有可记的东西"。

    只有 `{"memories": []}` 和 `[]` 算明确 —— 空字典、缺字段、缺 key 的条目
    都不算：那些是"我们没读懂"，不是"模型说没有"。
    """
    if isinstance(data, list):
        return not data
    if isinstance(data, dict):
        rows = data.get("memories")
        return isinstance(rows, list) and not rows
    return False


def parse_candidates(data: Any) -> list[MemoryCandidate]:
    """把模型输出整形成候选列表。任何形状不对的条目单独丢掉，不牵连其它条目。

    对**形状**尽量宽容、对**内容**尽量严谨。宽容的部分：

      * 顶层可以是 `{"memories": [...]}`、`{"data": {"memories": [...]}}`，
        或者干脆就是个列表 —— 不同模型和中转的包装习惯不一样，
        为这个丢掉整个功能不值当；
      * 条目缺 `key` 时**不再整条丢弃**，改用 text 派生一个稳定键。
        `key` 只是给去重用的内部字段，而提示词要求"稳定的英文短键"
        恰恰是模型最容易不写的一项 —— 拿它当准入门槛，等于把用户档案
        的存亡押在模型记不记得填一个内部字段上。

    严谨的部分：`text` 为空才丢 —— 那才是真的没东西可记。
    """
    for candidate in (data, data.get("data") if isinstance(data, dict) else None):
        if isinstance(candidate, list):
            rows = candidate
            break
        if isinstance(candidate, dict) and isinstance(candidate.get("memories"), list):
            rows = candidate["memories"]
            break
    else:
        return []

    out: list[MemoryCandidate] = []
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or "").strip()
        kind = str(raw.get("kind") or KIND_PREFERENCE).strip()
        if not text:
            continue
        if kind not in (KIND_ORDER, KIND_PREFERENCE, KIND_MANUAL):
            kind = KIND_PREFERENCE
        key = str(raw.get("key") or "").strip() or _derived_key(kind, text)
        if key in seen:
            continue
        text = text[:MAX_TEXT_CHARS]
        seen.add(key)
        out.append(MemoryCandidate(kind=kind, key=key[:64], text=text))
        if len(out) >= MAX_PER_TURN:
            break
    return out


def _derived_key(kind: str, text: str) -> str:
    """模型没给 key 时，从内容派生一个稳定键。

    必须**稳定**：同一个事实换个说法要能落到同一条记忆上（`upsert` 靠 key
    去重），所以用内容哈希而不是自增序号 —— 序号每次调用都在变，
    同一句话会被记成两条。
    """
    return f"{kind}:" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def candidates_from_json(raw: str) -> list[MemoryCandidate]:
    """给测试和离线脚本用：从一段 JSON 文本直接构造候选。"""
    try:
        return parse_candidates(json.loads(raw))
    except (json.JSONDecodeError, TypeError):
        return []


# ============================================================
# 命中判断
# ============================================================


def memory_value(text: str) -> str:
    """取出记忆里那个**具体的值**（"尺码：L" → "L"）。

    没有分隔符时退回整条 —— 宁可整条比对，也不要凭空造一个能匹配到别的
    记忆上的值。这个值是 `mark_used` 唯一的判断依据。
    """
    if LABEL_SEP in text:
        return text.split(LABEL_SEP, 1)[1].strip()
    return text.strip()


def used_memory_ids(
    memories: Sequence[Any],
    question: str,
    resolved_question: str,
) -> list[int]:
    """哪些记忆**真的帮上忙了**。

    判据刻意很窄：这个值出现在补全后的问题里、却不在用户原话里 ——
    说明它是系统自己补进去的。这一条同时挡掉两种虚高：
      * 用户自己说出来的订单号（原话里就有）不算记忆的功劳；
      * 每轮都被塞进 prompt（那只是"被带上"，不是"被用上"）。
    """
    used: list[int] = []
    original = question or ""
    resolved = resolved_question or ""
    if not resolved or resolved == original:
        return used
    for mem in memories or []:
        value = memory_value(getattr(mem, "text", "") or "")
        if not value or value in original:
            continue
        if value in resolved:
            used.append(int(getattr(mem, "id", 0) or 0))
    return [i for i in used if i]


def prompt_lines(memories: Iterable[Any]) -> list[str]:
    """记忆 → 喂进 prompt 的文本行。

    保持"标签：值"的原样，不做任何加工：模型看到的和用户面板里看到的
    应该是同一句话，否则会出现"面板上写着 L 码、模型却按 XL 回答"这种
    谁也说不清的偏差。
    """
    return [str(getattr(m, "text", "") or "").strip() for m in memories if getattr(m, "text", "")]


__all__ = [
    "KIND_MANUAL",
    "KIND_ORDER",
    "KIND_PREFERENCE",
    "LABEL_SEP",
    "MAX_PER_TURN",
    "MAX_TEXT_CHARS",
    "ORDER_MEMORY_LABEL",
    "MemoryCandidate",
    "candidates_from_json",
    "extract_by_rules",
    "extract_candidates",
    "memory_value",
    "parse_candidates",
    "prompt_lines",
    "used_memory_ids",
]
