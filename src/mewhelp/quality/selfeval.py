"""生成后事实自评 —— 闸门漏过去的东西，这里再兜一次。

## 它和闸门的分工

`ConfidenceGate` 判的是"有没有资格开口"，只看检索结果，零成本。
本模块判的是"已经开的口有没有跑偏"，必须等 `answer` 出来才能跑。

这两件事无法合并：一份完全合格、分数很高的证据，模型照样可能往里塞一个
证据里没有的天数或金额。闸门看不到这种错，因为它没见过回答。

## 为什么两层校验（模型 + 规则）

`llm.chat(task="self_eval")` 给的是语义判断（"这句话是不是被证据支撑"），
但**模型自评本身也是模型**，它可能被自己的上下文说服、可能返回格式垃圾、
可能直接调用失败。所以本地永远再跑一遍**确定性规则校验**（数字类声明能不能
在证据里逐字找到），最后取两者中**更保守**的那一份：

    unsupported = 模型说没支撑的 ∪ 规则说找不到的
    grounded    = 模型说过了 且 规则也过了
    coverage    = min(模型给的, 规则算的)

宁可误判成"没通过"（代价是一次转人工），也不放过一处编造（代价是用户照着
一个假的天数去申请退款）。这个取舍是单向的，不能对称处理。

异常一律降级成"规则判定的结果"，绝不外泄 —— 自评崩了就当它不存在，
但绝不允许它因为崩了而静默放行。

## 与 mock 的约定

prompt 里回答部分用 `<answer>...</answer>` 包起来。`MockLLMClient` 的
`_mock_self_eval` 就是靠这组标签把"回答"和"证据"切开的：它只用正则抽回答里
的数字，再回到证据里找。约定破了，mock 会把整段 prompt 当回答，
把证据里的数字也算成"回答里编的" —— 所以标签是接口的一部分，不是格式偏好。
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..config import Settings, get_settings
from ..llm import LLMClient
from ..schemas import SelfEvalResult

#: 答案里"可能被编造"的量化声明：天数 / 金额 / 小时 / 工作日 / 折扣 / 百分比。
#: 只盯数字，是因为它是幻觉里唯一**可以被机器判定**的部分。
#:
#: 定性的编造（"我们支持无条件退货"）机器判不了 —— 那需要理解语义，
#: 交给模型那一路。规则这路的定位是"绝不会漏掉任何一个数字"，不是"什么都懂"。
_CLAIM_PATTERNS = (
    re.compile(r"\d+(?:\.\d+)?\s*个工作日"),
    re.compile(r"\d+(?:\.\d+)?\s*小时"),
    re.compile(r"\d+(?:\.\d+)?\s*(?:天|日)"),
    re.compile(r"\d+(?:\.\d+)?\s*元"),
    re.compile(r"\d+(?:\.\d+)?\s*%"),
    re.compile(r"\d+(?:\.\d+)?\s*折"),
)

#: 规则这路的每条未支撑声明扣多少覆盖率。扣 0.34 而不是 0.5：
#: 一处编造就已经足以判定不通过（grounded=False），coverage 只是给人工
#: 一个"编得多离谱"的刻度，扣得太狠会让"一处"和"五处"看起来一样烂。
_PENALTY_PER_CLAIM = 0.34

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


class SelfEvaluator:
    """回答 → 是否被证据支撑。任何异常都降级，不外泄。"""

    def __init__(self, llm: LLMClient, settings: Settings | None = None) -> None:
        self.llm = llm
        self.settings = settings or get_settings()

    # --------------------------------------------------------
    def evaluate(self, answer: str, context: str) -> SelfEvalResult:
        """核对 `answer` 里的每处量化声明是否都能在 `context` 里找到。

        `context` 就是 `RetrievalResult.to_context()` 的产物 —— 用同一份渲染文本，
        是因为模型生成时看到的就是它。用别的形态（比如原始 chunk 列表）做校验，
        会出现"生成时看到了、校验时找不到"的假阳性。
        """
        local = _rule_based(answer, context)

        try:
            judged = self._ask_model(answer, context)
        except Exception as exc:  # noqa: BLE001 — 自评崩了不能拖垮主链路
            return local.model_copy(
                update={
                    "note": _join_note(
                        local.note,
                        f"自评模型不可用（{type(exc).__name__}），已降级为规则校验",
                    )
                }
            )

        if judged is None:
            return local.model_copy(
                update={"note": _join_note(local.note, "自评模型返回不可解析，已降级为规则校验")}
            )

        return _merge(local, judged)

    # --------------------------------------------------------
    def _ask_model(self, answer: str, context: str) -> SelfEvalResult | None:
        """问模型一次。返回 None 表示"回答不可用"，交给规则那路兜。

        不在这里吞异常：抛出去由 `evaluate` 统一收敛，是为了让"调用失败"
        和"返回垃圾"走出同一条降级路径 —— 两条路径会分叉出两种行为，
        而这两种情况的处理方式本来就该完全一样。
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "你是事实校验器。只判断回答里的每一处量化声明（天数、金额、"
                    "时效、折扣）是否能在给定证据里逐字找到，不要用你自己的知识补充。"
                    "只输出 JSON。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"证据：\n{context}\n\n"
                    f"<answer>\n{answer}\n</answer>\n\n"
                    "逐条核对上面的回答。要求：\n"
                    "1. grounded：回答里所有量化声明都能在证据中找到才为 true；\n"
                    "2. coverage：被证据支撑的声明占比（0~1）；\n"
                    '3. unsupported_claims：找不到依据的声明原文列表；\n'
                    '4. note：一句话说明。\n'
                    '输出格式：{"grounded": bool, "coverage": float, '
                    '"unsupported_claims": [str], "note": str}'
                ),
            },
        ]
        resp = self.llm.chat(messages, json_mode=True, task="self_eval")
        payload = _load_payload(resp)
        if payload is None:
            return None
        return _coerce(payload)


# ------------------------------------------------------------
# 本地确定性校验
# ------------------------------------------------------------


def _rule_based(answer: str, context: str) -> SelfEvalResult:
    """纯规则的兜底校验：回答里的数字能不能在证据里找到。

    这一路永远不会失败、不会超时、不需要 API Key，所以它可以作为
    "模型那路挂掉时"的唯一依靠 —— 这也是把它写成纯函数的原因。
    """
    answer = answer or ""
    context = context or ""

    # 空证据 + 有内容的回答 = 无支撑。这是必须显式拦的一种情况：
    # 规则只能证明"某个数字没找到"，证明不了"整段话没依据"；
    # 而 context 为空时后者恰恰是最要命的问题。
    if not context.strip():
        return SelfEvalResult(
            grounded=not answer.strip(),
            coverage=0.0,
            unsupported_claims=[answer.strip()[:80]] if answer.strip() else [],
            note="没有可校验的证据，无法确认回答有依据",
        )
    if not answer.strip():
        return SelfEvalResult(
            grounded=False,
            coverage=0.0,
            unsupported_claims=[],
            note="回答为空，无从校验",
        )

    # 归一化空白后再比：证据写 "7 天"、回答写 "7天" 不该算两回事。
    # 这是最常见的一类"看着像幻觉其实不是"，不归一化会把它误判成编造。
    norm_ctx = _squash(context)
    claims = _claims(answer)
    missing = [c for c in claims if _squash(c) not in norm_ctx]

    coverage = 1.0 if not claims else max(0.0, 1.0 - len(missing) * _PENALTY_PER_CLAIM)
    note = (
        f"规则校验通过，{len(claims)} 处量化声明均在证据中找到"
        if not missing
        else f"规则校验发现 {len(missing)} 处量化声明未在证据中出现"
    )
    return SelfEvalResult(
        grounded=not missing,
        coverage=round(coverage, 4),
        unsupported_claims=missing,
        note=note,
    )


def _claims(answer: str) -> list[str]:
    """抽出回答里的量化声明，保序去重。"""
    found: list[str] = []
    for pattern in _CLAIM_PATTERNS:
        found.extend(m.group(0) for m in pattern.finditer(answer))
    return list(dict.fromkeys(found))


def _squash(text: str) -> str:
    """去掉所有空白 —— 用于宽松匹配。"""
    return re.sub(r"\s+", "", text or "")


# ------------------------------------------------------------
# 合并 / 解析
# ------------------------------------------------------------


def _merge(local: SelfEvalResult, judged: SelfEvalResult) -> SelfEvalResult:
    """取两者中更保守的一份。理由见模块 docstring。"""
    unsupported = list(dict.fromkeys(judged.unsupported_claims + local.unsupported_claims))
    grounded = bool(judged.grounded) and bool(local.grounded)
    coverage = min(float(judged.coverage), float(local.coverage))

    if not grounded:
        # 不通过时把 coverage 也压到不满分：不通过和"覆盖 1.0"同时出现，
        # 在审计表里读起来是自相矛盾的，人工审核会不知道该信哪个。
        coverage = min(coverage, 0.99)

    if bool(judged.grounded) != bool(local.grounded):
        # 两路结论打架时必须把这事写进 note。这条 note 会跟着 trace 落库，
        # 是事后判断"是模型太松还是规则太严"的唯一线索 ——
        # 只报最终结论而不报分歧，等于把调参需要的信息丢掉了。
        note = _join_note(
            "模型与规则结论不一致，已按更保守的一侧处理",
            f"模型：{'通过' if judged.grounded else '不通过'}（{judged.note}）",
            f"规则：{'通过' if local.grounded else '不通过'}（{local.note}）",
        )
    elif grounded:
        note = _join_note("模型与规则校验一致", judged.note or local.note)
    else:
        note = _join_note("模型与规则校验一致", judged.note or local.note)

    return SelfEvalResult(
        grounded=grounded,
        coverage=round(max(0.0, min(1.0, coverage)), 4),
        unsupported_claims=unsupported,
        note=note,
    )


def _load_payload(resp: Any) -> dict[str, Any] | None:
    """从 LLMResponse 里挖出 dict。挖不到就返回 None（走降级）。

    先用 `resp.json()`（它已经处理了 ```json 围栏），失败再自己兜一层 ——
    自评的输出格式是模型最容易写歪的地方，这里多一层容错能少一次无谓降级。
    """
    try:
        payload = resp.json()
    except Exception:  # noqa: BLE001 — 解析失败是预期内的，不当异常往外扔
        payload = None

    if not isinstance(payload, dict):
        m = _FENCE.search(getattr(resp, "text", "") or "")
        if m:
            try:
                payload = json.loads(m.group(1))
            except json.JSONDecodeError:
                payload = None
        if not isinstance(payload, dict):
            return None

    # 有些实现会把结果包一层，比如 {"self_eval": {...}} —— 多探一层比
    # 因为外层包了一层就整体降级划算。
    for key in ("self_eval", "result", "data"):
        inner = payload.get(key)
        if isinstance(inner, dict) and "grounded" in inner:
            return inner
    return payload


def _coerce(payload: dict[str, Any]) -> SelfEvalResult:
    """把模型给的字段强制成契约里的类型。

    模型经常把 grounded 写成字符串 "true"、把 coverage 写成百分比 85，
    直接塞进 Pydantic 要么报错要么静默变成 85.0（覆盖率 8500%）——
    后者更糟，因为它不报错。
    """
    coverage = _as_float(payload.get("coverage"), default=0.0)
    if coverage > 1.0:  # 写成 85 的情况
        coverage = coverage / 100.0

    return SelfEvalResult(
        grounded=_as_bool(payload.get("grounded")),
        coverage=round(max(0.0, min(1.0, coverage)), 4),
        unsupported_claims=_as_str_list(payload.get("unsupported_claims")),
        note=str(payload.get("note") or "")[:500],
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "是", "通过"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    return []


def _join_note(*parts: str) -> str:
    """拼接备注，去掉空段 —— 备注是给人工看的，空段会造成"两条 blank"的噪音。"""
    return "；".join(p for p in parts if p and p.strip())
