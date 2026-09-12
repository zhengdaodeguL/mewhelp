"""会话上下文层：双层上下文（近处滑窗保原文，更早的压成摘要）。

为什么要有这一层：

  * 多轮问答里，最早的几轮往往携带"订单号 / 结论 / 用户诉求"这类关键实体，
    直接截断会让模型在第 10 轮突然失忆；
  * 但把全部历史塞进 prompt，token 成本随轮数线性增长，且越靠后的信噪比越低。

所以分成两层：近处 `window_turns` 轮保留原文（指代消解要靠它做字面替换，
摘要过的文字没法做精确匹配）；更早的轮次压成**一段可增量更新的摘要**，
成本从 O(轮数) 降到 O(1)。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from ..config import Settings, get_settings
from ..llm import LLMClient
from ..schemas import ContextBundle, Turn

logger = logging.getLogger(__name__)

#: 附加在 ContextBundle 上的内部状态字段名（Pydantic v2 的 extra 默认丢弃，
#: 所以不会污染 as_messages()，也不会破坏"ContextBundle 只有契约字段"这条约定）。
_META = "__mewhelp_cm__"

#: 摘要 prompt 的分节标记。
#: 存在的意义：模型（和 mock）能分辨"已经压过的内容"和"这次新滑出的内容"。
#: 缺了它，增量更新就退化成"把全量历史重摘一遍"，token 成本又回到线性。
#: 两个标记都写成常量并被测试引用 —— 段名一改，测试会立刻炸，
#: 而不是悄悄退化成"每次都全量重摘"这种看不见的浪费。
PREVIOUS_HEADER = "【已有摘要】"
NEW_TURNS_HEADER = "【本次新增轮次】"
REQUIREMENT_HEADER = "【要求】"

_REQUIREMENT = (
    "请把以上内容压缩成一段简洁的中文摘要，用于后续对话的上下文。"
    "必须保留：订单号、金额、商品名、用户诉求、已经给出的结论；"
    "可以丢弃：寒暄和重复表述。直接输出摘要正文。"
)


def _render_transcript(turns: Sequence[Turn]) -> str:
    return "\n".join(turn_to_text(t) for t in turns)


def _prompt_summary(previous: str, excerpt: Sequence[Turn]) -> list[dict[str, str]]:
    """构造摘要 prompt。

    刻意把"旧摘要"和"新增轮次"分成两段呈现：模型只需要做一次**合并压缩**，
    而不是把全部历史重新读一遍。这是增量摘要在 prompt 层面的落点。
    """
    blocks: list[str] = []
    if previous:
        blocks.append(f"{PREVIOUS_HEADER}\n{previous}")
    blocks.append(f"{NEW_TURNS_HEADER}\n{_render_transcript(excerpt)}")
    blocks.append(f"{REQUIREMENT_HEADER}\n{_REQUIREMENT}")
    return [{"role": "user", "content": "\n\n".join(blocks)}]


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：中文约 1 字 1 token，英文约 4 字符 1 token。

    公式逐字对齐 `llm._rough_tokens`（含末尾那个常量 +1，它近似代表单条消息的
    角色名和分隔符开销）。口径必须一致：否则"摘要省了多少 token"这个数字在
    mock 记账和本模块记账之间对不上，度量本身就没有可信度。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    other = len(text) - cjk
    return cjk + max(1, other // 4)


def turns_token_estimate(turns: Sequence[Turn]) -> int:
    """一整段对话的 token 估算 —— "全量保留"的代价基线。

    按 `ContextBundle` 实际会发送的形态（roles + 逐条 content）逐条累计，
    这样"省了多少"的分子分母才是同一把尺子量出来的。
    """
    total = 0
    for t in turns:
        total += estimate_tokens(t.role) + estimate_tokens(t.content)
    return total


def turn_to_text(turn: Turn) -> str:
    """把一轮对话渲染成摘要 prompt 里的转写行。"""
    role = {"user": "用户", "assistant": "客服", "tool": "工具", "system": "系统"}.get(
        turn.role, turn.role
    )
    return f"{role}：{turn.content}"


class ContextManager:
    """双层会话上下文的构造器。

    这个类**不是有状态的服务**：`build()` 每次从传入的完整历史重建结果，
    因此同一个 ContextManager 可以安全地服务多个并发会话。跨轮增量所需
    的状态挂在上一轮产出的 `ContextBundle` 上（见 `build()` 的 `prev` 参数），
    会话的存储责任交给 storage 层，而不是藏在这里。
    """

    def __init__(self, llm: LLMClient, settings: Settings | None = None) -> None:
        self.llm = llm
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    def build(
        self,
        turns: list[Turn],
        prev: ContextBundle | None = None,
    ) -> ContextBundle:
        """把完整历史折叠成 (summary, window) 双层上下文。

        `prev` 是上一轮 `build()` 的产出。传了它才能做真正的增量摘要 ——
        只把"这次新滑出滑窗"的轮次并进旧摘要；不传则每次都要重读更早的全部
        历史，token 成本随对话轮数线性增长，正是要避免的那件事。

        任何异常都在这里兜底成"无摘要 + 完整滑窗"的降级结果：
        上下文质量下降是可接受的，整条请求链路因为摘要挂掉是不可接受的。
        """
        turns = list(turns or [])
        window_size = max(0, int(self.settings.window_turns))

        # 切分：近处原文 / 更早待压缩部分。
        # 窗口单位是**轮**（一问一答），不是消息条数，并且必须切在轮边界上：
        # 从半截开始（只剩 assistant 的半句）会让指代消解和历史引用都错位。
        window = self._tail_window(turns, window_size)
        older = turns[: len(turns) - len(window)]

        # 旧摘要的覆盖范围只信内部状态里的绝对下标，并且必须"没有越界"：
        # 历史被重放/截断/回滚时下标会失配，继续增量就会把实际没被摘要的
        # 轮次永久跳过 —— 那是静默的信息丢失，比多花一次模型调用严重得多。
        base_summary = ""
        base_covered = 0
        if prev is not None and prev.summary:
            covered = _coverage_index(prev, len(older))
            if covered:
                base_summary = prev.summary
                base_covered = covered

        newly_older = older[base_covered:]
        # 触发判据只看"还没被摘要、又即将滑出滑窗"的那部分 —— 已经压过的内容
        # 不重复计费，成本才不随轮数增长。
        pending_tokens = turns_token_estimate(newly_older)

        summary = base_summary
        summarized_turns = base_covered
        summary_generated = False

        if newly_older and pending_tokens > self.settings.summary_trigger_tokens:
            merged = self._summarize(base_summary, newly_older)
            if merged is not None:
                summary = merged
                summarized_turns = len(older)
                summary_generated = True

        bundle = ContextBundle(
            summary=summary,
            window=window,
            summarized_turns=summarized_turns,
            summary_generated=summary_generated,
        )
        # 摘要生成本身要调模型（多一跳延迟），已覆盖又被重新摘一遍会浪费，
        # 所以用私有属性记住"摘要覆盖到第几轮、内容是什么、内容是否可信"，
        # 让下一轮可以直接续着做增量。
        #
        # 三个数字各有各的用途，不能合并：
        #   covered_index —— 摘要在**本轮** turns 数组里的覆盖下标（绝对量）
        #   covered       —— 对外口径的累计轮数（契约里的可度量字段）
        #   window_len    —— 本轮滑窗长度，诊断用
        bundle.__dict__[_META] = {
            "summary": summary,
            "covered": summarized_turns,
            "covered_index": len(older) if summary_generated else base_covered,
            "window_len": len(window),
            # 降级得到的空摘要不能当作"已覆盖"的证明，否则后续早期轮次
            # 会被永久跳过摘要。
            "trusted": bool(summary) and summarized_turns > 0,
        }
        bundle.token_estimate = self.estimate_bundle_tokens(bundle)
        return bundle

    def append(self, session: list[Turn], turn: Turn) -> list[Turn]:
        """把一轮对话并进会话历史。

        两个刻意的取舍：
          1. 不修改传入的 list —— 调用方（图上的节点）拿着的 history 可能正被
             另一条分支读，原地 append 会制造难查的幽灵 bug；
          2. 历史长度有上限，超出时按对截断最早的轮次。没有上限的会话列表
             是内存泄漏，"用摘要换长度"仍然会在内存里留下全量历史。
             滑窗/摘要只需要最近若干轮，留一个远大于窗口的余量即可。
        """
        history = list(session or [])
        history.append(turn)
        limit = max(int(self.settings.window_turns) * 20, 60)
        if len(history) > limit:
            history = history[-limit:]
        return history

    # ------------------------------------------------------------------
    # 度量
    # ------------------------------------------------------------------
    def estimate_bundle_tokens(self, bundle: ContextBundle) -> int:
        """这次真实要发给模型的 token 估算。

        只算摘要 + 滑窗，**不算被滑出且未摘要的轮次** —— 那些确实没发出去。
        这个数字就是"省了多少"的分子，必须如实反映发送内容。
        """
        total = estimate_tokens(bundle.summary) if bundle.summary else 0
        total += turns_token_estimate(bundle.window)
        return total

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    @staticmethod
    def _tail_window(turns: Sequence[Turn], window_size: int) -> list[Turn]:
        """取最近 window_size 轮（轮 = 一次 user 开头到下一个 user 之前）。

        以 user 为轮起点而不是简单的"每 2 条一组"：真实会话里会出现
        工具消息、system 提示、连续两条 assistant，按固定步长切会在这些
        位置切歪，而且是那种"大部分情况对、偶尔错一轮"的难查错误。
        """
        if window_size <= 0 or not turns:
            return []
        starts = [i for i, t in enumerate(turns) if t.role == "user"]
        if len(starts) <= window_size:
            return list(turns)
        return list(turns[starts[-window_size] :])

    def _summarize(self, previous: str, excerpt: Sequence[Turn]) -> str | None:
        """调模型做一次增量压缩。失败返回 None（调用方据此降级）。

        这里把异常收成 None 而不是让它冒出去：摘要属于"锦上添花"，
        它失败时正确的行为是带着滑窗原文继续，而不是让用户看到 500。
        """
        text = _render_transcript(excerpt)
        messages = _prompt_summary(previous, excerpt)
        try:
            resp = self.llm.chat(
                messages,
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens,
                task="summarize",
            )
        except Exception as exc:  # noqa: BLE001 — 兜底是契约要求
            logger.warning("摘要生成失败，降级为无摘要：%s", exc)
            return None

        merged = (getattr(resp, "text", "") or "").strip()
        if not merged:
            # 空摘要同样视为失败：留着它会让下一轮误以为早期内容已经压过了。
            logger.warning("摘要模型返回空文本，降级为无摘要（excerpt_tokens=%d）", estimate_tokens(text))
            return None
        return merged


def _coverage_index(prev: ContextBundle, older_len: int) -> int:
    """把上一轮 bundle 的摘要覆盖范围换算成"当前早期部分的已覆盖下标"。

    返回 0 表示"这份旧摘要不可用，必须重摘"。

    只认内部状态里记的**绝对下标**，不拿 summarized_turns + len(window) 去反推 ——
    后者只在"每次滑出的轮数正好等于窗口长度"时才成立，窗口还没长满或阈值
    没触发时会算多，进而把实际没被摘要的轮次永久跳过（静默丢信息）。

    `covered <= older_len` 是流式场景下天然成立的不变式：历史只增不减时，
    已覆盖下标一定不超过新的早期部分长度。反过来说，一旦历史被截断/回滚，
    这个判据立刻失效并强制重摘 —— 这正是我们要的那道护栏。
    """
    meta = context_of(prev)
    if not meta.get("trusted"):
        return 0
    covered = int(meta.get("covered_index", 0) or 0)
    if 0 < covered <= older_len:
        return covered
    return 0


def context_of(bundle: ContextBundle) -> dict:
    """读取 ContextBundle 上挂的内部状态（没有则返回空字典）。

    给 storage 层持久化用：读出来的 bundle 是"死"的，拿回这个状态才能
    继续做增量摘要。
    """
    meta = bundle.__dict__.get(_META)
    return dict(meta) if isinstance(meta, dict) else {}


def link_prev(bundle: ContextBundle, prev: ContextBundle | None) -> ContextBundle:
    """把上一轮 bundle 的摘要状态补挂到本轮 bundle 上。

    用于 storage 层从库里重建会话：库里存的是 summary 文本，重建时用这个函数
    把状态接回去，增量摘要就能继续。返回入参本身，方便链式使用。

    注意恢复出来的 bundle 必须自己带齐 window 和 summarized_turns ——
    覆盖下标要靠它们换算，这就是为什么 storage 表里 turns_json 和 summary
    必须一起存，只存摘要文本是接不回增量链的。
    """
    prev_meta = context_of(prev) if prev is not None else {}
    trusted = bool(prev_meta.get("trusted", bool(prev and prev.summary)))
    covered_index = int(prev_meta.get("covered_index", 0) or 0)
    if not covered_index and trusted:
        # 只拿到了契约字段（比如从库里读出来的），退回按 摘要覆盖轮数 + 窗口长度 估算。
        # 这是近似值：早期部分不足一个窗口时它偏大，宁可多判一次"不可信"重摘，
        # 也不要少摘。
        covered_index = (
            (prev.summarized_turns + len(prev.window)) if prev is not None else 0
        )
    summary = bundle.summary or (prev.summary if prev else "")
    bundle.__dict__[_META] = {
        "summary": summary,
        "covered": int(prev.summarized_turns if prev is not None else 0),
        "covered_index": covered_index,
        "window_len": len(bundle.window),
        "trusted": bool(trusted and summary),
    }
    return bundle


__all__ = [
    "ContextManager",
    "context_of",
    "estimate_tokens",
    "link_prev",
    "turn_to_text",
    "turns_token_estimate",
]
