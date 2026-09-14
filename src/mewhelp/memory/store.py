"""用户长期记忆的读写。

和同目录 `context.py` 的分工值得先说清楚，因为它们名字很像：

  * `context.py` 管的是**一次会话内**的上下文（近处原文滑窗 + 更早的摘要）。
    会话一结束，它承载的东西就该被忘掉。
  * 这个模块管的是**跨会话**的稳定事实（"他穿 L 码""他关注过 ORD-1007"）。

为什么记忆要单独落库，而不是"扔进 prompt 的历史里就行"：历史会被滑窗和摘要
丢掉（这正是省 token 的手段），而"用户是谁"恰恰不该因为聊得久了就被忘掉。
用户抱怨的"聊到第十轮它就不认识我了"，根因就在这里。

三条实现纪律，和 `SessionStore` 一致：

1. **异常不外泄。** 记忆读写失败就当没有记忆 / 没记住，记一条 warning。
   它是锦上添花，挂了不该让用户连一句答复都拿不到。
2. **去重靠数据库唯一约束**（`(user_id, mem_key)`），不是"先查再写"。
   后者在并发下会漏，而漏出来的表现是面板里堆着一串自相矛盾的旧值。
3. **上限管住的是"人"，不是"库"。** 每个用户的记忆条数有上限，超出时先淘汰
   最久没更新的、且没被钉住的那条 —— 没有上限的记忆列表会变成第二份聊天记录。
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable

from sqlalchemy import delete, func, select, update

from ..config import Settings, get_settings
from ..schemas import MemoryItem
from ..storage import UserMemoryRow, ensure_utc, session_scope
from .extract import MemoryCandidate, candidates_from_json  # noqa: F401  (re-export 给调用方)

logger = logging.getLogger(__name__)

#: 每个用户保留的记忆条数上限。50 条已经远超"模型真正用得上的量"（进 prompt
#: 的最多 8 条），留这个余量是为了让用户自己写下的备忘不会被自动淘汰掉。
DEFAULT_MAX_ITEMS = 50

#: 每次进 prompt 的记忆条数。上限压得低是有意的：记忆是**背景**，
#: 它挤掉的是检索证据的位置 —— 8 条档案把 5 条证据挤出上下文，
#: 是典型的"优化了次要矛盾"。
DEFAULT_PROMPT_ITEMS = 8

MAX_TEXT_CHARS = 60


def _stable_key(text: str) -> str:
    """没有给 key 时按内容生成一个。

    用 sha1 而不是内置 `hash()`：后者带进程级随机盐，重启一次同样的备忘
    就会变成两条。这个坑在飞轮那边已经踩过一次（见 `flywheel/store.py`）。
    """
    return "t:" + hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:12]


class MemoryStore:
    """长期记忆的增删改查。

    所有方法都收口在 `session_scope` 里，不持有连接、不带游标 ——
    和 `SessionStore` 一样，接口上没有任何"内存实现"的味道，
    将来换成 Redis 或独立的用户画像服务，外面这几个方法的语义都不用动。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        max_items: int = DEFAULT_MAX_ITEMS,
        prompt_items: int = DEFAULT_PROMPT_ITEMS,
    ) -> None:
        self.settings = settings or get_settings()
        self.max_items = max(1, int(max_items))
        self.prompt_items = max(1, int(prompt_items))
        #: 最近一次失败的原因。不抛异常，但必须能被诊断 ——
        #: 否则"记忆没生效"只会表现为"它怎么又忘了"，没有任何线索。
        self.last_error: str | None = None

    # ---------------------------------------------------------------
    # 读
    # ---------------------------------------------------------------
    def list(self, user_id: str, *, limit: int | None = None) -> list[MemoryItem]:
        """一个用户的全部记忆：钉住的排前面，其余按最近更新排。"""
        if not user_id:
            return []
        try:
            with session_scope(settings=self.settings) as s:
                stmt = (
                    select(UserMemoryRow)
                    .where(UserMemoryRow.user_id == user_id)
                    .order_by(
                        UserMemoryRow.pinned.desc(),
                        UserMemoryRow.updated_at.desc(),
                        UserMemoryRow.id.desc(),
                    )
                )
                if limit is not None:
                    stmt = stmt.limit(max(0, int(limit)))
                return [_to_item(r) for r in s.scalars(stmt)]
        except Exception as exc:  # noqa: BLE001 —— 读不到记忆 = 没有记忆
            self._warn("list", exc)
            return []

    def for_prompt(self, user_id: str, *, limit: int | None = None) -> list[MemoryItem]:
        """要写进 prompt 的那几条。

        排序和 `list` 一致（钉住的优先），但**条数上限更小**：
        界面要展示全部，模型只需要最相关的那几条。
        """
        return self.list(user_id, limit=limit or self.prompt_items)

    def count(self, user_id: str) -> int:
        if not user_id:
            return 0
        try:
            with session_scope(settings=self.settings) as s:
                return int(
                    s.scalar(
                        select(func.count())
                        .select_from(UserMemoryRow)
                        .where(UserMemoryRow.user_id == user_id)
                    )
                    or 0
                )
        except Exception as exc:  # noqa: BLE001
            self._warn("count", exc)
            return 0

    # ---------------------------------------------------------------
    # 写
    # ---------------------------------------------------------------
    def remember(
        self,
        user_id: str,
        candidate: MemoryCandidate,
        *,
        source: str = "auto",
        pinned: bool = False,
    ) -> MemoryItem | None:
        """写入一条记忆；同键已存在则**覆盖内容、保留计数**。

        保留 `hits` 和 `pinned` 是关键：用户已经把"尺码"钉在面板上了，
        这次只是把 L 改成 XL，钉住的状态不该被抹掉 —— 那看起来就像
        "我的设置自己丢了"，是最让人不信任的一类 bug。
        """
        return self.upsert(
            user_id,
            kind=candidate.kind,
            key=candidate.key,
            text=candidate.text,
            source=source,
            pinned=pinned,
        )

    def upsert(
        self,
        user_id: str,
        *,
        kind: str,
        text: str,
        key: str = "",
        source: str = "auto",
        pinned: bool = False,
    ) -> MemoryItem | None:
        text = (text or "").strip()[:MAX_TEXT_CHARS]
        if not user_id or not text:
            return None
        mem_key = (key or "").strip()[:64] or _stable_key(text)
        try:
            with session_scope(settings=self.settings) as s:
                row = s.scalar(
                    select(UserMemoryRow).where(
                        UserMemoryRow.user_id == user_id,
                        UserMemoryRow.mem_key == mem_key,
                    )
                )
                if row is None:
                    row = UserMemoryRow(
                        user_id=user_id,
                        kind=kind or "preference",
                        mem_key=mem_key,
                        text=text,
                        source=source,
                        pinned=bool(pinned),
                        hits=0,
                    )
                    s.add(row)
                    s.flush()
                else:
                    row.text = text
                    row.kind = kind or row.kind
                    # 手动写的东西不会被后续的自动抽取降级成 auto ——
                    # "这条是我自己加的"这个标记必须稳，否则用户会看到
                    # 自己写的备忘悄悄变成了系统猜的。
                    if source == "manual":
                        row.source = "manual"
                    if pinned:
                        row.pinned = True
                    s.flush()
                item = _to_item(row)
            self._enforce_limit(user_id)
            return item
        except Exception as exc:  # noqa: BLE001 —— 记不住不影响这一轮答复
            self._warn("upsert", exc)
            return None

    def remember_many(
        self, user_id: str, candidates: Iterable[MemoryCandidate], *, source: str = "auto"
    ) -> list[MemoryItem]:
        """批量写入，返回真正落库的那些。单条失败不牵连其它条。"""
        out: list[MemoryItem] = []
        for cand in candidates or []:
            item = self.remember(user_id, cand, source=source)
            if item is not None:
                out.append(item)
        return out

    def delete(self, user_id: str, memory_id: int) -> bool:
        """删一条。必须带 user_id —— 只按主键删就等于"谁都能删别人的记忆"，
        演示环境没有登录态，这个参数是眼下唯一的所有权校验。
        """
        if not user_id:
            return False
        try:
            with session_scope(settings=self.settings) as s:
                result = s.execute(
                    delete(UserMemoryRow).where(
                        UserMemoryRow.id == int(memory_id),
                        UserMemoryRow.user_id == user_id,
                    )
                )
                return bool(result.rowcount)
        except Exception as exc:  # noqa: BLE001
            self._warn("delete", exc)
            return False

    def set_pinned(self, user_id: str, memory_id: int, pinned: bool) -> bool:
        if not user_id:
            return False
        try:
            with session_scope(settings=self.settings) as s:
                result = s.execute(
                    update(UserMemoryRow)
                    .where(
                        UserMemoryRow.id == int(memory_id),
                        UserMemoryRow.user_id == user_id,
                    )
                    .values(pinned=bool(pinned))
                )
                return bool(result.rowcount)
        except Exception as exc:  # noqa: BLE001
            self._warn("set_pinned", exc)
            return False

    def clear(self, user_id: str) -> int:
        """清空一个用户的记忆，返回删掉的条数。"""
        if not user_id:
            return 0
        try:
            with session_scope(settings=self.settings) as s:
                result = s.execute(
                    delete(UserMemoryRow).where(UserMemoryRow.user_id == user_id)
                )
                return int(result.rowcount or 0)
        except Exception as exc:  # noqa: BLE001
            self._warn("clear", exc)
            return 0

    def mark_used(self, user_id: str, memory_ids: Iterable[int]) -> int:
        """给"真的被用上"的记忆各加一次命中。

        调用方负责判断哪几条算是被用上了（判据在 `extract.used_memory_ids`），
        这里只负责累加 —— 口径判断和存储分开，将来换判据不用动存储层。
        """
        ids = [int(i) for i in (memory_ids or []) if i]
        if not user_id or not ids:
            return 0
        try:
            with session_scope(settings=self.settings) as s:
                result = s.execute(
                    update(UserMemoryRow)
                    .where(
                        UserMemoryRow.user_id == user_id,
                        UserMemoryRow.id.in_(ids),
                    )
                    .values(hits=UserMemoryRow.hits + 1)
                )
                return int(result.rowcount or 0)
        except Exception as exc:  # noqa: BLE001
            self._warn("mark_used", exc)
            return 0

    # ---------------------------------------------------------------
    # 内部
    # ---------------------------------------------------------------
    def _enforce_limit(self, user_id: str) -> int:
        """超出上限时淘汰最久未更新的**未被钉住**的记忆。

        为什么只淘汰没钉住的：钉住是用户明确的"这条别删"。一个会自动删掉
        用户亲手钉住内容的清理逻辑，比不做清理糟糕得多。
        """
        try:
            with session_scope(settings=self.settings) as s:
                total = int(
                    s.scalar(
                        select(func.count())
                        .select_from(UserMemoryRow)
                        .where(UserMemoryRow.user_id == user_id)
                    )
                    or 0
                )
                overflow = total - self.max_items
                if overflow <= 0:
                    return 0
                victims = list(
                    s.scalars(
                        select(UserMemoryRow.id)
                        .where(
                            UserMemoryRow.user_id == user_id,
                            UserMemoryRow.pinned.is_(False),
                        )
                        .order_by(UserMemoryRow.updated_at.asc(), UserMemoryRow.id.asc())
                        .limit(overflow)
                    )
                )
                if not victims:
                    return 0
                s.execute(delete(UserMemoryRow).where(UserMemoryRow.id.in_(victims)))
                return len(victims)
        except Exception as exc:  # noqa: BLE001 —— 清理失败只是让列表长一点
            self._warn("enforce_limit", exc)
            return 0

    def _warn(self, op: str, exc: Exception) -> None:
        self.last_error = f"{op} failed: {exc!r}"
        logger.warning("[memory] %s 失败（按无记忆继续）：%s", op, exc)


def _to_item(row: UserMemoryRow) -> MemoryItem:
    """ORM 行 → 契约模型。

    手工映射而不是 `model_validate(row)`：契约模型是"给界面看的"，
    它刻意不带 `mem_key` / `user_id`，从 ORM 直接校验会因为缺字段而失败。
    这个转换写在一处，改契约时只需要改这里。

    时间统一走 `ensure_utc`：从库里读回来的 datetime 是朴素的（SQLite 不存
    时区），而刚写进内存的那条是 tz-aware 的 —— 不统一的话，同一个列表里
    两条时间会按两把尺子序列化，前端显示出来的"几小时前"有时对有时错。
    """
    return MemoryItem(
        id=int(row.id or 0),
        kind=row.kind or "preference",
        text=row.text or "",
        source=row.source or "auto",
        pinned=bool(row.pinned),
        hits=int(row.hits or 0),
        created_at=ensure_utc(row.created_at),
        updated_at=ensure_utc(row.updated_at),
    )


__all__ = [
    "DEFAULT_MAX_ITEMS",
    "DEFAULT_PROMPT_ITEMS",
    "MemoryStore",
]
