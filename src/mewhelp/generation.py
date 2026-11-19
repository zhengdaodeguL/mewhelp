"""单次生成的取消上下文：连接同步图、异步 HTTP 请求和 SSE。"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import CancelledError, Future
from contextvars import ContextVar
from typing import Any


class GenerationContext:
    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.cancelled = threading.Event()
        self._lock = threading.Lock()
        self._pending: set[Future] = set()
        self.prepare: Any = None
        self.token: Any = None
        self.finish: Any = None
        self.learner: Any = None
        self.text = ""

    def check(self) -> None:
        if self.cancelled.is_set():
            # 不能被节点的 except Exception 当成模型故障吞掉并写入兜底答案。
            raise asyncio.CancelledError

    def run(self, coroutine: Any) -> Any:
        """同步节点等待异步请求；取消会传给请求任务，连等待响应头也可取消。"""
        with self._lock:
            if self.cancelled.is_set():
                coroutine.close()
                raise asyncio.CancelledError
            future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
            self._pending.add(future)
        try:
            return future.result()
        except CancelledError:
            raise asyncio.CancelledError from None
        finally:
            with self._lock:
                self._pending.discard(future)

    def cancel(self) -> None:
        with self._lock:
            self.cancelled.set()
            pending = list(self._pending)
        for future in pending:
            future.cancel()


current_generation: ContextVar[GenerationContext | None] = ContextVar(
    "mewhelp_generation", default=None
)


def check_cancelled() -> None:
    context = current_generation.get()
    if context is not None:
        context.check()
