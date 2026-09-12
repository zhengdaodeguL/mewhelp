"""可观测层：把一条请求还原成一棵调用树。

对外只暴露 `Tracer` / `build_tracer` —— 编排层、服务层都从这里拿，
不直接碰内部实现。
"""

from __future__ import annotations

from .tracing import Tracer, build_tracer, digest, get_tracer, reset_tracer

__all__ = ["Tracer", "build_tracer", "get_tracer", "reset_tracer", "digest"]
