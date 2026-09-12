"""服务层：FastAPI 应用 + SSE 事件流 + 演示页。

契约见 CONTRACT.md 第 I 节。对外只有 `create_app`：

    from mewhelp.api import create_app
    app = create_app()            # 全 mock，离线可跑

`create_app(deps)` 的口子是给测试和演示留的 —— 换掉 `GraphDeps` 里的任何一个
组件（检索器、飞轮、LLM），图和应用层都不用改。
"""

from .app import ChatRequest, FeedbackRequest, ReviewRequest, create_app
from .service import (
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_EVIDENCE,
    EVENT_META,
    EVENT_TOKEN,
    EVENT_TOOL,
    ChatService,
    ResultCache,
    SessionStore,
    stream_events,
)

__all__ = [
    "create_app",
    "ChatRequest",
    "FeedbackRequest",
    "ReviewRequest",
    "ChatService",
    "SessionStore",
    "ResultCache",
    "stream_events",
    "EVENT_META",
    "EVENT_EVIDENCE",
    "EVENT_TOOL",
    "EVENT_TOKEN",
    "EVENT_DONE",
    "EVENT_ERROR",
]
