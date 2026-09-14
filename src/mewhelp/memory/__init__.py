"""记忆模块：**会话内**的上下文（双层滑窗+摘要）与**跨会话**的长期记忆。

两者名字相近、职责相反，所以放在同一个包里对照着看：

  context.ContextManager  一次会话内的上下文。会话结束即失效，
                          目的是"在有限的 token 里保留尽量多的相关信息"。
  store.MemoryStore       跨会话的用户档案。会话结束了它还在，
                          目的是"记住这个用户是谁"，与 token 预算无关。

把它们混成一个概念，会得到两种都做不好的实现：要么把用户档案塞进
滑窗（聊久了就忘），要么把整段对话当成档案（档案变成第二份聊天记录）。
"""

from .context import (
    ContextManager,
    context_of,
    estimate_tokens,
    link_prev,
    turn_to_text,
    turns_token_estimate,
)
from .extract import (
    KIND_MANUAL,
    KIND_ORDER,
    KIND_PREFERENCE,
    MemoryCandidate,
    extract_by_rules,
    extract_candidates,
    memory_value,
    used_memory_ids,
)
from .store import DEFAULT_MAX_ITEMS, DEFAULT_PROMPT_ITEMS, MemoryStore

__all__ = [
    "ContextManager",
    "DEFAULT_MAX_ITEMS",
    "DEFAULT_PROMPT_ITEMS",
    "KIND_MANUAL",
    "KIND_ORDER",
    "KIND_PREFERENCE",
    "MemoryCandidate",
    "MemoryStore",
    "context_of",
    "estimate_tokens",
    "extract_by_rules",
    "extract_candidates",
    "link_prev",
    "memory_value",
    "turn_to_text",
    "turns_token_estimate",
    "used_memory_ids",
]
