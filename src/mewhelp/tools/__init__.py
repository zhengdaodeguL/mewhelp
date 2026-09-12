"""工具层：注册表 + 业务工具。

分工：`registry` 只管"怎么安全地调用一个工具"（超时/重试/审计），
`business` 只管"业务铁律长什么样"。两者分开，是为了让业务规则能被
单独评审和单独测试 —— 规则写错和调用机制写错，是两类完全不同的 bug。
"""

from .business import build_business_registry, register_business_tools
from .registry import ToolRegistry

__all__ = ["ToolRegistry", "build_business_registry", "register_business_tools"]
