"""测试环境护栏：把整套流水线钉在 mock 上。

仓库根的 `.env` 是**开发者本机**的配置。接真实模型时它里面会写 key、模型名、
以及 `MEWHELP_LLM_MODE=openai`；而 `Settings.model_config` 里恰好就配了读 `.env`，
于是测试里随手写的一句 `Settings()` 会把这份本机配置一起读进来。

后果不是"测试挂了"（挂了反而好办），而是**"跑测试"悄悄变成"拿真模型跑测试"**：

  * 每个文件慢几十秒（`tests/test_nlu.py` 单文件从 0.4s 涨到 51s）；
  * 结果随本机的 `.env` 漂移 —— 同一个 commit 在别人机器上是绿的，在这里是黄的；
  * 真要花钱和流量，CI 上则因为读不到 `.env` 而走另一条分支。

pydantic-settings 里**环境变量优先级高于 `.env`**，所以在收集阶段把这几项钉死就够了，
用例本身仍可覆盖：显式参数 > 环境变量 > `.env`。要跑真栈的用例（见 `tests/test_llm.py`）
一律显式传参或 `_env_file=None`，不受这里影响。
"""

from __future__ import annotations

import os

#: 被 `.env` 打开的开关，在这里逐个关回去。只列"切换实现"的四项 ——
#: 数值参数（阈值、top_k）本机怎么调都不该影响测试的结论，不必钉。
_FORCE_OFFLINE = {
    "MEWHELP_LLM_MODE": "mock",
    "MEWHELP_EMBEDDING_MODE": "mock",
    "MEWHELP_RERANKER_MODE": "mock",
    "MEWHELP_VECTOR_STORE_MODE": "memory",
}

for _key, _value in _FORCE_OFFLINE.items():
    os.environ[_key] = _value
