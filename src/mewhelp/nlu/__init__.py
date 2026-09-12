"""NLU 层：指代消解 + 意图识别 + 槽位抽取。

对外只暴露两个类，因为它们就是 CONTRACT.md 2.B 约定的接缝：

    CoreferenceResolver.resolve(question, history) -> ResolveResult
    IntentClassifier.classify(text)               -> ClassifyResult
    IntentClassifier.extract_slots(text, history) -> SlotExtraction

`resolve` 额外接受一个可选的 `context`（被指代那一轮的原文）。契约只要求
前两个位置参数，多一个带默认值的关键字参数不破坏任何调用方，但能让
"history 已被压成摘要"的场景少猜一次 —— 这是追加而非修改契约。

`entities` 和 `verbs` 是实现细节，但显式导出：测试和评估脚本要单独验证
正则口径（"消解补的单号和槽位抽的单号必须一致"这条约束，只能靠共用
同一份实现来保证），而 `PRONOUNS` 让评估脚本能对"指代覆盖率"打分。
"""

from __future__ import annotations

from . import entities, verbs
from .coreference import PRONOUNS, CoreferenceResolver
from .intent import IntentClassifier

__all__ = [
    "CoreferenceResolver",
    "IntentClassifier",
    "PRONOUNS",
    "entities",
    "verbs",
]
