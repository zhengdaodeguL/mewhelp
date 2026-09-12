"""Quality 层：生成前的置信度闸门 + 生成后的事实自评。

两道检查一前一后，分工不同：

  * `ConfidenceGate`（生成**前**）—— 只看检索证据，证据不行就根本不让模型开口。
  * `SelfEvaluator`（生成**后**）—— 回答已经写出来了，再核一遍它说的每个数字
    是否都能在证据里找到，兜住"证据够但模型跑偏"这一类问题。

时序本身就是设计：闸门在前，成本是零次模型调用；自评在后，成本是一次调用。
两者都不能省 —— 省掉闸门就是把幻觉直接推给用户，省掉自评就是放模型自由发挥。
"""

from .gate import ConfidenceGate, extract_entities
from .selfeval import SelfEvaluator

__all__ = ["ConfidenceGate", "SelfEvaluator", "extract_entities"]
