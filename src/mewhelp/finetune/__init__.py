"""第 10 章：RoBERTa 微调。

**这个模块最重要的不是代码，是它的取舍逻辑。**

先问三个问题，答案是"都能"就别微调：

  1. Prompt 能不能解决？   —— 改措辞、加 few-shot，成本几乎为零
  2. RAG 能不能解决？      —— 知识不够就补知识，别拿模型参数去记事实
  3. 规则能不能解决？      —— 有限类别、关键词明确时，规则比模型更准更便宜

只有当"任务固定、类别封闭、量足够大、且前三条都试过了"的时候，
微调才是对的答案。

**为什么是 RoBERTa 全参而不是 LoRA？**
LoRA / QLoRA 是给大模型（7B+）用的 PEFT —— 那些模型全参微调的成本高到不现实，
所以冻结原参数、只训一小撮低秩矩阵。而 RoBERTa 只有 ~1 亿参数，
**全参微调在 CPU 上十几分钟就能跑完**，根本不需要 PEFT 那套复杂度。
这个区分能讲清楚，是面试里的加分项。

一个小模型 + 一套干净的训练/评估流程，比一个大模型 + 一堆没验证的假设更有说服力。
"""

from .dataset import LABELS, build_dataset, load_labeled, split_dataset
from .train import TrainConfig, evaluate, predict, train

__all__ = [
    "LABELS",
    "build_dataset",
    "load_labeled",
    "split_dataset",
    "TrainConfig",
    "train",
    "evaluate",
    "predict",
]
