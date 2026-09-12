"""训练与评估。

**接口设计上的一个关键选择：训练循环不 import torch。**
torch 只在 `_load_torch()` 里 lazy import —— 这样没装 finetune 依赖的环境
仍然能 import 这个模块、跑数据准备和 baseline，不会因为缺 2GB 的包而整个项目挂掉。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .dataset import ID_TO_LABEL, LABELS, LABEL_TO_ID


def _load_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as exc:  # pragma: no cover - 取决于本机是否装了重依赖
        raise ImportError(
            "微调需要额外依赖。请先安装：\n"
            '    python -m pip install -e ".[finetune]"\n'
            "没显卡也能跑（CPU），RoBERTa 这个体量十几分钟就够。"
        ) from exc
    return torch, DataLoader, Dataset, AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class TrainConfig:
    model_name: str = "hfl/chinese-roberta-wwm-ext"
    #: 全参微调。RoBERTa 只有 ~1 亿参数，CPU 上跑得完，不需要 LoRA。
    full_finetune: bool = True
    max_len: int = 48            # 客服短句，48 足够，能省一大截算力
    batch_size: int = 32
    epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 20260911
    per_label: int = 200
    output_dir: Path = field(default_factory=lambda: Path("finetune/artifacts/roberta-topic"))

    def to_json(self) -> str:
        d = asdict(self)
        d["output_dir"] = str(self.output_dir)
        return json.dumps(d, ensure_ascii=False, indent=2)


@dataclass
class Metrics:
    accuracy: float = 0.0
    macro_f1: float = 0.0
    #: 每一类的 F1 —— 只看 Macro-F1 会掩盖"某个类完全没学会"
    per_class_f1: dict[str, float] = field(default_factory=dict)
    n: int = 0
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def _compute_metrics(golds: Sequence[int], preds: Sequence[int]) -> Metrics:
    """手写混淆矩阵 + F1，不依赖 sklearn —— 少一个重依赖，且口径自己说了算。"""
    n_labels = len(LABELS)
    confusion = [[0] * n_labels for _ in range(n_labels)]
    for g, p in zip(golds, preds):
        confusion[g][p] += 1

    correct = sum(1 for g, p in zip(golds, preds) if g == p)
    per_class_f1: dict[str, float] = {}
    for i, label in enumerate(LABELS):
        tp = confusion[i][i]
        fp = sum(confusion[r][i] for r in range(n_labels)) - tp
        fn = sum(confusion[i]) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class_f1[label] = round(f1, 4)

    macro_f1 = sum(per_class_f1.values()) / n_labels if n_labels else 0.0

    # 混淆矩阵只保留非零项，否则 17x17 里绝大多数是 0，报告没法看
    conf_out: dict[str, dict[str, int]] = {}
    for i, label in enumerate(LABELS):
        row = {LABELS[j]: confusion[i][j] for j in range(n_labels) if confusion[i][j]}
        if row:
            conf_out[label] = row

    return Metrics(
        accuracy=round(correct / len(golds), 4) if golds else 0.0,
        macro_f1=round(macro_f1, 4),
        per_class_f1=per_class_f1,
        n=len(golds),
        confusion=conf_out,
    )


def _make_dataset_class(Dataset, rows, tokenizer, max_len):
    class _TopicDataset(Dataset):
        def __init__(self):
            self.enc = tokenizer(
                [r["text"] for r in rows],
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )
            self.labels = [r["label_id"] for r in rows]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.enc.items()}
            item["labels"] = self.labels[idx]
            return item

    return _TopicDataset()


def train(
    train_rows: Sequence[dict],
    dev_rows: Sequence[dict],
    config: TrainConfig | None = None,
) -> tuple[Any, Any, Metrics, Metrics]:
    """全参微调。返回 (model, tokenizer, dev_metrics, train_metrics)。

    CPU / GPU 自适应：没显卡就用 CPU。RoBERTa 这个体量 CPU 十几分钟能跑完，
    把"需要显卡"从门槛里去掉，比任何性能优化都重要。
    """
    cfg = config or TrainConfig()
    torch, DataLoader, Dataset, AutoModel, AutoTokenizer = _load_torch()

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModel.from_pretrained(cfg.model_name, num_labels=len(LABELS))
    model.to(device)

    # 全参微调：所有参数都参与更新。LoRA 在这里是多余的复杂度 ——
    # PEFT 是为"全参训不起"的大模型准备的，RoBERTa 不在那个量级。
    if not cfg.full_finetune:  # pragma: no cover - 目前不推荐这条路
        for name, p in model.named_parameters():
            if "classifier" not in name:
                p.requires_grad = False

    train_ds = _make_dataset_class(Dataset, train_rows, tokenizer, cfg.max_len)
    dev_ds = _make_dataset_class(Dataset, dev_rows, tokenizer, cfg.max_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.batch_size)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, len(train_loader) * cfg.epochs)
    warmup = int(total_steps * cfg.warmup_ratio)

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_at)

    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            scheduler.step()
            optim.zero_grad()

    dev_metrics = evaluate(model, dev_loader, device, torch)
    train_metrics = evaluate(model, DataLoader(train_ds, batch_size=cfg.batch_size), device, torch)
    return model, tokenizer, dev_metrics, train_metrics


def evaluate(model, loader, device, torch) -> Metrics:
    model.eval()
    golds: list[int] = []
    preds: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits
            preds += logits.argmax(dim=-1).cpu().tolist()
            golds += labels.cpu().tolist()
    return _compute_metrics(golds, preds)


def predict(model, tokenizer, texts: Sequence[str], *, device=None, max_len: int = 48) -> list[str]:
    """推理 —— 供 API 旁路调用（主题统计）。"""
    torch, _, _, _, _ = _load_torch()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    enc = tokenizer(list(texts), truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    return [ID_TO_LABEL[i] for i in logits.argmax(dim=-1).cpu().tolist()]


# ============================================================
# 不依赖 torch 的轻量基线 —— 用来说明"微调到底带来了多少提升"
# ============================================================


def baseline_keyword(rows: Sequence[dict], seeds: dict[str, tuple[str, ...]]) -> Metrics:
    """关键词基线。

    没有基线的微调结果是没有意义的 —— "准确率 92%" 到底是模型的功劳，
    还是任务本来就简单？先跑一个不用训练的基线，差距才是微调的价值。
    """
    golds: list[int] = []
    preds: list[int] = []
    for r in rows:
        text = r["text"]
        best_label, best_score = LABELS[0], 0
        for label, examples in seeds.items():
            # 用字符重叠度做最近邻 —— 比关键词命中更接近"无训练"的性能上限
            score = max(_char_overlap(text, ex) for ex in examples)
            if score > best_score:
                best_score, best_label = score, label
        golds.append(r["label_id"])
        preds.append(LABEL_TO_ID[best_label])
    return _compute_metrics(golds, preds)


def _char_overlap(a: str, b: str) -> int:
    """字符 bigram 重叠数 —— 一个不训练也能用的相似度。"""
    def grams(s: str) -> set[str]:
        return {s[i : i + 2] for i in range(len(s) - 1)}

    return len(grams(a) & grams(b))


def run_pipeline(
    *,
    config: TrainConfig | None = None,
    seed_rows: dict[str, tuple[str, ...]] | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """完整流程：造数据 -> 分层切分 -> 基线 -> 微调 -> 评估 -> 落盘。

    返回一份可直接贴进报告的对照结果。
    """
    cfg = config or TrainConfig()
    from .dataset import SEEDS, build_dataset, split_dataset

    seeds = seed_rows or SEEDS
    out = Path(out_dir or cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    rows = build_dataset(per_label=cfg.per_label, seed=cfg.seed)
    train_rows, dev_rows, test_rows = split_dataset(rows, seed=cfg.seed)

    result: dict[str, Any] = {
        "config": json.loads(cfg.to_json()),
        "dataset": {
            "total": len(rows),
            "train": len(train_rows),
            "dev": len(dev_rows),
            "test": len(test_rows),
            "n_labels": len(LABELS),
        },
    }

    # 基线：不训练能到多少
    base = baseline_keyword(test_rows, seeds)
    result["baseline_keyword"] = json.loads(base.to_json())

    # 微调
    model, tokenizer, dev_m, train_m = train(train_rows, dev_rows, cfg)
    test_ds_loader = _make_test_loader(test_rows, tokenizer, cfg)
    torch, _, _, _, _ = _load_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_m = evaluate(model, test_ds_loader, device, torch)

    result["finetuned"] = json.loads(test_m.to_json())
    result["finetuned_dev"] = json.loads(dev_m.to_json())
    result["finetuned_train"] = json.loads(train_m.to_json())

    # 过拟合检查：train 远高于 dev 就是在背数据
    result["overfit_gap"] = round(train_m.macro_f1 - dev_m.macro_f1, 4)

    # 对比基线
    result["improvement_over_baseline"] = round(test_m.macro_f1 - base.macro_f1, 4)
    result["elapsed_s"] = round(time.perf_counter() - t0, 1)

    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _make_test_loader(rows, tokenizer, cfg):
    torch, DataLoader, Dataset, _, _ = _load_torch()
    ds = _make_dataset_class(Dataset, rows, tokenizer, cfg.max_len)
    return DataLoader(ds, batch_size=cfg.batch_size)
