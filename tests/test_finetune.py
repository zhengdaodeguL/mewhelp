"""微调模块测试。

**刻意不依赖 torch** —— 没装 finetune 依赖的环境也必须能跑这批测试。
torch 相关的部分靠 lazy import 隔离，所以这里只测数据管线和指标口径。

指标口径必须自己验证：混淆矩阵算错的话，"Macro-F1 0.92" 这个数字
就是错的，而且错得很隐蔽 —— 数据看起来完全正常。
"""

from __future__ import annotations

import json

import pytest

from mewhelp.finetune.dataset import (
    ID_TO_LABEL,
    LABELS,
    LABEL_TO_ID,
    SEEDS,
    build_dataset,
    export_jsonl,
    load_labeled,
    split_dataset,
)
from mewhelp.finetune.train import (
    TrainConfig,
    _char_overlap,
    _compute_metrics,
    baseline_keyword,
)


# ============================================================
# 标签体系
# ============================================================


def test_seventeen_labels_exactly():
    """文章说的是 17 类 —— 这个数字要能对上。"""
    assert len(LABELS) == 17
    assert len(SEEDS) == 17
    assert len(set(LABELS)) == 17  # 无重复


def test_label_maps_are_consistent_and_bijective():
    for i, lbl in enumerate(LABELS):
        assert LABEL_TO_ID[lbl] == i
        assert ID_TO_LABEL[i] == lbl
    assert len(ID_TO_LABEL) == len(LABELS)


def test_confusable_label_pairs_are_both_present():
    """退款/退货、物流/配送 这种近义类必须都在 —— 它们才是检验模型真学会了的题。"""
    assert "退款政策" in LABELS and "退货流程" in LABELS
    assert "物流时效" in LABELS and "物流异常" in LABELS


def test_every_label_has_seed_examples():
    for lbl in LABELS:
        assert len(SEEDS[lbl]) >= 3, f"{lbl} 种子样本太少"


# ============================================================
# 数据构建
# ============================================================


def test_build_dataset_is_deterministic():
    """同一 seed 必须产出同一份数据。

    不可复现的话，"这次效果好"就永远说不清是改了模型还是改了数据 ——
    微调实验的头号纪律。
    """
    a = build_dataset(per_label=10, seed=42)
    b = build_dataset(per_label=10, seed=42)
    assert [r["text"] for r in a] == [r["text"] for r in b]
    assert [r["label"] for r in a] == [r["label"] for r in b]


def test_build_dataset_covers_all_labels_balanced():
    rows = build_dataset(per_label=20, seed=1)
    assert len(rows) == 17 * 20
    from collections import Counter

    counts = Counter(r["label"] for r in rows)
    assert set(counts) == set(LABELS)
    assert set(counts.values()) == {20}  # 类别平衡


def test_build_dataset_label_id_matches_label():
    rows = build_dataset(per_label=5, seed=7)
    for r in rows:
        assert ID_TO_LABEL[r["label_id"]] == r["label"]


def test_split_is_stratified_every_label_appears_in_test():
    """分层切分的关键作用：每个类都要出现在测试集里。

    不做分层的话，小数据集上很容易"某类全在训练集"，然后报一个好看的
    Macro-F1，掩盖了那个类根本没被考到。
    """
    rows = build_dataset(per_label=30, seed=3)
    train, dev, test = split_dataset(rows, seed=3)

    assert len(train) + len(dev) + len(test) == len(rows)
    for name, part in (("train", train), ("dev", dev), ("test", test)):
        labels = {r["label"] for r in part}
        assert labels == set(LABELS), f"{name} 缺少这些类: {set(LABELS) - labels}"


def test_split_has_no_leakage():
    """同一份数据不能同时出现在两个切分里。

    模板扩充的空间有限，随机采样很容易撞重复项；重复项一旦被切分摊到
    训练集和测试集两边就是泄漏 —— 测试分会虚高且完全看不出来。
    所以 build_dataset 按标签去重，这条测试是它的守门人。
    """
    rows = build_dataset(per_label=20, seed=5)
    train, dev, test = split_dataset(rows, seed=5)
    key = lambda r: (r["text"], r["label"])  # noqa: E731
    tr, dv, te = {key(r) for r in train}, {key(r) for r in dev}, {key(r) for r in test}
    assert not (tr & dv), f"训练/验证重叠: {tr & dv}"
    assert not (tr & te), f"训练/测试重叠: {tr & te}"
    assert not (dv & te), f"验证/测试重叠: {dv & te}"


def test_build_dataset_has_no_duplicate_texts_within_label():
    """去重是上面那条防泄漏测试的前提，单独钉一下。"""
    rows = build_dataset(per_label=50, seed=17)
    from collections import defaultdict

    by_label: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r["text"])
    for label, texts in by_label.items():
        assert len(texts) == len(set(texts)), f"{label} 有重复文本"


def test_load_labeled_rejects_unknown_label(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(
        json.dumps({"text": "你好", "label": "不存在的类"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="未知标签"):
        load_labeled(p)


def test_export_and_load_roundtrip(tmp_path):
    rows = build_dataset(per_label=3, seed=9)
    p = tmp_path / "out.jsonl"
    export_jsonl(rows, p)
    back = load_labeled(p)
    assert len(back) == len(rows)
    assert {r["label"] for r in back} == set(LABELS)


# ============================================================
# 指标口径
# ============================================================


def test_metrics_perfect_prediction():
    """全对时 Macro-F1 是 1.0。

    注意 n=4 而标签有 17 类 —— 没出现过的类 F1 记 0，所以 Macro-F1 会
    被这 13 个空类拉到 4/17。这是 Macro-F1 的定义使然，也正是它的价值：
    它惩罚"只学会一部分类"。真正跑评估时必须 17 类都有样本。
    """
    golds = [0, 1, 2, 3]
    m = _compute_metrics(golds, golds)
    assert m.accuracy == 1.0
    assert m.macro_f1 == pytest.approx(4 / 17, abs=0.01)

    # 17 类全都有样本且全对 -> 才是真正的 1.0
    full = _compute_metrics(list(range(len(LABELS))), list(range(len(LABELS))))
    assert full.macro_f1 == 1.0


def test_metrics_macro_f1_penalizes_ignored_class():
    """这是 Macro-F1 存在的理由：完全忽略一个类时，准确率可能还很好看，
    但 Macro-F1 会立刻塌下来。

    用 17 类做：全预测成第 0 类，accuracy = 1/17，macro_f1 很低。
    """
    golds = list(range(len(LABELS)))
    preds = [0] * len(LABELS)
    m = _compute_metrics(golds, preds)
    assert m.accuracy == pytest.approx(1 / 17, abs=0.01)
    assert m.macro_f1 < 0.1
    # 第 0 类运气好 F1 不为 0，其余全是 0
    assert m.per_class_f1[LABELS[0]] > 0
    assert m.per_class_f1[LABELS[1]] == 0.0


def test_metrics_confusion_matrix_records_misclassification():
    """混淆矩阵要能指出"把 A 认成了 B" —— 这是改进方向唯一的线索。"""
    a, b = LABEL_TO_ID["退款政策"], LABEL_TO_ID["退货流程"]
    # 真实标签 [退款政策, 退款政策, 退货流程, 退货流程]
    # 预测     [退款政策, 退货流程, 退货流程, 退货流程]
    m = _compute_metrics([a, a, b, b], [a, b, b, b])
    # 退款政策被误判成退货流程 1 次
    assert m.confusion["退款政策"]["退货流程"] == 1
    # 退货流程全判对 2 次
    assert m.confusion["退货流程"]["退货流程"] == 2
    assert m.accuracy == pytest.approx(0.75)


def test_metrics_empty_input_is_safe():
    m = _compute_metrics([], [])
    assert m.n == 0
    assert m.accuracy == 0.0


def test_per_class_f1_covers_all_seventeen():
    golds = list(range(len(LABELS)))
    m = _compute_metrics(golds, golds)
    assert len(m.per_class_f1) == 17
    assert set(m.per_class_f1) == set(LABELS)


# ============================================================
# 基线
# ============================================================


def test_baseline_runs_without_torch():
    """基线刻意不依赖 torch —— 它存在的意义就是给微调结果当一个不用训练的参照。"""
    rows = build_dataset(per_label=3, seed=11)
    m = baseline_keyword(rows, SEEDS)
    assert m.n == len(rows)
    assert 0.0 <= m.macro_f1 <= 1.0


def test_char_overlap_sanity():
    assert _char_overlap("退款政策", "退款政策") > 0
    assert _char_overlap("退款政策", "完全无关的内容") == 0


# ============================================================
# 配置
# ============================================================


def test_train_config_defaults_are_cpu_feasible():
    cfg = TrainConfig()
    # 全参微调（RoBERTa 体量不需要 PEFT）
    assert cfg.full_finetune is True
    # 短句 48 就够 —— 拉长序列是 CPU 上最贵的浪费
    assert cfg.max_len <= 64
    assert cfg.epochs <= 5


def test_train_config_serializes_paths():
    cfg = TrainConfig()
    d = json.loads(cfg.to_json())
    assert isinstance(d["output_dir"], str)
    assert "roberta" in d["output_dir"]


def test_train_module_does_not_import_torch_at_module_level():
    """硬要求：import finetune.train 不能把 torch 拉进来。

    否则没装 2GB 重依赖的环境连数据准备脚本都跑不了 —— 那正是"离线可跑"
    这个目标最先崩掉的地方。
    """
    import subprocess
    import sys

    code = (
        "import sys; sys.path.insert(0,'src');"
        "import mewhelp.finetune.train;"
        "print('torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd="."
    )
    assert out.stdout.strip() == "False", f"torch 被提前 import 了: {out.stdout} {out.stderr}"
