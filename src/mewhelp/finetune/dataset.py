"""微调数据集：17 类主题分类。

为什么是"主题分类"而不是"意图识别"？
因为意图识别（9 类）已经被 LLM + 规则解决了，再训一个模型是重复投资。
主题分类是**旁路的统计需求** —— 运营要看"这个月用户在抱怨什么"，
这类分类不需要 LLM 的推理能力，只需要又快又便宜又稳定，正是判别式小模型的甜区。

**这也正是"什么时候才轮到微调"的示范**：不是核心链路，是旁路统计。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

#: 17 类主题。刻意包含几个语义上接近的类别（退款/退货、物流/配送），
#: 用来检验模型是不是真的学到了区别，而不是靠关键词糊过去。
LABELS: tuple[str, ...] = (
    "退款进度",
    "退款政策",
    "退货流程",
    "换货申请",
    "物流时效",
    "物流异常",
    "配送范围",
    "订单取消",
    "地址修改",
    "发票问题",
    "优惠券使用",
    "会员权益",
    "尺码咨询",
    "材质咨询",
    "保修售后",
    "支付问题",
    "账号安全",
)

LABEL_TO_ID: dict[str, int] = {lbl: i for i, lbl in enumerate(LABELS)}
ID_TO_LABEL: dict[int, str] = {i: lbl for lbl, i in LABEL_TO_ID.items()}

#: 每一类的人工种子样本。样本量刻意小 —— 这是"冷启动真实起点"，
#: 真正的数据应该从生产日志里挖（见 data flywheel 的问题池）。
SEEDS: dict[str, tuple[str, ...]] = {
    "退款进度": (
        "我申请的退款什么时候到账",
        "退款审核通过了吗",
        "钱还没退回来",
        "退款显示处理中已经三天了",
    ),
    "退款政策": (
        "生鲜类可以退款吗",
        "超过多久就不能退款了",
        "退款有什么条件",
        "拆封了还能退款吗",
    ),
    "退货流程": (
        "退货要怎么操作",
        "退货寄回的地址在哪",
        "七天无理由怎么退",
        "退货需要自己付运费吗",
    ),
    "换货申请": (
        "收到的东西尺码不对想换一个",
        "可以换颜色吗",
        "换货要多久",
        "换货运费谁承担",
    ),
    "物流时效": (
        "一般几天能到",
        "发的是哪家快递",
        "什么时候能发货",
        "偏远地区要多久",
    ),
    "物流异常": (
        "快递一直没更新",
        "包裹显示签收但我没收到",
        "快递被退回了",
        "物流信息停在转运中心",
    ),
    "配送范围": (
        "你们发新疆吗",
        "港澳台能发货吗",
        "农村地区送货上门吗",
        "海外可以寄吗",
    ),
    "订单取消": (
        "我想取消订单",
        "下单后还能撤吗",
        "已经发货了能取消吗",
        "不小心下错单了怎么办",
    ),
    "地址修改": (
        "收货地址写错了要改",
        "能改成公司地址吗",
        "发货后还能改地址吗",
        "电话号码填错了怎么改",
    ),
    "发票问题": (
        "怎么开发票",
        "发票什么时候能开出来",
        "可以开增值税专用发票吗",
        "发票抬头填错了能重开吗",
    ),
    "优惠券使用": (
        "优惠券怎么用不了",
        "两张券能一起用吗",
        "满减券的使用门槛是什么",
        "优惠券过期了能补吗",
    ),
    "会员权益": (
        "会员有什么优惠",
        "怎么升级会员等级",
        "会员包邮吗",
        "会员积分怎么算",
    ),
    "尺码咨询": (
        "这个尺码怎么选",
        "我身高170体重60公斤选什么码",
        "尺码表在哪看",
        "这个版型偏大还是偏小",
    ),
    "材质咨询": (
        "这个是什么材质的",
        "会起球吗",
        "面料透气吗",
        "能不能机洗",
    ),
    "保修售后": (
        "保修多久",
        "坏了可以修吗",
        "保修需要发票吗",
        "保修期内维修要钱吗",
    ),
    "支付问题": (
        "支付失败了怎么办",
        "支持哪些付款方式",
        "可以用花呗吗",
        "扣款了但订单没生成",
    ),
    "账号安全": (
        "账号被盗了怎么办",
        "怎么修改密码",
        "收到异常登录提醒",
        "怎么解绑手机号",
    ),
}

#: 模板化扩充 —— 用真实客服里的修饰语把种子扩到可训练的量级。
#: 这是"数据不够时怎么办"的标准做法，比无中生有地编样本可靠得多。
_PREFIXES = ("你好，", "请问", "想问一下", "", "麻烦问下", "客服你好，")
_SUFFIXES = ("", "谢谢", "急", "在线等", "麻烦了")


def build_dataset(
    *,
    per_label: int = 200,
    seed: int = 20260911,
    augment: bool = True,
) -> list[dict[str, Any]]:
    """生成带标签的数据集。

    确定性：同一个 seed 永远产出同一份数据。微调结果不可复现的话，
    "这次效果好"就永远说不清是改了模型还是改了数据。

    **同一标签内不产出重复文本。** 这不是洁癖：模板扩充的空间有限
    （种子 × 前缀 × 后缀），随机采样很容易撞出重复项，而重复项一旦被
    分层切分摊到训练集和测试集两边，就是数据泄漏 —— 测试分会虚高，
    且完全看不出来。所以这里按固定顺序枚举组合并去重，而不是随机采样。

    返回 `[{"text": ..., "label": ..., "label_id": ...}, ...]`
    """
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    for label in LABELS:
        seeds = SEEDS[label]
        variants: list[str] = []
        seen: set[str] = set()
        # 枚举顺序固定（种子 → 前缀 → 后缀），保证可复现
        for s in seeds:
            for pre in _PREFIXES:
                for suf in _SUFFIXES:
                    text = f"{pre}{s}{suf}" if augment else s
                    if text not in seen:
                        seen.add(text)
                        variants.append(text)

        for i in range(per_label):
            # 组合用尽后循环复用 —— 宁可重复，也不无中生有地编样本
            text = variants[i % len(variants)]
            rows.append({"text": text, "label": label, "label_id": LABEL_TO_ID[label]})

    rng.shuffle(rows)
    return rows


def split_dataset(
    rows: list[dict[str, Any]],
    *,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 20260911,
) -> tuple[list[dict], list[dict], list[dict]]:
    """分层切分。

    **按标签分层是必须的** —— 随机切分在小数据集上很容易出现"某类全在训练集里，
    测试集一条没有"，然后报出一个好看的 Macro-F1，掩盖了那个类根本没学会。
    """
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)

    train, dev, test = [], [], []
    for label, items in by_label.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * test_ratio))
        n_dev = max(1, int(n * dev_ratio))
        test += items[:n_test]
        dev += items[n_test : n_test + n_dev]
        train += items[n_test + n_dev :]

    for part in (train, dev, test):
        rng.shuffle(part)
    return train, dev, test


def load_labeled(path: Path) -> list[dict[str, Any]]:
    """读真实标注数据（JSONL）。生产环境应该从问题池导出。"""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            label = obj["label"]
            if label not in LABEL_TO_ID:
                raise ValueError(f"未知标签: {label}")
            rows.append({"text": obj["text"], "label": label, "label_id": LABEL_TO_ID[label]})
    return rows


def export_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({"text": r["text"], "label": r["label"]}, ensure_ascii=False) + "\n")
