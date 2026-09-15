"""NLU 规则层共用的实体抽取原语。

为什么单独抽一个文件：指代消解（把"它"补成实体）和槽位抽取（填 order_id）
找的其实是同一批实体。如果两份代码各写一套正则，迟早会出现"消解补的是
ORD-1001、槽位抽出来的是 ORD1001"这种前后不一致 —— 下游按字符串比对时
就会静默对不上（订单查不到、飞轮落错单），而且极难排查。

这里只放"纯函数式"的规则：输入一段文本，输出实体或 None。不做任何 IO、
不看配置、不调模型，保证确定性可测。
"""

from __future__ import annotations

import re
from typing import Iterator

from ..schemas import Turn

# ============================================================
# 订单号
# ============================================================

#: 全项目**唯一**的订单号口径。指代消解、槽位、高风险子流程、置信度闸门
#: 都从这一条正则派生 —— 三处各写一份的下场，是其中两份悄悄写坏、只有一份能跑。
#:
#: **不要用 `\b` 划边界。** Python 的 `\w` 包含中日韩字符，于是"订单ORD-1001"
#: 里"单"和"O"都是词字符、中间根本没有词边界，`\bORD` 匹配不到；
#: "ORD-1001的物流"末尾同理。而中文用户恰恰很少在单号前后敲空格 ——
#: 结果就是"用户明明报了单号，系统却追着要单号"。
#:
#: 改用两头都不依赖 `\w` 的定界：前面不能是 ASCII 字母数字（挡住 `xORD-1001`
#: 这种粘连），后面不能还是数字（挡住从更长的数字串里截一段）。
ORDER_RE = re.compile(r"(?<![A-Za-z0-9])ORD[-_\s]?(\d{4,8})(?!\d)", re.I)

#: 用户也常只说"订单 1001""单号：10012345"。这种裸数字必须**带上下文词**才认，
#: 否则"我家住 1001 室""电话 1380000"都会被当成订单号 —— 误抽订单号比抽不到
#: 更危险：下游会拿着一个不存在的单号去查，然后报"订单不存在"给用户。
#: 末尾的负向前瞻挡住"1001 年""3 月"这类日期。
_BARE_ORDER_RE = re.compile(
    r"(?:订单|单号|order)\s*(?:号码|号|编号)?\s*[：:#]?\s*(\d{4,12})(?![\d年月日元])",
    re.I,
)


def normalize_order_id(raw: str) -> str:
    """统一成 ORD-xxxx，作为全系统的订单号规范形式。"""
    return "ORD-{}".format(str(raw).lstrip("0") or "0")


def find_order_id(text: str) -> str | None:
    """文本里**最后一个**订单号。

    为什么取最后一个而不是第一个：指代消解要的是"最近提到的那一单"，
    用户后面说的单号总是更相关；一轮里说了两单时，最后那个才是他要问的。
    """
    if not text:
        return None
    hits = ORDER_RE.findall(text)
    if hits:
        return normalize_order_id(hits[-1])
    hits = _BARE_ORDER_RE.findall(text)
    if hits:
        return normalize_order_id(hits[-1])
    return None


# ============================================================
# 地址
# ============================================================

#: 只有"明显在说地址值"的引导词才允许往后抓，避免把"地址写错了"整句当地址
_ADDRESS_RE = re.compile(
    r"(?:收货地址|地址|寄到|送到|改成|换到|改为)"
    r"\s*(?:是|为|在)?\s*[：:]?\s*"
    r"([一-龥A-Za-z0-9\-]{4,40})"
)

#: 抓到的尾巴常粘着疑问/请求词（"…1号能改吗"），在这里切断
_ADDR_CUT = "吗呢吧怎能否可搞"

#: 抓到的开头的动作/连接词要剥掉（"改成杭州市西湖区…" → "杭州市西湖区…"）
_ADDR_LEAD_RE = re.compile(r"^(?:改[成为到]|换[成为到]|寄到|送到|是|为|在|的)+")

#: 这些短语说明抽到的根本不是一个值，而是"那个地址"这类**指代占位**。
#: 放它们过去，槽位就会被填成"那个地址"，missing_for() 便不再报缺，
#: 改地址流程会拿着一个假地址去执行 —— 静默错，比缺槽位严重得多。
#: 所以规则层和"模型返回值清洗"共用同一份判定，避免一边挡一边放。
_PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "那个地址", "这个地址", "那地址", "这地址", "上述地址", "原地址",
    "那个订单", "这个订单", "该订单", "那单", "这单", "如上", "同上",
    "那个商品", "这个商品", "那个东西", "这个东西",
)

#: 疑问词打头的地址值一定不是地址（"改成哪里"）
_ADDR_REJECT_PREFIX = ("哪", "什么", "多少", "怎么")


def is_placeholder(value: str) -> bool:
    """判断一个值是不是"指代占位"，而不是真实值。"""
    text = (value or "").strip()
    if not text:
        return True
    return any(text == m or text.startswith(m) for m in _PLACEHOLDER_MARKERS)


def _cut_address(v: str) -> str | None:
    idx = min((v.find(c) for c in _ADDR_CUT if v.find(c) > 0), default=-1)
    value = (v[:idx] if idx > 0 else v).strip(" 　的")
    value = _ADDR_LEAD_RE.sub("", value).strip(" 　的：:")
    if len(value) < 4:
        return None
    if is_placeholder(value) or any(value.startswith(p) for p in _ADDR_REJECT_PREFIX):
        return None
    return value


def find_address(text: str) -> str | None:
    """文本里最后一个像**地址值**的片段。

    取最后一个的理由和订单号一样：用户改口后的说法才是他真正要的。
    """
    if not text:
        return None
    for m in reversed(list(_ADDRESS_RE.finditer(text))):
        value = _cut_address(m.group(1))
        if value:
            return value
    return None


# ============================================================
# 商品名
# ============================================================

#: 电商商品名几乎都落在这些"品类字"上。这是中文里最稳的一个信号：
#: 没有形态学、没有词表也能用，且完全确定性。换成需要下载模型的 NER
#: 会破坏"离线可跑"，收益也不见得比这个字表高。
_PRODUCT_SUFFIXES = frozenset(
    "衫服裙裤鞋靴帽袜包机表镜杯灯枕被毯霜膜膏液油粉锅壶垫帘箱柜椅桌仪器刀剪盒罐袋瓶笔纸线卡盘"
)

#: 往前收集名字时，遇到这些字就断（"我买的羊毛衫" → 不要"我买的"）
_NAME_STOP = frozenset(
    "的了是我你他她它们和与及或个件款这那买有在把给从到就都还很也才又只没要想去来"
    "吗呢吧怎可否能多少什何时先再更挺真不"
)
_PUNCT = frozenset("，,。.？?！!；;：:、（）()【】[]{}<>《》\"'`~/\\|+=*&^%$#@ \t\n\r")

#: 指代词 + 量词后紧跟的通常就是商品名（"这个连衣裙"）
_PRODUCT_CUE_RE = re.compile(
    r"(?:那|这)(?:款|个|件|双|台|瓶|盒|包|条|本|套|只|把)\s*([一-龥A-Za-z0-9]{2,10})"
)
#: "我买的/刚下单的/入手的 X"
_PRODUCT_BOUGHT_RE = re.compile(
    r"(?:买了|买的|买|下单了|下单|入手了|入手|拍下了|拍了)"
    r"(?:的|了)?(?:那|这)?(?:款|个|件|双|台|瓶|盒|包|条|本|套)?"
    r"([一-龥A-Za-z0-9]{2,8})"
)

#: 商品属性词 —— 出现在商品名位置上时说明用户问的是属性，不是商品（"这个尺码"）
ATTR_WORDS: tuple[str, ...] = (
    "尺码", "尺寸", "材质", "面料", "颜色", "色号", "型号", "参数", "规格",
    "保修", "重量", "价格", "品牌", "大小", "版本", "口味", "含量", "成分",
    "功效", "适合", "能不能", "怎么", "质量", "保质期", "电池",
)

#: 疑问词打头的片段一律不可能是商品名 —— 用来挡住
#: "这个(哪个能退)" 里被误当成商品名的问句
_QUESTION_HEADS = frozenset("哪什么多少怎为何谁几哪能")

#: 谓语/形容词的字。商品名不会以它们收尾（"台灯坏""连衣裙脏"），
#: 出现了说明这个候选是从谓语里切出来的半截词。
_VERB_TAIL = frozenset(
    "坏脏破旧好差大烂碎丢缺漏断裂湿丑慢卡吵弱短紧松"
    "有点太很不都也还能要会想改退换修买"
)


def _cut_candidate(raw: str) -> str:
    """把候选片段截到第一个虚词/标点为止。

    "我买的羊毛衫有点问题" 里 "买的" 后面的捕获会一路吞到谓语（"羊毛衫有点问题"）。
    截断必须在**两侧**都做：只从右边裁（_trim_name）会在遇到"有"这种停用字时
    直接裁空，而商品名其实在它前面。
    """
    for i, ch in enumerate(raw):
        if ch in _NAME_STOP or ch in _PUNCT:
            return raw[:i]
    return raw


def _trim_name(raw: str, back: int = 5) -> str:
    """从右往左收集连续词字作为商品名，遇停用字/标点即止。"""
    chars: list[str] = []
    i = len(raw) - 1
    while i >= 0 and len(chars) < back:
        ch = raw[i]
        if ch in _NAME_STOP or ch in _PUNCT:
            break
        chars.append(ch)
        i -= 1
    return "".join(reversed(chars))


def _extract(raw: str) -> str | None:
    """捕获片段 → 商品名。

    两个方向都试：从右往左收（适合"买的羊毛衫"这种干净的片段）和从开头取
    最短的品类词前缀（适合"连衣裙色差太大了"这种后面拖着谓语的片段）。
    右收的结果比左取还长一截时，说明它把谓语吞进来了，改用左取。
    """
    right = _trim_name(_cut_candidate(raw))
    lead = _leading_name(raw)
    if lead and (not _plausible(right) or len(right) > len(lead) + 1):
        return lead
    if _plausible(right):
        # 右收的结果以动词/形容词收尾时说明它是从谓语里切出来的（"台灯坏"），
        # 这种半截词进检索只会污染查询 —— 宁可只留品类词（"台灯"）。
        if right[-1] in _NAME_STOP or right[-1] in _VERB_TAIL:
            return lead or right
        return right
    return lead


def _leading_name(raw: str) -> str | None:
    """从片段**开头**取商品名：最短的、以品类字结尾的合理前缀。

    "连衣裙色差太大" → "连衣裙"（而不是把后面的形容词一起吞进来）。
    为什么按"以品类字结尾"最短匹配：电商商品名几乎都以品类字收尾
    （衫/裙/鞋/机…），而形容词/谓语不会。找最短的那个前缀，就天然把
    "色差太大""有点问题"这类尾巴甩掉了。
    """
    raw = _cut_candidate(raw)
    for i in range(min(len(raw), 5), 1, -1):
        candidate = raw[:i]
        if candidate[-1] in _PRODUCT_SUFFIXES and _plausible(candidate):
            return candidate
    return None


def _plausible(name: str) -> bool:
    """过滤掉"尺码""怎么""哪个""哪个能退"这类冒充商品的片段。

    "哪个能退" 这种必须挡住：它会让"这个尺码"被判成限定词而放弃消解，
    也会把槽位 product 填成一句问句。商品名不可能以疑问词开头。
    """
    if len(name) < 2:
        return False
    if name[0] in _QUESTION_HEADS:
        return False
    return not any(name.startswith(w) for w in ATTR_WORDS)


def find_product(text: str) -> str | None:
    """文本里最后一个像商品名的片段。

    三条通道按可靠性排序：带量词的指代短语 > "买了个X" > 品类字后缀。
    最后一条最容易误判（"包邮"里的"包"），所以有额外的合理性过滤。
    """
    if not text:
        return None

    for pattern in (_PRODUCT_CUE_RE, _PRODUCT_BOUGHT_RE):
        for m in reversed(list(pattern.finditer(text))):
            name = _extract(m.group(1))
            if name:
                return name

    best: str | None = None
    best_pos = -1
    for i, ch in enumerate(text):
        if ch not in _PRODUCT_SUFFIXES:
            continue
        # 连续的品类字属于同一个词（"连衣裙"里的"衣"与"裙"），只从句尾那个字
        # 往前收集；否则会切出"衣"和"裙"两个半截词
        if i + 1 < len(text) and text[i + 1] in _PRODUCT_SUFFIXES:
            continue
        name = _trim_name(text[max(0, i - 5) : i]) + ch
        if _plausible(name) and i > best_pos:
            best, best_pos = name, i
    return best


# ============================================================
# 退款/退货原因
# ============================================================

_REASON_RE = re.compile(r"(?:因为|由于|原因(?:是|为)?[：:]?)\s*([^\n，,。？?！!；;]{2,20})")
_REASON_LEXICON: tuple[str, ...] = (
    "质量问题", "商品损坏", "破损", "坏了", "坏了", "有色差", "色差",
    "尺寸不合适", "尺码不合适", "拍错了", "买错了", "发错货", "少发", "漏发",
    "不想要了", "多拍了一件", "重复下单", "太贵", "和描述不符",
)


def find_reason(text: str) -> str | None:
    """优先抽"因为…"的显式说法；没有就取词典里命中的原话。

    两种都返回**用户原话片段**而不是我替他归纳的标签 —— 归因是人工审核和
    飞轮分析的事，NLU 层不该替他们下结论（编出来的原因会污染统计）。
    """
    if not text:
        return None
    hits = _REASON_RE.findall(text)
    if hits:
        return hits[-1].strip()
    for word in _REASON_LEXICON:
        if word in text:
            return word
    return None


# ============================================================
# 历史轮次的读取辅助
# ============================================================


def iter_turns(history: list[Turn] | None) -> Iterator[tuple[int, str, str]]:
    """把 history 规整成 (第几轮, 角色, 文本)，轮次从 1 开始。

    同时兼容 Turn 对象和裸 dict —— 反序列化过的会话（SQLite 里存的
    turns_json、API 传进来的 body）经常是 dict，为此崩掉不值得。
    """
    for i, turn in enumerate(history or [], 1):
        if isinstance(turn, dict):
            role = str(turn.get("role", ""))
            content = turn.get("content", "")
        else:
            role = str(getattr(turn, "role", ""))
            content = getattr(turn, "content", "")
        if isinstance(content, str) and content.strip():
            yield i, role, content


def nearest_entity(
    turns: list[tuple[int, str, str]], finder
) -> tuple[str | None, int | None]:
    """在历史里**从后往前**找实体，返回 (实体, 轮次)。

    从后往前是刻意的：指代消解的默认解永远是"最近一次提到的那个"。
    从前往后找会让用户第二次改口说的事被第一次的说法盖掉。
    """
    for idx, _role, text in reversed(turns):
        value = finder(text)
        if value:
            return value, idx
    return None, None


def turn_of(turns: list[tuple[int, str, str]], value: str) -> int | None:
    """反查某个实体值出现在第几轮（模型给出的补全结果也要能标引用来源）。"""
    for idx, _role, text in reversed(turns):
        if value and value in text:
            return idx
    return None
