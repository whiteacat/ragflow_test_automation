"""统一文本归一化与语境极性判定。

生成侧（metrics_generation）与检索侧（metrics_retrieval）共用同一套规则，
避免「18.84 MPa」与「18.84MPa」这类差异被判成不命中。

极性四态用于「否定先于命中」：
- warn：安全警示（严禁/禁止/不得…）→ 禁止项命中不算违规
- deny：事实被否定（不使用/不适用/并非…）→ 事实不算覆盖
- weak_neg：弱否定（不/未/无），不足以判定，按命中处理但可人工复核
- affirm：肯定陈述
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

UNIT_ALIASES = (("兆帕", "mpa"), ("千帕", "kpa"), ("兆瓦", "mw"),
                ("千瓦", "kw"), ("摄氏度", "°c"), ("公斤力", "kgf"))

CELL_SEP = "\u0001"

_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_SPACE_RE = re.compile(r"[\u00a0\u3000]")
_MARK_RE = re.compile(r"[*_`#]")
_THOUSANDS_RE = re.compile(r"(?<=\d)[,，](?=\d{3}(?!\d))")
_RANGE_RE = re.compile(r"(?<=\d)\s*[至到～—–]\s*(?=\d)")
_WS_RE = re.compile(r"\s+")
_SERIF_RE = re.compile(r"[的是为]")

NEG_SAFE = ("严禁", "禁止", "不得", "切勿", "避免", "绝不能", "不允许", "不准", "勿")
NEG_DENY = ("不适用", "未使用", "不使用", "并非", "不属于", "没有采用",
            "未采用", "不需要", "无需", "未曾使用")
NEG_WEAK = ("不", "未", "无", "非", "禁", "勿")

_CLAUSE_DELIMS = "，,；;。！？!?\n"


def normalize(s: str) -> str:
    """归一化：全半角、零宽、千分位、数字间范围符、表格竖线、单位别名、空白、系词。"""
    s = unicodedata.normalize("NFKC", s or "")
    s = _ZERO_WIDTH_RE.sub("", s)
    s = _SPACE_RE.sub("", s)
    s = _MARK_RE.sub("", s)                 # 去 Markdown 加粗/代码/标题符号
    s = _THOUSANDS_RE.sub("", s)            # 1,234 → 1234（小数逗号不受影响）
    s = _RANGE_RE.sub("~", s)               # 18.84至23.55 → 18.84~23.55（仅数字之间）
    s = s.replace("|", CELL_SEP).replace("｜", CELL_SEP)
    s = s.lower()
    for cn, en in UNIT_ALIASES:
        s = s.replace(cn, en)
    s = _WS_RE.sub("", s)
    s = _SERIF_RE.sub("", s)                # 双侧同去，防“压力为23.55”误判
    return s


def clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """返回命中位置所在分句的 [lo, hi) 区间（按中文标点/换行切）。"""
    lo = max([text.rfind(c, 0, start) for c in _CLAUSE_DELIMS] + [-1]) + 1
    his = [text.find(c, end) for c in _CLAUSE_DELIMS]
    hi = min([h for h in his if h >= 0], default=len(text))
    return lo, hi


def polarity(text: str, start: int, end: int, span: int = 15) -> str:
    """判定命中处的语境极性。只看同一分句内的前后 span 字符，跨句不算。"""
    lo, hi = clause_bounds(text, start, end)
    win = text[max(lo, start - span):min(hi, end + span)]
    if any(k in win for k in NEG_SAFE):
        return "warn"
    if any(k in win for k in NEG_DENY):
        return "deny"
    if any(k in win for k in NEG_WEAK):
        return "weak_neg"
    return "affirm"

UNIT_DIM = {
    "pa": ("pressure", Decimal("1")), "kpa": ("pressure", Decimal("1000")),
    "mpa": ("pressure", Decimal("1000000")), "bar": ("pressure", Decimal("100000")),
    "kgf/cm2": ("pressure", Decimal("98066.5")),
    "kg/cm2": ("pressure", Decimal("98066.5")),
    "v": ("voltage", Decimal("1")), "kv": ("voltage", Decimal("1000")),
    "a": ("current", Decimal("1")), "ma": ("current", Decimal("0.001")),
    "w": ("power", Decimal("1")), "kw": ("power", Decimal("1000")),
    "mw": ("power", Decimal("1000000")),
    "mm": ("length", Decimal("1")), "t/h": ("flow", Decimal("1")),
    "r/min": ("speed", Decimal("1")), "n·m": ("torque", Decimal("1")),
    "nm": ("torque", Decimal("1")),
    "°c": ("temperature", Decimal("1")), "%": ("percent", Decimal("1")),
}

_UNIT_ALT = ("kgf/cm2|kg/cm2|r/min|n·m|t/h|mpa|kpa|bar|kw|mw|kv|ma|mm|nm|pa|°c|%|w|v|a")
_NUM_UNIT_RE = re.compile(rf"(\d+(?:\.\d+)?)({_UNIT_ALT})?")
_RANGE_PAIR_RE = re.compile(rf"(\d+(?:\.\d+)?)~(\d+(?:\.\d+)?)({_UNIT_ALT})?")


@dataclass(frozen=True)
class Num:
    raw: str
    value: Decimal
    unit: str
    start: int


def numbers(text: str) -> list[Num]:
    out: list[Num] = []
    for m in _NUM_UNIT_RE.finditer(text):
        raw = m.group(1)
        try:
            value = Decimal(raw)
        except InvalidOperation:
            continue
        out.append(Num(raw, value, m.group(2) or "", m.start()))
    return out


def canon(n: Num) -> tuple[str, Decimal] | None:
    spec = UNIT_DIM.get(n.unit)
    if not spec:
        return None
    dim, factor = spec
    return dim, n.value * factor


def num_matches(fact: Num, answer: Num) -> bool:
    """完全一致：同量纲按基准单位精确比较，不做任何舍入或容差。"""
    cf, ca = canon(fact), canon(answer)
    if cf and ca:
        return cf[0] == ca[0] and cf[1] == ca[1]
    return fact.value == answer.value


def range_endpoint_hit(fact_norm: str, answer_numbers: list[Num]) -> bool | None:
    """事实给出 a~b 区间时，答案须精确等于某个端点才算覆盖（中间值不算）；无区间返回 None。"""
    m = _RANGE_PAIR_RE.search(fact_norm)
    if not m:
        return None
    unit = m.group(3) or ""
    spec = UNIT_DIM.get(unit)
    endpoints: list[tuple[str, Decimal]] = []
    for raw in (m.group(1), m.group(2)):
        value = Decimal(raw)
        endpoints.append((spec[0], value * spec[1]) if spec else ("", value))
    for n in answer_numbers:
        c = canon(n)
        if spec:
            if not c or c[0] != spec[0]:
                continue
            value = c[1]
        else:
            value = n.value
        if any(value == ep[1] for ep in endpoints):
            return True
    return False
