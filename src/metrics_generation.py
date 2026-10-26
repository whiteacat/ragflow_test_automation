"""规则可自动计算的生成/平台指标。

规范键见 docs/DATASET_SPEC.md：expected_answer_points（旧 required_facts 只读兼容）、
forbidden_claims（旧 forbidden_facts 只读兼容）、should_refuse（旧 must_abstain 只读兼容）。
"""
from __future__ import annotations

import re
from statistics import mean
from typing import Dict, List, Optional

from . import embeddings, synonyms
from .textnorm import (
    CELL_SEP,
    normalize,
    num_matches,
    numbers,
    polarity,
    range_endpoint_hit,
)

ABSTAIN_PATTERNS = [
    "无法回答", "没有找到", "未找到相关", "知识库中没有", "不确定", "建议咨询",
    "转人工", "无法确认", "证据不足", "抱歉，我无法", "cannot answer",
    "not found in", "insufficient evidence", "不知道",
    "未提及", "资料中未", "暂无相关", "暂无资料", "无法提供", "超出知识范围",
    "建议联系", "需咨询", "暂无法", "no information", "not mentioned",
    "cannot provide", "unable to answer",
]

# 计量术语（不确定度/不确定因素）不是拒答，此前被「不确定」子串误伤
ABSTAIN_MISFIRE_RE = re.compile(r"不确定(度|性|因素|区间|分析|来源|评定|原理)")

ABSTAIN_LONG_CHARS = 200

STRONG_CITATION_PATTERNS = [
    r"【[^】]{1,80}】",
    r"《[^》]{1,80}》",
    r"[\w\-]+\.(?:pdf|docx?|xlsx?)",
    r"\[[^\]]{1,60}(pdf|docx|doc|条款|第.{1,10}条|页)\]",
    r"来源[：:]", r"引用[：:]",
    r"chunk[_-]?[a-z0-9]+", r"doc[_-]?[a-z0-9]+",
    r"\[ID[:：\s]*\d+\]",
    r"(?:依据|根据|见|详见|参见)[^。；\n]{0,20}第?\d+(?:\.\d+)*(?:条|款|章|节)",
    r"(?:GB|DL|NB)/?T?\s*\d{2,}",
]

WEAK_CITATION_PATTERNS = [
    r"\[\^?\d{1,3}(?:[,，\-–]\s*\d{1,3})*\]",
    r"[（(]\d{1,2}[)）]",
]

CITATION_TAIL_CHARS = 200


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_UNIT_RE = re.compile(r"(mpa|kpa|kgf|kn|kv|kw|mw|ma|r/min|t/h|n·m|nm|mm|℃|%|pa|v)")
_FILLER_RE = re.compile(r"(两值均适用于|以上均适用于|均适用于|两值均|均适用|适用于|两值|均相同|相同|一致)")
_LABEL_SPLIT = re.compile(r"[、，,；;：:/\u0001与和及~]")
_CELL_SEP = CELL_SEP


def normalize_text(s: str) -> str:
    return normalize(s)


def _labels(text: str) -> List[str]:
    t = _UNIT_RE.sub("", _NUM_RE.sub("", text))
    t = _FILLER_RE.sub("、", t)
    return [x for x in _LABEL_SPLIT.split(t) if len(x) >= 2]


def fact_labels(text: str) -> List[str]:
    """事实中的标签词（去掉数值与单位），供校准/调试复用同一口径。"""
    return _labels(normalize_text(text))


def _spans(text: str, tokens: List[str]) -> List[tuple[int, int]]:
    out = []
    for t in tokens:
        if not t:
            continue
        i = text.find(t)
        if i >= 0:
            out.append((i, i + len(t)))
    return out


PROXIMITY_WINDOW = 40          # 标签与数值的邻近窗口（字符）
CREDIT_EXACT = 1.0             # 数值 + 标签（字面/同义词）命中且相邻
CREDIT_NEAR = 0.8              # 数值命中，标签仅核心词/语义近似
CREDIT_UNCERTAIN = 0.5         # 标签命中但数值缺失或离得太远
FACT_COVERED_CREDIT = CREDIT_NEAR   # 严格口径：≥0.8 才算覆盖


def _numbers_covered(fact_norm: str, f_nums, a_nums) -> bool:
    if not a_nums:
        return False
    if all(any(num_matches(fn, an) for an in a_nums) for fn in f_nums):
        return True
    return range_endpoint_hit(fact_norm, a_nums) is True


def _positions(text: str, tokens: List[str]) -> List[int]:
    out: List[int] = []
    for token in tokens:
        if not token:
            continue
        start = 0
        while True:
            idx = text.find(token, start)
            if idx < 0:
                break
            out.append(idx)
            start = idx + 1
    return out


def _label_levels(labels: List[str], answer: str, answer_norm: str) -> List[int]:
    """每个标签的命中等级：2=字面/同义词，1=核心词共现，0=语义，-1=未命中。"""
    levels: List[int] = []
    semantic_threshold = (embeddings.settings().fact_label_threshold
                          if embeddings.enabled() else None)
    sentences = embeddings.sentences(answer) if semantic_threshold is not None else []
    for label in labels:
        if any(v in answer_norm for v in synonyms.variants(label)):
            levels.append(2)
            continue
        cores = synonyms.core(label)
        if cores and all(c in answer_norm for c in cores):
            levels.append(1)
            continue
        if semantic_threshold is not None and sentences:
            sim = embeddings.best_similarity(label, sentences)
            if sim is not None and sim >= semantic_threshold:
                levels.append(0)
                continue
        levels.append(-1)
    return levels


def fact_credit(fact: str, answer: str) -> float:
    """单条事实的覆盖分档：1.0 / 0.8 / 0.5 / 0。

    - 1.0：数值(若有)全部精确命中 + 标签字面或同义词命中，且数值与标签在
      ``PROXIMITY_WINDOW`` 字内共现
    - 0.8：数值命中，但标签只靠核心词/语义近似
    - 0.5：标签命中，数值缺失或离得太远（计入不确定，进复核而非直接判错）
    - 0：标签未命中，或被否定语境（“不使用/不适用”）否定
    """
    f, a = normalize_text(fact), normalize_text(answer)
    if not f:
        return CREDIT_EXACT
    idx = a.find(f)
    if idx >= 0:
        return 0.0 if polarity(a, idx, idx + len(f)) == "deny" else CREDIT_EXACT
    labels = _labels(f)
    f_nums = numbers(f)
    a_nums = numbers(a)
    levels = _label_levels(labels, answer, a) if labels else []
    if any(level < 0 for level in levels):
        return 0.0
    if f_nums and not _numbers_covered(f, f_nums, a_nums):
        return CREDIT_UNCERTAIN if labels else 0.0
    if f_nums and labels:
        variants = [v for label in labels for v in synonyms.variants(label)]
        label_pos = _positions(a, variants)
        num_pos = _positions(a, [n.raw for n in f_nums])
        if label_pos and num_pos and not all(
                min(abs(num - lab) for lab in label_pos) <= PROXIMITY_WINDOW
                for num in num_pos):
            return CREDIT_UNCERTAIN
    spans = _spans(a, labels + [n.raw for n in f_nums])
    if any(polarity(a, s, e) == "deny" for s, e in spans):
        return 0.0
    if not labels:
        return CREDIT_EXACT if f_nums else 0.0
    return CREDIT_NEAR if any(level <= 1 for level in levels) else CREDIT_EXACT


def fact_covered(fact: str, answer: str) -> bool:
    """严格口径：分档分 ≥0.8 才算覆盖（数值缺失或离得太远记 0.5，不算覆盖）。"""
    return fact_credit(fact, answer) >= FACT_COVERED_CREDIT


def fact_uncertain(fact: str, answer: str) -> bool:
    """标签命中但数值存疑（缺失/离得太远），应进复核队列而非直接判错。"""
    return abs(fact_credit(fact, answer) - CREDIT_UNCERTAIN) < 1e-9


def fact_recall(required_facts: List[str], answer: str) -> Optional[float]:
    """GEN-02 严格口径（二值）。无 required_facts 返回 None（不计入）。"""
    if not required_facts:
        return None
    hits = sum(1 for f in required_facts if fact_covered(f, answer))
    return hits / len(required_facts)


def fact_recall_relaxed(required_facts: List[str], answer: str) -> Optional[float]:
    """GEN-02 放宽口径（分档加权）：1.0/0.8/0.5/0，供与严格口径对照。"""
    if not required_facts:
        return None
    return sum(fact_credit(f, answer) for f in required_facts) / len(required_facts)


def fact_uncertain_count(required_facts: List[str], answer: str) -> int:
    """需人工复核的事实条数（标签命中但数值存疑）。"""
    return sum(1 for f in required_facts if fact_uncertain(f, answer))


ABSTAIN_HEAD_CHARS = 60
ABSTAIN_SHORT_CHARS = 80


def abstention_state(answer: str) -> Optional[int]:
    """三态拒答：1=拒答，0=作答，None=长答案开头拒答后继续作答（转人工复核）。"""
    a = normalize_text(answer)
    if not a:
        return None
    if ABSTAIN_MISFIRE_RE.search(a):
        return 0
    head = a if len(a) <= ABSTAIN_SHORT_CHARS else a[:ABSTAIN_HEAD_CHARS]
    if not any(normalize_text(p) in head for p in ABSTAIN_PATTERNS):
        return 0
    return 1 if len(a) <= ABSTAIN_LONG_CHARS else None


def is_abstention(answer: str) -> bool:
    return abstention_state(answer) == 1


def abstention_correct(answerable: bool, must_abstain: bool, answer: str) -> Optional[int]:
    """GEN-05：应拒答(must_abstain 或 answerable=False)→拒答了=1；应回答→未拒答=1。"""
    state = abstention_state(answer)
    if state is None:
        return None
    should_abstain = (not answerable) or must_abstain
    refused = bool(state)
    if should_abstain:
        return 1 if refused else 0
    return 1 if not refused else 0


def forbidden_scan(forbidden_facts: List[str], answer: str) -> Optional[Dict[str, List[str]]]:
    """区分「真违规(hard)」与「安全警示/否定软命中(soft)」。

    “严禁按 5.7MPa 执行”是正确警示，不应触发一票否决。
    """
    if not forbidden_facts:
        return None
    a = normalize_text(answer)
    hard: List[str] = []
    soft: List[str] = []
    for f in forbidden_facts:
        fn = normalize_text(f)
        if not fn:
            continue
        i = a.find(fn)
        if i < 0:
            continue
        if polarity(a, i, i + len(fn)) == "affirm":
            hard.append(f)
        else:
            soft.append(f)
    return {"hard": hard, "soft": soft}


def forbidden_hit(forbidden_facts: List[str], answer: str) -> Optional[int]:
    """GEN-09 规则版：真违规=1；安全警示不算违规；无 forbidden 标注返回 None。"""
    scan = forbidden_scan(forbidden_facts, answer)
    if scan is None:
        return None
    return 1 if scan["hard"] else 0


def has_citation(answer: str) -> bool:
    text = answer or ""
    if any(re.search(p, text, re.IGNORECASE) for p in STRONG_CITATION_PATTERNS):
        return True
    tail = text[-CITATION_TAIL_CHARS:]
    return any(re.search(p, tail) for p in WEAK_CITATION_PATTERNS)


def citation_complete(must_cite: bool, answer: str, answerable: bool = True,
                      abstained: bool = False) -> Optional[int]:
    """GEN-10：仅 must_cite=True 且应回答的题计入；有引用标记=1。"""
    if not must_cite or not answerable or abstained:
        return None
    return 1 if has_citation(answer) else 0


def percentile(data: List[float], q: float) -> Optional[float]:
    """q in [0,100]，线性插值；空返回 None（不依赖 numpy）。"""
    if not data:
        return None
    s = sorted(data)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * q / 100.0
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def latency_stats(e2e_ms: List[float]) -> Dict[str, Optional[float]]:
    return {
        "count": float(len(e2e_ms)),
        "p50": percentile(e2e_ms, 50),
        "p95": percentile(e2e_ms, 95),
        "mean": float(mean(e2e_ms)) if e2e_ms else None,
        "max": float(max(e2e_ms)) if e2e_ms else None,
    }


UNIT_PATTERN = re.compile(r"(MPa|kPa|Pa|kV|V|A|mA|℃|°C|N·m|Nm|r/min|mm|t/h|MW|kW|%)")


def _fact_anchor(answer_norm: str, fact_norm: str, labels: List[str], nums) -> int:
    """事实在答案中的锚点：优先整串连续段，其次「数值+标签」40 字内聚簇，最后单侧。

    只取最早出现的**连续段**，避免开头「如前所述，水压试验…」的顺带提及
    把位置拉到正文之前，造成逆序误判（O2）。
    """
    if fact_norm:
        idx = answer_norm.find(fact_norm)
        if idx >= 0:
            return idx
    label_pos = _positions(answer_norm, labels)
    num_pos = _positions(answer_norm, [n.raw for n in nums])
    if label_pos and num_pos:
        starts = []
        for num in num_pos:
            nearest = min(label_pos, key=lambda lab: abs(lab - num))
            if abs(nearest - num) <= PROXIMITY_WINDOW:
                starts.append(min(num, nearest))
        if starts:
            return min(starts)
    if label_pos:
        return min(label_pos)
    if num_pos:
        return min(num_pos)
    return -1


def order_correct(required_facts: List[str], answer: str) -> Optional[int]:
    """GEN-ORDER：事实锚点单调不减 → 1。

    事实只有标签命中（数值缺失）时用标签位置代替，不再整题返回 None（O1）。
    """
    if len(required_facts) < 2:
        return None
    a = normalize_text(answer)
    pos: List[int] = []
    for f in required_facts:
        fn = normalize_text(f)
        idx = _fact_anchor(a, fn, _labels(fn), numbers(fn))
        if idx < 0:
            return None
        pos.append(idx)
    return 1 if pos == sorted(pos) else 0


def _fact_present(fact: str, answer: str) -> bool:
    """事实是否在答案中出现：整串 / 标签全中 / 数值全中，三者任一即可。"""
    fn, a = normalize_text(fact), normalize_text(answer)
    if not fn:
        return True
    if fn in a:
        return True
    labels = _labels(fn)
    if labels and all(label in a for label in labels):
        return True
    f_nums = numbers(fn)
    a_nums = numbers(a)
    return bool(f_nums) and all(any(num_matches(n, an) for an in a_nums) for n in f_nums)


def unit_complete(required_facts: List[str], answer: str) -> Optional[int]:
    """GEN-UNIT：只统计确实出现的事实的单位，整条缺失的事实不拉低该项（O1）。"""
    units = set()
    for f in required_facts:
        found = {u for u in UNIT_PATTERN.findall(f or "")}
        if not found:
            continue
        if not _fact_present(f, answer):
            continue
        units |= {normalize_text(u) for u in found}
    if not units:
        return None
    a = normalize_text(answer)
    return 1 if all(u in a for u in units) else 0


def mandatory_recall(mandatory_points: List[str], answer: str) -> Optional[float]:
    if not mandatory_points:
        return None
    return fact_recall(mandatory_points, answer)


def score_generation_rule(question: dict, answer: str) -> Dict[str, Optional[float]]:
    required = (question.get("expected_answer_points")
                or question.get("required_facts") or [])
    mandatory = question.get("mandatory_points") or []
    forbidden = (question.get("forbidden_claims")
                 or question.get("forbidden_facts") or [])
    state = abstention_state(answer)
    abstained = state == 1
    scan = forbidden_scan(forbidden, answer)
    out: Dict[str, Optional[float]] = {
        "fact_recall_rule": fact_recall(required, answer),
        "fact_recall_relaxed": fact_recall_relaxed(required, answer),
        "fact_uncertain_count": None if not required else fact_uncertain_count(required, answer),
        "abstention_correct": abstention_correct(
            bool(question.get("answerable", True)),
            bool(question.get("must_abstain", question.get("should_refuse", False))), answer),
        "abstention_state": state,
        "forbidden_fact_hit": None if scan is None else (1 if scan["hard"] else 0),
        "forbidden_soft": None if scan is None else (1 if scan["soft"] else 0),
        "citation_complete_rule": citation_complete(
            bool(question.get("must_cite", True)), answer,
            bool(question.get("answerable", True)), abstained),
        "abstained": 1 if abstained else 0,
    }
    if mandatory:
        out["mandatory_recall_rule"] = mandatory_recall(mandatory, answer)
    if question.get("requires_order"):
        out["order_correct_rule"] = order_correct(required, answer)
    if question.get("requires_unit"):
        out["unit_complete_rule"] = unit_complete(required, answer)
    return out
