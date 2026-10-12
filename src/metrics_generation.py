"""规则可自动计算的生成/平台指标。

规范键见 docs/DATASET_SPEC.md：expected_answer_points（旧 required_facts 只读兼容）、
forbidden_claims（旧 forbidden_facts 只读兼容）、should_refuse（旧 must_abstain 只读兼容）。
"""
from __future__ import annotations

import re
import unicodedata
from statistics import mean
from typing import Dict, List, Optional

ABSTAIN_PATTERNS = [
    "无法回答", "没有找到", "未找到相关", "知识库中没有", "不确定", "建议咨询",
    "转人工", "无法确认", "证据不足", "抱歉，我无法", "cannot answer",
    "not found in", "insufficient evidence", "不知道",
]

CITATION_PATTERNS = [
    r"【[^】]{1,80}】", r"\[[^\]]{1,60}(pdf|docx|doc|条款|第.{1,10}条|页)\]",
    r"来源[：:]", r"引用[：:]", r"chunk[_-]?[a-z0-9]+", r"doc[_-]?[a-z0-9]+",
]


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\s+", "", s)
    return s.lower()


def fact_covered(fact: str, answer: str) -> bool:
    """单条 required_fact 是否被答案覆盖（规则版：归一化包含）。"""
    f, a = normalize_text(fact), normalize_text(answer)
    if not f:
        return True
    if f in a:
        return True
    # 长事实允许关键词级回退：分句后任一长度>=6的子句命中即算覆盖（防一刀切）
    parts = re.split(r"[；;，。,、]", fact)
    parts = [normalize_text(p) for p in parts if len(normalize_text(p)) >= 6]
    return any(p in a for p in parts)


def fact_recall(required_facts: List[str], answer: str) -> Optional[float]:
    """GEN-02 规则版。无 required_facts 返回 None（不计入）。"""
    if not required_facts:
        return None
    hits = sum(1 for f in required_facts if fact_covered(f, answer))
    return hits / len(required_facts)


def is_abstention(answer: str) -> bool:
    a = normalize_text(answer)
    return any(normalize_text(p) in a for p in ABSTAIN_PATTERNS)


def abstention_correct(answerable: bool, must_abstain: bool, answer: str) -> Optional[int]:
    """GEN-05：应拒答(must_abstain 或 answerable=False)→拒答了=1；应回答→未拒答=1。"""
    should_abstain = (not answerable) or must_abstain
    refused = is_abstention(answer)
    if should_abstain:
        return 1 if refused else 0
    return 1 if not refused else 0


def forbidden_hit(forbidden_facts: List[str], answer: str) -> Optional[int]:
    """GEN-09 规则版：出现任何禁止事项=1（失败）；无 forbidden 标注返回 None。"""
    if not forbidden_facts:
        return None
    a = normalize_text(answer)
    for f in forbidden_facts:
        if normalize_text(f) and normalize_text(f) in a:
            return 1
    return 0


def has_citation(answer: str) -> bool:
    return any(re.search(p, answer or "", re.IGNORECASE) for p in CITATION_PATTERNS)


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


UNIT_PATTERN = re.compile(r"(MPa|kPa|Pa|kV|V|A|mA|℃|°C|N·m|Nm|r/min|mm|m\b|t/h|MW|kW|%)")


def order_correct(required_facts: List[str], answer: str) -> Optional[int]:
    if len(required_facts) < 2:
        return None
    a = normalize_text(answer)
    pos = []
    for f in required_facts:
        f_n = normalize_text(f)
        idx = a.find(f_n) if f_n else -1
        if idx < 0:
            parts = [p for p in re.split(r"[；;，。,、]", f) if len(normalize_text(p)) >= 6]
            hits = [a.find(normalize_text(p)) for p in parts]
            hits = [h for h in hits if h >= 0]
            idx = min(hits) if hits else -1
        if idx < 0:
            return None
        pos.append(idx)
    return 1 if pos == sorted(pos) else 0


def unit_complete(required_facts: List[str], answer: str) -> Optional[int]:
    units = {u for f in required_facts for u in UNIT_PATTERN.findall(f or "")}
    if not units:
        return None
    a = normalize_text(answer)
    return 1 if all(normalize_text(u) in a for u in units) else 0


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
    abstained = is_abstention(answer)
    out: Dict[str, Optional[float]] = {
        "fact_recall_rule": fact_recall(required, answer),
        "abstention_correct": abstention_correct(
            bool(question.get("answerable", True)),
            bool(question.get("must_abstain", question.get("should_refuse", False))), answer),
        "forbidden_fact_hit": forbidden_hit(forbidden, answer),
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
