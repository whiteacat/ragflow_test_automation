"""阈值校准：对已人工标注的样本做阈值扫描、指标评估与推荐。

纯函数，无 IO、无网络，供 scripts/calibrate_thresholds.py 与测试复用。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class Sample:
    """一条校准样本：score 为规则/语义给出的打分，label 为人工结论（1=应命中）。"""

    sample_id: str
    kind: str
    score: float
    label: int
    source: str = "rule"


def grid_points(start: float, stop: float, step: float) -> List[float]:
    """生成闭区间 [start, stop] 的等距阈值网格（含端点）。"""
    if step <= 0:
        raise ValueError("step 必须大于 0")
    if stop < start:
        raise ValueError("stop 必须大于等于 start")
    count = int(round((stop - start) / step))
    return [round(start + i * step, 4) for i in range(count + 1)]


def prf(tp: int, fp: int, fn: int) -> tuple:
    """precision / recall / F1；分母为 0 时返回 0。"""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def sweep(samples: Sequence[Sample], grid: Sequence[float]) -> List[Dict[str, Any]]:
    """对每个阈值统计混淆矩阵与 P/R/F1（判定规则：score >= threshold 记命中）。"""
    pairs = [(s.label, s.score) for s in samples]
    rows: List[Dict[str, Any]] = []
    for threshold in grid:
        tp = fp = fn = tn = 0
        for label, score in pairs:
            predicted = 1 if score >= threshold else 0
            if label == 1 and predicted == 1:
                tp += 1
            elif label == 0 and predicted == 1:
                fp += 1
            elif label == 1:
                fn += 1
            else:
                tn += 1
        precision, recall, f1 = prf(tp, fp, fn)
        rows.append({
            "threshold": threshold,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })
    return rows


def recommend(rows: Sequence[Dict[str, Any]], min_precision: float = 0.95,
              min_recall: float = 0.0) -> Optional[Dict[str, Any]]:
    """在满足 precision/recall 下限的阈值中取 F1 最高者（并列取更高阈值）。

    都不满足时退回全表 F1 最高并标注原因，提示样本量/标注不足。
    """
    if not rows:
        return None
    ok = [r for r in rows
          if r["precision"] >= min_precision and r["recall"] >= min_recall]
    if ok:
        best = max(ok, key=lambda r: (r["f1"], r["recall"], r["threshold"]))
        return {**best,
                "reason": f"precision≥{min_precision} 且 recall≥{min_recall} 中 F1 最高"}
    best = max(rows, key=lambda r: (r["f1"], r["precision"], r["threshold"]))
    return {**best,
            "reason": (f"无阈值同时满足 precision≥{min_precision}、recall≥{min_recall}，"
                       "退回 F1 最高；建议先扩充/复核样本再定阈值")}


def summarize_scores(samples: Sequence[Sample]) -> Dict[str, Any]:
    """按人工标签分组给出分数的分布，用于判断阈值区间是否可分。"""
    out: Dict[str, Any] = {"n": len(samples), "n_pos": 0, "n_neg": 0}
    for label, key in ((1, "pos"), (0, "neg")):
        values = sorted(s.score for s in samples if s.label == label)
        if label == 1:
            out["n_pos"] = len(values)
        else:
            out["n_neg"] = len(values)
        if values:
            out[f"{key}_min"] = round(values[0], 4)
            out[f"{key}_max"] = round(values[-1], 4)
            out[f"{key}_median"] = round(values[len(values) // 2], 4)
    return out
