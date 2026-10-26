#!/usr/bin/env python3
"""阈值校准：用人工标注样本扫描 fact / evidence / fuzzy 三类阈值，
输出 precision / recall / F1 与推荐阈值。

标注格式（jsonl，一行一条，label 1=人工判断应命中，0=不应命中）：
    {"kind":"fact","fact":"给水泵出口压力0.6MPa","answer":"泵出口压力为0.6MPa","label":1}
    {"kind":"evidence","evidence":"给水泵出口压力不低于0.6MPa","chunk":"泵的出口压力须保持在0.6MPa以上","label":0}
    {"kind":"fuzzy","clause":"给水泵出口压力不低于0.6MPa","chunk":"泵出口压力保持0.6MPa以上","label":1}
建议：fact/evidence 样本每条只放一个要点/一个分句，标注才可解释。

用法：
    python scripts/calibrate_thresholds.py --emit-template
    python scripts/calibrate_thresholds.py --config config/test.yaml \\
        --data datasets/calibration.jsonl --kind all \\
        --grid 0.60:0.95:0.05 --min-precision 0.95 --out results/calibration

口径说明：
    fact     —— 先跑纯规则（数值+标签，关闭 embedding）；规则未命中时才取语义相似度，
                生产口径要求所有标签都达阈值，故取标签相似度最小值。
    evidence —— 分句先走字面/二元组（阈值 CLAUSE_FUZZY_THRESHOLD）；未命中才取语义相似度；
                全部分句覆盖才算命中，故取分句最小值。
    fuzzy    —— 只算字二元组召回率，用于校准 CLAUSE_FUZZY_THRESHOLD。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import embeddings  # noqa: E402
from src.calibration import (  # noqa: E402
    Sample,
    grid_points,
    recommend,
    summarize_scores,
    sweep,
)
from src.config import load_config  # noqa: E402
from src.metrics_generation import fact_covered, fact_labels, normalize_text  # noqa: E402
from src.metrics_retrieval import (  # noqa: E402
    bigram_recall,
    clause_hit,
    split_clauses,
)

TEMPLATE = [
    '{"kind":"fact","fact":"给水泵出口压力0.6MPa","answer":"泵出口压力为0.6MPa","label":1}',
    '{"kind":"fact","fact":"水压试验压力18.84MPa","answer":"水压试验压力为18.8MPa","label":0}',
    '{"kind":"evidence","evidence":"给水泵出口压力不低于0.6MPa",'
    '"chunk":"泵的出口压力须保持在0.6MPa以上","label":1}',
    '{"kind":"fuzzy","clause":"给水泵出口压力不低于0.6MPa",'
    '"chunk":"泵出口压力保持0.6MPa以上","label":0}',
]
KINDS = ("fact", "evidence", "fuzzy")


def _parse_grid(spec: str) -> List[float]:
    try:
        start, stop, step = (float(x) for x in spec.split(":"))
    except ValueError:
        raise SystemExit(f"--grid 格式应为 start:stop:step，收到 {spec!r}")
    return grid_points(start, stop, step)


def load_records(path: str, kinds: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {k: [] for k in kinds}
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] 第 {line_no} 行不是合法 JSON，已跳过")
                skipped += 1
                continue
            kind = rec.get("kind")
            if kind not in kinds:
                continue
            label = rec.get("label")
            if label not in (0, 1):
                print(f"[warn] 第 {line_no} 行 label 必须是 0/1，已跳过")
                skipped += 1
                continue
            rec.setdefault("id", f"L{line_no}")
            grouped[kind].append(rec)
    if skipped:
        print(f"[info] 跳过 {skipped} 条")
    return grouped


def _require_embeddings() -> None:
    if not embeddings.enabled():
        raise SystemExit(
            "embedding 未启用：请在 config 中设置 embedding.enabled=true 并填好 base_url"
            f"（当前 last_error: {embeddings.last_error() or '未发起请求'}）"
        )


def _score_fact(records: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Sample]:
    embeddings.reset()                      # 纯规则口径
    rule_hits = [fact_covered(r.get("fact", ""), r.get("answer", "")) for r in records]
    embeddings.configure(cfg.get("embedding"))
    if not all(rule_hits):
        _require_embeddings()
    out: List[Sample] = []
    for rec, hit in zip(records, rule_hits):
        if hit:
            out.append(Sample(rec["id"], "fact", 1.0, rec["label"], "rule"))
            continue
        labels = fact_labels(rec.get("fact", "")) or [normalize_text(rec.get("fact", ""))]
        sentences = embeddings.sentences(rec.get("answer", ""))
        sims = [embeddings.best_similarity(lbl, sentences) for lbl in labels]
        if any(s is None for s in sims):
            raise SystemExit(f"语义请求失败（样本 {rec['id']}）: {embeddings.last_error()}")
        out.append(Sample(rec["id"], "fact", min(sims), rec["label"], "semantic"))
    return out


def _score_evidence(records: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Sample]:
    out: List[Sample] = []
    for rec in records:
        evidence = rec.get("evidence", "")
        chunk = rec.get("chunk", "")
        clauses = split_clauses(evidence) or [evidence]
        full = normalize_text(chunk)
        values: List[Optional[float]] = []
        for clause in clauses:
            normalized = normalize_text(clause)
            values.append(1.0 if clause_hit(normalized, full) else None)
        if any(v is None for v in values):
            _require_embeddings()
            sentences = embeddings.sentences(chunk) or [chunk]
            for idx, value in enumerate(values):
                if value is not None:
                    continue
                sim = embeddings.best_similarity(normalize_text(clauses[idx]), sentences)
                if sim is None:
                    raise SystemExit(
                        f"语义请求失败（样本 {rec['id']}）: {embeddings.last_error()}"
                    )
                values[idx] = sim
        out.append(Sample(rec["id"], "evidence", min(values), rec["label"], "semantic"))
    return out


def _score_fuzzy(records: List[Dict[str, Any]]) -> List[Sample]:
    out: List[Sample] = []
    for rec in records:
        score = bigram_recall(normalize_text(rec.get("clause", "")),
                              normalize_text(rec.get("chunk", "")))
        out.append(Sample(rec["id"], "fuzzy", score, rec["label"], "bigram"))
    return out


def _print_table(kind: str, rows: List[Dict[str, Any]], chosen: Dict[str, Any]) -> None:
    print(f"\n### {kind}（阈值点 {len(rows)} 个）")
    print("| 阈值 | TP | FP | FN | TN | precision | recall | F1 |")
    print("|---|---|---|---|---|---|---|---|")
    for row in rows:
        mark = " ✅" if row["threshold"] == chosen["threshold"] else ""
        print(f"| {row['threshold']:.2f}{mark} | {row['tp']} | {row['fp']} | {row['fn']} | "
              f"{row['tn']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |")
    print(f"推荐阈值 **{chosen['threshold']:.2f}**：{chosen['reason']}；"
          f"precision={chosen['precision']:.3f}, recall={chosen['recall']:.3f}, "
          f"F1={chosen['f1']:.3f}")


def _write_json(path: str, kind: str, samples: List[Sample], rows: List[Dict[str, Any]],
                chosen: Dict[str, Any], grid: List[float]) -> None:
    payload = {
        "kind": kind,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "n": len(samples),
        "summary": summarize_scores(samples),
        "grid": grid,
        "rows": rows,
        "recommended": chosen,
        "samples": [
            {"id": s.sample_id, "score": round(s.score, 4), "label": s.label,
             "source": s.source}
            for s in samples
        ],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="阈值校准（fact / evidence / fuzzy）")
    ap.add_argument("--config", help="config/test.yaml（fact/evidence 需 embedding.enabled=true）")
    ap.add_argument("--data", help="人工标注样本 jsonl")
    ap.add_argument("--kind", default="all", choices=[*KINDS, "all"])
    ap.add_argument("--grid", default="0.60:0.95:0.05", help="阈值网格 start:stop:step")
    ap.add_argument("--min-precision", type=float, default=0.95)
    ap.add_argument("--min-recall", type=float, default=0.0)
    ap.add_argument("--out", default="results/calibration", help="JSON 输出目录")
    ap.add_argument("--emit-template", action="store_true", help="打印标注样例后退出")
    ap.add_argument("--limit", type=int, default=0, help="每类最多用多少条（0=全部）")
    args = ap.parse_args(argv)

    if args.emit_template:
        print("\n".join(TEMPLATE))
        return 0
    if not args.data:
        ap.error("需要 --data；不确定格式时先跑 --emit-template")
    if not os.path.exists(args.data):
        ap.error(f"样本文件不存在: {args.data}")

    kinds = list(KINDS) if args.kind == "all" else [args.kind]
    cfg: Dict[str, Any] = {}
    if args.config and os.path.exists(args.config):
        cfg = load_config(args.config)

    grouped = load_records(args.data, kinds)
    if args.limit > 0:
        grouped = {k: v[:args.limit] for k, v in grouped.items()}

    grid = _parse_grid(args.grid)
    produced: Dict[str, Any] = {}
    for kind in kinds:
        records = grouped.get(kind) or []
        if not records:
            print(f"[info] 无 {kind} 样本，跳过")
            continue
        if kind == "fact":
            samples = _score_fact(records, cfg)
        elif kind == "evidence":
            samples = _score_evidence(records, cfg)
        else:
            samples = _score_fuzzy(records)
        rows = sweep(samples, grid)
        chosen = recommend(rows, args.min_precision, args.min_recall)
        stats = summarize_scores(samples)
        print(f"\n## {kind} 样本分布: {json.dumps(stats, ensure_ascii=False)}")
        _print_table(kind, rows, chosen)
        out_path = os.path.join(args.out, f"{kind}_calibration.json")
        _write_json(out_path, kind, samples, rows, chosen, grid)
        print(f"明细已写入 {out_path}")
        produced[kind] = chosen["threshold"]

    if not produced:
        print("[error] 没有任何样本被评估，请检查 --data 与 --kind")
        return 1
    print("\n建议写入配置：")
    for kind, threshold in produced.items():
        key = {"fact": "fact_label_threshold", "evidence": "evidence_threshold",
               "fuzzy": "CLAUSE_FUZZY_THRESHOLD（代码常量，需改 src/metrics_retrieval.py）"}[kind]
        print(f"  {kind:8s} → {key}: {threshold:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
