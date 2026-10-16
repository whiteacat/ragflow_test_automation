"""类 Ragas 四指标（规则版）：Context Precision / Context Recall /
Faithfulness / Answer Relevancy。

规则版只做可判定部分（归一化包含匹配），语义改写一律视为未覆盖——
数值天然保守，适合当下限基线；终值以 LLM 复核为准。
输入为 run_all 产出的 per_question.jsonl（附 golden 取证据与要点）。

用法：
    python scripts/ragas_like.py --per-question results/<exp>/per_question.jsonl \\
        --golden datasets/xlsx_golden.jsonl --k 5 --out /tmp/ragas.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics_generation import normalize_text  # noqa: E402
from src.metrics_retrieval import split_clauses  # noqa: E402

def char_bigrams(s: str) -> list:
    t = re.sub(r"[，、。；：|｜·—…？！“”‘’（）【】\[\]]", "", normalize_text(s))
    grams = [t[i:i + 2] for i in range(len(t) - 1)]
    return list(dict.fromkeys(grams))


def bigram_recall(hyp: str, ref_norm: str) -> float | None:
    grams = char_bigrams(hyp)
    if not grams:
        return None
    return sum(1 for g in grams if g in ref_norm) / len(grams)


def parse_snippet(snippet: str):
    chunks = []
    for line in (snippet or "").splitlines():
        m = re.match(r"\[(\\d+)\]\s*(.*?)#([^:]+):\s?(.*)$", line)
        if m:
            chunks.append({"rank": int(m.group(1)), "doc": m.group(2),
                           "chunk_id": m.group(3), "content": m.group(4)})
        elif line.strip():
            chunks.append({"rank": len(chunks) + 1, "doc": "",
                           "chunk_id": "", "content": line.strip()})
    return chunks


def context_precision(chunks, clauses, gold_ids, k: int):
    top = chunks[:k]
    if not top or not clauses:
        return None
    flags = []
    for c in top:
        hit = normalize_text(c["chunk_id"]) in gold_ids if c["chunk_id"] else False
        if not hit:
            cc = normalize_text(c["content"])
            hit = any(cl in cc for cl in clauses)
        flags.append(1 if hit else 0)
    if sum(flags) == 0:
        return 0.0
    precisions = [sum(flags[: i + 1]) / (i + 1) for i, v in enumerate(flags) if v]
    return sum(precisions) / len(precisions)


def context_recall(chunks, clauses) -> float | None:
    if not clauses:
        return None
    full = normalize_text("\n".join(c["content"] for c in chunks))
    return sum(1 for cl in clauses if cl in full) / len(clauses)


def faithfulness(answer: str, chunks) -> float | None:
    """答案每条断言相对检索内容的字二元组召回均值（规则版忠实度）。"""
    claims = [c for c in split_clauses(answer or "") if len(normalize_text(c)) >= 8]
    if not claims:
        return None
    full = normalize_text("\n".join(c["content"] for c in chunks))
    scores = [bigram_recall(c, full) for c in claims]
    scores = [s for s in scores if s is not None]
    return sum(scores) / len(scores) if scores else None


def answer_relevancy(question: str, answer: str) -> float | None:
    """题干字二元组在答案中的召回率（规则版相关度代理）。"""
    if not (answer or "").strip():
        return None
    return bigram_recall(question or "", normalize_text(answer))


def score_row(row: dict, golden: dict, k: int) -> dict:
    if row.get("run_error"):
        return {"question_id": row.get("question_id"),
                "context_precision": None, "context_recall": None,
                "faithfulness": None, "answer_relevancy": None,
                "note": "run_error"}
    chunks = parse_snippet(row.get("evidence_snippet", ""))
    clauses = [normalize_text(c) for c in
               split_clauses(golden.get("evidence_text", ""))]
    gold_ids = {normalize_text(c) for c in
                (golden.get("chunk_ids") or []) + (golden.get("acceptable_chunks") or [])
                + ([golden["expected_chunk"]] if golden.get("expected_chunk") else [])}
    gold_ids.discard("")
    answer = row.get("predicted_answer", "")
    return {
        "question_id": row.get("question_id"),
        "test_stage": row.get("test_stage"),
        "context_precision": context_precision(chunks, clauses, gold_ids, k),
        "context_recall": context_recall(chunks, clauses),
        "faithfulness": faithfulness(answer, chunks),
        "answer_relevancy": answer_relevancy(row.get("question", ""), answer),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-question", required=True)
    ap.add_argument("--golden", default="")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    gold = {}
    if args.golden and os.path.exists(args.golden):
        for line in open(args.golden, encoding="utf-8"):
            if line.strip():
                g = json.loads(line)
                gold[g.get("question_id")] = g
    rows = []
    for line in open(args.per_question, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append(score_row(r, gold.get(r.get("question_id"), {}), args.k))
    summary = {}
    for m in ("context_precision", "context_recall", "faithfulness", "answer_relevancy"):
        vals = [r[m] for r in rows if r[m] is not None]
        summary[m] = {"value": sum(vals) / len(vals) if vals else None, "n": len(vals)}
    summary["note"] = ("规则版：改写/同义一律计未覆盖，数值为下限；"
                       "answer_relevancy 为题干词覆盖代理，语义终值以 LLM 复核为准。")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
