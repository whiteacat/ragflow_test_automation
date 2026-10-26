#!/usr/bin/env python3
"""从跑批结果生成阈值校准候选样本：自动预标 + 待人工确认清单。

用法：
    # 1) 生成候选（fact/evidence/fuzzy 三类）
    python scripts/make_calibration_set.py \\
        --per-question results/<exp>/per_question.jsonl \\
        --out datasets/calibration_candidates.jsonl \\
        --review-out datasets/calibration_review.jsonl \\
        --max-per-kind 80 --cross-negatives 1

    # 2) 只改 label 为 null 的行（review 文件已按类分组），确认后
    python scripts/calibrate_thresholds.py --config config/test.yaml \\
        --data datasets/calibration_candidates.jsonl --kind all

    # 片段内容被 evidence_snippet 截断到 300 字时，可用完整 chunk 文件替换
    python scripts/make_calibration_set.py --per-question ... \\
        --chunks /tmp/chunk_candidates.jsonl --out ...

预标判据只使用与「待校准的语义/模糊阈值」无关的信号，避免自证：
    literal          字面命中（归一化后包含）                → label 1
    number_mismatch  数值完全对不上（十进制精确比较）        → label 0
    cross_question   与另一题（不同改写组/文档）配对          → label 0
    needs_review     规则无法判断（恰是需要人判的语义样本）  → label null

输出每行在 calibrate_thresholds.py 所需字段之外，附带：
    prelabel / basis / needs_review / question_id / question_type /
    paraphrase_group / source
label 为 null 的行会被 calibrate_thresholds.py 跳过并提示条数。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics_generation import fact_labels, normalize_text  # noqa: E402
from src.metrics_retrieval import bigram_recall, split_clauses  # noqa: E402
from src.textnorm import num_matches, numbers, range_endpoint_hit  # noqa: E402

_SNIPPET_HEAD = re.compile(r"(?m)^\[(\d+)\]\s*(.*?)#([^:\s]+):\s*")
KINDS = ("fact", "evidence", "fuzzy")
PRIORITY = {"needs_review": 0, "cross_question": 1, "number_mismatch": 2, "literal": 3}
TRUNCATED_LEN = 300


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_snippet(text: str) -> List[Dict[str, str]]:
    """解析 per_question.evidence_snippet（'[1] 文档#chunk: 内容' 逐行）。"""
    matches = list(_SNIPPET_HEAD.finditer(text or ""))
    out: List[Dict[str, str]] = []
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        out.append({
            "chunk_id": match.group(3),
            "document": match.group(2),
            "content": text[match.end():end].strip(),
        })
    return out


def load_chunks(path: str) -> Dict[str, List[Dict[str, str]]]:
    """读取完整 chunk：兼容 assist_chunk_mapping.py 的 candidates 结构。"""
    mapping: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in load_jsonl(path):
        qid = str(row.get("question_id") or "")
        for cand in row.get("candidates") or []:
            content = (cand.get("content") or "").strip()
            if not content:
                continue
            mapping[qid].append({
                "chunk_id": str(cand.get("chunk_id") or ""),
                "document": str(cand.get("document_name") or cand.get("document_id") or ""),
                "content": content,
            })
    return mapping


def prelabel_fact(fact: str, answer: str) -> Tuple[Optional[int], str]:
    """fact 预标：先比数值（精确），再比字面；都不定则交人工。"""
    f, a = normalize_text(fact), normalize_text(answer)
    f_nums = numbers(f)
    if f_nums:
        a_nums = numbers(a)
        number_ok = all(any(num_matches(fn, an) for an in a_nums) for fn in f_nums)
        if not number_ok and range_endpoint_hit(f, a_nums) is not True:
            return 0, "number_mismatch"
    labels = fact_labels(fact)
    if (f and f in a) or (labels and all(lbl in a for lbl in labels)):
        return 1, "literal"
    return None, "needs_review"


def prelabel_evidence(clause: str, chunk: str,
                      full_content: bool = True) -> Tuple[Optional[int], str]:
    """evidence 预标：字面 → 数字冲突 → 人工（不用二元组打分，避免与 fuzzy 目标循环论证）。"""
    c, ch = normalize_text(clause), normalize_text(chunk)
    if c and c in ch:
        return 1, "literal"
    if full_content and len(chunk) < TRUNCATED_LEN:
        c_nums, ch_nums = numbers(c), numbers(ch)
        if c_nums and ch_nums and not any(num_matches(cn, xn)
                                         for cn in c_nums for xn in ch_nums):
            return 0, "number_mismatch"
    return None, "needs_review"


def _row(kind: str, payload: Dict[str, Any], prelabel: Optional[int], basis: str,
         src: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "kind": kind,
        **payload,
        "label": prelabel,
        "prelabel": prelabel,
        "basis": basis,
        "needs_review": prelabel is None,
        "question_id": src.get("question_id", ""),
        "question_type": src.get("question_type", ""),
        "paraphrase_group": src.get("paraphrase_group", ""),
        "source": src.get("source", "per_question"),
    }


def build(rows: List[Dict[str, Any]], chunks: Optional[Dict[str, List[Dict[str, str]]]],
          kinds: List[str], cross_negatives: int, rng: random.Random
          ) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    fact_pool: List[Tuple[Dict[str, Any], str]] = []
    chunk_pool: List[Tuple[Dict[str, Any], Dict[str, str]]] = []

    for row in rows:
        qid = str(row.get("question_id") or "")
        answer = row.get("predicted_answer") or ""
        points = list(row.get("expected_answer_points") or [])
        points += [p for p in (row.get("mandatory_points") or []) if p not in points]

        if "fact" in kinds and answer:
            for point in points:
                label, basis = prelabel_fact(point, answer)
                out.append(_row("fact", {"fact": point, "answer": answer}, label, basis, row))
                fact_pool.append((row, answer))

        if not any(k in kinds for k in ("evidence", "fuzzy")):
            continue
        contents = (chunks or {}).get(qid) or parse_snippet(row.get("evidence_snippet") or "")
        full = bool((chunks or {}).get(qid))
        for clause in split_clauses(row.get("evidence_text") or ""):
            if not contents:
                break
            best = max(contents, key=lambda c: bigram_recall(normalize_text(clause),
                                                             normalize_text(c["content"])))
            label, basis = prelabel_evidence(clause, best["content"], full_content=full)
            payload = {"clause": clause, "chunk": best["content"], "chunk_id": best["chunk_id"]}
            if "evidence" in kinds:
                out.append(_row("evidence", {"evidence": clause, "chunk": best["content"],
                                             "chunk_id": best["chunk_id"]}, label, basis, row))
            if "fuzzy" in kinds:
                out.append(_row("fuzzy", payload, label, basis, row))
            chunk_pool.append((row, best))

    # 跨题负例：与另一题（不同改写组、不同文档）配对，天然应判不命中
    if cross_negatives > 0:
        if "fact" in kinds:
            for row, _answer in fact_pool:
                other = _pick_other(rows, row, rng)
                if other is None:
                    continue
                for point in (row.get("expected_answer_points") or [])[:cross_negatives]:
                    out.append(_row("fact", {"fact": point, "answer": other.get("predicted_answer") or ""},
                                    0, "cross_question", row))
        if chunk_pool:
            for row, _chunk in chunk_pool:
                other = _pick_other(rows, row, rng)
                if other is None:
                    continue
                other_chunks = (chunks or {}).get(str(other.get("question_id") or "")) \
                    or parse_snippet(other.get("evidence_snippet") or "")
                if not other_chunks:
                    continue
                foreign = rng.choice(other_chunks)
                for clause in split_clauses(row.get("evidence_text") or "")[:cross_negatives]:
                    payload = {"clause": clause, "chunk": foreign["content"],
                               "chunk_id": foreign["chunk_id"]}
                    if "evidence" in kinds:
                        out.append(_row("evidence", {"evidence": clause, "chunk": foreign["content"],
                                                     "chunk_id": foreign["chunk_id"]},
                                        0, "cross_question", row))
                    if "fuzzy" in kinds:
                        out.append(_row("fuzzy", payload, 0, "cross_question", row))
    deduped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for item in out:
        key = _dedupe_key(item)
        if key not in deduped:
            deduped[key] = item
    return list(deduped.values())


def _dedupe_key(item: Dict[str, Any]) -> Tuple[str, str, str]:
    """按 kind 取各自的文本字段去重（evidence 用 evidence，fuzzy 用 clause）。"""
    kind = item["kind"]
    if kind == "fact":
        return kind, normalize_text(item.get("fact", "")), normalize_text(item.get("answer", ""))
    left = item.get("evidence") if kind == "evidence" else item.get("clause")
    return kind, normalize_text(left or ""), normalize_text(item.get("chunk", ""))


def _pick_other(rows: List[Dict[str, Any]], row: Dict[str, Any],
                rng: random.Random) -> Optional[Dict[str, Any]]:
    group = row.get("paraphrase_group") or ""
    doc = row.get("expected_document") or ""
    candidates = [r for r in rows
                  if str(r.get("question_id")) != str(row.get("question_id"))
                  and r.get("predicted_answer")
                  and (not group or r.get("paraphrase_group") != group)
                  and (not doc or r.get("expected_document") != doc)]
    return rng.choice(candidates) if candidates else None


def select(items: List[Dict[str, Any]], max_per_kind: int,
           rng: random.Random) -> List[Dict[str, Any]]:
    """按 kind 限额，优先保留信息量大的样本（待人工 > 跨题负例 > 数值冲突 > 字面）。"""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[item["kind"]].append(item)
    picked: List[Dict[str, Any]] = []
    for kind, group in grouped.items():
        if max_per_kind > 0 and len(group) > max_per_kind:
            shuffled = group[:]
            rng.shuffle(shuffled)
            shuffled.sort(key=lambda r: PRIORITY.get(r["basis"], 9))
            group = shuffled[:max_per_kind]
        picked.extend(group)
    return picked


def stats(items: List[Dict[str, Any]]) -> str:
    table = Counter((i["kind"], i["basis"]) for i in items)
    lines = ["| kind | basis | 条数 |", "|---|---|---|"]
    for (kind, basis), count in sorted(table.items()):
        lines.append(f"| {kind} | {basis} | {count} |")
    review = sum(1 for i in items if i["needs_review"])
    lines.append(f"\n合计 {len(items)} 条，其中待人工确认 {review} 条")
    return "\n".join(lines)


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="生成阈值校准候选样本（自动预标 + 待确认清单）")
    ap.add_argument("--per-question", required=True, help="results/<exp>/per_question.jsonl")
    ap.add_argument("--chunks", help="可选：含 candidates[].content 的完整 chunk 文件")
    ap.add_argument("--out", default="datasets/calibration_candidates.jsonl")
    ap.add_argument("--review-out", default="datasets/calibration_review.jsonl")
    ap.add_argument("--kinds", default="fact,evidence,fuzzy")
    ap.add_argument("--max-per-kind", type=int, default=80, help="每类上限（0=不限）")
    ap.add_argument("--cross-negatives", type=int, default=1, help="每题构造几条跨题负例（0=不造）")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args(argv)

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip() in KINDS]
    if not kinds:
        ap.error(f"--kinds 只能是 {','.join(KINDS)} 的组合")
    if not os.path.exists(args.per_question):
        ap.error(f"文件不存在: {args.per_question}")

    rows = load_jsonl(args.per_question)
    chunks = load_chunks(args.chunks) if args.chunks else None
    rng = random.Random(args.seed)

    items = build(rows, chunks, kinds, args.cross_negatives, rng)
    items = select(items, args.max_per_kind, rng)
    review = [i for i in items if i["needs_review"]]

    write_jsonl(args.out, items)
    write_jsonl(args.review_out, review)
    print(stats(items))
    print(f"\n候选样本：{args.out}")
    print(f"待人工确认：{args.review_out}（把 label 的 null 改成 0/1 后即可用于校准）")
    if not chunks:
        print("\n[提示] 未提供 --chunks，证据片段来自 evidence_snippet（每条截断 300 字）："
              "数字冲突判据已自动跳过，跨分句命中的判断可能偏保守。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
