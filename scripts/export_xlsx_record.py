"""把一次运行结果写回 xlsx 执行记录（填写说明步骤 5）。

默认写副本不破坏原表；--in-place 才覆盖原文件（请先备份）。
用例编号直接对 question_id；测试环节由 test_stage 回映射，
qa 环节按 must_cite 区分回答/引用。

用法：
    python scripts/export_xlsx_record.py --xlsx 企业RAG全环节测试集_600MW.xlsx \\
        --per-question results/<exp>/per_question.jsonl \\
        --summary results/<exp>/summary.json --batch EXP-001 --tester 测试员
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

STAGE_CN = {"parse": "原文解析", "chunk": "切分", "retrieval": "召回",
            "rerank": "重排", "qa": "回答", "refusal": "拒答"}

ATTRIB_CN = {"source": "原文问题", "parse": "解析问题", "chunk": "切块问题",
             "index": "索引问题", "recall": "召回问题", "rerank": "重排问题",
             "generate": "生成问题", "policy": "策略问题"}


def verdict(row: dict, veto_ids: set) -> str:
    if row.get("question_id") in veto_ids or row.get("forbidden_fact_hit") == 1:
        return "不通过"
    if row.get("abstention_correct") == 0:
        return "不通过"
    if row.get("failure_attribution") or row.get("needs_chunk_mapping"):
        return "待复核"
    return "通过"


def metrics_str(row: dict) -> str:
    parts = []
    for k in ("Top@5", "Top@10", "MRR", "NDCG@10", "hit_rank",
              "fact_recall_rule", "abstention_correct", "citation_complete_rule",
              "order_correct_rule", "unit_complete_rule", "rerank_gain"):
        v = row.get(k)
        if v is not None:
            parts.append(f"{k}={v}")
    return ";".join(parts)[:500]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--per-question", required=True)
    ap.add_argument("--summary", default="")
    ap.add_argument("--batch", default="")
    ap.add_argument("--tester", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--in-place", action="store_true")
    args = ap.parse_args()
    try:
        import openpyxl  # noqa: E402
    except ImportError:
        print("需安装 openpyxl", file=sys.stderr)
        sys.exit(2)

    rows = [json.loads(l) for l in open(args.per_question, encoding="utf-8")
            if l.strip()]
    veto_ids: set = set()
    cfg_desc = ""
    if args.summary and os.path.exists(args.summary):
        s = json.load(open(args.summary, encoding="utf-8"))
        veto_ids = {v["question_id"] for v in
                    s.get("metrics", {}).get("_vetos", [])}
        r = s.get("retrieval") or {}
        ds = s.get("dataset") or {}
        cfg_desc = (f"top_k={r.get('top_k')};similarity={r.get('similarity_threshold')};"
                    f"rerank={r.get('rerank_id')};dataset={ds.get('path')}")
    batch = args.batch or datetime.date.today().strftime("BATCH-%Y%m%d")
    day = datetime.date.today().isoformat()

    wb = openpyxl.load_workbook(args.xlsx)
    ws = wb["执行记录"]
    last = 3
    for i in range(4, ws.max_row + 1):
        if ws.cell(row=i, column=2).value:
            last = i
    for r in rows:
        stage = STAGE_CN.get(str(r.get("test_stage", "")), "")
        if r.get("test_stage") == "qa" and r.get("must_cite"):
            stage = "引用"
        ws.append([batch, r.get("question_id"), stage, day, cfg_desc,
                   (r.get("predicted_answer") or "")[:800] or
                   (f"Top@5={r.get('Top@5')}" if r.get("Top@5") is not None else ""),
                   (r.get("evidence_snippet") or "")[:800],
                   r.get("hit_rank"), metrics_str(r), verdict(r, veto_ids),
                   ATTRIB_CN.get(str(r.get("failure_attribution") or ""), ""),
                   args.tester, r.get("source_ref", "")])
    out = args.xlsx if args.in_place else (
        args.out or os.path.join(os.path.dirname(os.path.abspath(args.xlsx)),
                                 f"{os.path.splitext(os.path.basename(args.xlsx))[0]}"
                                 f"_执行记录_{batch}.xlsx"))
    wb.save(out)
    print(f"写入 {len(rows)} 条执行记录（自 R{last + 1} 起）→ {out}")


if __name__ == "__main__":
    main()
