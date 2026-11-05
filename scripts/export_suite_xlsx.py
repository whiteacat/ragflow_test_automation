"""把一次测试运行的全部产物汇成一个 xlsx（测试主线的唯一交付物）。

覆盖 `results/<exp>/` 下的 per_question / summary / parse_summary /
review_queue / pipeline_stages 等产物，输出 7 个 sheet：

    运行信息 · 流程记录 · 指标汇总 · 逐题结果 · 人工评分表 · 评分规则 · 异常与复核

再跑一次不会丢人工评分：脚本会先读回旧文件里已填的分数再写回。

用法：
    python scripts/export_suite_xlsx.py --dir results/exp_full_84
    python scripts/export_suite_xlsx.py --dir results/exp_full_84 \\
        --config config/test.yaml --batch EXP-084 --out results/exp_full_84/suite.xlsx
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openpyxl import Workbook  # noqa: E402
from openpyxl.styles import PatternFill  # noqa: E402

from src.suite import (  # noqa: E402
    STAGES,
    aggregate_subjective,
    collect,
    config_snapshot,
    existing_scores,
    latest_stages,
)

try:  # 作为包导入（pytest）
    from scripts.export_review_xlsx import (  # noqa: E402
        OBJECTIVE_METRICS,
        SUBJECTIVE_METRICS,
        _set_col_widths,
        _style_cell,
        _style_header,
        sheet_review,
        sheet_rules,
    )
except ImportError:  # 直接运行脚本
    from export_review_xlsx import (  # type: ignore  # noqa: E402
        OBJECTIVE_METRICS,
        SUBJECTIVE_METRICS,
        _set_col_widths,
        _style_cell,
        _style_header,
        sheet_review,
        sheet_rules,
    )

OK_FILL = PatternFill("solid", fgColor="E2EFDA")
BAD_FILL = PatternFill("solid", fgColor="FCE4E4")
SKIP_FILL = PatternFill("solid", fgColor="FFF2CC")

# 逐题结果优先列顺序，其余字段追加在后
QUESTION_COLS = [
    "question_id", "scenario_id", "test_stage", "question_type", "question",
    "golden_answer", "predicted_answer",
    "Top@5", "Top@10", "Top@20", "MRR", "NDCG@10",
    "evidence_recall@5", "evidence_recall@10",
    "fact_recall_rule", "fact_recall_relaxed",
    "abstention_correct", "citation_complete",
    "e2e_latency_ms", "failure_attribution", "run_error",
]


# ---------------------------------------------------------------------------
# Sheet: 运行信息
# ---------------------------------------------------------------------------
def sheet_run_info(wb: Workbook, data: Dict[str, Any], out_path: str,
                   config_path: str, batch: str):
    ws = wb.create_sheet("运行信息")
    ws.append(["项", "值"])
    _style_header(ws)

    summary = data.get("summary") or {}
    ds = summary.get("dataset") or {}
    lat = summary.get("latency") or {}
    cfg = config_snapshot(config_path)
    rag = (cfg.get("ragflow") or {}) if cfg else {}

    items = [
        ("实验编号", os.path.basename(os.path.normpath(data["dir"]))),
        ("批次", batch),
        ("导出时间", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("运行目录", data["dir"]),
        ("交付文件", out_path),
        ("运行模式", summary.get("mode", "")),
        ("题目数", summary.get("n_questions", len(data["per_question"]))),
        ("数据集", ds.get("path", "")),
        ("数据集 sha256", str(ds.get("sha256", ""))[:16]),
        ("RAGFlow", rag.get("base_url", "")),
        ("Chat Assistant", rag.get("chat_id", "")),
        ("重排模型", (cfg.get("retrieval") or {}).get("rerank_id") if cfg else ""),
        ("相似度阈值", (cfg.get("retrieval") or {}).get("similarity_threshold") if cfg else ""),
        ("向量权重", (cfg.get("retrieval") or {}).get("vector_similarity_weight") if cfg else ""),
        ("P50/P95/Max(s)", " / ".join(str(lat.get(k, "")) for k in
                                      ("p50", "p95", "max")) if lat else ""),
        ("一票否决疑似", summary.get("veto_count", "")),
        ("阶段完成", " ".join(f"{k}:{v['status']}" for k, v in
                              sorted(latest_stages(data["stages"]).items()))),
        ("配置快照", json.dumps(cfg, ensure_ascii=False) if cfg else "（未提供 --config）"),
        ("产物清单", "  ".join(sorted(os.path.basename(p) for p in
                                      (os.path.join(data["dir"], n) for n in
                                       ("per_question.jsonl", "summary.json", "report.md",
                                        "parse_summary.json", "review_queue.jsonl"))
                                      if os.path.exists(p)))),
    ]
    for k, v in items:
        ws.append([k, v])
        for cell in ws[ws.max_row]:
            _style_cell(cell)
    _set_col_widths(ws, {"A": 16, "B": 110})
    ws.freeze_panes = "A2"
    return ws


# ---------------------------------------------------------------------------
# Sheet: 流程记录
# ---------------------------------------------------------------------------
def sheet_stages(wb: Workbook, stages: List[Dict[str, Any]]):
    ws = wb.create_sheet("流程记录")
    ws.append(["阶段", "名称", "状态", "耗时(s)", "产物", "说明", "时间"])
    _style_header(ws)

    by_stage: Dict[str, List[Dict[str, Any]]] = {}
    for s in stages:
        by_stage.setdefault(str(s.get("stage")), []).append(s)

    rows: List[Dict[str, Any]] = []
    for sid, name, desc in STAGES:
        if by_stage.get(sid):
            rows.extend(by_stage[sid])
        else:
            rows.append({"stage": sid, "name": name, "status": "missing",
                         "elapsed_s": "", "artifact": "", "note": desc, "ts": ""})
    known = {s for s, _, _ in STAGES}
    rows.extend(s for s in stages if str(s.get("stage")) not in known)

    for s in rows:
        status = str(s.get("status", ""))
        ws.append([s.get("stage"), s.get("name"), status, s.get("elapsed_s"),
                   s.get("artifact"), s.get("note"), s.get("ts")])
        fill = OK_FILL if status == "ok" else (BAD_FILL if status == "fail" else SKIP_FILL)
        for cell in ws[ws.max_row]:
            _style_cell(cell, fill)
    _set_col_widths(ws, {"A": 8, "B": 14, "C": 10, "D": 10, "E": 34, "F": 46, "G": 20})
    ws.freeze_panes = "A2"
    return ws


# ---------------------------------------------------------------------------
# Sheet: 指标汇总（客观 + 主观人工聚合）
# ---------------------------------------------------------------------------
def sheet_metrics(wb: Workbook, data: Dict[str, Any], scores: Dict[tuple, tuple]):
    ws = wb.create_sheet("指标汇总")
    ws.append(["类别", "指标编号", "指标名称", "指标值/均值", "样本数",
               "满分/达标率", "分布", "阈值", "判定", "备注"])
    _style_header(ws)

    summary = data.get("summary") or {}
    metrics = summary.get("metrics") or {}
    verdicts = summary.get("verdicts") or {}

    for mid, meta in OBJECTIVE_METRICS.items():
        m = metrics.get(mid)
        value = m.get("value") if isinstance(m, dict) else m
        n = m.get("n", "") if isinstance(m, dict) else ""
        note = m.get("note", "") if isinstance(m, dict) else ""
        v = verdicts.get(mid) or {}
        verdict = "✅" if v.get("pass") else ("❌" if v.get("pass") is False else "—")
        ws.append(["客观", mid, meta["name"], value, n, "", "", meta["threshold"],
                   f"{verdict} {v.get('desc', '')}".strip(), note])
        for cell in ws[ws.max_row]:
            _style_cell(cell)

    for r in aggregate_subjective(scores, SUBJECTIVE_METRICS):
        mid = r["指标编号"]
        meta = SUBJECTIVE_METRICS.get(mid, {})
        lower = meta.get("direction") == "lower"
        thr = str(meta.get("threshold", ""))
        rate = r["满分/达标率"]
        try:
            limit = float(thr.split()[-1])
        except (ValueError, IndexError):
            limit = None
        if limit is None:
            verdict = "—"
        elif lower:
            verdict = "✅" if r["均值"] <= limit else "❌"
        else:
            verdict = "✅" if r["均值"] >= limit * (2 if meta.get("scale") == "0/1/2" else 1) else "❌"
        ws.append(["主观(人工)", mid, meta.get("name", ""), r["均值"], r["样本数"],
                   rate, r["分布"], thr, verdict, "人工评分均值口径"])
        for cell in ws[ws.max_row]:
            _style_cell(cell)

    if not scores:
        ws.append(["主观(人工)", "—", "尚未填写人工评分", "", 0, "", "", "",
                   "—", "在「人工评分表」填分后重跑 S6 即可回填"])
        for cell in ws[ws.max_row]:
            _style_cell(cell, SKIP_FILL)

    _set_col_widths(ws, {"A": 12, "B": 12, "C": 18, "D": 14, "E": 8, "F": 12,
                         "G": 22, "H": 12, "I": 22, "J": 40})
    ws.freeze_panes = "A2"
    return ws


# ---------------------------------------------------------------------------
# Sheet: 逐题结果
# ---------------------------------------------------------------------------
def sheet_questions(wb: Workbook, rows: List[Dict[str, Any]]):
    ws = wb.create_sheet("逐题结果")
    extra = []
    for r in rows:
        for k in r:
            if k not in QUESTION_COLS and k not in extra:
                extra.append(k)
    cols = QUESTION_COLS + extra
    ws.append(cols)
    _style_header(ws)

    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, (list, dict)):
                v = json.dumps(v, ensure_ascii=False)
            vals.append(v)
        ws.append(vals)
        if r.get("run_error"):
            for cell in ws[ws.max_row]:
                _style_cell(cell, BAD_FILL)

    widths = {"A": 12, "B": 12, "C": 10, "D": 10, "E": 46, "F": 46, "G": 60}
    _set_col_widths(ws, widths)
    ws.freeze_panes = "B2"
    return ws


# ---------------------------------------------------------------------------
# Sheet: 异常与复核
# ---------------------------------------------------------------------------
def sheet_issues(wb: Workbook, data: Dict[str, Any]):
    ws = wb.create_sheet("异常与复核")
    ws.append(["类型", "题目编号", "内容", "来源"])
    _style_header(ws)

    for r in data["per_question"]:
        if r.get("run_error"):
            ws.append(["运行异常", r.get("question_id"), str(r.get("run_error"))[:500],
                       "per_question.run_error"])
        elif r.get("predicted_answer") in (None, "") and r.get("test_stage") in ("qa", "refusal"):
            ws.append(["空答案", r.get("question_id"), "predicted_answer 为空，需查 chat 接口",
                       "per_question"])

    for row in data["parse_anomalies"]:
        ws.append(["解析异常", row.get("doc_name", ""),
                   json.dumps(row, ensure_ascii=False)[:500], "parse_anomalies.jsonl"])
    for row in data["review_queue"]:
        ws.append(["待复核", row.get("question_id", ""),
                   f"{row.get('judge_task', '')} 风险={row.get('risk', '')}",
                   "review_queue.jsonl"])
    for row in data["review_scores"]:
        ws.append(["复核分", row.get("question_id", ""),
                   f"{row.get('judge_task', '')}={row.get('score', '')}",
                   "review_scores.jsonl"])

    if ws.max_row == 1:
        ws.append(["—", "", "无异常/待复核项", ""])
    _set_col_widths(ws, {"A": 12, "B": 14, "C": 100, "D": 26})
    ws.freeze_panes = "A2"
    return ws


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def build(exp_dir: str, out_path: str, config_path: str = "",
          batch: str = "") -> str:
    data = collect(exp_dir)
    prev = existing_scores(out_path)
    wb = Workbook()
    wb.remove(wb.active)

    sheet_run_info(wb, data, out_path, config_path, batch)
    sheet_stages(wb, data["stages"])
    sheet_metrics(wb, data, prev)
    sheet_questions(wb, data["per_question"])
    sheet_review(wb, data["per_question"])
    sheet_rules(wb)

    # 回填历史人工评分（键：题目编号 + 指标编号）
    ws = wb["人工评分表"]
    for row in ws.iter_rows(min_row=2, max_col=18):
        key = (str(row[0].value or ""), str(row[12].value or ""))
        if key in prev:
            row[16].value, row[17].value = prev[key]

    sheet_issues(wb, data)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    wb.save(out_path)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="results/<exp> 运行目录")
    ap.add_argument("--out", default="", help="默认 <dir>/suite.xlsx")
    ap.add_argument("--config", default="", help="写入配置快照（api_key 脱敏）")
    ap.add_argument("--batch", default="", help="批次号，写进「运行信息」")
    args = ap.parse_args()

    out = args.out or os.path.join(args.dir, "suite.xlsx")
    if not os.path.isdir(args.dir):
        print(f"目录不存在：{args.dir}")
        return 1
    build(args.dir, out, args.config, args.batch)
    print(f"已生成: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
