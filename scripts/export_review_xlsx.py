"""生成人工评审 xlsx：客观指标结果 + 主观指标评分表 + 评分规则。

客观可算的指标已在 per_question.jsonl 中落盘，本脚本把它们整理成表；
不可客观计算的主观指标（GEN-01/03/04/06/07/08、RAG-06/07/08、TRACE-02）
逐题展开成待评分行，附带完整上下文（问题/参考答案/系统答案/证据/事实点）
和打分规则，供人工审查打分。

用法：
    python scripts/export_review_xlsx.py \\
        --per-question results/<exp>/per_question.jsonl \\
        --out results/<exp>/review.xlsx

    # 可选：附带 summary.json 的聚合结果
    python scripts/export_review_xlsx.py \\
        --per-question results/<exp>/per_question.jsonl \\
        --summary results/<exp>/summary.json \\
        --out results/<exp>/review.xlsx
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

# ---------------------------------------------------------------------------
# 主观指标定义（与 src/judge.py 的 JUDGE_TASKS 保持一致，此处展开打分规则）
# ---------------------------------------------------------------------------
SUBJECTIVE_METRICS: Dict[str, Dict[str, Any]] = {
    "GEN-01": {
        "name": "答案准确率",
        "口径": "对照参考答案与必须覆盖事实，判断答案是否正确",
        "scale": "0/1/2",
        "scale_desc": "0=错误或关键事实错误；1=基本正确但缺次要信息；2=正确完整",
        "direction": "higher",
        "veto": False,
        "threshold": ">= 0.95",
    },
    "GEN-03": {
        "name": "答案完整率",
        "口径": "必须覆盖事实是否全部完整表达（含条件、例外、限定语）",
        "scale": "0/1/2",
        "scale_desc": "0=缺关键内容；1=基本完整；2=完整",
        "direction": "higher",
        "veto": False,
        "threshold": ">= 0.90",
    },
    "GEN-04": {
        "name": "幻觉率",
        "口径": "答案中是否有检索证据或参考答案不支持的事实",
        "scale": "0/1",
        "scale_desc": "0=无幻觉；1=有幻觉（越小越好）",
        "direction": "lower",
        "veto": True,
        "threshold": "<= 0.02",
    },
    "GEN-06": {
        "name": "错误事实率",
        "口径": "答案中是否有与参考答案/证据矛盾的事实",
        "scale": "0/1",
        "scale_desc": "0=无矛盾；1=有矛盾（越小越好）",
        "direction": "lower",
        "veto": False,
        "threshold": "<= 0.02",
    },
    "GEN-07": {
        "name": "参数准确率",
        "口径": "答案中的关键参数（压力/温度/流量等）是否与参考答案精确一致",
        "scale": "0/1",
        "scale_desc": "0=参数错误；1=参数正确（高风险题必须 100%）",
        "direction": "higher",
        "veto": True,
        "threshold": ">= 0.99",
    },
    "GEN-08": {
        "name": "操作步骤正确率",
        "口径": "步骤顺序、前置条件、状态确认是否正确",
        "scale": "0/1/2",
        "scale_desc": "0=错误；1=基本正确；2=正确",
        "direction": "higher",
        "veto": False,
        "threshold": ">= 0.98",
    },
    "RAG-06": {
        "name": "引用准确率",
        "口径": "每条引用是否真正支持对应结论",
        "scale": "0/1",
        "scale_desc": "0=不支持；1=支持",
        "direction": "higher",
        "veto": False,
        "threshold": ">= 0.95",
    },
    "RAG-07": {
        "name": "证据覆盖率",
        "口径": "关键结论是否有有效源文件/条款支撑",
        "scale": "0/1",
        "scale_desc": "0=无支撑；1=有支撑",
        "direction": "higher",
        "veto": False,
        "threshold": ">= 0.95",
    },
    "RAG-08": {
        "name": "版本正确率",
        "口径": "引用的文档版本是否为当前有效版本（旧/失效版本记 0）",
        "scale": "0/1",
        "scale_desc": "0=版本错误；1=版本正确",
        "direction": "higher",
        "veto": True,
        "threshold": ">= 1.00",
    },
    "TRACE-02": {
        "name": "证据可追溯率",
        "口径": "关键结论能否追溯到具体文档/片段",
        "scale": "0/1",
        "scale_desc": "0=不能追溯；1=能追溯",
        "direction": "higher",
        "veto": False,
        "threshold": ">= 0.95",
    },
}

# 客观指标（自动计算，无需人工评分）
OBJECTIVE_METRICS: Dict[str, Dict[str, Any]] = {
    "RAG-01": {"name": "Top@5 召回率", "direction": "higher", "threshold": ">= 0.90"},
    "RAG-02": {"name": "Top@10 召回率", "direction": "higher", "threshold": ">= 0.95"},
    "RAG-03": {"name": "Top@20 召回率", "direction": "higher", "threshold": ">= 0.98"},
    "RAG-04": {"name": "MRR", "direction": "higher", "threshold": ">= 0.80"},
    "RAG-05": {"name": "NDCG@10", "direction": "higher", "threshold": ">= 0.85"},
    "RAG-09": {"name": "附件召回率", "direction": "higher", "threshold": ">= 0.95"},
    "RAG-10": {"name": "跨文档证据完整率", "direction": "higher", "threshold": ">= 0.90"},
    "GEN-02": {"name": "关键事实召回率", "direction": "higher", "threshold": ">= 0.95"},
    "GEN-05": {"name": "拒答正确率", "direction": "higher", "threshold": ">= 0.98"},
    "GEN-09": {"name": "禁止事项识别率", "direction": "higher", "threshold": ">= 1.00"},
    "GEN-10": {"name": "引用完整率", "direction": "higher", "threshold": ">= 0.95"},
    "GEN-ORDER": {"name": "步骤顺序正确率", "direction": "higher", "threshold": ">= 0.98"},
    "GEN-UNIT": {"name": "单位完整率", "direction": "higher", "threshold": ">= 0.99"},
    "TRACE-01": {"name": "链路完整率", "direction": "higher", "threshold": ">= 0.99"},
}

# GEN-08 仅适用于步骤/事故类题型
STEP_TYPES_FOR_GEN08 = ("procedure", "accident", "accident_action",
                        "accident_cause", "accident_symptom")

# 客观指标逐题列（per_question.jsonl 中的 key → 表头）
OBJ_COLUMNS = [
    ("question_id", "题目编号"),
    ("scenario_id", "场景编号"),
    ("test_stage", "测试环节"),
    ("question_type", "题型"),
    ("Top@5", "Top@5"),
    ("Top@10", "Top@10"),
    ("Top@20", "Top@20"),
    ("MRR", "MRR"),
    ("NDCG@10", "NDCG@10"),
    ("evidence_recall@5", "evidence_recall@5"),
    ("evidence_recall@10", "evidence_recall@10"),
    ("fact_recall_rule", "fact_recall_rule"),
    ("fact_recall_relaxed", "fact_recall_relaxed"),
    ("fact_uncertain_count", "fact_uncertain_count"),
    ("abstention_correct", "abstention_correct"),
    ("forbidden_fact_hit", "forbidden_fact_hit"),
    ("citation_complete_rule", "citation_complete_rule"),
    ("trace_ok", "trace_ok"),
    ("run_error", "run_error"),
    ("chat_error", "chat_error"),
]

# 样式
_HEADER_FILL = PatternFill("solid", fgColor="4472C4")
_HEADER_FONT = Font(bold=True, color="FFFFFF", size=10)
_INPUT_FILL = PatternFill("solid", fgColor="E2EFDA")   # 绿底=输入/依据
_OUTPUT_FILL = PatternFill("solid", fgColor="FFF2CC")  # 黄底=输出/待评
_SCORE_FILL = PatternFill("solid", fgColor="FCE4D6")   # 橙底=人工填写
_RULE_FILL = PatternFill("solid", fgColor="D6DCE4")    # 灰底=规则参考
_THIN = Border(*[Side(style="thin", color="B0B0B0")] * 4)
_WRAP = Alignment(wrap_text=True, vertical="top")


def _style_header(ws, row: int = 1):
    for cell in ws[row]:
        cell.fill = _HEADER_FILL
        cell.font = _HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = _THIN


def _style_cell(cell, fill: Optional[PatternFill] = None):
    cell.alignment = _WRAP
    cell.border = _THIN
    if fill:
        cell.fill = fill


def _set_col_widths(ws, widths: Dict[str, int]):
    for col, w in widths.items():
        ws.column_dimensions[col].width = w


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_per_question(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_summary(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Sheet builders
# ---------------------------------------------------------------------------
def sheet_rules(wb: Workbook):
    """评分规则：所有指标的口径、打分规则、方向、阈值。"""
    ws = wb.create_sheet("评分规则")
    headers = ["指标编号", "指标名称", "指标口径", "打分规则", "分数方向",
               "计算方式", "一票否决", "阈值"]
    ws.append(headers)
    _style_header(ws)

    for mid, m in SUBJECTIVE_METRICS.items():
        ws.append([
            mid, m["name"], m["口径"], m["scale_desc"],
            "越高越好" if m["direction"] == "higher" else "越低越好",
            "人工评审", "是" if m["veto"] else "否", m["threshold"],
        ])
        for cell in ws[ws.max_row]:
            _style_cell(cell, _RULE_FILL)

    for mid, m in OBJECTIVE_METRICS.items():
        ws.append([
            mid, m["name"], "（自动计算，见「客观指标结果」sheet）",
            "见 per_question.jsonl / summary.json",
            "越高越好" if m["direction"] == "higher" else "越低越好",
            "自动计算", "是" if mid in ("RAG-08", "GEN-09") else "否",
            m["threshold"],
        ])
        for cell in ws[ws.max_row]:
            _style_cell(cell)

    _set_col_widths(ws, {"A": 12, "B": 16, "C": 45, "D": 45, "E": 10,
                         "F": 10, "G": 10, "H": 12})
    ws.freeze_panes = "A2"
    return ws


def sheet_review(wb: Workbook, rows: List[Dict[str, Any]]):
    """人工评分表：逐题 × 主观指标，附完整上下文 + 打分列。"""
    ws = wb.create_sheet("人工评分表")
    headers = [
        "题目编号", "场景编号", "测试环节", "题型",
        "问题（输入）", "参考答案（输入）", "必须覆盖事实（输入）",
        "禁止事项（输入）", "检索证据（输入）",
        "系统答案（输出·待评审）",
        "应拒答", "拒答原因",
        "指标编号", "指标名称", "指标口径", "打分规则",
        "人工评分", "评分理由",
    ]
    ws.append(headers)
    _style_header(ws)

    # 列索引（1-based）
    COL_INPUT = {"问题（输入）": 5, "参考答案（输入）": 6,
                 "必须覆盖事实（输入）": 7, "禁止事项（输入）": 8,
                 "检索证据（输入）": 9}
    COL_OUTPUT = {"系统答案（输出·待评审）": 10}
    COL_SCORE = {"人工评分": 17, "评分理由": 18}

    score_validations: Dict[str, DataValidation] = {}
    for mid, m in SUBJECTIVE_METRICS.items():
        dv = DataValidation(
            type="list",
            formula1='"' + ",".join(m["scale"].split("/")) + '"',
            allow_blank=True,
        )
        dv.error = f"只允许 {m['scale']}"
        dv.prompt = m["scale_desc"]
        score_validations[mid] = dv

    row_idx = 1  # 当前 xlsx 行号（header 占 1）
    for item in rows:
        qid = item.get("question_id", "")
        # 运行异常的题也保留，方便人工看失败原因
        qtype = item.get("question_type", "")
        should_refuse = bool(item.get("should_refuse") or not item.get("answerable", True))
        refusal_reason = item.get("refusal_reason", "") or ""
        has_cite = bool(item.get("must_cite"))

        for mid, m in SUBJECTIVE_METRICS.items():
            # 适用性过滤
            if mid == "GEN-08" and qtype not in STEP_TYPES_FOR_GEN08:
                continue
            if mid == "RAG-06" and not has_cite:
                continue
            if mid == "RAG-08":
                continue  # 版本正确率需人工对照版本号，单独处理

            row_idx += 1
            facts = item.get("expected_answer_points") or item.get("required_facts") or []
            forbidden = item.get("forbidden_claims") or item.get("forbidden_facts") or []
            evidence = (item.get("evidence_snippet") or "")[:2000]

            ws.append([
                qid, item.get("scenario_id", ""), item.get("test_stage", ""), qtype,
                item.get("question", ""),
                item.get("golden_answer", "") or item.get("golden", ""),
                "\n".join(f"• {f}" for f in facts) if facts else "（无）",
                "\n".join(f"• {f}" for f in forbidden) if forbidden else "（无）",
                evidence,
                item.get("predicted_answer", "") or item.get("answer", ""),
                "是" if should_refuse else "否",
                refusal_reason or "—",
                mid, m["name"], m["口径"], m["scale_desc"],
                None, None,
            ])

            # 样式
            for col_idx in COL_INPUT.values():
                _style_cell(ws.cell(row=row_idx, column=col_idx), _INPUT_FILL)
            for col_idx in COL_OUTPUT.values():
                _style_cell(ws.cell(row=row_idx, column=col_idx), _OUTPUT_FILL)
            for col_idx in COL_SCORE.values():
                _style_cell(ws.cell(row=row_idx, column=col_idx), _SCORE_FILL)
            for col in range(1, 19):
                if col not in COL_INPUT.values() and col not in COL_OUTPUT.values() \
                        and col not in COL_SCORE.values():
                    _style_cell(ws.cell(row=row_idx, column=col))

            # 打分下拉
            dv = score_validations[mid]
            dv.add(ws.cell(row=row_idx, column=COL_SCORE["人工评分"]))

    for dv in score_validations.values():
        ws.add_data_validation(dv)

    _set_col_widths(ws, {
        "A": 12, "B": 12, "C": 10, "D": 14,
        "E": 35, "F": 35, "G": 30, "H": 20, "I": 40,
        "J": 40,
        "K": 8, "L": 12,
        "M": 10, "N": 14, "O": 35, "P": 35,
        "Q": 10, "R": 25,
    })
    ws.freeze_panes = "E2"
    return ws


def sheet_objective(wb: Workbook, rows: List[Dict[str, Any]]):
    """客观指标结果：自动计算的逐题指标。"""
    ws = wb.create_sheet("客观指标结果")
    headers = [label for _, label in OBJ_COLUMNS]
    ws.append(headers)
    _style_header(ws)

    for item in rows:
        vals = []
        for key, _ in OBJ_COLUMNS:
            v = item.get(key)
            if isinstance(v, bool):
                v = "是" if v else "否"
            vals.append(v)
        ws.append(vals)
        for cell in ws[ws.max_row]:
            _style_cell(cell)

    _set_col_widths(ws, {
        "A": 12, "B": 12, "C": 10, "D": 14,
        "E": 8, "F": 8, "G": 8, "H": 8, "I": 8,
        "J": 14, "K": 14, "L": 14, "M": 14, "N": 12,
        "O": 12, "P": 12, "Q": 12, "R": 8,
        "S": 25, "T": 25,
    })
    ws.freeze_panes = "E2"
    return ws


def sheet_summary(wb: Workbook, summary: Dict[str, Any]):
    """汇总：聚合指标 + 阈值判定。"""
    ws = wb.create_sheet("汇总")
    headers = ["指标编号", "指标名称", "指标值", "样本数", "阈值", "判定", "备注"]
    ws.append(headers)
    _style_header(ws)

    metrics = summary.get("metrics", {}) if summary else {}
    verdicts = summary.get("verdicts", {}) if summary else {}

    all_mids = list(OBJECTIVE_METRICS) + list(SUBJECTIVE_METRICS)
    for mid in all_mids:
        meta = OBJECTIVE_METRICS.get(mid) or SUBJECTIVE_METRICS.get(mid)
        m = metrics.get(mid)
        if isinstance(m, dict):
            value = m.get("value")
            n = m.get("n")
            note = m.get("note", "")
        else:
            value = m
            n = ""
            note = ""
        v = verdicts.get(mid, {})
        verdict_icon = "✅" if v.get("pass") else ("❌" if v.get("pass") is False else "—")
        desc = v.get("desc", "")
        ws.append([mid, meta["name"], value, n, meta["threshold"],
                   f"{verdict_icon} {desc}" if desc else verdict_icon, note])
        for cell in ws[ws.max_row]:
            _style_cell(cell)

    _set_col_widths(ws, {"A": 12, "B": 18, "C": 12, "D": 8,
                         "E": 12, "F": 25, "G": 30})
    ws.freeze_panes = "A2"
    return ws


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="生成人工评审 xlsx")
    ap.add_argument("--per-question", required=True,
                    help="per_question.jsonl 路径")
    ap.add_argument("--summary", default="",
                    help="summary.json 路径（可选，聚合结果进「汇总」sheet）")
    ap.add_argument("--out", required=True,
                    help="输出 xlsx 路径，如 results/<exp>/review.xlsx")
    args = ap.parse_args()

    if not os.path.exists(args.per_question):
        print(f"错误: 找不到 {args.per_question}", file=sys.stderr)
        return 1

    rows = load_per_question(args.per_question)
    summary = load_summary(args.summary)
    print(f"读取 {len(rows)} 条逐题记录")

    wb = Workbook()
    wb.remove(wb.active)  # 去掉默认空 sheet

    sheet_rules(wb)
    print("  ✓ 评分规则")
    sheet_review(wb, rows)
    n_review = wb["人工评分表"].max_row - 1
    print(f"  ✓ 人工评分表（{n_review} 条待评分行）")
    sheet_objective(wb, rows)
    print(f"  ✓ 客观指标结果（{len(rows)} 题）")
    sheet_summary(wb, summary)
    print("  ✓ 汇总" + ("（含 summary.json 聚合）" if summary else "（无 summary.json，仅列指标定义）"))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    wb.save(args.out)
    print(f"已保存 → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
