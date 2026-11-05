"""测试主线（src/suite.py + export_suite_xlsx）的离线回归。"""
import json
import os

import pytest

from src.suite import (
    aggregate_subjective,
    collect,
    existing_scores,
    exp_dir_from_config,
    latest_stages,
    load_stages,
    record,
)
from scripts.export_review_xlsx import SUBJECTIVE_METRICS
from scripts.export_suite_xlsx import build


def _rows():
    return [
        {
            "question_id": "ANS-001", "scenario_id": "SCN-001", "test_stage": "qa",
            "question_type": "table", "question": "水压试验压力是多少？",
            "golden_answer": "水压试验18.84MPa",
            "expected_answer_points": ["水压试验18.84MPa"],
            "predicted_answer": "水压试验压力为18.84MPa",
            "evidence_snippet": "[1] doc.pdf#c1: 18.84MPa",
            "Top@5": 1, "run_error": "",
        },
        {
            "question_id": "RET-001", "scenario_id": "SCN-001", "test_stage": "retrieval",
            "question_type": "param", "question": "水压试验多大压力？",
            "golden_answer": "18.84MPa", "predicted_answer": "",
            "run_error": "timeout",
        },
        {
            "question_id": "CIT-001", "scenario_id": "SCN-001", "test_stage": "qa",
            "question_type": "param", "question": "水压试验多大压力？",
            "golden_answer": "18.84MPa", "predicted_answer": "",
        },
        {
            "question_id": "ANS-002", "scenario_id": "SCN-002", "test_stage": "qa",
            "question_type": "param", "question": "润滑油压动作值？",
            "golden_answer": "0.082/0.076MPa", "predicted_answer": "未找到",
            "needs_chunk_mapping": True, "Top@5": None,
        },
    ]


def _write_run(tmp_path, rows=None, summary=None, stages=()):
    exp = tmp_path / "exp"
    exp.mkdir()
    with open(exp / "per_question.jsonl", "w", encoding="utf-8") as f:
        for r in (rows if rows is not None else _rows()):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(exp / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary or {
            "n_questions": 2, "mode": "full",
            "metrics": {"RAG-01": {"value": 0.5, "n": 2}},
            "verdicts": {"RAG-01": {"pass": False, "desc": "0.5 < 0.90"}},
            "latency": {"p50": 1.0},
        }, f)
    for stage, status in stages:
        record(str(exp), stage, status, 0.1, note="t")
    return str(exp)


def test_record_appends_and_orders(tmp_path):
    exp = str(tmp_path / "e")
    record(exp, "S4", "ok", 1.5)
    record(exp, "S0", "ok", 0.2)
    record(exp, "S4", "fail", 0.1)
    stages = load_stages(exp)
    assert [s["stage"] for s in stages] == ["S0", "S4", "S4"]
    assert latest_stages(stages)["S4"]["status"] == "fail"


def test_collect_on_empty_dir(tmp_path):
    data = collect(str(tmp_path))
    assert data["per_question"] == [] and data["summary"] == {}
    assert data["stages"] == []


def test_aggregate_subjective_lower_is_better():
    scores = {("A", "GEN-04"): (0, ""), ("B", "GEN-04"): (1, ""),
              ("C", "GEN-04"): (0, "")}
    rows = aggregate_subjective(scores, SUBJECTIVE_METRICS)
    gen04 = next(r for r in rows if r["指标编号"] == "GEN-04")
    assert gen04["样本数"] == 3
    assert gen04["均值"] == pytest.approx(0.3333, abs=1e-4)
    assert gen04["满分/达标率"] == pytest.approx(2 / 3, abs=1e-4)


def test_build_creates_all_sheets_and_keeps_scores(tmp_path):
    exp = _write_run(tmp_path, stages=[("S0", "ok"), ("S4", "ok"), ("S6", "ok")])
    out = os.path.join(exp, "suite.xlsx")
    build(exp, out, batch="EXP-T")

    from openpyxl import load_workbook
    wb = load_workbook(out)
    assert wb.sheetnames == ["运行信息", "流程记录", "指标汇总", "逐题结果",
                             "人工评分表", "评分规则", "异常与复核"]

    ws = wb["人工评分表"]
    header = [c.value for c in ws[1]]
    qid_col = header.index("题目编号") + 1
    mid_col = header.index("指标编号") + 1
    score_col = header.index("人工评分") + 1
    reason_col = header.index("评分理由") + 1

    target = None
    for r in range(2, ws.max_row + 1):
        if (ws.cell(r, qid_col).value == "ANS-001"
                and ws.cell(r, mid_col).value == "GEN-01"):
            target = r
            break
    assert target, "ANS-001 × GEN-01 行应存在"
    ws.cell(target, score_col).value = 2
    ws.cell(target, reason_col).value = "完整"
    wb.save(out)

    build(exp, out, batch="EXP-T")
    assert existing_scores(out)[("ANS-001", "GEN-01")] == (2, "完整")


def test_exp_dir_reads_run_section(tmp_path):
    cfg = {"run": {"output_dir": "out", "experiment_id": "exp_full_84"},
           "output_dir": "ignored", "experiment_id": "ignored"}
    assert exp_dir_from_config(cfg, str(tmp_path)) == str(
        tmp_path / "out" / "exp_full_84")


def test_issues_sheet_lists_errors_and_empty_answers(tmp_path):
    exp = _write_run(tmp_path)
    with open(os.path.join(exp, "dataset_warnings.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"question_id": "ANS-002",
                            "message": "1 个事实点含多个数值"},
                           ensure_ascii=False) + "\n")
    out = os.path.join(exp, "suite.xlsx")
    build(exp, out)
    from openpyxl import load_workbook
    ws = load_workbook(out)["异常与复核"]
    text = "\n".join(" ".join(str(c.value or "") for c in row)
                     for row in ws.iter_rows())
    assert "RET-001" in text and "运行异常" in text
    assert "CIT-001" in text and "空答案" in text
    assert "ANS-002" in text and "无Chunk映射" in text
    assert "1 个事实点含多个数值" in text
