"""xlsx 全环节用例导入/场景汇总单测（无需 RAGFlow 服务）。"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.import_xlsx_cases import (  # noqa: E402
    build_baseline,
    build_golden,
    split_expectation,
    strip_version,
)
from src.metrics_retrieval import evidence_cover, evidence_recall_at_k  # noqa: E402
from src.models import RetrievedChunk  # noqa: E402
from src.report import build_report  # noqa: E402
from src.metrics_generation import latency_stats  # noqa: E402


def test_split_expectation():
    pts, forb = split_expectation(
        "水压试验18.84MPa；超压试验23.55MPa。 禁止：不得用再热器5.7MPa替代。")
    assert pts == ["水压试验18.84MPa", "超压试验23.55MPa"]
    assert forb == ["不得用再热器5.7MPa替代"]
    pts, _ = split_expectation("能够支持以下要点的依据进入候选集：A；B。")
    assert pts == ["A", "B"]
    assert split_expectation("") == ([], [])


def test_build_golden_recall_row():
    g = build_golden("SCN-001", "RET-001", "召回", "表格参数", "水冷壁压力到多少？",
                     "进入候选集：18.84MPa；23.55MPa。", "证据", "1 机组启动；表T002第1行",
                     "600MW机组集控运行规程20230831.docx", "20230831（文件日期标识）",
                     "自动测试", "待审核", "", 42, "abc123")
    assert g["test_stage"] == "retrieval"
    assert g["question_type"] == "table"
    assert g["expected_answer_points"] == ["18.84MPa", "23.55MPa"]
    assert g["table_or_item"] == "表T002"
    assert g["source_version"] == "20230831"
    assert g["scenario_id"] == "SCN-001" and g["paraphrase_group"] == "SCN-001"
    assert g["needs_chunk_mapping"] is True
    assert g["source_ref"] == "测试用例总表#R42"


def test_build_golden_refusal_row():
    g = build_golden("SCN-019", "REF-001", "拒答", "含糊问题", "油压低了怎么办？",
                     "要求补充油系统；不得自行假定。", "证据", "2.2.1.11",
                     "规程.docx", "20230831", "自动测试+人工复核", "待审核",
                     "缺少系统", 99, "")
    assert g["should_refuse"] is True and g["refusal_reason"] == "ambiguous"
    assert g["answerable"] is False and g["expected_answer_points"] == []
    assert "阶段预期" in g["remarks"]


def test_build_baseline_row():
    b = build_baseline("PAR-001", "表格参数", "水压试验表…", "完整保留…",
                       "省煤器 | 18.84MPa", "1 机组启动；表T002第1行",
                       "原文解析", "SCN-001", 6)
    assert b["record_id"] == "PAR-001" and b["kind"] == "table_cell"
    assert b["text"] == "省煤器 | 18.84MPa"


def test_import_roundtrip_tmp_xlsx(tmp_path):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "测试用例总表"
    for _ in range(4):
        ws.append([])
    ws.append(["场景编号", "用例编号", "测试集", "测试环节", "问题类型",
               "测试问题/输入对象", "阶段预期结果", "原文证据", "原文位置",
               "文档名称", "文档版本", "验证方式", "审核状态", "备注"])
    ws.append(["SCN-001", "PAR-001", "解析与切分", "原文解析", "表格参数",
               "对象", "完整保留", "证据文本", "位置", " doc.docx",
               "20230831", "自动比对+人工复核", "待审核", ""])
    ws.append(["SCN-001", "RET-001", "召回与重排", "召回", "表格参数",
               "问题？", "要点：A；B。", "证据", "位置", "doc.docx",
               "20230831", "自动测试", "待审核", ""])
    ws.append([])  # 空行应跳过
    p = tmp_path / "cases.xlsx"
    wb.save(p)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from import_xlsx_cases import read_case_rows
    _, rows = read_case_rows(str(p))
    assert len(rows) == 2
    assert rows[0][1][1] == "PAR-001"


def test_report_scenario_rollup_and_rerank_top3():
    rows = [
        {"question_id": "RET-001", "test_stage": "retrieval", "Top@5": 1,
         "hit_rank": 2, "scenario_id": "SCN-001", "answerable": True},
        {"question_id": "RER-001", "test_stage": "rerank", "Top@5": 1,
         "hit_rank": 2, "scenario_id": "SCN-001", "answerable": True},
        {"question_id": "ANS-001", "test_stage": "qa", "fact_recall_rule": 0.5,
         "failure_attribution": "generate", "scenario_id": "SCN-001",
         "answerable": True, "abstention_correct": 1},
    ]
    rep = build_report(rows, latency_stats([10, 20, 30]), False)
    assert rep["metrics"]["RER-Top3"] == {"value": 1.0, "n": 1,
                                          "note": rep["metrics"]["RER-Top3"]["note"]}
    scn = rep["metrics"]["_scenarios"]["SCN-001"]
    assert scn["n"] == 3 and scn["pass"] is False
    assert scn["failed"] == ["ANS-001"]


def test_evidence_cover():
    assert evidence_cover("", ["anything"]) is None
    assert evidence_cover("水压试验18.84MPa；超压试验23.55MPa。",
                          ["水压试验18.84MPa，超压试验23.55MPa。"]) == 1.0
    assert evidence_cover("A1111；B2222；C3333。", ["A1111"]) == 1 / 3
    chunks = [RetrievedChunk(chunk_id="C1", rank=1, content="超压试验23.55MPa")]
    assert evidence_recall_at_k("超压试验23.55MPa。", chunks, 5) == 1.0
    assert evidence_recall_at_k("", chunks, 5) is None


def test_apply_auto_mapping_marks_provenance(tmp_path):
    from scripts.assist_chunk_mapping import apply
    golden = tmp_path / "g.jsonl"
    golden.write_text(json.dumps({
        "question_id": "RET-001", "needs_chunk_mapping": True,
        "remarks": "测试集环节：召回"}) + "\n",
        encoding="utf-8")
    mapping = tmp_path / "m.jsonl"
    mapping.write_text(json.dumps({
        "question_id": "RET-001", "expected_chunk": "CHK-1",
        "expected_document": "DOC-1",
        "auto_coverage": 0.95}) + "\n",
        encoding="utf-8")
    out = tmp_path / "o.jsonl"
    apply(str(mapping), str(golden), str(out))
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["needs_chunk_mapping"] is False
    assert d["expected_chunk"] == "CHK-1"
    assert "auto-mapped覆盖率0.95" in d["remarks"]


def test_gen08_gate_covers_new_accident_types():
    from src.judge import build_review_queue
    rows = [
        {"question_id": "A1", "question_type": "accident_action",
         "question": "q", "predicted_answer": "a", "golden_answer": "g",
         "required_facts": [], "forbidden_facts": [], "evidence_snippet": ""},
        {"question_id": "F1", "question_type": "fact",
         "question": "q", "predicted_answer": "a", "golden_answer": "g",
         "required_facts": [], "forbidden_facts": [], "evidence_snippet": ""},
    ]
    tasks = {(q["question_id"], q["judge_task"]) for q in build_review_queue(rows)}
    assert ("A1", "GEN-08") in tasks
    assert ("F1", "GEN-08") not in tasks


def test_must_cite_default_true_in_scoring():
    from src.metrics_generation import score_generation_rule
    s = score_generation_rule({"answerable": True}, "结论正确【检修规程V3】。")
    assert s["citation_complete_rule"] == 1


def test_document_name_match_and_ndcg_expected_only():
    from src.metrics_retrieval import hit_at_k, ndcg_at_10
    from src.models import GoldenItem
    g = GoldenItem.from_dict({"question_id": "D", "question": "q",
                              "expected_document": "规程20230831"})
    cs = [RetrievedChunk(chunk_id="X", document_id="opaque-id",
                         document_name="规程20230831", rank=2)]
    assert hit_at_k(g, cs, 5) == 1
    g2 = GoldenItem.from_dict({"question_id": "E", "question": "q",
                               "expected_chunk": "CX"})
    cs2 = [RetrievedChunk(chunk_id="CX", document_id="D", rank=1)]
    assert abs(ndcg_at_10(g2, cs2) - 1.0) < 1e-9


def test_review_status_mapping():
    g = build_golden("SCN-001", "RET-001", "召回", "表格参数", "问题？",
                     "要点：A。", "证据", "位置", "doc.docx", "20230831",
                     "自动测试", "审核通过", "", 42, "")
    assert g["review_status"] == "approved"


def test_pipeline_report_dir_parsing():
    from scripts.run_pipeline import report_dir_from_output
    assert report_dir_from_output("...\n报告目录: results/exp1  待复核") == "results/exp1"
    assert report_dir_from_output("no marker") == ""


def test_runner_isolates_retrieval_failure(tmp_path):
    import yaml
    import src.runner as runner
    from src.models import QAOutcome

    class DeadClient:
        def __init__(self, *a, **k):
            pass

        def retrieve(self, *a, **k):
            raise ConnectionError("服务不通")

        def chat(self, *a, **k):
            return QAOutcome(question_id="", answer="x")

    golden = tmp_path / "g.jsonl"
    golden.write_text(
        json.dumps({"question_id": "Q1", "question": "q1",
                    "answerable": True, "chunk_ids": ["C1"]}) + "\n"
        + json.dumps({"question_id": "Q2", "question": "q2",
                      "answerable": True, "chunk_ids": ["C2"]}) + "\n",
        encoding="utf-8")
    cfg = {"ragflow": {"base_url": "http://x", "api_key": "k",
                       "dataset_ids": ["d"], "timeout_s": 5},
           "retrieval": {"page_size": 5},
           "run": {"dataset": str(golden), "output_dir": str(tmp_path / "res"),
                   "experiment_id": "e1", "retrieval_only": True, "qps": 0}}
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    old = runner.RAGFlowClient
    runner.RAGFlowClient = DeadClient
    try:
        rep = runner.run(str(cfg_path))
    finally:
        runner.RAGFlowClient = old
    assert rep["n_questions"] == 2
    assert len(rep["metrics"]["_errors"]) == 2
    rows = [json.loads(l) for l in
            open(tmp_path / "res" / "e1" / "per_question.jsonl", encoding="utf-8")]
    assert all("服务不通" in r.get("run_error", "") for r in rows)
    assert rep["metrics"]["RAG-01"]["value"] is None


def test_report_error_rows_counted_in_scenarios():
    rows = [{"question_id": "R1", "test_stage": "retrieval",
             "scenario_id": "SCN-X", "answerable": True,
             "run_error": "retrieval: boom"}]
    rep = build_report(rows, latency_stats([]), True)
    assert len(rep["metrics"]["_errors"]) == 1
    assert rep["metrics"]["_scenarios"]["SCN-X"]["failed"] == ["R1"]


def test_export_verdict_run_error():
    from scripts.export_xlsx_record import fail_reason, verdict
    r = {"question_id": "Q", "run_error": "retrieval: 401"}
    assert verdict(r, set()) == "待复核"
    assert fail_reason(r).startswith("运行异常")
