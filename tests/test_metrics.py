"""全自动指标单元测试（无需 RAGFlow 服务，纯本地验证）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics_generation import (  # noqa: E402
    abstention_correct,
    citation_complete,
    forbidden_hit,
    fact_recall,
    mandatory_recall,
    order_correct,
    unit_complete,
    percentile,
)
from src.metrics_retrieval import (  # noqa: E402
    first_hit_rank,
    hit_at_k,
    ndcg_at_10,
    paraphrase_group_recall,
    reciprocal_rank,
    rerank_gain,
)
from src.attribution import attribute_failure  # noqa: E402
from src.models import GoldenItem, RetrievedChunk  # noqa: E402
from src.report import build_report  # noqa: E402
from src.runner import dataset_fingerprint  # noqa: E402


def mk(chunks):
    return [RetrievedChunk(chunk_id=c, document_id="D1", rank=i + 1)
            for i, c in enumerate(chunks)]


def test_topk_mrr():
    g = GoldenItem(question_id="T", question="q", chunk_ids=["C3"])
    cs = mk(["C1", "C2", "C3", "C4"])
    assert hit_at_k(g, cs, 5) == 1 and hit_at_k(g, cs, 2) == 0
    assert first_hit_rank(g, cs) == 3
    assert abs(reciprocal_rank(g, cs) - 1 / 3) < 1e-9


def test_miss_zero():
    g = GoldenItem(question_id="T", question="q", chunk_ids=["CX"])
    assert hit_at_k(g, mk(["C1"]), 5) == 0
    assert reciprocal_rank(g, mk(["C1"])) == 0.0


def test_no_golden_excluded():
    g = GoldenItem(question_id="T", question="q")  # 不可答题无证据
    assert hit_at_k(g, mk(["C1"]), 5) is None
    assert ndcg_at_10(g, mk(["C1"])) is None


def test_ndcg_perfect():
    g = GoldenItem(question_id="T", question="q", chunk_ids=["C1", "C2"])
    assert abs(ndcg_at_10(g, mk(["C1", "C2"])) - 1.0) < 1e-9


def test_fact_recall():
    assert fact_recall(["额定电压25kV"], "答：额定电压 25kV，见规程。") == 1.0
    assert fact_recall(["A", "B"], "只有A") == 0.5
    assert fact_recall([], "anything") is None


def test_abstention():
    assert abstention_correct(False, True, "知识库中没有相关内容，无法回答。") == 1
    assert abstention_correct(False, True, "额定电压25kV。") == 0
    assert abstention_correct(True, False, "额定电压25kV。") == 1


def test_forbidden():
    assert forbidden_hit(["带电拆卸"], "可以带电拆卸气路接头。") == 1
    assert forbidden_hit(["带电拆卸"], "严禁带电作业，先断电。") == 0
    assert forbidden_hit([], "anything") is None


def test_citation():
    assert citation_complete(True, "结论正确【检修规程V3第3.2条】。") == 1
    assert citation_complete(True, "结论正确。") == 0
    assert citation_complete(False, "结论正确。") is None


def test_percentile():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5.0
    assert percentile([], 50) is None


def test_dataset_fingerprint(tmp_path):
    import hashlib

    p = tmp_path / "golden.jsonl"
    p.write_text('{"question_id": "Q1"}\n', encoding="utf-8")
    fp = dataset_fingerprint(str(p), [{"question_id": "Q1"}])
    assert fp["path"] == str(p)
    assert fp["n_items"] == 1
    assert fp["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert isinstance(fp["mtime"], float)


def test_build_report_dataset_passthrough():
    from src.metrics_generation import latency_stats

    rep = build_report([], latency_stats([]), True)
    assert rep["n_questions"] == 0
    assert rep["dataset"] is None  # 老调用方式兼容
    ds = {"path": "d.jsonl", "sha256": "abc", "n_items": 6, "mtime": 1.0}
    rep2 = build_report([], latency_stats([]), True, dataset_info=ds,
                        retrieval_config={"top_k": 1024})
    assert rep2["dataset"] == ds
    assert rep2["retrieval"] == {"top_k": 1024}


def test_table9_compat_old_keys():
    g = GoldenItem.from_dict({"question_id": "Q1", "question": "q",
                              "question_type": "accident",
                              "required_facts": ["A", "B"],
                              "chunk_ids": ["C1"], "must_abstain": True})
    assert g.question_type == "accident_action"
    assert g.expected_answer_points == ["A", "B"]
    assert g.should_refuse is True and g.test_stage == "refusal"


def test_table9_new_keys():
    g = GoldenItem.from_dict({"case_id": "PAR-1", "question": "q",
                              "test_stage": "qa", "question_type": "table",
                              "expected_answer_points": ["23.55MPa"],
                              "expected_chunk": "CX", "acceptable_chunks": ["CY"],
                              "refusal_reason": "", "review_status": "approved",
                              "reviewer": "张工"})
    assert g.question_id == "PAR-1" and "CX" in g.chunk_ids
    cs = mk(["CY"])
    assert hit_at_k(g, cs, 5) == 1


def test_order_unit_mandatory():
    req = ["先查风源压力", "再查升弓电磁阀", "最后查控制回路"]
    assert order_correct(req, "先查风源压力，再查升弓电磁阀，最后查控制回路。") == 1
    assert order_correct(req, "最后查控制回路，再查升弓电磁阀，先查风源压力。") == 0
    assert unit_complete(["超压试验压力23.55MPa"], "压力为23.55 MPa。") == 1
    assert unit_complete(["超压试验压力23.55MPa"], "压力为23.55。") == 0
    assert mandatory_recall(["M12", "85N·m"], "M12 螺栓 85N·m") == 1.0


def test_rerank_gain_and_paraphrase():
    assert rerank_gain(5, 1) == 1
    assert rerank_gain(2, 2) == 0
    assert rerank_gain(1, 8) == -1
    rows = [{"question_id": "a", "paraphrase_group": "G1", "Top@5": 1},
            {"question_id": "b", "paraphrase_group": "G1", "Top@5": 0}]
    assert paraphrase_group_recall(rows) == {"G1": 0.5}


def test_attribution_eight_classes():
    assert attribute_failure({"answerable": True, "Top@5": 0,
                              "hit_rank": None}) == "recall"
    assert attribute_failure({"answerable": True, "Top@5": 1, "Top@20": 1,
                              "hit_rank": 8,
                              "fact_recall_rule": 1.0}) == "rerank"
    assert attribute_failure({"answerable": True, "Top@5": 1, "hit_rank": 1,
                              "fact_recall_rule": 0.5}) == "generate"
    assert attribute_failure({"answerable": False, "should_refuse": True,
                              "abstention_correct": 0}) == "policy"
