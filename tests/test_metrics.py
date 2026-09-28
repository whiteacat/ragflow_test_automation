"""全自动指标单元测试（无需 RAGFlow 服务，纯本地验证）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics_generation import (  # noqa: E402
    abstention_correct,
    citation_complete,
    forbidden_hit,
    fact_recall,
    percentile,
)
from src.metrics_retrieval import (  # noqa: E402
    first_hit_rank,
    hit_at_k,
    ndcg_at_10,
    reciprocal_rank,
)
from src.models import GoldenItem, RetrievedChunk  # noqa: E402


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
