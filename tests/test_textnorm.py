"""指标口径修正用例：归一化、否定语境、拒答三态、数值容差、引用识别。"""
from src.metrics_generation import (
    abstention_correct,
    abstention_state,
    fact_covered,
    forbidden_hit,
    forbidden_scan,
    has_citation,
    is_abstention,
    normalize_text,
)
from src.metrics_retrieval import cross_doc_coverage, evidence_cover
from src.models import GoldenItem, RetrievedChunk
from src.textnorm import normalize, polarity


def test_normalize_ignores_zero_width_and_hidden_spaces():
    assert normalize("1\u200b8.84\u00a0MPa") == normalize("18.84MPa")
    assert normalize("18.84 MPa") == "18.84mpa"


def test_normalize_strips_thousands_separator_only():
    assert normalize("1,234kV") == "1234kv"
    assert normalize("18,84MPa") == "18,84mpa"


def test_normalize_range_symbol_only_between_digits():
    assert normalize("18.84至23.55MPa") == "18.84~23.55mpa"
    assert normalize("至少检查支吊架") == "至少检查支吊架"


def test_normalize_unit_alias_equivalence():
    assert normalize("23.55兆帕") == normalize("23.55 MPa")
    assert normalize("540摄氏度") == normalize("540℃")


def test_polarity_warn_deny_affirm():
    a = normalize("严禁按再热器5.7MPa执行")
    i = a.find("5.7mpa")
    assert polarity(a, i, i + len("5.7mpa")) == "warn"

    b = normalize("水压试验不使用18.84MPa")
    j = b.find("18.84")
    assert polarity(b, j, j + len("18.84")) == "deny"

    c = normalize("水压试验压力为18.84MPa")
    k = c.find("18.84")
    assert polarity(c, k, k + len("18.84")) == "affirm"


def test_polarity_does_not_cross_clause():
    s = normalize("严禁超压；再热器5.7MPa见附录")
    i = s.find("5.7mpa")
    assert polarity(s, i, i + len("5.7mpa")) == "affirm"


def test_fact_covered_rejects_denied_claim():
    assert not fact_covered("水压试验18.84MPa", "水压试验不使用18.84MPa")
    assert not fact_covered("水压试验18.84MPa", "水压试验不适用18.84MPa")
    assert fact_covered("水压试验18.84MPa", "水压试验压力为18.84MPa")


def test_fact_covered_keeps_requirement_wording():
    assert fact_covered("水压试验18.84MPa", "水压试验压力不得低于18.84MPa")


def test_forbidden_warning_is_not_violation():
    assert forbidden_hit(["再热器5.7MPa"], "严禁按再热器5.7MPa执行") == 0
    assert forbidden_hit(["再热器5.7MPa"], "不得使用再热器5.7MPa的数据") == 0
    assert forbidden_hit(["带电拆卸"], "可以带电拆卸气路接头。") == 1


def test_forbidden_scan_separates_hard_and_soft():
    assert forbidden_scan(["再热器5.7MPa"], "严禁按再热器5.7MPa执行") == {
        "hard": [], "soft": ["再热器5.7MPa"]}
    assert forbidden_scan(["再热器5.7MPa"], "按再热器5.7MPa执行") == {
        "hard": ["再热器5.7MPa"], "soft": []}
    assert forbidden_scan([], "anything") is None


def test_abstention_ignores_uncertainty_terms():
    assert abstention_state("测量不确定度为±0.5%。") == 0
    assert abstention_state("不确定因素包括煤质波动。") == 0
    assert not is_abstention("测量不确定度为±0.5%。")


def test_abstention_detects_extended_phrases():
    assert abstention_correct(False, True, "资料中未提及该参数。") == 1
    assert abstention_correct(False, True, "该问题超出知识范围。") == 1
    assert abstention_correct(False, True, "暂无法提供该参数。") == 1


def test_abstention_long_preamble_is_uncertain():
    ans = "无法回答：" + "现有资料仅覆盖2023版，" * 20 + "建议补充入库后再查询。"
    assert len(normalize_text(ans)) > 200
    assert abstention_state(ans) is None
    assert abstention_correct(False, True, ans) is None


def test_fact_covered_requires_exact_numbers():
    assert not fact_covered("水压试验18.84MPa", "水压试验压力为18.8MPa")
    assert fact_covered("水压试验18.84MPa", "水压试验压力为18.840MPa")
    assert not fact_covered("水压试验18.84MPa", "水压试验压力为18.9MPa")
    assert fact_covered("过热器出口温度541℃", "过热器出口温度为541°C")


def test_fact_covered_unit_conversion():
    assert fact_covered("给水泵出口压力0.6MPa", "给水泵出口压力600kPa")
    assert fact_covered("给水泵电机功率1kW", "给水泵电机功率1000W")
    assert not fact_covered("给水泵出口压力0.6MPa", "给水泵出口压力0.8MPa")


def test_fact_covered_rejects_dimension_mismatch():
    assert not fact_covered("额定电压5.7MPa", "额定电压5.7kV")


def test_fact_covered_range_endpoints_only():
    assert fact_covered("水压试验18.84~23.55MPa", "水压试验压力为18.84MPa")
    assert fact_covered("水压试验18.84~23.55MPa", "水压试验压力为23.55MPa")
    assert not fact_covered("水压试验18.84~23.55MPa", "水压试验压力为20MPa")
    assert not fact_covered("水压试验18.84~23.55MPa", "水压试验压力为30MPa")


def test_citation_detects_numbered_reference_in_tail():
    assert has_citation("过热器出口温度541℃，详见运行规程。[1]")
    assert has_citation("过热器出口温度541℃[^1]")


def test_citation_ignores_numbering_in_body():
    body = "第一步检查支吊架（1），第二步缓慢升压。" + "升压过程中注意监视压力变化，" * 20
    assert not has_citation(body)


def test_citation_detects_book_title_and_clause():
    assert has_citation("依据《600MW机组运行规程》4.2条执行。")
    assert has_citation("根据 DL/T 1234 规定执行。")


def test_evidence_cover_normalizes_units_and_spaces():
    assert evidence_cover("18.84 MPa", ["…试验压力为18.84MPa…"]) == 1.0
    assert evidence_cover("18.84 MPa", ["…试验压力为23.55MPa…"]) == 0.0


def test_evidence_cover_fuzzy_clause():
    assert evidence_cover("水压试验压力为18.84MPa", ["水压试验压力应为18.84MPa"]) == 1.0


def test_cross_doc_uses_acceptable_chunks():
    golden = GoldenItem(question_id="q1", question_type="multi_document",
                        acceptable_chunks=["c2", "c3"])
    chunks = [RetrievedChunk(chunk_id="c2", document_id="d1", rank=1),
              RetrievedChunk(chunk_id="c3", document_id="d2", rank=2)]
    assert cross_doc_coverage(golden, chunks) == 1
