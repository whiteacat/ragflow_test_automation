"""校准候选抽取：片段解析、预标判据、去重键、限额优先级。"""
import json

from scripts.make_calibration_set import (
    _dedupe_key,
    load_chunks,
    parse_snippet,
    prelabel_evidence,
    prelabel_fact,
    select,
)


def test_parse_snippet_splits_chunks():
    snippet = ("[1] 汽机规程#c1: 给水泵出口压力为0.6MPa，转速2980r/min。\n"
               "[2] 锅炉规程#c9: 水压试验压力为18.84MPa：表4-2。")
    chunks = parse_snippet(snippet)
    assert [c["chunk_id"] for c in chunks] == ["c1", "c9"]
    assert chunks[0]["document"] == "汽机规程"
    assert chunks[1]["content"] == "水压试验压力为18.84MPa：表4-2。"


def test_parse_snippet_empty():
    assert parse_snippet("") == []


def test_prelabel_fact_literal_and_number_mismatch():
    assert prelabel_fact("给水泵出口压力0.6MPa", "给水泵出口压力为0.6MPa") == (1, "literal")
    assert prelabel_fact("水压试验压力18.84MPa", "水压试验压力为18.8MPa") == (0, "number_mismatch")


def test_prelabel_fact_needs_review_for_paraphrase():
    assert prelabel_fact("给水泵出口压力0.6MPa", "泵的出口压力须保持在0.6MPa") == (None, "needs_review")


def test_prelabel_evidence_literal_after_normalization():
    assert prelabel_evidence("给水泵出口压力不低于0.6MPa",
                             "给水泵出口压力不低于0.6MPa。") == (1, "literal")
    # 归一化会去掉“的/为”，写法差异不应掉成人工
    assert prelabel_evidence("给水泵出口压力不低于0.6MPa",
                             "给水泵的出口压力不低于0.6MPa") == (1, "literal")


def test_prelabel_evidence_number_conflict_only_with_full_content():
    clause, chunk = "水压试验压力18.84MPa", "水压试验压力为23.55MPa"
    assert prelabel_evidence(clause, chunk, full_content=True) == (0, "number_mismatch")
    assert prelabel_evidence(clause, chunk, full_content=False) == (None, "needs_review")


def test_dedupe_key_separates_evidence_and_fuzzy():
    evidence = {"kind": "evidence", "evidence": "同一句", "chunk": "同一片段"}
    fuzzy = {"kind": "fuzzy", "clause": "同一句", "chunk": "同一片段"}
    fact = {"kind": "fact", "fact": "同一句", "answer": "同一片段"}
    keys = {_dedupe_key(evidence), _dedupe_key(fuzzy), _dedupe_key(fact)}
    assert len(keys) == 3


def test_select_keeps_review_items_first():
    items = [
        {"kind": "fact", "basis": "literal", "needs_review": False},
        {"kind": "fact", "basis": "needs_review", "needs_review": True},
        {"kind": "fact", "basis": "cross_question", "needs_review": False},
        {"kind": "fact", "basis": "number_mismatch", "needs_review": False},
    ]
    import random

    picked = select(items, max_per_kind=2, rng=random.Random(1))
    assert len(picked) == 2
    assert picked[0]["basis"] == "needs_review"


def test_load_chunks_reads_candidates(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text(json.dumps({
        "question_id": "Q1",
        "candidates": [
            {"rank": 1, "chunk_id": "c1", "document_name": "汽机规程", "content": "全文一"},
            {"rank": 2, "chunk_id": "c2", "document_name": "汽机规程", "content": ""},
        ],
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    mapping = load_chunks(str(path))
    assert [c["chunk_id"] for c in mapping["Q1"]] == ["c1"]
