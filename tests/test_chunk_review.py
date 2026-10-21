"""人工勾选 xlsx 的导出/回收单测（无需 RAGFlow 服务）。"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openpyxl import Workbook  # noqa: E402

from scripts.export_chunk_candidates_xlsx import (  # noqa: E402
    HEADERS,
    read_selection,
    write_mapping,
)


def _mk_xlsx(path, rows):
    wb = Workbook()
    ws = wb.active
    ws.title = "候选清单"
    ws.append(HEADERS)
    for r in rows:
        ws.append(r)
    wb.save(path)


def _row(rank, chunk_id, selected, qid="Q1"):
    r = [""] * len(HEADERS)
    r[HEADERS.index("题号")] = qid
    r[HEADERS.index("候选排名")] = rank
    r[HEADERS.index("chunk_id")] = chunk_id
    r[HEADERS.index("选中(填1)")] = selected
    return r


def test_read_selection_sorts_by_rank_and_skips_unchecked(tmp_path):
    p = tmp_path / "r.xlsx"
    _mk_xlsx(p, [_row(3, "C3", 1), _row(1, "C1", "1"),
                 _row(2, "C2", None), _row(5, "C5", "是")])
    assert read_selection(p) == {"Q1": ["C1", "C3", "C5"]}


def test_read_selection_zero_and_blank_chunk_ignored(tmp_path):
    p = tmp_path / "r.xlsx"
    _mk_xlsx(p, [_row(1, "C1", 0), _row(2, "", 1), _row(3, "C3", "yes")])
    assert read_selection(p) == {"Q1": ["C3"]}


def test_read_selection_none_returns_empty(tmp_path):
    p = tmp_path / "r.xlsx"
    _mk_xlsx(p, [_row(1, "C1", None)])
    assert read_selection(p) == {}


def test_write_mapping_first_is_expected_rest_acceptable(tmp_path):
    out = tmp_path / "m.jsonl"
    write_mapping({"Q1": ["C1", "C2"], "Q2": ["C9"]}, out)
    rows = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    assert rows[0] == {"question_id": "Q1", "expected_chunk": "C1",
                       "acceptable_chunks": ["C2"]}
    assert rows[1] == {"question_id": "Q2", "expected_chunk": "C9",
                       "acceptable_chunks": []}
