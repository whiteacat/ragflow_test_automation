"""候选清单生成/自动映射的离线回归（不打网络，用假客户端）。"""
import json

import scripts.assist_chunk_mapping as acm
from src.models import RetrievedChunk


def _golden(tmp_path):
    rows = [
        {"question_id": "Q1", "question": "水压试验压力是多少？",
         "evidence_text": "水压试验压力18.84MPa", "needs_chunk_mapping": True},
        {"question_id": "Q2", "question": "已映射，不再处理",
         "needs_chunk_mapping": False},
        {"question_id": "Q3", "question": "检索会失败的题",
         "evidence_text": "无关证据", "needs_chunk_mapping": True},
    ]
    path = tmp_path / "golden.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")
    return str(path)


def _fake_env(monkeypatch):
    monkeypatch.setattr("src.config.load_config", lambda _p: {
        "ragflow": {"base_url": "http://stub", "api_key": "k",
                    "dataset_ids": ["ds1"]},
        "retrieval": {}})


class _Client:
    def __init__(self, *_a, **_k):
        pass

    def retrieve(self, question, ds, **_kw):
        if "失败" in question:
            raise RuntimeError("boom")
        return {"chunks": [
            RetrievedChunk(chunk_id="c1", document_id="d1",
                           document_name="doc.pdf", content="水压试验压力18.84MPa",
                           score=0.91234, rank=1),
            RetrievedChunk(chunk_id="c2", document_id="d1",
                           document_name="doc.pdf", content="其他内容",
                           score=0.5, rank=2),
        ]}


def test_assist_writes_candidates_and_survives_failures(tmp_path, monkeypatch, capsys):
    """assist 的计数变量曾是 list，`n += 1` 直接 TypeError 崩掉全部候选生成。"""
    _fake_env(monkeypatch)
    monkeypatch.setattr("src.ragflow_client.RAGFlowClient", _Client)
    out = tmp_path / "cand.jsonl"

    acm.assist(_golden(tmp_path), "unused.yaml", str(out), 10)

    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert [x["question_id"] for x in lines] == ["Q1"]
    assert [c["chunk_id"] for c in lines[0]["candidates"]] == ["c1", "c2"]
    assert lines[0]["candidates"][0]["score"] == 0.9123
    text = capsys.readouterr().out
    assert "待映射 1 题" in text
    assert "失败 1 题" in text


def test_auto_map_marks_manual_when_below_threshold(tmp_path, monkeypatch, capsys):
    _fake_env(monkeypatch)
    monkeypatch.setattr("src.ragflow_client.RAGFlowClient", _Client)
    out = tmp_path / "mapping.jsonl"

    acm.auto_map(_golden(tmp_path), "unused.yaml", str(out), 0.8, 10)

    got = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert [x["question_id"] for x in got] == ["Q1"]
    assert got[0]["expected_chunk"] == "c1"
    assert got[0]["auto_coverage"] == 1.0
    assert "低于阈值转人工 1 题" in capsys.readouterr().out
