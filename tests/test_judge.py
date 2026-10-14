"""LLM 复核执行器单测（不调真实模型，桩掉网络调用）。"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts import judge_review as jr  # noqa: E402


def _write_dir(d, queue, prompts, perq=()):
    with open(os.path.join(d, "review_queue.jsonl"), "w", encoding="utf-8") as f:
        for q in queue:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(os.path.join(d, "review_prompts.jsonl"), "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    if perq:
        with open(os.path.join(d, "per_question.jsonl"), "w", encoding="utf-8") as f:
            for r in perq:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_select_filters_tasks_and_risk(tmp_path):
    d = str(tmp_path)
    queue = [
        {"question_id": "Q1", "judge_task": "GEN-01"},
        {"question_id": "Q1", "judge_task": "GEN-04"},
        {"question_id": "Q2", "judge_task": "GEN-01"},
    ]
    _write_dir(d, queue, [], [{"question_id": "Q1", "risk_level": "critical"},
                              {"question_id": "Q2", "risk_level": "low"}])
    _, q, perq = jr.load_inputs(d)
    assert len(jr.select(q, perq, {"GEN-01"}, set())) == 2
    assert len(jr.select(q, perq, set(), {"critical"})) == 2
    assert len(jr.select(q, perq, {"GEN-04"}, {"low"})) == 0


def test_resume_skips_scored(tmp_path):
    d = str(tmp_path)
    queue = [{"question_id": "Q1", "judge_task": "GEN-01"}]
    _write_dir(d, queue, [])
    with open(os.path.join(d, "review_scores.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"question_id": "Q1", "judge_task": "GEN-01",
                            "score": 2}) + "\n")
    done = jr.load_done(d)
    todo = [q for q in queue
            if (q["question_id"], q["judge_task"]) not in done]
    assert todo == []


def test_call_with_retry_gives_up(monkeypatch):
    import scripts.judge_review as m
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise ConnectionError("down")

    monkeypatch.setattr(m, "call_openai_compatible", boom)
    res = jr.call_with_retry("http://x", "k", "mdl", "p", 1, 2)
    assert res["score"] is None and "down" in res["error"]
    assert len(calls) == 3
