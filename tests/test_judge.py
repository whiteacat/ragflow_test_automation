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


def test_parse_judge_json_variants():
    from src.judge import parse_judge_json
    assert parse_judge_json('{"score": 2, "reason": "ok"}')["score"] == 2
    fenced = '分析如下：\n```json\n{"score": 1, "reason": "缺条件"}\n```'
    assert parse_judge_json(fenced)["score"] == 1
    prose = '我认为 score: 0，因为无依据。'
    assert parse_judge_json(prose)["score"] == 0
    assert parse_judge_json('完全无法判断')['score'] is None


def test_call_response_format_passthrough(monkeypatch):
    import scripts.judge_review as m
    bodies = []

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": '{"score": 2}'}}]}

    class Post:
        def post(self, url, headers=None, json=None, timeout=None):
            bodies.append(json)
            return Resp()

    import sys as _sys
    _sys.modules.setdefault("requests", Post())
    monkeypatch.setattr("requests.post", Post().post, raising=False)
    import src.judge as j
    out = j.call_openai_compatible("http://x", "k", "mdl", "p", 5, "json_object")
    assert out["score"] == 2
    assert bodies[0]["response_format"] == {"type": "json_object"}
    out2 = j.call_openai_compatible("http://x", "k", "mdl", "p", 5)
    assert "response_format" not in bodies[1]


def test_refusal_iron_rule_in_prompt():
    from src.judge import build_prompt
    p = build_prompt("油压低了怎么办？", "润滑油压0.06时停机。", "", [], [],
                     "证据", "GEN-01", True, "ambiguous")
    assert "应拒答" in p and "一律 0 分" in p
    p2 = build_prompt("压力多少？", "23.55MPa。", "23.55MPa。", ["23.55MPa"],
                      [], "证据", "GEN-01")
    assert "应正常回答" in p2 and "一律 0 分" not in p2


def test_task_scales_and_directions():
    from src.judge import JUDGE_TASKS, SCORE_DIRECTIONS
    assert "只取0/1/2" in JUDGE_TASKS["GEN-01"]
    assert "只取0/1" in JUDGE_TASKS["GEN-04"]
    assert SCORE_DIRECTIONS["GEN-04"] == "lower"
    assert SCORE_DIRECTIONS["GEN-01"] == "higher"


def test_queue_carries_refusal_context():
    from src.judge import build_review_queue
    rows = [{"question_id": "R1", "question_type": "negative",
             "question": "q", "predicted_answer": "a", "golden_answer": "",
             "expected_answer_points": [], "forbidden_claims": [],
             "evidence_snippet": "", "answerable": False,
             "refusal_reason": "ambiguous"}]
    q = build_review_queue(rows, tasks=["GEN-01"])
    assert "应拒答" in q[0]["prompt"]


def test_ragas_like_metrics():
    from scripts import ragas_like as rl
    chunks = [{"rank": 1, "doc": "D", "chunk_id": "C1", "content": "超压试验23.55MPa"},
              {"rank": 2, "doc": "D", "chunk_id": "CX", "content": "无关内容"}]
    clauses = ["超压试验23.55mpa"]
    assert rl.context_precision(chunks, clauses, set(), 5) == 1.0
    assert rl.context_recall(chunks, clauses) == 1.0
    assert rl.faithfulness("超压试验23.55MPa。", chunks) == 1.0
    rel = rl.answer_relevancy("水冷壁超压试验压力是多少？", "超压试验压力23.55MPa。")
    assert 0.0 < rel < 1.0  # 部分覆盖：规则版取二元组召回
    assert rl.context_precision(chunks, [], set(), 5) is None
    assert rl.answer_relevancy("?", "答") is None
    row = {"question_id": "Q", "evidence_snippet": "[1] D#C1: 超压试验23.55MPa",
           "predicted_answer": "超压试验23.55MPa。", "question": "压力是多少？"}
    s = rl.score_row(row, {"evidence_text": "超压试验23.55MPa",
                           "chunk_ids": ["C1"]}, 5)
    assert s["context_precision"] == 1.0 and s["faithfulness"] == 1.0
