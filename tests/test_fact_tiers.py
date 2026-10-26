"""P2：同义词表、邻近窗口、事实分档评分。"""
import time

import pytest

from src import synonyms
from src.metrics_generation import (
    PROXIMITY_WINDOW,
    fact_covered,
    fact_credit,
    fact_recall,
    fact_recall_relaxed,
    fact_uncertain,
    fact_uncertain_count,
    order_correct,
    score_generation_rule,
    unit_complete,
)


@pytest.fixture(autouse=True)
def _reset_synonyms():
    synonyms.reset()
    yield
    synonyms.reset()


def _write(tmp_path, body: str) -> str:
    path = tmp_path / "synonyms.yaml"
    path.write_text(body, encoding="utf-8")
    return str(path)


def test_parse_list_form_and_transitive_closure(tmp_path):
    synonyms.configure(_write(tmp_path, "A: [B]\nB: [C]\n"))
    assert set(synonyms.expand("a")) == {"b", "c"}
    assert set(synonyms.expand("c")) == {"a", "b"}


def test_parse_dict_form_with_core(tmp_path):
    synonyms.configure(_write(tmp_path, "水压试验:\n  synonyms: [打压试验]\n  core: [水压, 试验]\n"))
    assert synonyms.expand("水压试验") == ("打压试验",)
    assert synonyms.core("打压试验") == ("水压", "试验")


def test_variants_replace_term_inside_longer_label(tmp_path):
    synonyms.configure(_write(tmp_path, "汽轮机: [汽机]\n"))
    assert "汽机润滑油压" in synonyms.variants("汽轮机润滑油压")


def test_hot_reload_picks_up_new_file(tmp_path, monkeypatch):
    path = _write(tmp_path, "A: [B]\n")
    synonyms.configure(path)
    assert synonyms.expand("a") == ("b",)

    with open(path, "w", encoding="utf-8") as f:
        f.write("A: [C]\n")
    future = time.time() + 5
    import os
    os.utime(path, (future, future))
    monkeypatch.setattr(synonyms, "_last_check", 0.0)
    assert synonyms.expand("a") == ("c",)


def test_credit_exact_and_synonym_are_full_score(tmp_path):
    synonyms.configure(_write(tmp_path, "汽轮机: [汽机]\n"))
    assert fact_credit("给水泵出口压力0.6MPa", "给水泵出口压力为0.6MPa") == 1.0
    assert fact_credit("汽轮机润滑油压0.15MPa", "汽机润滑油压为0.15MPa") == 1.0


def test_credit_core_word_match_is_near(tmp_path):
    synonyms.configure(_write(tmp_path, "水压试验:\n  core: [水压, 试验]\n"))
    assert fact_credit("水压试验18.84MPa", "水压合格，试验记录18.84MPa") == 0.8


def test_credit_labels_without_numbers_is_uncertain():
    assert fact_credit("水压试验18.84MPa", "水压试验合格") == 0.5
    assert fact_covered("水压试验18.84MPa", "水压试验合格") is False
    assert fact_uncertain("水压试验18.84MPa", "水压试验合格") is True


def test_credit_window_boundary():
    fact = "水压试验18.84MPa"
    near = "水压试验" + "说" * (PROXIMITY_WINDOW - len("水压试验")) + "18.84MPa"
    far = "水压试验" + "说" * (PROXIMITY_WINDOW - len("水压试验") + 1) + "18.84MPa"
    assert fact_credit(fact, near) == 1.0
    assert fact_credit(fact, far) == 0.5


def test_credit_zero_when_label_absent_or_denied():
    assert fact_credit("给水泵出口压力0.6MPa", "凝结水泵出口压力0.6MPa") == 0.0
    assert fact_credit("水压试验18.84MPa", "水压试验不使用18.84MPa") == 0.0


def test_relaxed_recall_and_uncertain_count():
    facts = ["给水泵出口压力0.6MPa", "水压试验18.84MPa"]
    answer = "给水泵出口压力为0.6MPa；水压试验合格"
    assert fact_recall(facts, answer) == 0.5
    assert fact_recall_relaxed(facts, answer) == pytest.approx(0.75)
    assert fact_uncertain_count(facts, answer) == 1


def test_score_generation_rule_exposes_both_metrics():
    question = {"expected_answer_points": ["给水泵出口压力0.6MPa"], "answerable": True}
    out = score_generation_rule(question, "给水泵出口压力为0.6MPa")
    assert out["fact_recall_rule"] == 1.0
    assert out["fact_recall_relaxed"] == 1.0
    assert out["fact_uncertain_count"] == 0


def test_order_ignores_passing_mention_before_body():
    req = ["水压试验18.84MPa", "再热器温度540℃"]
    answer = ("如前所述，水压试验要求严格。1) 开启进水阀；2) 水压试验压力为18.84MPa；"
              "3) 关闭阀门，记录再热器温度540℃。")
    assert order_correct(req, answer) == 1


def test_order_falls_back_to_label_when_number_missing():
    req = ["水压试验18.84MPa", "再热器温度540℃"]
    assert order_correct(req, "水压试验合格；再热器温度540℃") == 1


def test_order_none_only_when_fact_absent():
    req = ["水压试验18.84MPa", "再热器温度540℃"]
    assert order_correct(req, "水压试验压力为18.84MPa") is None


def test_unit_ignores_absent_fact_units():
    req = ["水压试验18.84MPa", "凝结水泵流量100t/h"]
    assert unit_complete(req, "水压试验压力为18.84MPa") == 1
