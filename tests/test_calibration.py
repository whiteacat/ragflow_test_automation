"""阈值校准纯函数：网格、混淆矩阵、推荐规则。"""
import pytest

from src.calibration import Sample, grid_points, prf, recommend, summarize_scores, sweep


def _samples():
    return [
        Sample("a", "fact", 0.95, 1),
        Sample("b", "fact", 0.80, 1),
        Sample("c", "fact", 0.70, 0),
        Sample("d", "fact", 0.60, 0),
    ]


def test_grid_points_inclusive():
    assert grid_points(0.6, 0.9, 0.1) == [0.6, 0.7, 0.8, 0.9]
    assert grid_points(0.85, 0.85, 0.05) == [0.85]


def test_grid_points_rejects_bad_step():
    with pytest.raises(ValueError):
        grid_points(0.5, 0.9, 0)
    with pytest.raises(ValueError):
        grid_points(0.9, 0.5, 0.1)


def test_prf_zero_denominator():
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)


def test_sweep_confusion_matrix():
    rows = {r["threshold"]: r for r in sweep(_samples(), [0.9, 0.75])}
    assert rows[0.9] == {"threshold": 0.9, "tp": 1, "fp": 0, "fn": 1, "tn": 2,
                         "precision": 1.0, "recall": 0.5, "f1": 0.6667}
    assert rows[0.75] == {"threshold": 0.75, "tp": 2, "fp": 0, "fn": 0, "tn": 2,
                          "precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_recommend_respects_precision_floor():
    rows = sweep([*_samples(), Sample("e", "fact", 0.78, 0)], [0.9, 0.75, 0.65])
    chosen = recommend(rows, min_precision=0.95)
    assert chosen["threshold"] == 0.9
    assert chosen["precision"] >= 0.95


def test_recommend_falls_back_with_reason():
    rows = [{"threshold": 0.6, "tp": 1, "fp": 3, "fn": 1, "tn": 0,
             "precision": 0.25, "recall": 0.5, "f1": 0.3333}]
    chosen = recommend(rows, min_precision=0.95)
    assert chosen["threshold"] == 0.6
    assert "建议先扩充" in chosen["reason"]


def test_summarize_scores_groups_by_label():
    stats = summarize_scores(_samples())
    assert stats["n"] == 4 and stats["n_pos"] == 2 and stats["n_neg"] == 2
    assert stats["pos_min"] == 0.8 and stats["neg_max"] == 0.7
