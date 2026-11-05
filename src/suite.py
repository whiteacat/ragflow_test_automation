"""测试主线：统一的阶段清单与结果装配。

主线固定 7 个阶段（S0~S6），由 ``scripts/run_pipeline.py`` 按序执行；
每个阶段结束后把一行记录追加到 ``results/<exp>/pipeline_stages.jsonl``，
最后由 ``scripts/export_suite_xlsx.py`` 把全部产物汇成一个 ``suite.xlsx``。

这里只放"与执行环境无关"的纯逻辑（阶段表、读写、聚合），
不 import 任何脚本，方便单测。
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# 主线阶段：顺序即依赖顺序，不可颠倒
# ---------------------------------------------------------------------------
STAGES: List[tuple] = [
    ("S0", "数据准备", "xlsx → golden/baseline，字段校验 + 覆盖率"),
    ("S1", "解析评测", "parse_benchmark → DATA-01 四项 + 异常清单"),
    ("S2", "Chunk映射", "候选导出 → 人工勾选 → 写回 golden"),
    ("S3", "环境检查", "health_check 探活 RAGFlow / chat"),
    ("S4", "全链路运行", "run_all → per_question / summary / report"),
    ("S5", "复核", "judge_review（可选）→ 逐条复核分"),
    ("S6", "结果归档", "export_suite_xlsx → 单一 suite.xlsx"),
]
STAGE_NAMES: Dict[str, str] = {s: n for s, n, _ in STAGES}
STAGE_DESC: Dict[str, str] = {s: d for s, _, d in STAGES}
MANIFEST = "pipeline_stages.jsonl"


def exp_dir_from_config(cfg: Dict[str, Any], root: str = ".") -> str:
    """与 src/runner.py 的产物目录口径保持一致（run 段优先）。"""
    run = cfg.get("run") or {}
    out = run.get("output_dir", cfg.get("output_dir", "results"))
    exp = run.get("experiment_id", cfg.get("experiment_id", "exp"))
    return os.path.join(root, out, str(exp))


def record(exp_dir: str, stage: str, status: str, elapsed_s: float = 0.0,
           artifact: str = "", note: str = "") -> Dict[str, Any]:
    """追加一条阶段记录。status: ok / skip / fail。"""
    os.makedirs(exp_dir, exist_ok=True)
    row = {
        "stage": stage,
        "name": STAGE_NAMES.get(stage, stage),
        "status": status,
        "elapsed_s": round(float(elapsed_s), 2),
        "artifact": artifact,
        "note": note,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(exp_dir, MANIFEST), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def load_json(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_stages(exp_dir: str) -> List[Dict[str, Any]]:
    """按 S0..S6 排序返回全部记录（同一阶段的多个子步骤都保留）。"""
    order = {s: i for i, (s, _, _) in enumerate(STAGES)}
    rows = load_jsonl(os.path.join(exp_dir, MANIFEST))
    return sorted(rows, key=lambda r: (order.get(str(r.get("stage")), 99),
                                       str(r.get("ts", ""))))


def latest_stages(stages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """每个阶段取最后一次记录，用于「运行信息」的完成情况一行。"""
    last: Dict[str, Dict[str, Any]] = {}
    for row in stages:
        last[str(row.get("stage"))] = row
    return last


# ---------------------------------------------------------------------------
# 人工评分回收：重新生成 suite.xlsx 时不能丢掉已填的分数
# ---------------------------------------------------------------------------
SCORE_SHEET = "人工评分表"


def existing_scores(path: str) -> Dict[tuple, tuple]:
    """读回已有 xlsx 的（题目编号, 指标编号）→（人工评分, 评分理由）。"""
    if not os.path.exists(path):
        return {}
    try:
        from openpyxl import load_workbook
    except ImportError:  # pragma: no cover - openpyxl 是硬依赖
        return {}
    wb = load_workbook(path, data_only=True)
    if SCORE_SHEET not in wb.sheetnames:
        return {}
    ws = wb[SCORE_SHEET]
    out: Dict[tuple, tuple] = {}
    for row in ws.iter_rows(min_row=2, max_col=18, values_only=True):
        qid, mid = row[0], row[12]
        score, reason = row[16], row[17]
        if not qid or not mid or score in (None, ""):
            continue
        out[(str(qid), str(mid))] = (score, reason or "")
    return out


# ---------------------------------------------------------------------------
# 主观指标聚合（人工填分后重跑 S6 即可得到）
# ---------------------------------------------------------------------------
def aggregate_subjective(scores: Dict[tuple, tuple],
                         meta: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按指标聚合人工分：n / 分布 / 均值 / 满分率（越低越好的指标看 0 分率）。"""
    buckets: Dict[str, List[Any]] = {}
    for (_, mid), (score, _) in scores.items():
        try:
            buckets.setdefault(mid, []).append(float(score))
        except (TypeError, ValueError):
            continue
    rows = []
    for mid, vals in buckets.items():
        if not vals:
            continue
        m = meta.get(mid, {})
        lower_better = m.get("direction") == "lower"
        top = min(vals) if lower_better else max(vals)
        dist: Dict[str, int] = {}
        for v in vals:
            key = str(int(v)) if v == int(v) else str(v)
            dist[key] = dist.get(key, 0) + 1
        rows.append({
            "指标编号": mid,
            "指标名称": m.get("name", ""),
            "样本数": len(vals),
            "均值": round(sum(vals) / len(vals), 4),
            "满分/达标率": round(sum(1 for v in vals if v == top) / len(vals), 4),
            "分布": " / ".join(f"{k}:{v}" for k, v in sorted(dist.items())),
            "阈值": m.get("threshold", ""),
        })
    return sorted(rows, key=lambda r: r["指标编号"])


# ---------------------------------------------------------------------------
# 产物装配
# ---------------------------------------------------------------------------
ARTIFACTS = {
    "per_question": "per_question.jsonl",
    "summary": "summary.json",
    "parse_summary": "parse_summary.json",
    "parse_anomalies": "parse_anomalies.jsonl",
    "dataset_warnings": "dataset_warnings.jsonl",
    "review_queue": "review_queue.jsonl",
    "review_scores": "review_scores.jsonl",
    "report": "report.md",
}


def collect(exp_dir: str) -> Dict[str, Any]:
    """一次性读齐 run 目录下的全部产物，缺文件返回空值不报错。"""
    data: Dict[str, Any] = {"dir": exp_dir, "stages": load_stages(exp_dir)}
    for key, name in ARTIFACTS.items():
        path = os.path.join(exp_dir, name)
        if key in ("summary", "parse_summary"):
            data[key] = load_json(path)
        elif key == "report":
            data[key] = path if os.path.exists(path) else ""
        else:
            data[key] = load_jsonl(path)
    return data


def config_snapshot(config_path: str) -> Dict[str, Any]:
    """配置快照，敏感字段脱敏。"""
    if not config_path or not os.path.exists(config_path):
        return {}
    try:
        import yaml
    except ImportError:  # pragma: no cover
        return {}
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    rag = cfg.get("ragflow") or {}
    if rag.get("api_key"):
        rag["api_key"] = "***"
    return cfg
