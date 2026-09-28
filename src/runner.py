"""一键执行：检索基准 + 生成基准 + 聚合（全自动指标部分）。

流程：
1. 读 golden JSONL
2. retrieval_only=True  → 只调 /api/v1/retrieval，算 RAG-01~05/09/10 + 延迟
3. 全链路 → 检索 + chat 问答，叠加 GEN 规则分 + TRACE-01 + review_queue（半自动）
4. 落盘 per_question.jsonl / summary.json / review_queue.jsonl
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from .config import load_config
from .judge import build_review_queue
from .metrics_generation import latency_stats, score_generation_rule
from .metrics_retrieval import score_retrieval
from .models import GoldenItem
from .ragflow_client import RAGFlowClient
from .report import build_report


def load_golden(path: str, limit=None) -> List[Dict[str, Any]]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items[:limit] if limit else items


def run(cfg_path: str) -> Dict[str, Any]:
    cfg = load_config(cfg_path)
    rag = cfg["ragflow"]
    ret_cfg = cfg.get("retrieval", {})
    run_cfg = cfg.get("run", {})
    client = RAGFlowClient(rag["base_url"], rag["api_key"],
                           timeout_s=int(rag.get("timeout_s", 60)))
    dataset_ids: List[str] = list(rag.get("dataset_ids") or [])
    chat_id: str = str(rag.get("chat_id") or "")

    golden_raw = load_golden(run_cfg["dataset"], run_cfg.get("max_questions"))
    retrieval_only = bool(run_cfg.get("retrieval_only", False))
    qps = float(run_cfg.get("qps", 2) or 2)
    out_dir = os.path.join(run_cfg.get("output_dir", "results"),
                           str(run_cfg.get("experiment_id", "exp")))
    os.makedirs(out_dir, exist_ok=True)

    per_q: List[Dict[str, Any]] = []
    e2e_all: List[float] = []
    for g in golden_raw:
        item = GoldenItem.from_dict(g)
        # --- 检索 ---
        ret = client.retrieve(
            item.question, dataset_ids,
            page=int(ret_cfg.get("page", 1)),
            page_size=int(ret_cfg.get("page_size", 30)),
            top_k=int(ret_cfg.get("top_k", 1024)),
            similarity_threshold=float(ret_cfg.get("similarity_threshold", 0.2)),
            vector_similarity_weight=float(ret_cfg.get("vector_similarity_weight", 0.3)),
            keyword=bool(ret_cfg.get("keyword", False)),
            highlight=bool(ret_cfg.get("highlight", False)),
            rerank_id=ret_cfg.get("rerank_id"))
        chunks = ret["chunks"]
        r_scores = score_retrieval(item, chunks)
        row: Dict[str, Any] = {
            "question_id": item.question_id,
            "experiment_id": run_cfg.get("experiment_id"),
            "question_type": item.question_type,
            "risk_level": item.risk_level,
            "answerable": item.answerable,
            **{k: v for k, v in r_scores.items()},
            "retrieval_latency_ms": ret["latency_ms"],
            "evidence_snippet": "\n".join(
                f"[{c.rank}] {c.document_name}#{c.chunk_id}: {c.content[:300]}"
                for c in chunks[:10]),
        }
        # --- 生成（全链路） ---
        if not retrieval_only:
            t0 = time.perf_counter()
            qa = client.chat(chat_id, item.question, with_reference=True)
            gen_ms = int((time.perf_counter() - t0) * 1000)
            g_scores = score_generation_rule(g, qa.answer or "")
            row.update({
                "predicted_answer": qa.answer,
                "generation_latency_ms": qa.generation_latency_ms,
                "e2e_latency_ms": ret["latency_ms"] + gen_ms,
                "trace_ok": qa.trace_ok,
                "chat_error": qa.error,
                **g_scores,
                "golden_answer": item.golden_answer,
                "required_facts": item.required_facts,
                "forbidden_facts": item.forbidden_facts,
                "must_cite": item.must_cite,
                "must_abstain": item.must_abstain,
            })
            e2e_all.append(row["e2e_latency_ms"])
        else:
            row["e2e_latency_ms"] = ret["latency_ms"]
            e2e_all.append(row["e2e_latency_ms"])
        per_q.append(row)
        if qps > 0:
            time.sleep(1.0 / qps)

    with open(os.path.join(out_dir, "per_question.jsonl"), "w", encoding="utf-8") as f:
        for r in per_q:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    review_queue: List[Dict[str, Any]] = []
    if not retrieval_only:
        review_queue = build_review_queue(per_q)
        with open(os.path.join(out_dir, "review_queue.jsonl"), "w", encoding="utf-8") as f:
            for q in review_queue:
                q2 = {k: v for k, v in q.items() if k != "prompt"}
                f.write(json.dumps(q2, ensure_ascii=False) + "\n")
        with open(os.path.join(out_dir, "review_prompts.jsonl"), "w", encoding="utf-8") as f:
            for q in review_queue:
                f.write(json.dumps({"question_id": q["question_id"],
                                    "judge_task": q["judge_task"],
                                    "prompt": q["prompt"]},
                                   ensure_ascii=False) + "\n")

    report = build_report(per_q, latency_stats(e2e_all), retrieval_only)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    from .report import render_markdown
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(render_markdown(report))
    report["_output_dir"] = out_dir
    report["_review_items"] = len(review_queue)
    return report
