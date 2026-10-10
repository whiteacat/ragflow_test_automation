"""一键执行：检索基准 + 生成基准 + 聚合（全自动指标部分）。

流程：
1. 读 golden JSONL
2. retrieval_only=True  → 只调 /api/v1/retrieval，算 RAG-01~05/09/10 + 延迟
3. 全链路 → 检索 + chat 问答，叠加 GEN 规则分 + TRACE-01 + review_queue（半自动）
4. 落盘 per_question.jsonl / summary.json / review_queue.jsonl
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List

from .config import load_config
from .attribution import attribute_failure
from .judge import build_review_queue
from .metrics_generation import latency_stats, score_generation_rule
from .metrics_retrieval import first_hit_rank, score_retrieval
from .models import GoldenItem
from .ragflow_client import RAGFlowClient
from .report import build_report


def _retrieve_with(client: RAGFlowClient, question: str, dataset_ids, ret_cfg: dict,
                   rerank_override=None):
    kw = dict(
        page=int(ret_cfg.get("page", 1)),
        page_size=int(ret_cfg.get("page_size", 30)),
        top_k=int(ret_cfg.get("top_k", 1024)),
        similarity_threshold=float(ret_cfg.get("similarity_threshold", 0.2)),
        vector_similarity_weight=float(ret_cfg.get("vector_similarity_weight", 0.3)),
        keyword=bool(ret_cfg.get("keyword", False)),
        highlight=bool(ret_cfg.get("highlight", False)),
        rerank_id=rerank_override if rerank_override != "__default__"
        else ret_cfg.get("rerank_id"))
    if "document_ids" in ret_cfg and ret_cfg["document_ids"]:
        kw["document_ids"] = list(ret_cfg["document_ids"])
    return client.retrieve(question, dataset_ids, **kw)


def load_golden(path: str, limit=None) -> List[Dict[str, Any]]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items[:limit] if limit else items


def dataset_fingerprint(path: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """数据集指纹：用于追溯某次运行结果出自哪版 golden 数据。

    记录文件级 sha256（内容变更即变）+ 实际载入题数 + mtime，
    写入 summary.json；行级题干快照见 per_question 的 question 字段。
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return {
        "path": path,
        "sha256": h.hexdigest(),
        "n_items": len(items),
        "mtime": os.path.getmtime(path),
    }


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
    stages = run_cfg.get("stages")  # 如 [retrieval, rerank] 则只跑对应环节
    if stages:
        wanted = {str(s) for s in stages}
        golden_raw = [g for g in golden_raw
                      if str(GoldenItem.from_dict(g).test_stage) in wanted]
    ds_info = dataset_fingerprint(run_cfg["dataset"], golden_raw)
    retrieval_only = bool(run_cfg.get("retrieval_only", False))
    qps = float(run_cfg.get("qps", 2) or 2)
    out_dir = os.path.join(run_cfg.get("output_dir", "results"),
                           str(run_cfg.get("experiment_id", "exp")))
    os.makedirs(out_dir, exist_ok=True)

    per_q: List[Dict[str, Any]] = []
    e2e_all: List[float] = []
    for g in golden_raw:
        item = GoldenItem.from_dict(g)
        # --- 检索（主配置；rerank_compare 开启时再跑一次做 A/B） ---
        ret = _retrieve_with(client, item.question, dataset_ids, ret_cfg,
                             rerank_override="__default__")
        chunks = ret["chunks"]
        r_scores = score_retrieval(item, chunks)
        rerank_cmp = ret_cfg.get("rerank_compare")
        rank_base = r_scores.get("hit_rank")
        rerank_info: dict = {}
        if isinstance(rerank_cmp, list) and len(rerank_cmp) == 2:
            other = rerank_cmp[1] if ret_cfg.get("rerank_id") == rerank_cmp[0] else rerank_cmp[0]
            ret2 = _retrieve_with(client, item.question, dataset_ids, ret_cfg,
                                  rerank_override=other)
            rank_after = first_hit_rank(item, ret2["chunks"])
            from .metrics_retrieval import rerank_gain as _rg
            rerank_info = {"hit_rank_base": rank_base, "hit_rank_rerank": rank_after,
                           "rerank_gain": _rg(rank_base, rank_after),
                           "rerank_pair": list(rerank_cmp)}
        row: Dict[str, Any] = {
            "question_id": item.question_id,
            "question": item.question,
            "experiment_id": run_cfg.get("experiment_id"),
            "test_stage": item.test_stage,
            "question_type": item.question_type,
            "paraphrase_group": item.paraphrase_group,
            "difficulty": item.difficulty,
            "risk_level": item.risk_level,
            "answerable": item.answerable,
            **{k: v for k, v in r_scores.items()},
            **rerank_info,
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
                "expected_answer_points": item.expected_answer_points,
                "mandatory_points": item.mandatory_points,
                "forbidden_claims": item.forbidden_claims,
                "must_cite": item.must_cite,
                "should_refuse": item.should_refuse,
                "refusal_reason": item.refusal_reason,
                "evidence_text": item.evidence_text,
                "chapter_path": item.chapter_path,
                "table_or_item": item.table_or_item,
                "expected_document": item.expected_document,
                "expected_chunk": item.expected_chunk,
                "requires_order": item.requires_order,
                "requires_unit": item.requires_unit,
                "source_version": item.source_version,
                "reviewer": item.reviewer,
                "review_status": item.review_status,
                "scenario_id": item.scenario_id,
                "verify_method": item.verify_method,
                "source_ref": item.source_ref,
                "needs_chunk_mapping": item.needs_chunk_mapping,
            })
            row["failure_attribution"] = attribute_failure(row)
            e2e_all.append(row["e2e_latency_ms"])
        else:
            row.update({
                "evidence_text": item.evidence_text,
                "chapter_path": item.chapter_path,
                "table_or_item": item.table_or_item,
                "expected_document": item.expected_document,
                "expected_chunk": item.expected_chunk,
                "refusal_reason": item.refusal_reason,
                "source_version": item.source_version,
                "review_status": item.review_status,
                "scenario_id": item.scenario_id,
                "verify_method": item.verify_method,
                "source_ref": item.source_ref,
                "needs_chunk_mapping": item.needs_chunk_mapping,
            })
            row["failure_attribution"] = attribute_failure(row)
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

    report = build_report(per_q, latency_stats(e2e_all), retrieval_only,
                          dataset_info=ds_info,
                          retrieval_config=dict(ret_cfg))
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    from .report import render_markdown
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(render_markdown(report))
    report["_output_dir"] = out_dir
    report["_review_items"] = len(review_queue)
    return report
