"""Chunk 映射辅助：入库后为待映射用例找候选 Chunk（填写说明步骤 8）。

两步走：
1. assist：对 needs_chunk_mapping=true 的用例跑检索，输出候选清单供人工确认
     python scripts/assist_chunk_mapping.py --golden datasets/xlsx_golden.jsonl \\
         --config config/test.yaml --out /tmp/chunk_candidates.jsonl
2. apply：人工确认后写回 golden（被映射题自动 needs_chunk_mapping=false）
     python scripts/assist_chunk_mapping.py --apply mapping.jsonl \\
         --in-golden datasets/xlsx_golden.jsonl --out-golden datasets/xlsx_golden.jsonl
   mapping.jsonl 每行：{"question_id": "RET-001", "expected_chunk": "CHK-..",
                        "acceptable_chunks": ["CHK-.."], "expected_document": "DOC-.."}
3. auto-map：用证据分句覆盖率自动对齐，高于阈值直接生成 mapping 文件
     python scripts/assist_chunk_mapping.py --auto-map mapping_auto.jsonl \\
         --golden datasets/xlsx_golden.jsonl --threshold 0.8
   （之后人工抽查，再走第 2 步 apply；低于阈值的题留在文件内待人工）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics_retrieval import evidence_cover  # noqa: E402


def _retrieve_top(client, ds, ret_cfg, question):
    return client.retrieve(
        question, ds,
        page=int(ret_cfg.get("page", 1)),
        page_size=int(ret_cfg.get("page_size", 30)),
        top_k=int(ret_cfg.get("top_k", 1024)),
        similarity_threshold=float(ret_cfg.get("similarity_threshold", 0.2)),
        vector_similarity_weight=float(
            ret_cfg.get("vector_similarity_weight", 0.3)),
        keyword=bool(ret_cfg.get("keyword", False)),
        highlight=bool(ret_cfg.get("highlight", False)),
        rerank_id=ret_cfg.get("rerank_id"))


def assist(golden_path: str, cfg_path: str, out: str, topn: int) -> None:
    from src.config import load_config  # noqa: E402
    from src.ragflow_client import RAGFlowClient  # noqa: E402
    cfg = load_config(cfg_path)
    rag, ret_cfg = cfg["ragflow"], cfg.get("retrieval", {})
    client = RAGFlowClient(rag["base_url"], rag["api_key"],
                           timeout_s=int(rag.get("timeout_s", 60)))
    ds = list(rag.get("dataset_ids") or [])
    n, failed = 0, []
    with open(out, "w", encoding="utf-8") as f:
        for line in open(golden_path, encoding="utf-8"):
            if not line.strip():
                continue
            g = json.loads(line)
            if not g.get("needs_chunk_mapping"):
                continue
            try:
                ret = _retrieve_top(client, ds, ret_cfg, g["question"])
            except Exception as exc:  # noqa: BLE001 — 单题失败记下继续
                failed.append((g["question_id"], str(exc)[:80]))
                continue
            f.write(json.dumps({
                "question_id": g["question_id"],
                "question": g["question"],
                "chapter_path": g.get("chapter_path", ""),
                "candidates": [{
                    "rank": c.rank, "chunk_id": c.chunk_id,
                    "document_id": c.document_id, "document_name": c.document_name,
                    "score": round(c.score, 4), "content": c.content[:300]}
                    for c in ret["chunks"][:topn]]}, ensure_ascii=False) + "\n")
            n += 1
    print(f"待映射 {n} 题，候选清单 → {out}"
          + (f"；失败 {len(failed)} 题：{failed[:5]}" if failed else ""))


def apply(mapping_path: str, in_golden: str, out_golden: str) -> None:
    mp = {}
    for line in open(mapping_path, encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            mp[d["question_id"]] = d
    rows, hit = [], 0
    for line in open(in_golden, encoding="utf-8"):
        if not line.strip():
            continue
        g = json.loads(line)
        m = mp.get(g.get("question_id", ""))
        if m:
            if m.get("expected_chunk"):
                g["expected_chunk"] = m["expected_chunk"]
            if m.get("acceptable_chunks"):
                g["acceptable_chunks"] = m["acceptable_chunks"]
            if m.get("expected_document"):
                g["expected_document"] = m["expected_document"]
            if g.get("expected_chunk") or g.get("acceptable_chunks"):
                g["needs_chunk_mapping"] = False
            if m.get("auto_coverage") is not None:
                g["remarks"] = (g.get("remarks") or "") + (
                    f"；auto-mapped覆盖率{m['auto_coverage']}" if g.get("remarks")
                    else f"auto-mapped覆盖率{m['auto_coverage']}")
            hit += 1
        rows.append(g)
    with open(out_golden, "w", encoding="utf-8") as f:
        for g in rows:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"映射 {hit}/{len(mp)} 题 → {out_golden}")


def auto_map(golden_path: str, cfg_path: str, out: str, threshold: float,
             topn: int) -> None:
    from src.config import load_config  # noqa: E402
    from src.ragflow_client import RAGFlowClient  # noqa: E402
    cfg = load_config(cfg_path)
    rag, ret_cfg = cfg["ragflow"], cfg.get("retrieval", {})
    client = RAGFlowClient(rag["base_url"], rag["api_key"],
                           timeout_s=int(rag.get("timeout_s", 60)))
    ds = list(rag.get("dataset_ids") or [])
    auto, manual = 0, []
    with open(out, "w", encoding="utf-8") as f:
        for line in open(golden_path, encoding="utf-8"):
            if not line.strip():
                continue
            g = json.loads(line)
            if not g.get("needs_chunk_mapping") or not (g.get("evidence_text") or ""):
                continue
            try:
                ret = _retrieve_top(client, ds, ret_cfg, g["question"])
            except Exception as exc:  # noqa: BLE001 — 单题失败记下继续，不崩全场
                manual.append((g["question_id"], f"检索失败: {exc}"[:80]))
                continue
            scored = [(evidence_cover(g["evidence_text"], [c.content or ""]) or 0.0, c)
                      for c in ret["chunks"][:topn]]
            if not scored:
                manual.append((g["question_id"], "无召回候选"))
                continue
            scored.sort(key=lambda t: -t[0])
            best, chunk = scored[0] if scored else (0.0, None)
            if chunk is not None and best >= threshold:
                f.write(json.dumps({
                    "question_id": g["question_id"],
                    "expected_chunk": chunk.chunk_id,
                    "expected_document": chunk.document_id,
                    "auto_coverage": round(best, 3)}, ensure_ascii=False) + "\n")
                auto += 1
            else:
                manual.append((g["question_id"], round(best, 3)))
    print(f"自动映射 {auto} 题 → {out}；低于阈值转人工 {len(manual)} 题：{manual[:10]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="datasets/xlsx_golden.jsonl")
    ap.add_argument("--config", default="config/test.yaml")
    ap.add_argument("--out", default="/tmp/chunk_candidates.jsonl")
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--apply", default="")
    ap.add_argument("--in-golden", default="")
    ap.add_argument("--out-golden", default="")
    ap.add_argument("--auto-map", default="")
    ap.add_argument("--threshold", type=float, default=0.8)
    args = ap.parse_args()
    if args.apply:
        apply(args.apply, args.in_golden or args.golden,
              args.out_golden or args.golden)
    elif args.auto_map:
        auto_map(args.golden, args.config, args.auto_map,
                 args.threshold, args.topn)
    else:
        assist(args.golden, args.config, args.out, args.topn)


if __name__ == "__main__":
    main()
