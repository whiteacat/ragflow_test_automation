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
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def assist(golden_path: str, cfg_path: str, out: str, topn: int) -> None:
    from src.config import load_config  # noqa: E402
    from src.ragflow_client import RAGFlowClient  # noqa: E402
    cfg = load_config(cfg_path)
    rag, ret_cfg = cfg["ragflow"], cfg.get("retrieval", {})
    client = RAGFlowClient(rag["base_url"], rag["api_key"],
                           timeout_s=int(rag.get("timeout_s", 60)))
    ds = list(rag.get("dataset_ids") or [])
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for line in open(golden_path, encoding="utf-8"):
            if not line.strip():
                continue
            g = json.loads(line)
            if not g.get("needs_chunk_mapping"):
                continue
            ret = client.retrieve(
                g["question"], ds,
                page=int(ret_cfg.get("page", 1)),
                page_size=int(ret_cfg.get("page_size", 30)),
                top_k=int(ret_cfg.get("top_k", 1024)),
                similarity_threshold=float(ret_cfg.get("similarity_threshold", 0.2)),
                vector_similarity_weight=float(
                    ret_cfg.get("vector_similarity_weight", 0.3)),
                keyword=bool(ret_cfg.get("keyword", False)),
                highlight=bool(ret_cfg.get("highlight", False)),
                rerank_id=ret_cfg.get("rerank_id"))
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
    print(f"待映射 {n} 题，候选清单 → {out}")


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
            hit += 1
        rows.append(g)
    with open(out_golden, "w", encoding="utf-8") as f:
        for g in rows:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"映射 {hit}/{len(mp)} 题 → {out_golden}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="datasets/xlsx_golden.jsonl")
    ap.add_argument("--config", default="config/test.yaml")
    ap.add_argument("--out", default="/tmp/chunk_candidates.jsonl")
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--apply", default="")
    ap.add_argument("--in-golden", default="")
    ap.add_argument("--out-golden", default="")
    args = ap.parse_args()
    if args.apply:
        apply(args.apply, args.in_golden or args.golden,
              args.out_golden or args.golden)
    else:
        assist(args.golden, args.config, args.out, args.topn)


if __name__ == "__main__":
    main()
