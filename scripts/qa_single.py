"""单条问答全流程测试：一次看完检索→重排→问答→打分。

用法：
    python scripts/qa_single.py --config config/test.yaml \\
        --question "水冷壁超压试验压力是多少？" \\
        --expect "超压试验压力23.55MPa" --expect "适用于省煤器" \\
        --forbidden "再热器5.7MPa" --evidence "省煤器 | 23.55MPa" \\
        --rerank-compare
输出：TopK  chunk 列表、问答原文、四项规则分与证据覆盖率。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import load_config  # noqa: E402
from src.metrics_generation import (  # noqa: E402
    citation_complete,
    fact_recall,
    forbidden_hit,
    is_abstention,
    order_correct,
    unit_complete,
)
from src.metrics_retrieval import evidence_recall_at_k  # noqa: E402
from src.ragflow_client import RAGFlowClient  # noqa: E402


def show_chunks(title: str, chunks, topn: int) -> None:
    print(f"\n--- {title} ---")
    for c in chunks[:topn]:
        print(f"[{c.rank}] {c.document_name}#{c.chunk_id} "
              f"score={c.score:.3f}\n    {(c.content or '')[:200]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/test.yaml")
    ap.add_argument("--question", required=True)
    ap.add_argument("--expect", action="append", default=[])
    ap.add_argument("--forbidden", action="append", default=[])
    ap.add_argument("--evidence", default="")
    ap.add_argument("--must-cite", action="store_true")
    ap.add_argument("--topn", type=int, default=5)
    ap.add_argument("--rerank-compare", action="store_true")
    ap.add_argument("--rerank-id", default="",
                    help="对比侧重排模型ID（默认与主配置做开/关对比）")
    ap.add_argument("--requires-order", action="store_true")
    ap.add_argument("--requires-unit", action="store_true")
    ap.add_argument("--full", action="store_true",
                    help="同时跑问答（需 chat_id），默认只跑检索")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rag, ret_cfg = cfg["ragflow"], cfg.get("retrieval", {})
    client = RAGFlowClient(rag["base_url"], rag["api_key"],
                           timeout_s=int(rag.get("timeout_s", 60)))
    ds = list(rag.get("dataset_ids") or [])
    kw = dict(page=1, page_size=max(args.topn, 10),
              top_k=int(ret_cfg.get("top_k", 1024)),
              similarity_threshold=float(ret_cfg.get("similarity_threshold", 0.2)),
              vector_similarity_weight=float(
                  ret_cfg.get("vector_similarity_weight", 0.3)),
              keyword=bool(ret_cfg.get("keyword", False)),
              highlight=bool(ret_cfg.get("highlight", False)),
              rerank_id=ret_cfg.get("rerank_id"))
    if "document_ids" in ret_cfg and ret_cfg["document_ids"]:
        kw["document_ids"] = list(ret_cfg["document_ids"])
    ret = client.retrieve(args.question, ds, **kw)
    show_chunks(f"检索（rerank={ret_cfg.get('rerank_id')})", ret["chunks"], args.topn)

    if args.rerank_compare:
        probe = args.rerank_id or ret_cfg.get("rerank_id")
        if not probe:
            print("\n提示：主配置与 --rerank-id 均未给重排模型，无法对比")
        else:
            other = None if probe == ret_cfg.get("rerank_id") else probe
            ret2 = client.retrieve(args.question, ds, **{**kw, "rerank_id": other})
            show_chunks(f"重排对比（rerank={other})", ret2["chunks"], args.topn)

    if args.evidence:
        print(f"\n证据覆盖率@5/@10: "
              f"{evidence_recall_at_k(args.evidence, ret['chunks'], 5)}/"
              f"{evidence_recall_at_k(args.evidence, ret['chunks'], 10)}")
    if args.expect or args.forbidden or args.must_cite:
        print("\n(已给要点/禁止/引用期望，加 --full 跑出答案后打分)")

    if not args.full:
        print("\n只跑检索（加 --full 同时问答打分）。")
        return 0
    qa = client.chat(str(rag.get("chat_id") or ""), args.question,
                     with_reference=True)
    print(f"\n--- 问答（{qa.generation_latency_ms}ms）---\n{qa.answer[:1500]}")
    if qa.error:
        print(f"问答失败: {qa.error}")
        return 1
    print("\n--- 规则打分 ---")
    print(f"要点召回: {fact_recall(args.expect, qa.answer or '')} "
          f"(n={len(args.expect)})")
    print(f"禁止命中: {forbidden_hit(args.forbidden, qa.answer or '')} "
          f"(1=触发失败)")
    print(f"引用完整: {citation_complete(args.must_cite, qa.answer or '')}")
    print(f"是否拒答: {is_abstention(qa.answer or '')}")
    if args.requires_order:
        print(f"顺序正确: {order_correct(args.expect, qa.answer or '')}")
    if args.requires_unit:
        print(f"单位完整: {unit_complete(args.expect, qa.answer or '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
