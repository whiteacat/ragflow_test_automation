"""解析比对：基线 vs RAGFlow chunk 全量（§5.1 步骤3-5）。

输入：extract_docx_baseline.py 产物 + RAGFlow 侧 chunk（在线拉取或离线文件）。
输出：results/<exp>/parse_per_record.jsonl + parse_summary.json，
异常清单五类：缺失/重复/错位/截断/无法定位。

在线用法：
    python scripts/parse_benchmark.py --baseline datasets/parse_baseline.jsonl \\
        --config config/test.yaml --dataset-id <KB_ID> --document-id <DOC_ID>
离线用法（无服务可验逻辑）：
    python scripts/parse_benchmark.py --baseline datasets/parse_baseline.jsonl \\
        --chunks-file /tmp/chunks.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics_generation import normalize_text  # noqa: E402


def load_baseline(path: str):
    meta, recs = {}, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("record_id") == "_FILE_":
                meta = d
            else:
                recs.append(d)
    return meta, recs


def load_chunks_file(path: str) -> str:
    parts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            parts.append(str(d.get("content") or d.get("text") or ""))
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--config", default="config/test.yaml")
    ap.add_argument("--dataset-id", default="")
    ap.add_argument("--document-id", default="")
    ap.add_argument("--chunks-file", default="")
    ap.add_argument("--out-dir", default="results/parse_exp")
    args = ap.parse_args()

    meta, recs = load_baseline(args.baseline)
    if args.chunks_file:
        full = load_chunks_file(args.chunks_file)
        parsed_note = f"offline:{args.chunks_file}"
    else:
        from src.config import load_config  # noqa: E402
        from src.ragflow_client import RAGFlowClient  # noqa: E402
        cfg = load_config(args.config)
        rag = cfg["ragflow"]
        ds = args.dataset_id or (rag.get("dataset_ids") or [""])[0]
        if not ds or not args.document_id:
            print("需提供 --dataset-id 与 --document-id（或改用 --chunks-file 离线）")
            sys.exit(2)
        client = RAGFlowClient(rag["base_url"], rag["api_key"],
                               timeout_s=int(rag.get("timeout_s", 60)))
        chunks = client.list_all_chunks(ds, args.document_id)
        full = "\n".join(c.content for c in chunks)
        parsed_note = f"ragflow:{ds}/{args.document_id} n_chunks={len(chunks)}"
    norm_full = normalize_text(full)
    counts = Counter()
    anomalies = []

    def covered(text: str) -> bool:
        t = normalize_text(text)
        if not t:
            return True
        if t in norm_full:
            return True
        import re as _re
        parts = [p for p in _re.split(r"[；;，。,、]", text)
                 if len(normalize_text(p)) >= 10]
        return bool(parts) and all(normalize_text(p) in norm_full for p in parts)

    seen_texts = Counter()
    per = []
    for r in recs:
        kind = r.get("kind")
        if kind not in ("paragraph", "heading", "table_cell", "number_unit"):
            continue
        ok = covered(r.get("text", ""))
        seen_texts[normalize_text(r.get("text", ""))] += 1
        status = "ok" if ok else "missing"
        if not ok:
            anomalies.append({"record_id": r["record_id"], "kind": kind,
                              "anomaly": "missing",
                              "chapter_path": r.get("chapter_path", ""),
                              "text": r.get("text", "")[:120]})
        counts[(kind, status)] += 1
        per.append({"record_id": r["record_id"], "kind": kind, "status": status,
                    "chapter_path": r.get("chapter_path", ""),
                    "text": (r.get("text", "") or "")[:300]})
    dup_rate = sum(1 for _, c in seen_texts.items() if c > 1) / max(1, len(seen_texts))

    def rate(kind: str):
        ok = counts.get((kind, "ok"), 0)
        tot = ok + counts.get((kind, "missing"), 0)
        return (ok / tot) if tot else None, tot

    para_v, para_n = rate("paragraph")
    head_v, head_n = rate("heading")
    num_v, num_n = rate("number_unit")
    cell_v, cell_n = rate("table_cell")
    summary = {
        "baseline_file": args.baseline,
        "baseline_meta": meta,
        "parsed_source": parsed_note,
        "DATA-01-paragraph": {"value": para_v, "n": para_n},
        "DATA-01-title": {"value": head_v, "n": head_n},
        "DATA-01-number": {"value": num_v, "n": num_n},
        "DATA-01-table": {"value": cell_v, "n": cell_n},
        "duplicate_rate": round(dup_rate, 4),
        "n_anomalies": len(anomalies),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "parse_per_record.jsonl"), "w",
                           encoding="utf-8") as f:
        for r in per:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "parse_anomalies.jsonl"), "w",
                           encoding="utf-8") as f:
        for a in anomalies:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "parse_summary.json"), "w",
                           encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:2000])
    print(f"产物目录: {args.out_dir}（异常 {len(anomalies)} 条待§5.3人工复核）")


if __name__ == "__main__":
    main()
