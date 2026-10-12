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
                d.setdefault("record_id", d.get("block_id", ""))
                recs.append(d)
    return meta, recs


def load_chunks_file(path: str) -> str:
    return "\n".join(load_chunks_list(path))


def load_chunks_list(path: str) -> list:
    parts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            parts.append(str(d.get("content") or d.get("text") or ""))
    return parts


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
        contents = load_chunks_list(args.chunks_file)
        full = "\n".join(contents)
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
        try:
            info = client.get_document(ds, args.document_id)
            doc = info.get("data", info) if isinstance(info, dict) else {}
            prog = doc.get("progress")
            if isinstance(prog, (int, float)) and prog < 1.0:
                print(f"警告：文档解析进度 {prog} 未完成，比对结果可能误报缺失，"
                      f"待解析完成后重跑。")
        except Exception as exc:  # noqa: BLE001 — 进度查询失败不阻断比对
            print(f"提示：文档进度查询失败（{exc}），直接比对。")
        chunks = client.list_all_chunks(ds, args.document_id)
        contents = [c.content or "" for c in chunks]
        full = "\n".join(contents)
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
    bl_clauses = set()
    for r in recs:
        t = normalize_text(r.get("text", ""))
        bl_clauses.add(t)
        import re as _re2
        bl_clauses.update(normalize_text(p) for p in
                          _re2.split(r"[；;，。,、|\n｜]", r.get("text", "") or "")
                          if len(normalize_text(p)) >= 10)
    bl_clauses.discard("")
    non_empty = [c for c in contents if normalize_text(c)]
    loc_ok = sum(1 for c in non_empty
                 if any(cl in normalize_text(c) for cl in bl_clauses))
    loc_v = (loc_ok / len(non_empty)) if non_empty else None
    summary = {
        "baseline_file": args.baseline,
        "baseline_meta": meta,
        "parsed_source": parsed_note,
        "DATA-01-paragraph": {"value": para_v, "n": para_n},
        "DATA-01-title": {"value": head_v, "n": head_n},
        "DATA-01-number": {"value": num_v, "n": num_n},
        "DATA-01-table": {"value": cell_v, "n": cell_n},
        "DATA-01-locate": {"value": loc_v, "n": len(non_empty),
                           "note": "Chunk正向可在基线中找到出处占比"},
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
