"""手写 jsonl 旧键 → DATASET_SPEC V2 规范键迁移。

只做无损改写：required_facts→expected_answer_points、
forbidden_facts→forbidden_claims、must_abstain→should_refuse；
删除无判读方的 golden_facts/optional_facts/must_answer/requires_attachment；
清洗 chunk_ids 空串；补 test_stage 缺省值。改写后原语义不变。
evidence_text/chapter_path/refusal_reason 缺失仍需人工补（会列出清单）。

用法（默认试运行只报告，--write 才落盘）：
    python scripts/migrate_dataset.py datasets/golden_sample.jsonl --write
"""
from __future__ import annotations

import argparse
import json

RENAME = {"required_facts": "expected_answer_points",
          "forbidden_facts": "forbidden_claims",
          "must_abstain": "should_refuse"}
DROP = {"golden_facts", "optional_facts", "must_answer", "requires_attachment"}


def migrate(d: dict) -> tuple[dict, list[str]]:
    notes: list[str] = []
    d = dict(d)
    for old, new in RENAME.items():
        if old in d:
            if new not in d or not d[new]:
                d[new] = d.pop(old)
            else:
                d.pop(old)
            notes.append(f"{old}→{new}")
    for k in DROP:
        if k in d:
            d.pop(k)
            notes.append(f"删{k}")
    if d.get("question_type") == "accident":
        d["question_type"] = "accident_action"
        notes.append("accident→accident_action")
    for k in ("chunk_ids", "document_ids", "acceptable_chunks",
              "attachment_chunk_ids"):
        if isinstance(d.get(k), list):
            d[k] = [c for c in d[k] if c]
    if not d.get("test_stage"):
        d["test_stage"] = ("refusal" if d.get("should_refuse")
                           or d.get("answerable") is False else "qa")
    if "must_cite" not in d and d.get("answerable", True):
        d["must_cite"] = True
    missing = [k for k in ("evidence_text", "chapter_path")
               if d.get("answerable", True) and not d.get(k)]
    if d.get("should_refuse") and not d.get("refusal_reason"):
        missing.append("refusal_reason")
    return d, notes + ([f"待人工补：{','.join(missing)}"] if missing else [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.path, encoding="utf-8") if l.strip()]
    out = []
    for d in rows:
        nd, notes = migrate(d)
        out.append(nd)
        if notes:
            print(f"{d.get('question_id')}: {'; '.join(notes)}")
    if args.write:
        with open(args.path, "w", encoding="utf-8") as f:
            for d in out:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"已写回 {args.path}")
    else:
        print("试运行（加 --write 落盘）")


if __name__ == "__main__":
    main()
