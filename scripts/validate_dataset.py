"""Dataset 校验：字段完整性 + 口径一致性（跑测试前先跑这个）。"""
from __future__ import annotations

import json
import sys
from collections import Counter

REQUIRED = ["question_id", "question"]
VALID_TYPES = {"fact", "parameter", "procedure", "accident", "condition",
               "multi_document", "table", "negative"}
VALID_RISK = {"low", "medium", "high", "critical"}


def main(path: str) -> int:
    n, errs = 0, []
    seen = set()
    types, risks = Counter(), Counter()
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            n += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                errs.append(f"行{i}: JSON 解析失败 {e}")
                continue
            for k in REQUIRED:
                if not d.get(k):
                    errs.append(f"行{i}({d.get('question_id')}): 缺 {k}")
            qid = d.get("question_id", f"行{i}")
            if qid in seen:
                errs.append(f"{qid}: question_id 重复")
            seen.add(qid)
            if d.get("question_type") not in VALID_TYPES:
                errs.append(f"{qid}: question_type 非法 {d.get('question_type')}")
            if d.get("risk_level") not in VALID_RISK:
                errs.append(f"{qid}: risk_level 非法 {d.get('risk_level')}")
            types[d.get("question_type")] += 1
            risks[d.get("risk_level")] += 1
            if d.get("answerable") and not (d.get("chunk_ids") or d.get("document_ids")):
                errs.append(f"{qid}: 可答题但无 chunk_ids/document_ids（RAG 指标无法计算）")
            if not d.get("answerable") and not d.get("must_abstain"):
                errs.append(f"{qid}: 不可答题建议 must_abstain=true（GEN-05）")
            if d.get("requires_attachment") and not d.get("attachment_chunk_ids"):
                errs.append(f"{qid}: 附件题缺 attachment_chunk_ids（RAG-09）")
            if d.get("question_type") == "multi_document" and len(d.get("document_ids") or []) < 2:
                errs.append(f"{qid}: 跨文档题 document_ids 应>=2（RAG-10）")
    print(f"共 {n} 题；题型 {dict(types)}；风险 {dict(risks)}")
    if errs:
        print(f"发现 {len(errs)} 个问题：")
        for e in errs:
            print(" -", e)
        return 1
    print("校验通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "datasets/golden_sample.jsonl"))
