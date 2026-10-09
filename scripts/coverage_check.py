"""覆盖率检查（§11 Table11）：120 条核心配额 + 表格≥20%。

用法：
    python scripts/coverage_check.py datasets/golden_sample.jsonl
按 question_type / chapter_path 首段 / refusal_reason / review_status
统计分布，对照 Table11 建议数量给出缺口清单。
"""
from __future__ import annotations

import json
import sys
from collections import Counter

QUOTA = {
    "parameter": 20, "condition": 15, "procedure": 20,
    "accident_symptom": 6, "accident_cause": 7, "accident_action": 7,
    "table": 15, "paraphrase_group": 10, "multi_document": 8,
    "negative": 7, "compare": 3, "range": 2,
}
TABLE_MIN_RATIO = 0.20
CHAPTERS = ["机组启动", "正常运行", "机组停运", "事故处理",
            "升压站", "主要辅机", "继电保护", "厂用电"]


def main(path: str) -> int:
    types, chapters, refusals, reviews = Counter(), Counter(), Counter(), Counter()
    groups: set[str] = set()
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            n += 1
            qt = str(d.get("question_type", ""))
            if qt == "accident":
                qt = "accident_action"
            types[qt] += 1
            cp = str(d.get("chapter_path", ""))
            hit = next((c for c in CHAPTERS if c in cp), "未标注章节")
            chapters[hit] += 1
            if d.get("should_refuse"):
                refusals[str(d.get("refusal_reason", "?"))] += 1
            reviews[str(d.get("review_status", "pending"))] += 1
            if d.get("paraphrase_group"):
                groups.add(str(d["paraphrase_group"]))
    print(f"共 {n} 条（目标120核心；表格占比目标≥{TABLE_MIN_RATIO:.0%}）")
    print(f"题型分布 {dict(types)}")
    print(f"章节分布 {dict(chapters)}")
    print(f"拒答分布 {dict(refusals)}；改写组 {len(groups)} 个；审核态 {dict(reviews)}")
    gaps = []
    for k, want in QUOTA.items():
        if k == "paraphrase_group":
            continue
        have = types.get(k, 0)
        if k == "table" and n and have / max(1, n) < TABLE_MIN_RATIO:
            gaps.append(f"table 占比 {have}/{n} 低于{TABLE_MIN_RATIO:.0%}")
        if have < want and n >= 60:
            gaps.append(f"{k}: 现{have}/建议{want}")
    missing_ch = [c for c in CHAPTERS if chapters.get(c, 0) == 0]
    if missing_ch:
        gaps.append(f"章节零覆盖: {missing_ch}")
    if gaps:
        print("缺口：")
        for g in gaps:
            print(" -", g)
        return 1
    print("覆盖达标（小样本阶段仅提示，不强制120）。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "datasets/golden_sample.jsonl"))
