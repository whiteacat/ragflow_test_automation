"""Dataset 校验：字段完整性 + 口径一致性（跑测试前先跑这个）。

口径来源：《企业RAG测试集构建要求》Table7/Table8/Table9/§8.2/§11。
旧 6 条示例保持兼容（缺新字段只告警，不判错）；新用例按正式口径校验。
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter

REQUIRED = ["question_id", "question"]
VALID_TYPES = {"fact", "parameter", "condition", "procedure",
               "accident_symptom", "accident_cause", "accident_action",
               "accident", "compare", "table", "range",
               "multi_document", "negative"}
VALID_RISK = {"low", "medium", "high", "critical"}
VALID_STAGES = {"parse", "chunk", "retrieval", "rerank", "qa", "refusal", "permission"}
VALID_REFUSALS = {"no_answer", "insufficient", "ambiguous",
                  "false_premise", "stale_version", "no_permission"}
VALID_REVIEW = {"pending", "approved", "rejected", "deprecated"}
STEP_TYPES = {"procedure", "accident_action", "accident_cause", "accident_symptom", "accident"}
UNIT_TYPES = {"parameter", "table", "range"}
# 已废弃：仍可读（from_dict 兼容），但新数据不应再写
DEPRECATED_KEYS = {"golden_facts", "optional_facts", "required_facts",
                   "forbidden_facts", "must_answer", "must_abstain",
                   "requires_attachment", "allowed_paraphrase"}


def lint_answer_points(points, qid: str, warns: list) -> None:
    """事实点可匹配性告警：多值点、无数字长句、只有数值缺标签。"""
    multi = [p for p in points if len(re.findall(r"\d+(?:\.\d+)?", p or "")) >= 2]
    prose = [p for p in points if not re.search(r"\d", p or "") and len(p or "") > 30]
    bare = [p for p in points
            if re.search(r"\d", p or "") and not re.search(r"[\u4e00-\u9fff]", p or "")]
    if multi:
        warns.append(f"{qid}: {len(multi)} 个事实点含多个数值，建议一点一值，如 {multi[0]!r}")
    if prose:
        warns.append(f"{qid}: {len(prose)} 个事实点为无数字长句，建议改关键词集，如 {prose[0]!r}")
    if bare:
        warns.append(f"{qid}: {len(bare)} 个事实点只有数值无中文标签，换表述即无法定位，如 {bare[0]!r}")


def main(path: str) -> int:
    n, errs, warns = 0, [], []
    seen = set()
    types, risks, stages = Counter(), Counter(), Counter()
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
                if not d.get(k) and not (k == "question" and (d.get("should_refuse") or d.get("must_abstain"))):
                    errs.append(f"行{i}({d.get('question_id')}): 缺 {k}")
            qid = d.get("question_id") or d.get("case_id") or f"行{i}"
            if qid in seen:
                errs.append(f"{qid}: question_id 重复")
            seen.add(qid)
            qt = d.get("question_type", "fact")
            if qt not in VALID_TYPES:
                errs.append(f"{qid}: question_type 非法 {qt}")
            if d.get("risk_level") not in VALID_RISK:
                errs.append(f"{qid}: risk_level 非法 {d.get('risk_level')}")
            st = d.get("test_stage", "qa")
            if st not in VALID_STAGES:
                errs.append(f"{qid}: test_stage 非法 {st}")
            types[qt] += 1
            risks[d.get("risk_level")] += 1
            stages[st] += 1
            for dep in sorted(DEPRECATED_KEYS & set(d)):
                warns.append(f"{qid}: 字段 {dep} 已废弃（见 DATASET_SPEC.md），请改用规范键")
            should_refuse = bool(d.get("should_refuse", d.get("must_abstain", False))
                                 or d.get("answerable") is False)
            has_evidence_link = bool(d.get("chunk_ids") or d.get("document_ids")
                                         or d.get("expected_chunk")
                                         or d.get("acceptable_chunks"))
            if d.get("answerable") and not has_evidence_link:
                if d.get("needs_chunk_mapping"):
                    warns.append(f"{qid}: 待入库映射预期Chunk（检索指标暂不计分母）")
                elif str(d.get("test_stage", "qa")) in ("retrieval", "rerank"):
                    errs.append(f"{qid}: 召回/重排题无 chunk 映射（RAG 指标无法计算）")
                else:
                    warns.append(f"{qid}: 无 chunk 映射（RAG 指标暂不计，仅可跑问答规则分）")
            if (not d.get("answerable", True) or d.get("should_refuse")) and not should_refuse:
                errs.append(f"{qid}: 不可答题请置 should_refuse=true（GEN-05）")
            if should_refuse and not d.get("refusal_reason"):
                warns.append(f"{qid}: 拒答题缺 refusal_reason（Table7 六类）")
            elif d.get("refusal_reason") and d["refusal_reason"] not in VALID_REFUSALS:
                errs.append(f"{qid}: refusal_reason 非法 {d.get('refusal_reason')}")
            if d.get("attachment_chunk_ids") and not d.get("table_or_item"):
                warns.append(f"{qid}: 附件题建议补 table_or_item（附件表行列定位）")
            if qt == "multi_document" and (d.get("chunk_ids") or d.get("document_ids")
                    or d.get("expected_chunk") or d.get("acceptable_chunks")) \
                    and len(d.get("document_ids") or []) < 2:
                errs.append(f"{qid}: 跨文档题 document_ids 应>=2（RAG-10）")
            if qt == "table" and not (d.get("table_or_item") or d.get("evidence_text")):
                warns.append(f"{qid}: 表格题缺 table_or_item/evidence_text（§8.2-6 行列关联）")
            if qt in STEP_TYPES and not d.get("requires_order"):
                warns.append(f"{qid}: 步骤/事故题建议 requires_order=true（§7.1）")
            if qt in UNIT_TYPES and not d.get("requires_unit"):
                warns.append(f"{qid}: 参数/表格/区间题建议 requires_unit=true（§7.1）")
            if d.get("answerable", True) and not d.get("evidence_text"):
                warns.append(f"{qid}: 可答题缺 evidence_text（§8.2-1 每题须有明确证据）")
            if d.get("answerable", True) and not d.get("chapter_path"):
                warns.append(f"{qid}: 缺 chapter_path（§9 证据链/§11 章节覆盖）")
            if d.get("review_status", "pending") not in VALID_REVIEW:
                errs.append(f"{qid}: review_status 非法")
            if d.get("review_status") == "approved" and not d.get("reviewer"):
                errs.append(f"{qid}: 已审核用例缺 reviewer（Table9 正式集必填）")
            if d.get("risk_level") in ("high", "critical") and not (
                    d.get("mandatory_points") or d.get("expected_answer_points")):
                if not should_refuse:
                    warns.append(f"{qid}: 高风险题缺 mandatory_points/expected_answer_points（§3-5 逐条复核）")
            if d.get("answerable", True) and d.get("expected_answer_points"):
                lint_answer_points(d["expected_answer_points"], qid, warns)
    print(f"共 {n} 题；题型 {dict(types)}；风险 {dict(risks)}；阶段 {dict(stages)}")
    for w in warns:
        print(" !", w)
    if errs:
        print(f"发现 {len(errs)} 个问题：")
        for e in errs:
            print(" -", e)
        return 1
    print("校验通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "datasets/golden_sample.jsonl"))
