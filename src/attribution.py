from __future__ import annotations

from typing import Dict, Optional

ATTRIBUTIONS = (
    "source",    # 原文问题：受控文件本身错误/缺页/版本不明
    "parse",     # 解析问题：正文/表格/图片/编号/单位未正确提取
    "chunk",     # 切块问题：关键内容拆散/混入无关/来源路径错
    "index",     # 索引问题：向量/全文/元数据未正确写入
    "recall",    # 召回问题：目标Chunk未进入候选集
    "rerank",    # 重排问题：已召回但排序靠后
    "generate",  # 生成问题：依据正确但遗漏/误解/引用错
    "policy",    # 策略问题：拒答/版本/权限规则未正确执行
)

ATTRIBUTION_LABELS: Dict[str, str] = {
    "source": "原文问题",
    "parse": "解析问题",
    "chunk": "切块问题",
    "index": "索引问题",
    "recall": "召回问题",
    "rerank": "重排问题",
    "generate": "生成问题",
    "policy": "策略问题",
}


def attribute_failure(row: Dict) -> Optional[str]:
    if row.get("forbidden_fact_hit") == 1:
        return "generate"
    if row.get("answerable") is False or row.get("should_refuse"):
        if row.get("abstention_correct") == 0:
            return "policy"
        return None
    if row.get("needs_chunk_mapping") and row.get("Top@5") is None:
        return None
    hit_rank = row.get("hit_rank")
    if hit_rank is None:
        if row.get("Top@5") == 0:
            return "recall"
        return "parse"
    top5 = row.get("Top@5")
    if top5 == 0:
        return "recall"
    if isinstance(hit_rank, int) and hit_rank > 5 and row.get("Top@20") == 1:
        return "rerank"
    if (row.get("fact_recall_rule") is not None
            and float(row["fact_recall_rule"]) < 1.0):
        return "generate"
    if row.get("citation_complete_rule") == 0:
        return "generate"
    return None
