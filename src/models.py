"""RAGFlow 本地自动化测试 — 数据模型（对齐 xlsx「测试集字段映射」+《企业RAG测试集构建要求》Table9）。

兼容说明：旧 golden 字段（golden_facts/required_facts/chunk_ids/must_abstain 等）
仍可读，新字段为 Table9 规范名；from_dict 同时接受两套键。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---- 《企业RAG测试集构建要求》Table8 问题类型（稳定英文 ID + 中文标签） ----
# 旧 8 类保留为别名：fact/parameter/procedure/accident/condition/multi_document/table/negative
QUESTION_TYPES = {
    "fact": "直接事实型",
    "parameter": "参数限值型",
    "condition": "条件型",
    "procedure": "步骤型",
    "accident_symptom": "事故现象型",
    "accident_cause": "事故原因型",
    "accident_action": "事故处置型",
    "accident": "事故处置型（旧）",
    "compare": "对比型",
    "table": "表格型",
    "range": "区间判断型",
    "multi_document": "跨段综合型",
    "negative": "拒答与澄清型",
}
# 旧 accident 一律视为事故处置型；multi_document 视为跨段综合型
QUESTION_TYPE_ALIASES = {"accident": "accident_action"}

# ---- Table9 test_stage 七层（§4） ----
TEST_STAGES = {"parse", "chunk", "retrieval", "rerank", "qa", "refusal", "permission"}

# ---- §7.2 拒答六类 ----
REFUSAL_REASONS = {"no_answer", "insufficient", "ambiguous",
                   "false_premise", "stale_version", "no_permission"}

# ---- §10 测试集分层 + 审核态 ----
REVIEW_STATUS = {"pending", "approved", "rejected", "deprecated"}
DATASET_TIERS = {"core", "extended", "adversarial"}  # 核心金标准/扩展回归/探索对抗


@dataclass
class GoldenItem:
    question_id: str
    question: str = ""
    # -- Table9 核心 --
    test_stage: str = "qa"                # parse/chunk/retrieval/rerank/qa/refusal/permission
    question_type: str = "fact"           # 见 QUESTION_TYPES
    paraphrase_group: str = ""            # 同一知识点改写组（§8.2-10，避免虚增覆盖）
    difficulty: str = "medium"            # easy/medium/hard
    risk_level: str = "low"              # low/medium/high/critical
    answerable: bool = True
    # 答案要点化（§7.1）：expected=必须表达；mandatory=不可遗漏；forbidden=禁止内容
    # 规范键：expected_answer_points / mandatory_points / forbidden_claims。
    # 旧键 required_facts / forbidden_facts / golden_facts 只读兼容，不再存储。
    expected_answer_points: List[str] = field(default_factory=list)
    mandatory_points: List[str] = field(default_factory=list)
    forbidden_claims: List[str] = field(default_factory=list)
    # 证据链（§6.1/§9）：原文证据 + 章节路径 + 表格/条款 + 预期命中
    evidence_text: str = ""
    chapter_path: str = ""
    table_or_item: str = ""
    expected_document: str = ""
    expected_chunk: str = ""
    acceptable_chunks: List[str] = field(default_factory=list)
    document_ids: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    relevance: Optional[Dict[str, int]] = None   # chunk_id -> 0/1/2，用于 NDCG；缺省全按 1
    golden_answer: str = ""
    must_cite: bool = True
    should_refuse: bool = False
    refusal_reason: str = ""              # 见 REFUSAL_REASONS
    requires_order: bool = False          # 步骤题顺序是否强制（§7.1）
    requires_unit: bool = False           # 参数题单位是否强制（§7.1）
    attachment_chunk_ids: List[str] = field(default_factory=list)
    # 版本题：期望的现行版本标识（如 doc_id / version），用于 RAG-08（半自动，需人工确认）
    expected_version: Optional[str] = None
    source_version: str = ""              # Table9：受控原文版本（如 20230831）
    source_hash: str = ""                 # §5.1/§10：原文文件哈希，可重复性绑定
    scenario_id: str = ""                 # 全环节场景编号（如 SCN-001；同一场景跨环节归组）
    verify_method: str = ""               # xlsx 验证方式（如 自动测试/自动比对+人工复核）
    source_ref: str = ""                  # 用例来源定位（如 测试用例总表#R42）
    needs_chunk_mapping: bool = False     # 入库后待映射预期 Chunk（映射前检索指标不计分母）
    reviewer: str = ""                    # 正式集必填
    review_status: str = "pending"        # pending/approved/rejected/deprecated
    remarks: str = ""
    notes: str = ""

    @staticmethod
    def _str_list(v: Any) -> List[str]:
        if not v:
            return []
        return [str(x) for x in v] if isinstance(v, list) else [str(v)]

    @property
    def requires_attachment(self) -> bool:
        return bool(self.attachment_chunk_ids)

    @property
    def must_abstain(self) -> bool:
        return self.should_refuse

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GoldenItem":
        expected_pts = list(d.get("expected_answer_points")
                            or d.get("required_facts")
                            or d.get("golden_facts") or [])
        mandatory = list(d.get("mandatory_points") or [])
        forbidden = list(d.get("forbidden_claims") or d.get("forbidden_facts") or [])
        chunk_ids = list(d.get("chunk_ids") or [])
        doc_ids = list(d.get("document_ids") or [])
        should_refuse = bool(d.get("should_refuse", d.get("must_abstain",
                               not d.get("answerable", True))))
        qtype = str(d.get("question_type", "fact"))
        qtype = QUESTION_TYPE_ALIASES.get(qtype, qtype)
        stage = str(d.get("test_stage") or ("refusal" if should_refuse else "qa"))
        if stage not in TEST_STAGES:
            stage = "qa"
        return GoldenItem(
            question_id=str(d.get("question_id") or d.get("case_id") or ""),
            question=str(d.get("question", "")),
            test_stage=stage,
            question_type=qtype,
            paraphrase_group=str(d.get("paraphrase_group", "")),
            difficulty=str(d.get("difficulty", "medium")),
            risk_level=str(d.get("risk_level", "low")),
            answerable=bool(d.get("answerable", not should_refuse)),
            expected_answer_points=[str(x) for x in expected_pts],
            mandatory_points=[str(x) for x in mandatory],
            forbidden_claims=[str(x) for x in forbidden],
            evidence_text=str(d.get("evidence_text", "")),
            chapter_path=str(d.get("chapter_path", "")),
            table_or_item=str(d.get("table_or_item", "")),
            expected_document=str(d.get("expected_document", doc_ids[0] if doc_ids else "")),
            expected_chunk=str(d.get("expected_chunk", chunk_ids[0] if chunk_ids else "")),
            acceptable_chunks=GoldenItem._str_list(d.get("acceptable_chunks")),
            document_ids=doc_ids,
            chunk_ids=chunk_ids,
            relevance=dict(d["relevance"]) if d.get("relevance") else None,
            golden_answer=str(d.get("golden_answer", "")),
            must_cite=bool(d.get("must_cite", True)),
            should_refuse=should_refuse,
            refusal_reason=str(d.get("refusal_reason", "")),
            requires_order=bool(d.get("requires_order", False)),
            requires_unit=bool(d.get("requires_unit", False)),
            attachment_chunk_ids=list(d.get("attachment_chunk_ids") or []),
            expected_version=d.get("expected_version"),
            source_version=str(d.get("source_version", "")),
            source_hash=str(d.get("source_hash", "")),
            scenario_id=str(d.get("scenario_id", "")),
            verify_method=str(d.get("verify_method", "")),
            source_ref=str(d.get("source_ref", "")),
            needs_chunk_mapping=bool(d.get("needs_chunk_mapping", False)),
            reviewer=str(d.get("reviewer", "")),
            review_status=str(d.get("review_status", "pending")),
            remarks=str(d.get("remarks", "")),
            notes=str(d.get("notes", "")),
        )


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str = ""
    document_name: str = ""
    content: str = ""
    score: float = 0.0
    rank: int = 0                       # 1-based
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QAOutcome:
    question_id: str
    answer: str = ""
    references: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_latency_ms: int = 0
    generation_latency_ms: int = 0
    e2e_latency_ms: int = 0
    trace_ok: bool = True               # TRACE-01 字段完整性（本轮调用层面）
    error: str = ""
