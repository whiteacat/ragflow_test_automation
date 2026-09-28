"""RAGFlow 本地自动化测试 — 数据模型（对齐 xlsx「测试集字段映射」）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GoldenItem:
    question_id: str
    question: str
    question_type: str = "fact"          # fact/parameter/procedure/accident/condition/multi_document/table/negative
    difficulty: str = "medium"           # easy/medium/hard
    risk_level: str = "low"              # low/medium/high/critical
    answerable: bool = True
    document_ids: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    relevance: Optional[Dict[str, int]] = None   # chunk_id -> 0/1/2，用于 NDCG；缺省全按 1
    golden_answer: str = ""
    golden_facts: List[str] = field(default_factory=list)
    required_facts: List[str] = field(default_factory=list)
    optional_facts: List[str] = field(default_factory=list)
    forbidden_facts: List[str] = field(default_factory=list)
    must_answer: bool = True
    must_cite: bool = False
    must_abstain: bool = False
    requires_attachment: bool = False
    attachment_chunk_ids: List[str] = field(default_factory=list)
    # 版本题：期望的现行版本标识（如 doc_id / version），用于 RAG-08（半自动，需人工确认）
    expected_version: Optional[str] = None
    notes: str = ""

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GoldenItem":
        return GoldenItem(
            question_id=str(d.get("question_id", "")),
            question=str(d.get("question", "")),
            question_type=str(d.get("question_type", "fact")),
            difficulty=str(d.get("difficulty", "medium")),
            risk_level=str(d.get("risk_level", "low")),
            answerable=bool(d.get("answerable", True)),
            document_ids=list(d.get("document_ids") or []),
            chunk_ids=list(d.get("chunk_ids") or []),
            relevance=dict(d["relevance"]) if d.get("relevance") else None,
            golden_answer=str(d.get("golden_answer", "")),
            golden_facts=list(d.get("golden_facts") or []),
            required_facts=list(d.get("required_facts") or []),
            optional_facts=list(d.get("optional_facts") or []),
            forbidden_facts=list(d.get("forbidden_facts") or []),
            must_answer=bool(d.get("must_answer", True)),
            must_cite=bool(d.get("must_cite", False)),
            must_abstain=bool(d.get("must_abstain", False)),
            requires_attachment=bool(d.get("requires_attachment", False)),
            attachment_chunk_ids=list(d.get("attachment_chunk_ids") or []),
            expected_version=d.get("expected_version"),
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
