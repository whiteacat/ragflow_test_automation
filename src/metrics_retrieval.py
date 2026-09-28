"""全自动检索指标：RAG-01/02/03/04/05/09/10。

输入：GoldenItem + 有序 RetrievedChunk 列表（rank 1-based）。
命中定义：chunk_id 命中，或 document_id 命中（二者任一即命中，chunk 优先定 rank）。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from .models import GoldenItem, RetrievedChunk


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def first_hit_rank(golden: GoldenItem, chunks: List[RetrievedChunk]) -> Optional[int]:
    """正确证据首次出现位置（1-based），未命中返回 None。"""
    gold_chunks = {_norm(c) for c in golden.chunk_ids if c}
    gold_docs = {_norm(d) for d in golden.document_ids if d}
    if not gold_chunks and not gold_docs:
        return None
    for c in chunks:
        if _norm(c.chunk_id) and _norm(c.chunk_id) in gold_chunks:
            return c.rank
        if _norm(c.document_id) and _norm(c.document_id) in gold_docs:
            return c.rank
    return None


def hit_at_k(golden: GoldenItem, chunks: List[RetrievedChunk], k: int) -> Optional[int]:
    """Top@k 命中=1/未命中=0；golden 无证据标注返回 None（不计入分母）。"""
    if not golden.chunk_ids and not golden.document_ids:
        return None
    rank = first_hit_rank(golden, chunks)
    if rank is None:
        return 0
    return 1 if rank <= k else 0


def reciprocal_rank(golden: GoldenItem, chunks: List[RetrievedChunk]) -> Optional[float]:
    rank = first_hit_rank(golden, chunks)
    if rank is None:
        if not golden.chunk_ids and not golden.document_ids:
            return None
        return 0.0
    return 1.0 / rank


def ndcg_at_10(golden: GoldenItem, chunks: List[RetrievedChunk]) -> Optional[float]:
    """NDCG@10。relevance 缺省时 golden chunk 全按等级 1 计。"""
    if not golden.chunk_ids and not golden.document_ids:
        return None
    if golden.relevance:
        rel = {_norm(k): int(v) for k, v in golden.relevance.items()}
    else:
        rel = {_norm(c): 1 for c in golden.chunk_ids}
        for d in golden.document_ids:
            rel.setdefault(_norm(d), 1)

    def gain(c: RetrievedChunk) -> int:
        g = rel.get(_norm(c.chunk_id), 0)
        if g == 0 and _norm(c.document_id) in rel:
            g = rel[_norm(c.document_id)]
        return g

    dcg = 0.0
    for c in chunks[:10]:
        g = gain(c)
        if g > 0:
            dcg += (2 ** g - 1) / math.log2(c.rank + 1)
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal) if g > 0)
    if idcg == 0:
        return None
    return dcg / idcg


def attachment_recall(golden: GoldenItem, chunks: List[RetrievedChunk]) -> Optional[int]:
    """RAG-09：仅 requires_attachment=True 的题计入；附件 chunk 任一命中=1。"""
    if not golden.requires_attachment:
        return None
    targets = {_norm(c) for c in golden.attachment_chunk_ids if c}
    if not targets:
        return None
    for c in chunks:
        if _norm(c.chunk_id) in targets:
            return 1
    return 0


def cross_doc_coverage(golden: GoldenItem, chunks: List[RetrievedChunk]) -> Optional[int]:
    """RAG-10：仅 multi_document 题且 required_facts 非空时计入。

    简化判定：required_facts 每条需在检索文本（含 chunk content）或 golden 证据中
    可定位。这里只做可全自动部分——chunk_id/doc 覆盖全部 golden chunk_ids=1。
    语义级覆盖（事实→证据映射）交由 judge.py 人工/LLM 复核。
    """
    if golden.question_type != "multi_document":
        return None
    if not golden.chunk_ids:
        return None
    gold = {_norm(c) for c in golden.chunk_ids}
    seen = {_norm(c.chunk_id) for c in chunks} | {_norm(c.document_id) for c in chunks}
    return 1 if gold <= seen else 0


def score_retrieval(golden: GoldenItem,
                    chunks: List[RetrievedChunk]) -> Dict[str, Optional[float]]:
    return {
        "Top@5": hit_at_k(golden, chunks, 5),
        "Top@10": hit_at_k(golden, chunks, 10),
        "Top@20": hit_at_k(golden, chunks, 20),
        "MRR": reciprocal_rank(golden, chunks),
        "NDCG@10": ndcg_at_10(golden, chunks),
        "attachment_recall": attachment_recall(golden, chunks),
        "cross_doc_complete": cross_doc_coverage(golden, chunks),
        "hit_rank": first_hit_rank(golden, chunks),
    }
