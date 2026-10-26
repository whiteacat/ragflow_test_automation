"""全自动检索指标：RAG-01/02/03/04/05/09/10。

输入：GoldenItem + 有序 RetrievedChunk 列表（rank 1-based）。
命中定义：chunk_id 命中，或 document_id 命中（二者任一即命中，chunk 优先定 rank）。
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

from . import embeddings
from .models import GoldenItem, RetrievedChunk
from .textnorm import normalize as normalize_text

CLAUSE_FUZZY_THRESHOLD = 0.85


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _gold_sets(golden: GoldenItem):
    if (golden.needs_chunk_mapping and not golden.chunk_ids
            and not golden.acceptable_chunks and not golden.expected_chunk):
        return set(), set()
    chunks = {_norm(c) for c in golden.chunk_ids if c}
    if golden.expected_chunk:
        chunks.add(_norm(golden.expected_chunk))
    chunks |= {_norm(c) for c in (golden.acceptable_chunks or []) if c}
    docs = {_norm(d) for d in golden.document_ids if d}
    if golden.expected_document:
        docs.add(_norm(golden.expected_document))
    if chunks:
        # 已有 chunk 级标注时，文档级不再作为独立相关项：否则同一证据被算两次，
        # IDCG 分母虚高（完美命中 NDCG 仅 ~0.6），且同文档其它 chunk 也能蹭分。
        docs = set()
    return chunks, docs


def first_hit_rank(golden: GoldenItem, chunks: List[RetrievedChunk]) -> Optional[int]:
    gold_chunks, gold_docs = _gold_sets(golden)
    if not gold_chunks and not gold_docs:
        return None
    for c in chunks:
        if _norm(c.chunk_id) and _norm(c.chunk_id) in gold_chunks:
            return c.rank
        if _norm(c.document_id) and _norm(c.document_id) in gold_docs:
            return c.rank
        if _norm(c.document_name) and _norm(c.document_name) in gold_docs:
            return c.rank
    return None


def hit_at_k(golden: GoldenItem, chunks: List[RetrievedChunk], k: int) -> Optional[int]:
    """Top@k 命中=1/未命中=0；golden 无证据标注返回 None（不计入分母）。"""
    gc, gd = _gold_sets(golden)
    if not gc and not gd:
        return None
    rank = first_hit_rank(golden, chunks)
    if rank is None:
        return 0
    return 1 if rank <= k else 0


def reciprocal_rank(golden: GoldenItem, chunks: List[RetrievedChunk]) -> Optional[float]:
    rank = first_hit_rank(golden, chunks)
    if rank is None:
        gc, gd = _gold_sets(golden)
        if not gc and not gd:
            return None
        return 0.0
    return 1.0 / rank


def ndcg_at_10(golden: GoldenItem, chunks: List[RetrievedChunk]) -> Optional[float]:
    """NDCG@10。relevance 缺省时 golden chunk 全按等级 1 计。"""
    gc, gd = _gold_sets(golden)
    if not gc and not gd:
        return None
    if golden.relevance:
        rel = {_norm(k): int(v) for k, v in golden.relevance.items()}
    else:
        gold_chunks, gold_docs = _gold_sets(golden)
        rel = {c: 1 for c in gold_chunks}
        for d in gold_docs:
            rel.setdefault(d, 1)

    def gain(c: RetrievedChunk) -> tuple[int, str]:
        """返回 (增益, 命中的 gold 键)。键用于去重：同一 gold 项只计一次。"""
        k = _norm(c.chunk_id)
        if k and rel.get(k, 0) > 0:
            return rel[k], k
        k = _norm(c.document_id)
        if k and rel.get(k, 0) > 0:
            return rel[k], k
        k = _norm(c.document_name)
        if k and rel.get(k, 0) > 0:
            return rel[k], k
        return 0, ""

    dcg = 0.0
    credited: set[str] = set()
    for c in chunks[:10]:
        g, key = gain(c)
        if g > 0 and key not in credited:
            credited.add(key)
            dcg += (2 ** g - 1) / math.log2(c.rank + 1)
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal) if g > 0)
    if idcg == 0:
        return None
    return dcg / idcg


def attachment_recall(golden: GoldenItem, chunks: List[RetrievedChunk]) -> Optional[int]:
    if not golden.attachment_chunk_ids:
        return None
    targets = {_norm(c) for c in golden.attachment_chunk_ids if c}
    for c in chunks:
        if _norm(c.chunk_id) in targets:
            return 1
    return 0


def cross_doc_coverage(golden: GoldenItem, chunks: List[RetrievedChunk]) -> Optional[int]:
    if golden.question_type not in ("multi_document",):
        return None
    gold = {_norm(c) for c in golden.chunk_ids if c}
    if golden.expected_chunk:
        gold.add(_norm(golden.expected_chunk))
    gold |= {_norm(c) for c in (golden.acceptable_chunks or []) if c}
    if not gold:
        return None
    seen = {_norm(c.chunk_id) for c in chunks} | {_norm(c.document_id) for c in chunks}
    return 1 if gold <= seen else 0


def rerank_gain(rank_before: Optional[int], rank_after: Optional[int]) -> Optional[int]:
    if rank_before is None or rank_after is None:
        return None
    if rank_after < rank_before:
        return 1
    if rank_after == rank_before:
        return 0
    return -1


def split_clauses(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"[；;。\n|｜]", text or "")
            if len(_norm(p)) >= 4]


def _bigrams(s: str) -> set:
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _clause_hit(clause: str, full: str, full_grams: set) -> bool:
    if clause in full:
        return True
    grams = _bigrams(clause)
    if not grams:
        return False
    return len(grams & full_grams) / len(grams) >= CLAUSE_FUZZY_THRESHOLD


def _clause_semantic_hit(clause: str, contents: List[str]) -> bool:
    """字面与二元组都没命中时的语义兜底（需配置 embedding）。"""
    if not embeddings.enabled():
        return False
    sentences: List[str] = []
    for content in contents:
        sentences.extend(embeddings.sentences(content))
        if len(sentences) >= 60:
            break
    if not sentences:
        return False
    sim = embeddings.best_similarity(clause, sentences[:60])
    return sim is not None and sim >= embeddings.settings().evidence_threshold


def evidence_cover(evidence: str, contents: List[str]) -> Optional[float]:
    clauses = split_clauses(evidence)
    if not clauses:
        return None
    full = normalize_text("\n".join(contents))
    full_grams = _bigrams(full)
    hits = 0
    for c in clauses:
        clause = normalize_text(c)
        if _clause_hit(clause, full, full_grams) or _clause_semantic_hit(clause, contents):
            hits += 1
    return hits / len(clauses)


def evidence_recall_at_k(evidence: str, chunks: List[RetrievedChunk],
                         k: int) -> Optional[float]:
    if not (evidence or "").strip():
        return None
    return evidence_cover(evidence, [c.content or "" for c in chunks[:k]])


def paraphrase_group_recall(rows: List[Dict]) -> Dict[str, float]:
    from collections import defaultdict
    groups: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        g = r.get("paraphrase_group") or r.get("question_id")
        v = r.get("Top@5")
        if v is not None:
            groups[g].append(float(v))
    return {g: (sum(v) / len(v)) for g, v in groups.items() if v}


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
