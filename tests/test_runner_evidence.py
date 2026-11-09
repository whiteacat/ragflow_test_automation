from src.models import RetrievedChunk
from src.runner import chunk_evidence


def _chunk(rank: int, score: float = 0.5, **meta) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=f"c{rank}", document_name="规程",
                          content="x" * 400, score=score, rank=rank,
                          metadata=meta)


def test_snippet_carries_score_and_caps_at_ten():
    snippet, scores = chunk_evidence([_chunk(i, i / 10) for i in range(1, 13)])
    assert len(scores) == 10
    assert len(snippet.splitlines()) == 10
    assert "[1] score=0.1000 规程#c1: " in snippet
    assert "c11" not in snippet
    # 正文截断在 300 字
    assert "x" * 300 in snippet and "x" * 301 not in snippet


def test_scores_keep_similarities_and_default_to_none():
    _, scores = chunk_evidence([
        _chunk(1, 0.123456, vector_similarity=0.7, term_similarity=0.2),
        _chunk(2, 0.5),
    ])
    assert scores[0] == {"rank": 1, "chunk_id": "c1", "score": 0.1235,
                         "vector_similarity": 0.7, "term_similarity": 0.2}
    assert scores[1]["vector_similarity"] is None
    assert scores[1]["term_similarity"] is None
