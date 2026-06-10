from __future__ import annotations

from scicustom.kb import KnowledgeUnit
from scicustom.retrieval import DataInstance, flesch_reading_ease, sort_candidates


def _ku(name: str) -> KnowledgeUnit:
    return KnowledgeUnit(id=name, name=name)


def test_sort_by_overlap_then_rank():
    required = [_ku("A"), _ku("B"), _ku("C")]
    data = [
        DataInstance(query="q1", answer="a1", knowledge_units=["A"]),
        DataInstance(query="q2", answer="a2", knowledge_units=["A", "B"]),
        DataInstance(query="q3", answer="a3", knowledge_units=["C"]),
        DataInstance(query="q4", answer="a4", knowledge_units=[]),
    ]
    out = sort_candidates(data, required)
    assert out[0].query == "q2"  # highest overlap
    # q1 has avg rank 0 (matches A which is rank 0), q3 matches C at rank 2.
    assert out[1].query == "q1"
    assert out[2].query == "q3"
    assert out[3].query == "q4"


def test_flesch_smoke():
    score = flesch_reading_ease("The cat sat on the mat.  The dog ran fast.")
    # Just sanity-check the formula returns a finite, positive number for
    # easy-to-read text.
    assert score > 60
