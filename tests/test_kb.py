"""Sanity tests for the knowledge base loader.

These do not require any model downloads - they only exercise the JSON
ingestion and fuzzy matching path.
"""
from __future__ import annotations

import pytest

from scicustom.kb import KnowledgeBase, KnowledgeUnit, load_knowledge_units


def test_load_default_kb():
    kb = load_knowledge_units()
    assert isinstance(kb, KnowledgeBase)
    assert len(kb) >= 600  # paper reports 642
    assert "Pericyclic reaction" in kb
    assert kb.get("organic chemistry").name == "Organic chemistry"


def test_fuzzy_match_alias():
    kb = load_knowledge_units()
    # rapidfuzz should map "Photo-physical process" to "Photophysical process"
    pytest.importorskip("rapidfuzz")
    hit = kb.fuzzy_match("Photo-physical process", threshold=80)
    assert hit is not None
    assert hit.name == "Photophysical process"


def test_fuzzy_match_below_threshold():
    kb = load_knowledge_units()
    assert kb.fuzzy_match("zzz totally unrelated zzz", threshold=85.0) is None


def test_non_scientific_unit_present():
    kb = load_knowledge_units()
    # The paper introduces a dedicated NON-SCIENTIFIC unit (Section 2.3).
    assert any(u.domain == "non-scientific" for u in kb)
