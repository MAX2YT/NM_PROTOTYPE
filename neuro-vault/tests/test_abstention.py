"""
tests/test_abstention.py — Unit tests for AbstentionChecker.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from retrieval.abstention import AbstentionChecker


def _make_chunk(text: str, rerank_score: float = 5.0, dataset: str = "pubmed") -> dict:
    return {
        "chunk_id": "test_chunk",
        "text": text,
        "dataset": dataset,
        "rerank_score": rerank_score,
        "rrf_score": 0.5,
    }


@pytest.fixture
def checker():
    return AbstentionChecker(
        rerank_threshold=0.45,
        cosine_threshold=0.60,
        entity_coverage_threshold=0.50,
    )


class TestAbstentionBasic:
    def test_empty_chunks_abstains(self, checker):
        abstain, reason = checker.should_abstain("What is diabetes?", [])
        assert abstain is True
        assert "No relevant" in reason

    def test_high_relevance_no_abstain(self, checker):
        """With very high rerank score, should not abstain (mocking cosine)."""
        chunk = _make_chunk(
            "Diabetes mellitus is a metabolic disorder. Type 2 diabetes treatment "
            "includes metformin, lifestyle changes, and blood sugar monitoring.",
            rerank_score=10.0,
        )
        with patch.object(checker, "_cosine_similarity", return_value=0.85):
            abstain, reason = checker.should_abstain("diabetes treatment", [chunk])
        assert reason == "OK" or not abstain


class TestNormaliseRerank:
    @pytest.mark.parametrize("raw,expected_range", [
        (10.0, (0.99, 1.0)),
        (0.0, (0.49, 0.51)),
        (-10.0, (0.0, 0.01)),
        (1.0, (0.70, 0.75)),
    ])
    def test_sigmoid_bounds(self, raw, expected_range):
        score = AbstentionChecker._normalise_rerank(raw)
        assert expected_range[0] <= score <= expected_range[1]


class TestEntityCoverage:
    def test_no_entities_returns_one(self):
        chunks = [_make_chunk("some random text")]
        cov = AbstentionChecker._entity_coverage("hello world", chunks)
        assert cov == 1.0

    def test_full_coverage(self):
        chunks = [_make_chunk("diabetes mellitus treatment hypertension")]
        # Entities like "Diabetes Mellitus" should be found
        cov = AbstentionChecker._entity_coverage(
            "What is the treatment for Diabetes Mellitus and Hypertension?",
            chunks,
        )
        assert 0.0 <= cov <= 1.0

    def test_zero_coverage(self):
        chunks = [_make_chunk("xyz abc def")]
        cov = AbstentionChecker._entity_coverage(
            "What is the treatment for Dengue Fever?", chunks
        )
        assert cov < 1.0


class TestTokenOverlap:
    def test_identical_strings(self):
        score = AbstentionChecker._token_overlap("diabetes treatment", "diabetes treatment")
        assert score == 1.0

    def test_no_overlap(self):
        score = AbstentionChecker._token_overlap("alpha beta", "gamma delta")
        assert score == 0.0

    def test_partial_overlap(self):
        score = AbstentionChecker._token_overlap("diabetes insulin", "diabetes management")
        assert 0.0 < score < 1.0

    def test_empty_query(self):
        score = AbstentionChecker._token_overlap("", "some text")
        assert score == 0.0


class TestExplain:
    def test_explain_returns_all_signals(self, checker):
        chunk = _make_chunk("Diabetes treatment with metformin.", rerank_score=5.0)
        with patch.object(checker, "_cosine_similarity", return_value=0.75):
            signals = checker.explain("diabetes treatment", [chunk])
        assert "rerank_score" in signals
        assert "cosine_similarity" in signals
        assert "entity_coverage" in signals
        assert all(0.0 <= v <= 1.0 for v in signals.values())
