"""
tests/test_retriever.py — Unit tests for HybridRetriever (mocked embedder).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from retrieval.retriever import HybridRetriever


def _make_chunk(chunk_id: str, text: str, dataset: str, score: float = 0.9) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": f"doc_{chunk_id}",
        "text": text,
        "dataset": dataset,
        "title": "Test Document",
        "doc_type": "test",
        "score": score,
        "rerank_score": score,
        "rrf_score": score,
        "keywords": "",
        "vector_index": 0,
    }


@pytest.fixture
def mock_embedder():
    """Create a mock Embedder that returns predefined chunks."""
    embedder = MagicMock()
    embedder._metadata = [
        _make_chunk("c1", "Diabetes mellitus treatment guidelines.", "pubmed"),
        _make_chunk("c2", "Q: What is diabetes?\nA: A metabolic disease.", "medquad"),
        _make_chunk("c3", "SUBJECTIVE: Patient with type 2 diabetes.", "mtsamples"),
        _make_chunk("c4", "Hypertension management in South Asia.", "pubmed"),
        _make_chunk("c5", "Blood pressure control guidelines.", "pubmed"),
    ]
    embedder.search.return_value = embedder._metadata[:3]
    return embedder


@pytest.fixture
def retriever(mock_embedder):
    with patch("retrieval.retriever.HybridRetriever._load_reranker"):
        r = HybridRetriever(
            embedder=mock_embedder,
            dense_top_k=5,
            sparse_top_k=5,
            final_top_k=3,
            rrf_k=60,
            reranker_model="none",
        )
        r._reranker = None  # Disable reranker for unit tests
        return r


class TestBM25Index:
    def test_bm25_built(self, retriever):
        assert retriever._bm25 is not None

    def test_bm25_returns_hits(self, retriever):
        hits = retriever._bm25_search("diabetes treatment")
        assert isinstance(hits, list)

    def test_bm25_dataset_filter(self, retriever):
        hits = retriever._bm25_search("diabetes", dataset_filter="pubmed")
        for h in hits:
            assert h.get("dataset") == "pubmed"


class TestRRFFusion:
    def test_rrf_deduplicates(self, retriever):
        chunk = _make_chunk("c1", "Some text", "pubmed")
        dense = [chunk]
        sparse = [chunk]
        fused = retriever._rrf_fusion(dense, sparse, {"pubmed": 1.0})
        chunk_ids = [c["chunk_id"] for c in fused]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_rrf_score_increases_with_weight(self, retriever):
        pubmed_chunk = _make_chunk("pb1", "Pubmed evidence text", "pubmed")
        mtsamples_chunk = _make_chunk("mt1", "Clinical transcription text", "mtsamples")

        dense = [pubmed_chunk, mtsamples_chunk]
        sparse = [pubmed_chunk, mtsamples_chunk]

        weights = {"pubmed": 2.0, "mtsamples": 1.0}
        fused = retriever._rrf_fusion(dense, sparse, weights)

        # PubMed should rank first due to higher weight
        datasets = [c["dataset"] for c in fused]
        if len(datasets) >= 2:
            assert datasets[0] == "pubmed"

    def test_rrf_score_present(self, retriever):
        chunk = _make_chunk("c1", "text", "pubmed")
        fused = retriever._rrf_fusion([chunk], [chunk], {"pubmed": 1.0})
        assert "rrf_score" in fused[0]


class TestDatasetBadge:
    @pytest.mark.parametrize("dataset,expected_badge", [
        ("mtsamples", "[🏥 MTSamples]"),
        ("medquad", "[❓ MedQuAD]"),
        ("pubmed", "[🔬 PubMed]"),
        ("local_pdf", "[📄 PDF]"),
        ("unknown", "[unknown]"),
    ])
    def test_badge_labels(self, dataset, expected_badge):
        chunk = {"dataset": dataset}
        badge = HybridRetriever.dataset_badge(chunk)
        assert badge == expected_badge


class TestRetrieve:
    def test_retrieve_returns_list(self, retriever):
        results = retriever.retrieve("What is the treatment for diabetes?")
        assert isinstance(results, list)

    def test_retrieve_respects_final_top_k(self, retriever):
        results = retriever.retrieve("diabetes")
        assert len(results) <= retriever.final_top_k

    def test_retrieve_has_rerank_score(self, retriever):
        results = retriever.retrieve("diabetes")
        for r in results:
            assert "rerank_score" in r
