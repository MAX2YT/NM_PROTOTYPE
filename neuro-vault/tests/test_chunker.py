"""
tests/test_chunker.py — Unit tests for ClinicalAwareChunker.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from ingest.clinical_chunker import ClinicalAwareChunker


@pytest.fixture
def chunker():
    return ClinicalAwareChunker(chunk_size=100, chunk_overlap=10, min_chunk_len=20)


def _make_doc(text: str, dataset: str = "generic", **kwargs) -> dict:
    return {
        "doc_id": "test_001",
        "title": "Test Document",
        "text": text,
        "source": "Test",
        "doc_type": "test",
        "dataset": dataset,
        **kwargs,
    }


class TestMTSamplesChunking:
    def test_short_doc_single_chunk(self, chunker):
        doc = _make_doc("This is a short clinical note.", dataset="mtsamples")
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "This is a short clinical note."

    def test_section_split(self, chunker):
        text = (
            "SUBJECTIVE: Patient reports headache for 3 days.\n"
            "OBJECTIVE: BP 120/80 HR 72.\n"
            "ASSESSMENT: Tension headache.\n"
            "PLAN: Ibuprofen 400mg TID."
        )
        doc = _make_doc(text, dataset="mtsamples")
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1

    def test_chunk_inherits_metadata(self, chunker):
        doc = _make_doc("Some sufficiently long text here to pass minimum length.", dataset="mtsamples",
                         keywords="headache, ibuprofen")
        chunks = chunker.chunk_document(doc)
        assert "doc_id" in chunks[0]
        assert "chunk_id" in chunks[0]
        assert chunks[0]["dataset"] == "mtsamples"


class TestMedQuADChunking:
    def test_short_qa_stays_together(self, chunker):
        doc = _make_doc(
            "Q: What is diabetes?\nA: A metabolic disorder characterized by high blood sugar.",
            dataset="medquad",
            question="What is diabetes?",
            answer="A metabolic disorder characterized by high blood sugar.",
        )
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert "Q:" in chunks[0]["text"]
        assert "A:" in chunks[0]["text"]

    def test_long_answer_splits(self, chunker):
        long_answer = " ".join(["word"] * 200)
        doc = _make_doc(
            f"Q: Explain?\nA: {long_answer}",
            dataset="medquad",
            question="Explain?",
            answer=long_answer,
        )
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1


class TestPubMedChunking:
    def test_structured_abstract_sections(self, chunker):
        text = (
            "Cancer treatment advances.\n\n"
            "BACKGROUND: Background context.\n"
            "METHODS: Methods used.\n"
            "RESULTS: Results found.\n"
            "CONCLUSIONS: Study conclusions."
        )
        doc = _make_doc(text, dataset="pubmed", mesh_terms=["Neoplasms", "Treatment"])
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1

    def test_mesh_terms_in_keywords(self, chunker):
        doc = _make_doc("Study abstract text.", dataset="pubmed",
                         mesh_terms=["Diabetes", "Insulin"])
        chunks = chunker.chunk_document(doc)
        assert "Diabetes" in chunks[0].get("keywords", "") or \
               "Insulin" in chunks[0].get("keywords", "")


class TestGenericChunking:
    def test_long_text_split(self, chunker):
        text = " ".join([f"word{i}" for i in range(300)])
        doc = _make_doc(text, dataset="local_pdf")
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1

    def test_chunk_ids_unique(self, chunker):
        text = " ".join([f"word{i}" for i in range(500)])
        doc = _make_doc(text, dataset="local_pdf")
        chunks = chunker.chunk_document(doc)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_min_len_filter(self, chunker):
        doc = _make_doc("Hi.", dataset="generic")
        chunks = chunker.chunk_document(doc)
        # "Hi." is < min_chunk_len=20, so should be empty
        assert len(chunks) == 0

    def test_total_chunks_annotation(self, chunker):
        text = " ".join([f"word{i}" for i in range(300)])
        doc = _make_doc(text)
        chunks = chunker.chunk_document(doc)
        for c in chunks:
            assert c["total_chunks"] == len(chunks)


class TestChunkDocuments:
    def test_bulk_chunking(self, chunker):
        docs = [
            _make_doc(f"Document {i} with some text content.", dataset="mtsamples")
            for i in range(10)
        ]
        chunks = chunker.chunk_documents(docs)
        assert len(chunks) == 10  # each short doc = 1 chunk

    def test_error_resilience(self, chunker):
        docs = [
            _make_doc("Good document with enough length.", dataset="mtsamples"),
            {"doc_id": "bad_doc"},  # Missing "text" key
            _make_doc("Another good document with enough length.", dataset="mtsamples"),
        ]
        chunks = chunker.chunk_documents(docs)
        # Should NOT crash — bad doc is skipped
        assert len(chunks) >= 2
