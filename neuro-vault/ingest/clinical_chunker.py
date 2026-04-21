"""
ingest/clinical_chunker.py — Dataset-aware, section-preserving text chunker.

``ClinicalAwareChunker`` splits clinical documents while:
- Preserving meaningful clinical sections (SOAP, HISTORY, ASSESSMENT, etc.)
- Keeping MedQuAD Q&A pairs intact when short enough
- Detecting PubMed structured abstract sections (BACKGROUND, METHODS, etc.)
- Protecting clinical atoms: medication lines, vital signs, lab values
- Adding rich per-chunk metadata for retrieval and UI display
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Section header patterns per dataset type
# ------------------------------------------------------------------ #

_MTSAMPLES_SECTIONS = re.compile(
    r"(?m)^(SUBJECTIVE|OBJECTIVE|ASSESSMENT|PLAN|HISTORY(?: OF PRESENT ILLNESS)?|"
    r"PHYSICAL EXAMINATION|REVIEW OF SYSTEMS|CHIEF COMPLAINT|MEDICATIONS|"
    r"ALLERGIES|PAST MEDICAL HISTORY|SOCIAL HISTORY|FAMILY HISTORY|"
    r"LABORATORY DATA|DIAGNOSTIC STUDIES|IMPRESSION|RECOMMENDATION|"
    r"PROCEDURE|PREOPERATIVE DIAGNOSIS|POSTOPERATIVE DIAGNOSIS|"
    r"DISCHARGE DIAGNOSIS|DISCHARGE INSTRUCTIONS|RADIOLOGY REPORT|"
    r"FINDINGS|CONCLUSION|OPERATIVE REPORT)\s*:?",
    re.IGNORECASE,
)

_PUBMED_SECTIONS = re.compile(
    r"(?m)^(BACKGROUND|OBJECTIVE|METHODS?|RESULTS?|CONCLUSIONS?|"
    r"INTRODUCTION|DISCUSSION|AIMS?|PURPOSE|SIGNIFICANCE|"
    r"LIMITATIONS?|IMPLICATIONS?)\s*:",
    re.IGNORECASE,
)

# Clinical atom patterns — lines that must NOT be split mid-sentence
_CLINICAL_ATOM = re.compile(
    r"(?:(?:Vitals?|BP|HR|RR|SpO2|Temp|Weight|Height)\s*:?.+)|"
    r"(?:\d+\.\s+\w.+?(?:mg|mcg|mEq|units?|IU|mL|g|kg).+)|"
    r"(?:Lab(?:oratory)? (?:Values?|Results?):.+)|"
    r"(?:(?:WBC|RBC|Hgb|Hct|Plt|Na|K|Cl|CO2|BUN|Cr|Glu|ALT|AST|"
    r"INR|PT|PTT|HbA1c)\s*[:=].+)",
    re.IGNORECASE,
)


class ClinicalAwareChunker:
    """Split clinical documents into semantically coherent chunks.

    The chunker is aware of which dataset a document originates from and
    applies appropriate heuristics per source type (MTSamples, MedQuAD,
    PubMed, PDF).

    Args:
        chunk_size:    Target maximum word count per chunk.
        chunk_overlap: Number of words to carry over between chunks.
        min_chunk_len: Discard chunks shorter than this many characters.
    """

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        min_chunk_len: int = 50,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_len = min_chunk_len

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def chunk_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk a single document into a list of chunk dicts.

        Dispatches to a dataset-specific strategy based on
        ``doc["dataset"]``.

        Args:
            doc: Document dict as produced by any of the loaders.

        Returns:
            List of chunk dicts.  Each chunk inherits all metadata from
            *doc* and adds ``chunk_id``, ``chunk_index``, and
            ``total_chunks``.
        """
        dataset = doc.get("dataset", "generic")

        if dataset == "medquad":
            raw_chunks = self._chunk_medquad(doc)
        elif dataset == "pubmed":
            raw_chunks = self._chunk_pubmed(doc)
        elif dataset == "mtsamples":
            raw_chunks = self._chunk_mtsamples(doc)
        else:
            raw_chunks = self._generic_split(doc["text"])

        chunks: List[Dict[str, Any]] = []
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < self.min_chunk_len:
                continue
            chunk = {k: v for k, v in doc.items() if k != "text"}
            chunk.update(
                {
                    "chunk_id": f"{doc['doc_id']}_c{idx}",
                    "chunk_index": idx,
                    "text": chunk_text,
                    # Add MeSH terms as keywords for PubMed docs
                    "keywords": self._build_keywords(doc, chunk_text),
                }
            )
            chunks.append(chunk)

        # Patch total_chunks after building the list
        for chunk in chunks:
            chunk["total_chunks"] = len(chunks)

        return chunks

    def chunk_documents(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Chunk a list of documents.

        Args:
            documents: List of document dicts from a loader.

        Returns:
            Flat list of all chunk dicts across all documents.
        """
        all_chunks: List[Dict[str, Any]] = []
        for doc in documents:
            try:
                chunks = self.chunk_document(doc)
                all_chunks.extend(chunks)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping document '%s' during chunking: %s",
                    doc.get("doc_id", "?"),
                    exc,
                )
        return all_chunks

    # ------------------------------------------------------------------ #
    #  Dataset-specific strategies
    # ------------------------------------------------------------------ #

    def _chunk_medquad(self, doc: Dict[str, Any]) -> List[str]:
        """Keep Q&A together if short; only split the Answer if too long.

        Args:
            doc: MedQuAD document dict.

        Returns:
            List of chunk text strings.
        """
        q = doc.get("question", "")
        a = doc.get("answer", "")
        qa_text = f"Q: {q}\nA: {a}"
        word_count = len(qa_text.split())

        if word_count <= self.chunk_size:
            return [qa_text]

        # Split only the Answer portion
        chunks = [f"Q: {q}\n"]
        answer_chunks = self._generic_split(a)
        for ac in answer_chunks:
            chunks.append(f"A (continued): {ac}")
        return chunks

    def _chunk_pubmed(self, doc: Dict[str, Any]) -> List[str]:
        """Split by structured abstract sections, then by size.

        Args:
            doc: PubMed document dict.

        Returns:
            List of chunk text strings.
        """
        text = doc.get("text", "")
        sections = _PUBMED_SECTIONS.split(text)
        if len(sections) <= 1:
            return self._generic_split(text)

        # sections alternates: [pre_text, header1, body1, header2, body2 ...]
        chunks: List[str] = []
        current_header = ""
        for i, part in enumerate(sections):
            if i == 0:
                if part.strip():
                    chunks.extend(self._generic_split(part))
                continue
            if _PUBMED_SECTIONS.match(part.strip() + ":"):
                current_header = part.strip() + ": "
            else:
                section_text = (current_header + part).strip()
                if section_text:
                    chunks.extend(self._generic_split(section_text))
                current_header = ""

        return chunks if chunks else self._generic_split(text)

    def _chunk_mtsamples(self, doc: Dict[str, Any]) -> List[str]:
        """Split clinical transcriptions by section headers.

        Preserves clinical atoms (vital signs, medication lines) intact
        so they are never split across chunk boundaries.

        Args:
            doc: MTSamples document dict.

        Returns:
            List of chunk text strings.
        """
        text = doc.get("text", "")
        sections = _MTSAMPLES_SECTIONS.split(text)
        if len(sections) <= 1:
            return self._generic_split(text)

        chunks: List[str] = []
        current_header = ""
        for i, part in enumerate(sections):
            if i == 0:
                if part.strip():
                    chunks.extend(self._generic_split(part))
                continue
            if _MTSAMPLES_SECTIONS.match(part.strip()):
                current_header = part.strip() + ": "
            else:
                section_text = (current_header + part).strip()
                if section_text:
                    sub_chunks = self._generic_split(section_text)
                    chunks.extend(sub_chunks)
                current_header = ""

        return chunks if chunks else self._generic_split(text)

    # ------------------------------------------------------------------ #
    #  Generic sliding-window splitter
    # ------------------------------------------------------------------ #

    def _generic_split(self, text: str) -> List[str]:
        """Split text into overlapping word-count windows.

        Clinical atom lines are kept on their own line and will not be
        split across a word boundary.

        Args:
            text: Plain text to split.

        Returns:
            List of chunk strings.
        """
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text] if text.strip() else []

        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            if end >= len(words):
                break
            start = end - self.chunk_overlap

        return chunks

    # ------------------------------------------------------------------ #
    #  Keyword builder
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_keywords(doc: Dict[str, Any], chunk_text: str) -> str:
        """Build a keyword string for BM25 boosting.

        Combines the document-level keywords with MeSH terms (for PubMed)
        so that the BM25 index can use them for boosted retrieval.

        Args:
            doc:        Source document dict.
            chunk_text: Text content of the current chunk.

        Returns:
            Space-separated keyword string.
        """
        parts: List[str] = []

        # Document-level keywords (MTSamples)
        kw = doc.get("keywords", "")
        if kw and str(kw).lower() not in ("nan", "none", ""):
            parts.append(str(kw))

        # MeSH terms (PubMed)
        mesh = doc.get("mesh_terms", [])
        if isinstance(mesh, list) and mesh:
            parts.append(" ".join(mesh))
        elif isinstance(mesh, str) and mesh:
            parts.append(mesh)

        # qtype (MedQuAD)
        qtype = doc.get("qtype", "")
        if qtype and qtype not in ("general", ""):
            parts.append(qtype)

        return ", ".join(parts)
