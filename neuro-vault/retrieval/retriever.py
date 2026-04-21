"""
retrieval/retriever.py — Hybrid dense + sparse retrieval with RRF fusion.

Pipeline:
  1. Dense search via FAISS (BioClinicalBERT embeddings)
  2. Sparse search via BM25 (rank-bm25)
  3. Reciprocal Rank Fusion (RRF) with optional dataset authority weights
  4. Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
  5. Return final top-K results with dataset badges and provenance
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from config import Config
from ingest.embedder import Embedder

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid dense+sparse retriever with cross-encoder reranking.

    Args:
        embedder:        Loaded ``Embedder`` instance (FAISS index ready).
        dense_top_k:     Number of dense candidates to retrieve.
        sparse_top_k:    Number of BM25 candidates to retrieve.
        final_top_k:     Number of results returned after reranking.
        rrf_k:           RRF constant (typically 60).
        dataset_weights: Dict mapping dataset name to score multiplier.
        reranker_model:  HuggingFace cross-encoder model name.
    """

    def __init__(
        self,
        embedder: Embedder,
        dense_top_k: int = Config.DENSE_TOP_K,
        sparse_top_k: int = Config.SPARSE_TOP_K,
        final_top_k: int = Config.FINAL_TOP_K,
        rrf_k: int = Config.RRF_K,
        dataset_weights: Optional[Dict[str, float]] = None,
        reranker_model: str = Config.RERANKER_MODEL,
    ) -> None:
        self.embedder = embedder
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k
        self.dataset_weights = dataset_weights or Config.DATASET_WEIGHTS
        self.reranker_model = reranker_model

        self._bm25: Optional[BM25Okapi] = None
        self._bm25_corpus: List[dict] = []
        self._reranker = None  # lazy-loaded

        self._build_bm25()

    # ------------------------------------------------------------------ #
    #  BM25 index
    # ------------------------------------------------------------------ #

    def _build_bm25(self) -> None:
        """Build the BM25 index from the embedder's metadata corpus.

        Each chunk's ``text`` plus its ``keywords`` forms the BM25 token
        sequence so that MTSamples keyword fields boost relevant results.
        """
        metadata = self.embedder._metadata
        if not metadata:
            logger.warning("Embedder metadata is empty — BM25 index will be empty.")
            return

        self._bm25_corpus = metadata
        tokenized: List[List[str]] = []
        for chunk in metadata:
            text = chunk.get("text", "")
            keywords = chunk.get("keywords", "")
            combined = f"{text} {keywords}".lower().split()
            tokenized.append(combined)

        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built: %d documents", len(tokenized))

    # ------------------------------------------------------------------ #
    #  Cross-encoder reranker
    # ------------------------------------------------------------------ #

    def _load_reranker(self) -> None:
        """Lazy-load the cross-encoder reranker model."""
        if self._reranker is not None:
            return
        # Skip if reranker is explicitly disabled
        if not self.reranker_model or self.reranker_model.lower() in ("none", "disabled", ""):
            logger.info("Reranker disabled — skipping load.")
            return
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.reranker_model)
            logger.info("Cross-encoder reranker loaded: %s", self.reranker_model)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not load reranker '%s': %s — skipping reranking step.",
                self.reranker_model,
                exc,
            )
            self._reranker = None

    # ------------------------------------------------------------------ #
    #  Core retrieval
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        dataset_filter: Optional[str] = None,
        dataset_weights: Optional[Dict[str, float]] = None,
    ) -> List[dict]:
        """Retrieve the most relevant chunks for *query*.

        Steps:
          1. Dense FAISS search
          2. BM25 sparse search
          3. RRF fusion with dataset authority weights
          4. Cross-encoder reranking
          5. Return top-K with metadata

        Args:
            query:           Clinical question.
            dataset_filter:  Restrict to one dataset (``"pubmed"`` etc.).
            dataset_weights: Override default dataset authority weights.

        Returns:
            List of up to ``final_top_k`` result dicts, each including
            ``"text"``, ``"score"``, ``"rerank_score"``, ``"dataset"``,
            and all original chunk metadata.
        """
        weights = dataset_weights or self.dataset_weights

        # ── 1. Dense retrieval ──────────────────────────────────────────
        dense_hits = self.embedder.search(
            query, top_k=self.dense_top_k, dataset_filter=dataset_filter
        )

        # ── 2. Sparse (BM25) retrieval ──────────────────────────────────
        sparse_hits = self._bm25_search(query, dataset_filter=dataset_filter)

        # ── 3. RRF fusion ───────────────────────────────────────────────
        fused = self._rrf_fusion(dense_hits, sparse_hits, weights)

        # Take RRF top-(4*final_top_k) for reranking
        candidates = fused[: self.final_top_k * 4]

        # ── 4. Skip CPU-heavy cross-encoder — use RRF order directly ─────
        for c in candidates:
            c["rerank_score"] = c.get("rrf_score", 0.0)
        reranked = candidates

        # ── 5. Return final top-K ───────────────────────────────────────
        return reranked[: self.final_top_k]

    # ------------------------------------------------------------------ #
    #  BM25 search
    # ------------------------------------------------------------------ #

    def _bm25_search(
        self, query: str, dataset_filter: Optional[str] = None
    ) -> List[dict]:
        """Return top-K BM25 hits.

        Args:
            query:          Query string.
            dataset_filter: Optional dataset to restrict results.

        Returns:
            List of chunk dicts with ``"bm25_score"`` added.
        """
        if self._bm25 is None:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Pair scores with corpus metadata
        pairs = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )

        hits: List[dict] = []
        for idx, score in pairs:
            if score <= 0:
                continue
            meta = dict(self._bm25_corpus[idx])
            if dataset_filter and meta.get("dataset") != dataset_filter:
                continue
            meta["bm25_score"] = float(score)
            meta["vector_index"] = idx
            hits.append(meta)
            if len(hits) >= self.sparse_top_k:
                break

        return hits

    # ------------------------------------------------------------------ #
    #  RRF fusion
    # ------------------------------------------------------------------ #

    def _rrf_fusion(
        self,
        dense_hits: List[dict],
        sparse_hits: List[dict],
        dataset_weights: Dict[str, float],
    ) -> List[dict]:
        """Reciprocal Rank Fusion of dense and sparse hit lists.

        Score formula for position *r* (1-indexed):
          RRF_score += weight / (rrf_k + r)

        The authority weight is dataset-specific.

        Args:
            dense_hits:      List from FAISS search.
            sparse_hits:     List from BM25 search.
            dataset_weights: Multiplier per dataset name.

        Returns:
            Combined list of unique chunk dicts sorted by descending
            RRF score; includes ``"rrf_score"`` field.
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, dict] = {}

        def _key(chunk: dict) -> str:
            return chunk.get("chunk_id") or str(chunk.get("vector_index", id(chunk)))

        for rank, hit in enumerate(dense_hits, start=1):
            key = _key(hit)
            ds_weight = dataset_weights.get(hit.get("dataset", ""), 1.0)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + ds_weight / (
                self.rrf_k + rank
            )
            chunk_map[key] = hit

        for rank, hit in enumerate(sparse_hits, start=1):
            key = _key(hit)
            ds_weight = dataset_weights.get(hit.get("dataset", ""), 1.0)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + ds_weight / (
                self.rrf_k + rank
            )
            chunk_map[key] = hit

        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        results: List[dict] = []
        for key in sorted_keys:
            chunk = dict(chunk_map[key])
            chunk["rrf_score"] = rrf_scores[key]
            results.append(chunk)

        return results

    # ------------------------------------------------------------------ #
    #  Cross-encoder reranking
    # ------------------------------------------------------------------ #

    def _rerank(self, query: str, candidates: List[dict]) -> List[dict]:
        """Rerank candidates using a cross-encoder model.

        Falls back to RRF order if the reranker is unavailable.

        Args:
            query:      The original query string.
            candidates: Candidate chunk dicts from RRF fusion.

        Returns:
            Candidates sorted by descending ``"rerank_score"``.
        """
        self._load_reranker()

        if self._reranker is None or not candidates:
            for c in candidates:
                c["rerank_score"] = c.get("rrf_score", 0.0)
            return candidates

        pairs = [(query, c.get("text", "")) for c in candidates]
        try:
            scores = self._reranker.predict(pairs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reranker prediction failed: %s", exc)
            for c in candidates:
                c["rerank_score"] = c.get("rrf_score", 0.0)
            return candidates

        for chunk, score in zip(candidates, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked

    # ------------------------------------------------------------------ #
    #  Dataset badge helper
    # ------------------------------------------------------------------ #

    @staticmethod
    def dataset_badge(chunk: dict) -> str:
        """Return a display badge string for the chunk's dataset.

        Args:
            chunk: A result chunk dict with ``"dataset"`` key.

        Returns:
            Formatted badge string like ``"[🏥 MTSamples]"``.
        """
        ds = chunk.get("dataset", "unknown")
        badges = {
            "mtsamples": "[🏥 MTSamples]",
            "medquad": "[❓ MedQuAD]",
            "pubmed": "[🔬 PubMed]",
            "local_pdf": "[📄 PDF]",
        }
        return badges.get(ds, f"[{ds}]")
