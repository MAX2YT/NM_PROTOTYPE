"""
retrieval/abstention.py — Multi-signal abstention gating for Neuro-Vault.

The abstention checker prevents hallucinated responses by refusing to
generate an LLM answer when the retrieved context is deemed insufficient.

Three complementary signals are combined:
  1. Cross-encoder rerank score   (max score of top retrieved chunk)
  2. Cosine similarity            (max cosine between query and top chunk)
  3. Entity coverage              (fraction of query medical terms in context)

If *any* signal falls below its configured threshold the query is abstained.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import Config

logger = logging.getLogger(__name__)

# Improved medical entity pattern:
# 1. Acronyms (HIV, TB, DM, etc.)
# 2. Capitalized phrases with numbers (Type 2 Diabetes)
# 3. Suffix-based clinical terms (itis, osis, etc.)
# 4. Common single-word clinical terms
_MEDICAL_ENTITY = re.compile(
    r"\b(?:[A-Z]{2,6}|[A-Z][a-z0-9]+(?:\s+[A-Z0-9][a-z0-9]*)+|"
    r"[a-z]+(?:itis|osis|emia|pathy|ectomy|plasty|scopy|gram)|"
    r"diabetes|tuberculosis|dengue|malaria|cancer|infection|guideline|treatment)\b",
    re.IGNORECASE
)

# Common non-medical words that often start with capitals in queries
_NON_MEDICAL_FILTER = {
    "what", "how", "where", "when", "which", "who", "whom", "whose",
    "please", "tell", "give", "list", "show", "describe", "define",
    "the", "this", "that", "these", "those", "your", "my", "our"
}


class AbstentionChecker:
    """Decide whether the retriever found enough context to answer.

    All thresholds are loaded from ``Config`` but can be overridden at
    construction time for experimentation.

    Args:
        rerank_threshold:         Minimum rerank score to proceed.
        cosine_threshold:         Minimum cosine similarity to proceed.
        entity_coverage_threshold: Minimum entity coverage ratio.
    """

    def __init__(
        self,
        rerank_threshold: float = Config.RERANK_THRESHOLD,
        cosine_threshold: float = Config.COSINE_THRESHOLD,
        entity_coverage_threshold: float = Config.ENTITY_COVERAGE_THRESHOLD,
    ) -> None:
        self.rerank_threshold = rerank_threshold
        self.cosine_threshold = cosine_threshold
        self.entity_coverage_threshold = entity_coverage_threshold
        self._embed_model = None  # lazy-loaded for cosine computation

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def should_abstain(
        self,
        query: str,
        retrieved_chunks: List[dict],
    ) -> Tuple[bool, str]:
        """Evaluate whether a response should be generated or abstained.

        Args:
            query:            The user's clinical question.
            retrieved_chunks: Top-K chunk dicts from the retriever.

        Returns:
            Tuple ``(abstain: bool, reason: str)``.
            When ``abstain=False`` the reason is ``"OK"``.
        """
        if not retrieved_chunks:
            return True, "No relevant documents retrieved."

        signals = self._compute_signals(query, retrieved_chunks)
        return self._evaluate_signals(signals)

    def explain(
        self, query: str, retrieved_chunks: List[dict]
    ) -> Dict[str, float]:
        """Return the raw signal values for debugging / UI display.

        Args:
            query:            User query string.
            retrieved_chunks: Retriever results.

        Returns:
            dict with keys ``rerank_score``, ``cosine_similarity``,
            ``entity_coverage``.
        """
        return self._compute_signals(query, retrieved_chunks)

    # ------------------------------------------------------------------ #
    #  Signal computation
    # ------------------------------------------------------------------ #

    def _compute_signals(
        self, query: str, chunks: List[dict]
    ) -> Dict[str, float]:
        """Compute all three abstention signals.

        Args:
            query:  User query string.
            chunks: Retrieved chunk dicts.

        Returns:
            dict with signal values (all in [0, 1] range).
        """
        top_chunk = chunks[0]

        rerank_score = self._normalise_rerank(top_chunk.get("rerank_score", 0.0))
        cosine_sim = self._cosine_similarity(query, top_chunk.get("text", ""))
        entity_cov = self._entity_coverage(query, chunks)

        return {
            "rerank_score": rerank_score,
            "cosine_similarity": cosine_sim,
            "entity_coverage": entity_cov,
        }

    def _evaluate_signals(
        self, signals: Dict[str, float]
    ) -> Tuple[bool, str]:
        """Apply thresholding rules to computed signals.

        Args:
            signals: dict from ``_compute_signals``.

        Returns:
            ``(True, reason)`` if abstaining; ``(False, "OK")`` otherwise.
        """
        if signals["rerank_score"] < self.rerank_threshold:
            return True, (
                f"Retrieved context does not match your query well enough "
                f"(relevance score {signals['rerank_score']:.2f} < "
                f"threshold {self.rerank_threshold:.2f})."
            )

        if signals["cosine_similarity"] < self.cosine_threshold:
            return True, (
                f"Low semantic similarity between query and context "
                f"(cosine {signals['cosine_similarity']:.2f} < "
                f"threshold {self.cosine_threshold:.2f})."
            )

        if signals["entity_coverage"] < self.entity_coverage_threshold:
            return True, (
                f"Query medical entities not adequately covered by retrieved "
                f"documents (coverage {signals['entity_coverage']:.2f} < "
                f"threshold {self.entity_coverage_threshold:.2f})."
            )

        return False, "OK"

    # ------------------------------------------------------------------ #
    #  Individual signal implementations
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalise_rerank(raw_score: float) -> float:
        """Map raw cross-encoder logit score to [0, 1] via sigmoid.

        Args:
            raw_score: Cross-encoder output (can be negative).

        Returns:
            Float in [0, 1].
        """
        import math
        try:
            return 1.0 / (1.0 + math.exp(-raw_score))
        except OverflowError:
            return 0.0 if raw_score < 0 else 1.0

    def _cosine_similarity(self, query: str, chunk_text: str) -> float:
        """Compute cosine similarity between query and chunk embeddings.

        Uses the same BioClinicalBERT model as the main embedder for
        consistency.  Falls back to simple token overlap if the model
        cannot be loaded (e.g., first run before download).

        Args:
            query:      Query string.
            chunk_text: Chunk text string.

        Returns:
            Float in [0, 1].
        """
        try:
            self._load_embed_model()
            vecs = self._embed_model.encode(
                [query, chunk_text],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return float(np.dot(vecs[0], vecs[1]))
        except Exception:  # noqa: BLE001
            return self._token_overlap(query, chunk_text)

    def _load_embed_model(self) -> None:
        """Lazy-load the sentence-transformer model for cosine computation."""
        if self._embed_model is not None:
            return
        from sentence_transformers import SentenceTransformer
        from config import Config

        cache = Config.EMBEDDING_MODEL_DIR
        if cache.exists():
            self._embed_model = SentenceTransformer(str(cache))
        else:
            self._embed_model = SentenceTransformer(Config.EMBEDDING_MODEL)

    @staticmethod
    def _token_overlap(query: str, text: str) -> float:
        """Jaccard token overlap as a fallback similarity metric.

        Args:
            query: Query string.
            text:  Context text.

        Returns:
            Jaccard similarity in [0, 1].
        """
        q_tokens = set(query.lower().split())
        t_tokens = set(text.lower().split())
        if not q_tokens:
            return 0.0
        intersection = q_tokens & t_tokens
        union = q_tokens | t_tokens
        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def _entity_coverage(query: str, chunks: List[dict]) -> float:
        """Estimate how many query medical entities are present in context.

        Extracts candidate medical entities from the query using a regex
        heuristic and checks what fraction are mentioned in the retrieved
        chunks.

        Args:
            query:  User query string.
            chunks: List of retrieved chunk dicts.

        Returns:
            Coverage ratio in [0, 1].  Returns 1.0 if no entities found.
        """
        raw_entities = _MEDICAL_ENTITY.findall(query)
        entities = set()
        for ent in raw_entities:
            ent_low = ent.lower()
            if ent_low not in _NON_MEDICAL_FILTER:
                entities.add(ent_low)

        if not entities:
            return 1.0  # No recognisable entities → don't penalise

        combined_text = " ".join(c.get("text", "").lower() for c in chunks)
        
        covered_count = 0
        for ent in entities:
            # Flexible match: either the full phrase or the majority of its words
            if ent in combined_text:
                covered_count += 1
            else:
                # For multi-word entities, check if individual important words are there
                words = [w for w in ent.split() if len(w) > 3]
                if words:
                    word_matches = sum(1 for w in words if w in combined_text)
                    if word_matches / len(words) >= 0.5:
                        covered_count += 1

        return covered_count / len(entities)
