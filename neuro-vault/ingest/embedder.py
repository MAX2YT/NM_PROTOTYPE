"""
ingest/embedder.py — Embed chunks with BioClinicalBERT and store in FAISS.

The ``Embedder`` class:
- Downloads / loads ``Bio_ClinicalBERT`` from HuggingFace (cached locally)
- Encodes chunk texts in batches for memory efficiency
- Builds a ``faiss.IndexFlatL2`` index
- Persists the index and full metadata to disk
- Supports filtered retrieval by dataset name
- Tracks per-dataset chunk counts for UI display
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Embedder:
    """Encode documents and manage the FAISS vector index.

    Args:
        model_name:   HuggingFace model ID for sentence embeddings.
        model_dir:    Local cache directory for the model.
        index_path:   Path to save / load the FAISS ``.faiss`` file.
        metadata_path: Path to save / load the chunk metadata JSON.
        batch_size:   Number of chunks embedded per forward pass.
        embedding_dim: Expected output dimensionality (768 for BERT).
    """

    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        model_dir: Optional[Path] = None,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        batch_size: int = 32,
        embedding_dim: int = 768,
    ) -> None:
        self.model_name = model_name
        self.model_dir = model_dir
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim

        self._model = None   # lazy-loaded
        self._index: Optional[faiss.Index] = None
        self._metadata: List[dict] = []

    # ------------------------------------------------------------------ #
    #  Model loading
    # ------------------------------------------------------------------ #

    def _load_model(self) -> None:
        """Load BioClinicalBERT from local cache or download from HuggingFace.

        The model is stored at ``model_dir`` so subsequent runs are fully
        offline.
        """
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        cache_path = self.model_dir
        if cache_path and cache_path.exists():
            try:
                logger.info("Loading embedding model from local cache: %s", cache_path)
                self._model = SentenceTransformer(str(cache_path))
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Local embedding cache at '%s' is unusable (%s); "
                    "falling back to model name '%s'.",
                    cache_path,
                    exc,
                    self.model_name,
                )

        logger.info(
            "Downloading/loading embedding model '%s' (one-time setup)…",
            self.model_name,
        )
        self._model = SentenceTransformer(self.model_name)
        if cache_path:
            cache_path.mkdir(parents=True, exist_ok=True)
            self._model.save(str(cache_path))
            logger.info("Model saved to '%s' for offline use", cache_path)

    # ------------------------------------------------------------------ #
    #  Embedding
    # ------------------------------------------------------------------ #

    def embed_chunks(
        self,
        chunks: List[dict],
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, List[dict]]:
        """Encode a list of chunk dicts into a NumPy embedding matrix.

        Args:
            chunks:        List of chunk dicts (must have ``"text"`` key).
            show_progress: Display a tqdm progress bar.

        Returns:
            Tuple of ``(embeddings, metadata_list)`` where *embeddings* has
            shape ``(N, embedding_dim)`` and *metadata_list* mirrors *chunks*
            with the ``"text"`` key removed to save memory.
        """
        self._load_model()

        texts = [c.get("text", "") for c in chunks]
        all_embeddings: List[np.ndarray] = []

        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding chunks", unit="batch")

        for start in iterator:
            batch = texts[start : start + self.batch_size]
            vecs = self._model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embeddings.append(vecs.astype(np.float32))

        embeddings = np.vstack(all_embeddings)

        # Metadata: full chunk dict (keep text for retrieval display)
        metadata = [dict(c) for c in chunks]

        logger.info(
            "Embedded %d chunks → shape %s", len(chunks), embeddings.shape
        )
        return embeddings, metadata

    # ------------------------------------------------------------------ #
    #  Index construction
    # ------------------------------------------------------------------ #

    def build_index(self, chunks: List[dict], show_progress: bool = True) -> None:
        """Embed chunks and build the FAISS IndexFlatL2 index.

        Existing in-memory index is replaced.  Call ``save()`` afterwards
        to persist to disk.

        Args:
            chunks:        All chunk dicts to index.
            show_progress: Show embedding progress bar.
        """
        if not chunks:
            logger.warning("build_index called with empty chunk list — skipping.")
            return

        embeddings, metadata = self.embed_chunks(chunks, show_progress=show_progress)

        self._index = faiss.IndexFlatL2(self.embedding_dim)
        self._index.add(embeddings)
        self._metadata = metadata

        # Count per-dataset stats
        dataset_counts: Dict[str, int] = {}
        for m in metadata:
            ds = m.get("dataset", "unknown")
            dataset_counts[ds] = dataset_counts.get(ds, 0) + 1

        logger.info(
            "FAISS index built: %d vectors | %s",
            self._index.ntotal,
            " | ".join(f"{k}={v}" for k, v in dataset_counts.items()),
        )

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def save(self) -> None:
        """Persist the FAISS index and metadata to disk.

        Raises:
            RuntimeError: If ``build_index`` has not been called yet.
        """
        if self._index is None:
            raise RuntimeError("No index in memory. Call build_index() first.")

        if self.index_path:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self.index_path))
            logger.info("FAISS index saved: %s", self.index_path)

        if self.metadata_path:
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_path, "w", encoding="utf-8") as fh:
                json.dump(self._metadata, fh, ensure_ascii=False)
            logger.info("Metadata saved: %s (%d entries)", self.metadata_path, len(self._metadata))

    def load(self) -> None:
        """Load the FAISS index and metadata from disk.

        Raises:
            FileNotFoundError: If either file is missing.
        """
        if not self.index_path or not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{self.index_path}'.\n"
                "Run: python scripts/setup_datasets.py"
            )
        if not self.metadata_path or not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found at '{self.metadata_path}'."
            )

        self._index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "r", encoding="utf-8") as fh:
            self._metadata = json.load(fh)

        logger.info(
            "Loaded FAISS index: %d vectors | %d metadata entries",
            self._index.ntotal,
            len(self._metadata),
        )

    # ------------------------------------------------------------------ #
    #  Search
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        top_k: int = 20,
        dataset_filter: Optional[str] = None,
    ) -> List[dict]:
        """Dense vector search against the FAISS index.

        Args:
            query:          Natural-language query string.
            top_k:          Number of nearest neighbours to return.
            dataset_filter: If set (e.g. ``"pubmed"``), only return chunks
                            from that dataset.

        Returns:
            List of hit dicts, sorted by ascending L2 distance.
            Each hit adds ``"score"`` (negative L2 for ranking) and
            ``"rank"`` keys.
        """
        self._load_model()
        if self._index is None:
            raise RuntimeError("Index not loaded. Call load() or build_index() first.")

        query_vec = self._model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

        # Over-fetch to allow dataset filtering
        fetch_k = top_k * 5 if dataset_filter else top_k
        fetch_k = min(fetch_k, self._index.ntotal)

        distances, indices = self._index.search(query_vec, fetch_k)

        hits: List[dict] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = dict(self._metadata[idx])
            if dataset_filter and meta.get("dataset") != dataset_filter:
                continue
            meta["score"] = float(-dist)   # higher = better
            meta["vector_index"] = int(idx)
            hits.append(meta)
            if len(hits) >= top_k:
                break

        for rank, hit in enumerate(hits):
            hit["rank"] = rank

        return hits

    # ------------------------------------------------------------------ #
    #  Statistics
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        """Return index statistics for the UI info panel.

        Returns:
            dict with ``total_vectors``, ``dataset_counts``,
            ``index_size_mb``, ``embedding_dim``.
        """
        if self._index is None:
            return {"error": "Index not loaded"}

        dataset_counts: Dict[str, int] = {}
        for m in self._metadata:
            ds = m.get("dataset", "unknown")
            dataset_counts[ds] = dataset_counts.get(ds, 0) + 1

        size_mb = 0.0
        if self.index_path and self.index_path.exists():
            size_mb = self.index_path.stat().st_size / (1024 * 1024)

        return {
            "total_vectors": self._index.ntotal,
            "dataset_counts": dataset_counts,
            "index_size_mb": round(size_mb, 2),
            "embedding_dim": self.embedding_dim,
        }
