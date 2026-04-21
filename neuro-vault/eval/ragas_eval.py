"""
eval/ragas_eval.py — RAGAS-based evaluation of Neuro-Vault using MedQuAD test set.

Evaluation pipeline:
  1. Load ``data/eval/medquad_test.json`` (reserved 10% per source folder)
  2. For each QA pair: retrieve context → abstention check → LLM answer
  3. Collect RAGAS input: question, answer, contexts, ground_truth
  4. Compute RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  5. Compute custom metrics: abstention_rate, dataset_coverage
  6. Save full results to ``data/eval/ragas_results_{timestamp}.json``
  7. Return summary dict
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from config import Config
from ingest.embedder import Embedder
from retrieval.retriever import HybridRetriever
from retrieval.abstention import AbstentionChecker
from llm.ollama_client import OllamaClient
from llm.prompt_templates import build_prompt

logger = logging.getLogger(__name__)


def _load_test_set(test_file: Optional[str] = None) -> List[dict]:
    """Load the MedQuAD test set JSON.

    Args:
        test_file: Optional override path.  Defaults to
                   ``Config.MEDQUAD_TEST_PATH``.

    Returns:
        List of dicts with keys ``question``, ``answer``, ``source``, ``qtype``.

    Raises:
        FileNotFoundError: If the test file does not exist.
    """
    path = Path(test_file) if test_file else Config.MEDQUAD_TEST_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation test file not found at '{path}'.\n"
            "Run: python scripts/download_medquad.py  (generates the split)"
        )
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info("Loaded %d test pairs from '%s'", len(data), path)
    return data


def _build_pipeline() -> tuple:
    """Initialise and return the full retrieval + LLM pipeline.

    Returns:
        Tuple of ``(embedder, retriever, abstention_checker, llm_client)``.

    Raises:
        FileNotFoundError: If the FAISS index has not been built.
    """
    embedder = Embedder(
        model_name=Config.EMBEDDING_MODEL,
        model_dir=Config.EMBEDDING_MODEL_DIR,
        index_path=Config.FAISS_INDEX_PATH,
        metadata_path=Config.METADATA_PATH,
        batch_size=Config.EMBED_BATCH_SIZE,
        embedding_dim=Config.EMBEDDING_DIM,
    )
    embedder.load()

    retriever = HybridRetriever(embedder)
    abstention_checker = AbstentionChecker()
    llm_client = OllamaClient()

    return embedder, retriever, abstention_checker, llm_client


def run_evaluation(
    test_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    use_ragas: bool = True,
) -> dict:
    """Run full RAGAS evaluation on the MedQuAD test set.

    For each test QA pair:
      1. Retrieve context chunks for the question
      2. Apply abstention check
      3. If not abstaining: generate LLM answer
      4. Collect RAGAS-compatible sample dict

    Then compute RAGAS metrics + custom metrics.

    Args:
        test_file:   Path to test JSON.  Defaults to ``Config.MEDQUAD_TEST_PATH``.
        max_samples: Limit evaluation to first N samples (for quick runs).
        use_ragas:   If True, compute RAGAS metrics.  Set False for speed.

    Returns:
        Summary dict with all computed metrics.
    """
    logger.info("=== Starting Neuro-Vault RAGAS Evaluation ===")

    test_pairs = _load_test_set(test_file)
    if max_samples:
        test_pairs = test_pairs[:max_samples]
        logger.info("Limiting evaluation to %d samples", max_samples)

    embedder, retriever, abstention_checker, llm_client = _build_pipeline()

    # ── Collect RAGAS samples ──────────────────────────────────────────
    ragas_samples: List[dict] = []
    abstention_count = 0
    dataset_citation_counts: Dict[str, int] = {"pubmed": 0, "medquad": 0, "mtsamples": 0}

    for pair in tqdm(test_pairs, desc="Evaluating", unit="question"):
        question = pair["question"]
        ground_truth = pair["answer"]

        # Retrieve
        try:
            chunks = retriever.retrieve(question)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Retrieval failed for '%s…': %s", question[:40], exc)
            continue

        # Abstention check
        abstain, reason = abstention_checker.should_abstain(question, chunks)
        if abstain:
            abstention_count += 1
            ragas_samples.append(
                {
                    "question": question,
                    "answer": f"ABSTAINED: {reason}",
                    "contexts": [c.get("text", "") for c in chunks],
                    "ground_truth": ground_truth,
                    "abstained": True,
                    "source": pair.get("source", ""),
                    "qtype": pair.get("qtype", ""),
                }
            )
            continue

        # LLM answer
        try:
            prompt = build_prompt(question, chunks)
            answer = llm_client.generate(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM generation failed: %s", exc)
            answer = "LLM_ERROR"

        # Count dataset citations
        cited_datasets = set(c.get("dataset", "") for c in chunks)
        for ds in cited_datasets:
            if ds in dataset_citation_counts:
                dataset_citation_counts[ds] += 1

        ragas_samples.append(
            {
                "question": question,
                "answer": answer,
                "contexts": [c.get("text", "") for c in chunks[:3]],  # top-3 for RAGAS
                "ground_truth": ground_truth,
                "abstained": False,
                "source": pair.get("source", ""),
                "qtype": pair.get("qtype", ""),
                "dataset_sources": list(cited_datasets),
            }
        )

    # ── RAGAS metrics ──────────────────────────────────────────────────
    ragas_metrics: dict = {}
    if use_ragas and ragas_samples:
        ragas_metrics = _compute_ragas(ragas_samples)

    # ── Custom metrics ─────────────────────────────────────────────────
    answered = len([s for s in ragas_samples if not s.get("abstained")])
    total = len(ragas_samples)

    custom_metrics = {
        "total_evaluated": total,
        "answered": answered,
        "abstained": abstention_count,
        "abstention_rate": abstention_count / total if total > 0 else 0.0,
        "dataset_coverage": {
            ds: (count / max(answered, 1))
            for ds, count in dataset_citation_counts.items()
        },
    }

    # ── Persist results ────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = Config.EVAL_DIR / f"ragas_results_{ts}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    full_results = {
        "timestamp": ts,
        "custom_metrics": custom_metrics,
        "ragas_metrics": ragas_metrics,
        "samples": ragas_samples,
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(full_results, fh, indent=2, ensure_ascii=False)

    logger.info("Evaluation results saved to '%s'", output_path)

    summary = {**custom_metrics, **ragas_metrics, "results_path": str(output_path)}
    return summary


def _compute_ragas(samples: List[dict]) -> dict:
    """Compute RAGAS metrics on the collected samples.

    Skips abstained samples (they have no meaningful LLM answer).

    Args:
        samples: List of sample dicts with ``question``, ``answer``,
                 ``contexts``, ``ground_truth`` keys.

    Returns:
        dict with RAGAS metric scores (faithfulness, answer_relevancy,
        context_recall, context_precision) or error info if RAGAS fails.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
        from datasets import Dataset

        # Filter out abstained samples
        answerable = [s for s in samples if not s.get("abstained")]
        if not answerable:
            return {"error": "No answered samples to evaluate with RAGAS."}

        dataset = Dataset.from_list(
            [
                {
                    "question": s["question"],
                    "answer": s["answer"],
                    "contexts": s["contexts"],
                    "ground_truth": s["ground_truth"],
                }
                for s in answerable
            ]
        )

        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            ],
        )
        # Convert to plain dict of floats
        return {k: float(v) for k, v in result.items()}

    except ImportError:
        logger.warning("RAGAS not installed — skipping RAGAS metrics.")
        return {"error": "ragas not installed; run: pip install ragas"}
    except Exception as exc:  # noqa: BLE001
        logger.error("RAGAS evaluation failed: %s", exc)
        return {"error": str(exc)}
