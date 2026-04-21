"""
ingest/csv_loader.py — Load MTSamples CSV into Neuro-Vault document dicts.

The MTSamples dataset contains ~5,000 real clinical transcriptions covering
surgical notes, radiology reports, discharge summaries, psychiatry, and more.
Each row becomes one logical document before chunking.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def load_mtsamples(filepath: str | Path) -> List[dict]:
    """Load MTSamples CSV and convert rows to Neuro-Vault document dicts.

    Rows with null or very short transcriptions are dropped.  All resulting
    documents carry rich metadata so the retriever can apply filters and
    display informative source badges.

    Args:
        filepath: Absolute or relative path to ``mtsamples.csv``.

    Returns:
        List of document dicts, each with keys:
        ``doc_id``, ``title``, ``text``, ``source``, ``doc_type``,
        ``description``, ``keywords``, ``dataset``.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are absent.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"MTSamples CSV not found at '{filepath}'.\n"
            "Download from: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions\n"
            "Place the file at: data/raw/mtsamples.csv"
        )

    logger.info("Reading MTSamples CSV from %s", filepath)
    df = pd.read_csv(filepath, dtype=str)

    # ------------------------------------------------------------------ #
    #  Validate required columns
    # ------------------------------------------------------------------ #
    required_cols = {
        "description",
        "medical_specialty",
        "sample_name",
        "transcription",
        "keywords",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"MTSamples CSV is missing required columns: {missing}\n"
            "Ensure you downloaded the correct file from Kaggle."
        )

    original_count = len(df)

    # ------------------------------------------------------------------ #
    #  Filter invalid rows
    # ------------------------------------------------------------------ #
    df = df.dropna(subset=["transcription"])
    df["transcription"] = df["transcription"].astype(str)
    df = df[df["transcription"].str.strip().str.len() >= 100]

    logger.info(
        "MTSamples: kept %d / %d rows after filtering short/null transcriptions",
        len(df),
        original_count,
    )

    # ------------------------------------------------------------------ #
    #  Build document dicts
    # ------------------------------------------------------------------ #
    documents: List[dict] = []
    for idx, row in df.iterrows():
        specialty = str(row.get("medical_specialty", "General")).strip()
        if specialty.lower() in ("nan", "none", ""):
            specialty = "General"

        doc = {
            "doc_id": f"mtsamples_{idx}",
            "title": str(row.get("sample_name", f"MTSamples Record {idx}")).strip(),
            "text": str(row["transcription"]).strip(),
            "source": "MTSamples",
            "doc_type": specialty,
            "description": str(row.get("description", "")).strip(),
            "keywords": str(row.get("keywords", "")).strip(),
            "dataset": "mtsamples",
        }
        documents.append(doc)

    logger.info("MTSamples loader produced %d documents", len(documents))
    return documents


def validate_mtsamples(filepath: str | Path) -> dict:
    """Validate the MTSamples CSV and return a summary report.

    Args:
        filepath: Path to ``mtsamples.csv``.

    Returns:
        dict with keys: ``total_rows``, ``valid_rows``,
        ``specialty_distribution`` (top-10 as list of tuples).
    """
    filepath = Path(filepath)
    df = pd.read_csv(filepath, dtype=str)
    df = df.dropna(subset=["transcription"])
    df["transcription"] = df["transcription"].astype(str)
    valid_df = df[df["transcription"].str.strip().str.len() >= 100]

    specialty_counts = (
        valid_df["medical_specialty"]
        .fillna("Unknown")
        .str.strip()
        .value_counts()
        .head(10)
        .items()
    )

    return {
        "total_rows": len(df),
        "valid_rows": len(valid_df),
        "specialty_distribution": list(specialty_counts),
    }
