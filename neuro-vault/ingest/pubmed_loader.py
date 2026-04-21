"""
ingest/pubmed_loader.py — Load PubMed abstracts JSON into Neuro-Vault document dicts.

PubMed abstracts are high-authority biomedical evidence fetched via NCBI
E-utilities.  Each abstract document receives elevated retrieval weight
relative to other dataset sources.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def load_pubmed(filepath: str | Path) -> List[dict]:
    """Load ``pubmed_abstracts.json`` and convert records to document dicts.

    Each PubMed record becomes one document (title + abstract as the text
    body).  All available metadata (PMID, journal, year, MeSH terms) is
    stored so the UI can display rich source badges.

    Args:
        filepath: Path to ``data/raw/pubmed_abstracts.json``.

    Returns:
        List of document dicts with keys:
        ``doc_id``, ``title``, ``text``, ``source``, ``doc_type``,
        ``journal``, ``year``, ``authors``, ``mesh_terms``, ``dataset``.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        ValueError: If the JSON structure is unexpected.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"PubMed abstracts not found at '{filepath}'.\n"
            "Run: python scripts/fetch_pubmed.py"
        )

    logger.info("Loading PubMed abstracts from %s", filepath)

    with open(filepath, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, list):
        raise ValueError(
            f"Expected a JSON list in '{filepath}', got {type(raw).__name__}."
        )

    documents: List[dict] = []
    skipped = 0

    for record in raw:
        pmid = str(record.get("pmid", record.get("PMID", ""))).strip()
        title = str(record.get("title", record.get("Title", ""))).strip()
        abstract = str(record.get("abstract", record.get("Abstract", ""))).strip()

        if not abstract or abstract.lower() == "nan":
            skipped += 1
            continue

        journal = str(record.get("journal", record.get("Journal", ""))).strip()
        year = str(record.get("year", record.get("Year", ""))).strip()
        authors = record.get("authors", record.get("Authors", []))
        mesh_terms = record.get("mesh_terms", record.get("MeSH", []))

        # Ensure authors and mesh_terms are lists
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(";") if a.strip()]
        if isinstance(mesh_terms, str):
            mesh_terms = [m.strip() for m in mesh_terms.split(";") if m.strip()]

        # Compose the full text body: title + abstract
        if title and title.lower() != "nan":
            text_body = f"{title}\n\n{abstract}"
        else:
            text_body = abstract

        doc = {
            "doc_id": f"pubmed_{pmid}",
            "title": title or f"PubMed PMID {pmid}",
            "text": text_body,
            "source": f"PubMed PMID:{pmid}",
            "doc_type": "research_abstract",
            "journal": journal,
            "year": year,
            "authors": authors if isinstance(authors, list) else [],
            "mesh_terms": mesh_terms if isinstance(mesh_terms, list) else [],
            "dataset": "pubmed",
        }
        documents.append(doc)

    logger.info(
        "PubMed loader: %d documents loaded, %d skipped (no abstract)",
        len(documents),
        skipped,
    )
    return documents
