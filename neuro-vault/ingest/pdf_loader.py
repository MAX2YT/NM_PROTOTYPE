"""
ingest/pdf_loader.py — Load PDF files into Neuro-Vault document dicts.

Supports individual PDF files and batch loading from directories.
Uses PyMuPDF (fitz) for fast, accurate text extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def load_pdf(filepath: str | Path) -> List[dict]:
    """Extract text from a single PDF file and return as document dicts.

    Each page becomes one document dict to preserve page-level provenance.
    Pages with fewer than 50 characters of extracted text are skipped.

    Args:
        filepath: Absolute or relative path to the PDF file.

    Returns:
        List of document dicts, one per page.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ImportError: If PyMuPDF is not installed.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF loading.\n"
            "Install with: pip install pymupdf"
        ) from exc

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PDF not found: '{filepath}'")

    documents: List[dict] = []
    try:
        doc = fitz.open(str(filepath))
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()
            if len(text) < 50:
                continue
            documents.append(
                {
                    "doc_id": f"pdf_{filepath.stem}_p{page_num + 1}",
                    "title": f"{filepath.stem} — Page {page_num + 1}",
                    "text": text,
                    "source": str(filepath),
                    "doc_type": "pdf_document",
                    "page": page_num + 1,
                    "dataset": "local_pdf",
                }
            )
        doc.close()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to read PDF '%s': %s", filepath, exc)

    logger.info("PDF loader: %d pages from '%s'", len(documents), filepath.name)
    return documents


def load_pdf_directory(directory: str | Path) -> List[dict]:
    """Recursively load all PDF files from a directory.

    Args:
        directory: Path to a directory containing PDF files.

    Returns:
        Concatenated list of document dicts from all PDFs.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"PDF directory not found: '{directory}'")

    documents: List[dict] = []
    pdf_files = sorted(directory.rglob("*.pdf"))
    logger.info("Found %d PDF files in '%s'", len(pdf_files), directory)

    for pdf_path in pdf_files:
        try:
            docs = load_pdf(pdf_path)
            documents.extend(docs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping '%s': %s", pdf_path.name, exc)

    return documents
