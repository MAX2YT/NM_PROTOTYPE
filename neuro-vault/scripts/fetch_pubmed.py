"""
scripts/fetch_pubmed.py — Fetch PubMed abstracts via NCBI E-utilities API.

Queries NCBI for each term in Config.PUBMED_QUERIES, fetches the top-N
abstracts per query, deduplicates by PMID, and saves to
data/raw/pubmed_abstracts.json.

Rate limiting: max 3 requests/second (NCBI limit without API key).
Uses exponential back-off on HTTP 429 / 500 errors.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import requests
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

console = Console()
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Search  (esearch)
# ------------------------------------------------------------------ #


def search_pubmed(
    query: str,
    retmax: int = Config.PUBMED_RETMAX,
    sleep: float = Config.PUBMED_SLEEP,
    max_retries: int = Config.PUBMED_MAX_RETRIES,
) -> List[str]:
    """Search PubMed and return a list of PMIDs.

    Args:
        query:       Search term string.
        retmax:      Maximum number of results to return.
        sleep:       Seconds to sleep after the request.
        max_retries: Number of exponential back-off retries on error.

    Returns:
        List of PMID strings.  Returns empty list on failure.
    """
    url = f"{Config.PUBMED_BASE_URL}esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "sort": "relevance",
    }

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code in (429, 500):
                wait = (2 ** attempt) * 1.5
                logger.warning("HTTP %d — waiting %.1fs before retry", resp.status_code, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            ids = data.get("esearchresult", {}).get("idlist", [])
            time.sleep(sleep)
            return ids
        except requests.RequestException as exc:
            logger.warning("esearch attempt %d failed: %s", attempt + 1, exc)
            time.sleep((2 ** attempt) * 1.0)

    return []


# ------------------------------------------------------------------ #
#  Fetch abstracts  (efetch XML)
# ------------------------------------------------------------------ #


def fetch_abstracts(
    pmids: List[str],
    batch_size: int = Config.PUBMED_BATCH_SIZE,
    sleep: float = Config.PUBMED_SLEEP,
) -> List[dict]:
    """Fetch full abstract records for a list of PMIDs.

    Fetches in batches to stay within NCBI request limits.

    Args:
        pmids:      List of PMID strings.
        batch_size: Number of PMIDs per HTTP request.
        sleep:      Seconds to sleep between batch requests.

    Returns:
        List of abstract record dicts.
    """
    url = f"{Config.PUBMED_BASE_URL}efetch.fcgi"
    all_records: List[dict] = []

    for start in range(0, len(pmids), batch_size):
        batch = pmids[start : start + batch_size]
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            "rettype": "abstract",
        }
        for attempt in range(Config.PUBMED_MAX_RETRIES):
            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code in (429, 500):
                    wait = (2 ** attempt) * 1.5
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                records = parse_pubmed_xml(resp.text)
                all_records.extend(records)
                time.sleep(sleep)
                break
            except requests.RequestException as exc:
                logger.warning("efetch attempt %d failed: %s", attempt + 1, exc)
                time.sleep((2 ** attempt) * 1.0)

    return all_records


# ------------------------------------------------------------------ #
#  XML parser
# ------------------------------------------------------------------ #


def parse_pubmed_xml(xml_text: str) -> List[dict]:
    """Parse PubMed efetch XML response into record dicts.

    Handles structured abstracts (multiple ``AbstractText`` with ``Label``
    attributes) by joining labelled sections.

    Args:
        xml_text: Raw XML string from efetch endpoint.

    Returns:
        List of dicts with keys: ``pmid``, ``title``, ``abstract``,
        ``authors``, ``journal``, ``year``, ``mesh_terms``.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("Failed to parse PubMed XML: %s", exc)
        return []

    records: List[dict] = []

    for article in root.findall(".//PubmedArticle"):
        record = _parse_single_article(article)
        if record:
            records.append(record)

    return records


def _parse_single_article(article: ET.Element) -> Optional[dict]:
    """Extract fields from a single ``<PubmedArticle>`` XML element.

    Args:
        article: ``<PubmedArticle>`` ElementTree element.

    Returns:
        Record dict or ``None`` if critical fields are missing.
    """
    # PMID
    pmid_el = article.find(".//PMID")
    pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""
    if not pmid:
        return None

    # Title
    title_el = article.find(".//ArticleTitle")
    title = (title_el.text or "").strip() if title_el is not None else ""

    # Abstract — handle structured (labelled) abstracts
    abstract_parts: List[str] = []
    for ab_el in article.findall(".//AbstractText"):
        label = ab_el.get("Label", "")
        text = (ab_el.text or "").strip()
        if not text:
            continue
        if label:
            abstract_parts.append(f"{label}: {text}")
        else:
            abstract_parts.append(text)
    abstract = "\n".join(abstract_parts)

    if not abstract:
        return None  # Skip articles with no abstract

    # Authors
    authors: List[str] = []
    for author in article.findall(".//Author"):
        lastname = author.findtext("LastName", "")
        forename = author.findtext("ForeName", "")
        if lastname:
            authors.append(f"{lastname} {forename}".strip())

    # Journal
    journal_el = article.find(".//Journal/Title")
    journal = (journal_el.text or "").strip() if journal_el is not None else ""

    # Year
    year = ""
    for date_el_tag in [".//PubDate/Year", ".//PubDate/MedlineDate"]:
        year_el = article.find(date_el_tag)
        if year_el is not None and year_el.text:
            year = year_el.text[:4]
            break

    # MeSH terms
    mesh_terms: List[str] = []
    for mesh_el in article.findall(".//MeshHeading/DescriptorName"):
        if mesh_el.text:
            mesh_terms.append(mesh_el.text.strip())

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "journal": journal,
        "year": year,
        "mesh_terms": mesh_terms,
    }


# ------------------------------------------------------------------ #
#  Main orchestrator
# ------------------------------------------------------------------ #


def main(ingest: bool = False) -> dict:
    """Fetch PubMed abstracts for all configured queries and save to disk.

    Args:
        ingest: If True, also run the pubmed_loader and report chunk counts.

    Returns:
        dict with ``total_abstracts``, ``queries_processed``, ``chunk_count``.
    """
    console.rule("[bold cyan]Neuro-Vault — PubMed Abstracts[/bold cyan]")

    queries = Config.PUBMED_QUERIES
    console.print(
        f"[bold]{len(queries)} queries[/bold]  |  "
        f"[bold]{Config.PUBMED_RETMAX}[/bold] abstracts/query  |  "
        f"Rate: ≤3 req/s"
    )
    console.print(f"Estimated time: ~{len(queries) * 2} minutes\n")

    all_records: Dict[str, dict] = {}  # PMID → record (dedup)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Fetching PubMed…", total=len(queries))

        for i, query in enumerate(queries, start=1):
            progress.update(task, description=f"[{i}/{len(queries)}] {query[:50]}…")

            pmids = search_pubmed(query)
            if not pmids:
                progress.advance(task)
                continue

            records = fetch_abstracts(pmids)
            for rec in records:
                pmid = rec.get("pmid", "")
                if pmid and pmid not in all_records:
                    all_records[pmid] = rec

            progress.advance(task)

    # ── Save ───────────────────────────────────────────────────────────
    Config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Config.PUBMED_PATH
    records_list = list(all_records.values())

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(records_list, fh, indent=2, ensure_ascii=False)

    console.print(
        f"\n[green]✓[/green] Saved [bold]{len(records_list):,}[/bold] unique "
        f"abstracts to [cyan]{output_path}[/cyan]"
    )
    console.print(
        f"   (fetched from {len(queries)} queries, "
        f"deduplicated by PMID)"
    )

    result: dict = {
        "total_abstracts": len(records_list),
        "queries_processed": len(queries),
        "chunk_count": 0,
    }

    # ── Optional ingestion ─────────────────────────────────────────────
    if ingest and records_list:
        from ingest.pubmed_loader import load_pubmed
        from ingest.clinical_chunker import ClinicalAwareChunker

        docs = load_pubmed(output_path)
        chunker = ClinicalAwareChunker(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )
        chunks = chunker.chunk_documents(docs)
        console.print(
            f"[green]✓[/green] PubMed → [bold]{len(docs):,}[/bold] docs | "
            f"[bold]{len(chunks):,}[/bold] chunks"
        )
        result["chunk_count"] = len(chunks)

    return result


if __name__ == "__main__":
    main(ingest="--ingest" in sys.argv)
