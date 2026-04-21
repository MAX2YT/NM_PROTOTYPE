"""
scripts/setup_datasets.py — Master orchestrator for Neuro-Vault dataset ingestion.

Runs all three dataset pipelines in sequence, then builds the unified FAISS
index and prints a complete summary report.

Usage:
    python scripts/setup_datasets.py
    python scripts/setup_datasets.py --skip-pubmed    (skip online fetch)
    python scripts/setup_datasets.py --skip-medquad   (skip git clone)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from config import Config
from ingest.csv_loader import load_mtsamples
from ingest.xml_loader import load_medquad
from ingest.pubmed_loader import load_pubmed
from ingest.clinical_chunker import ClinicalAwareChunker
from ingest.embedder import Embedder

console = Console()


# ------------------------------------------------------------------ #
#  Banner
# ------------------------------------------------------------------ #


def print_banner() -> None:
    """Print the Neuro-Vault ASCII banner."""
    banner = """
███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗       ██╗   ██╗ █████╗ ██╗   ██╗██╗  ████████╗
████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗      ██║   ██║██╔══██╗██║   ██║██║  ╚══██╔══╝
██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║      ██║   ██║███████║██║   ██║██║     ██║   
██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║      ╚██╗ ██╔╝██╔══██║██║   ██║██║     ██║   
██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝       ╚████╔╝ ██║  ██║╚██████╔╝███████╗██║   
╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝         ╚═══╝  ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝   
    """
    console.print(Panel(
        Text(banner, style="bold cyan"),
        subtitle="[dim]Privacy-Preserving Clinical AI for Tamil Nadu Hospitals[/dim]",
        border_style="blue",
    ))
    console.print()


# ------------------------------------------------------------------ #
#  Per-dataset loading helpers
# ------------------------------------------------------------------ #


def _load_mtsamples_safe() -> tuple[list, list]:
    """Load and chunk MTSamples, returning (docs, chunks)."""
    if not Config.MTSAMPLES_PATH.exists():
        console.print("[yellow]⚠ MTSamples CSV not found — skipping.[/yellow]")
        return [], []
    try:
        docs = load_mtsamples(Config.MTSAMPLES_PATH)
        chunker = ClinicalAwareChunker(Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
        chunks = chunker.chunk_documents(docs)
        return docs, chunks
    except Exception as exc:
        console.print(f"[red]✗ MTSamples error:[/red] {exc}")
        return [], []


def _load_medquad_safe() -> tuple[list, list]:
    """Load and chunk MedQuAD train split, returning (docs, chunks)."""
    if not Config.MEDQUAD_DIR.exists() or not any(Config.MEDQUAD_DIR.rglob("*.xml")):
        console.print("[yellow]⚠ MedQuAD not found — skipping.[/yellow]")
        return [], []
    try:
        docs = load_medquad(
            Config.MEDQUAD_DIR,
            split="train",
            test_ratio=Config.MEDQUAD_TEST_RATIO,
            test_output_path=Config.MEDQUAD_TEST_PATH,
        )
        chunker = ClinicalAwareChunker(Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
        chunks = chunker.chunk_documents(docs)
        return docs, chunks
    except Exception as exc:
        console.print(f"[red]✗ MedQuAD error:[/red] {exc}")
        return [], []


def _load_pubmed_safe() -> tuple[list, list]:
    """Load and chunk PubMed abstracts, returning (docs, chunks)."""
    if not Config.PUBMED_PATH.exists():
        console.print("[yellow]⚠ PubMed abstracts not found — skipping.[/yellow]")
        return [], []
    try:
        docs = load_pubmed(Config.PUBMED_PATH)
        chunker = ClinicalAwareChunker(Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
        chunks = chunker.chunk_documents(docs)
        return docs, chunks
    except Exception as exc:
        console.print(f"[red]✗ PubMed error:[/red] {exc}")
        return [], []


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #


def main(
    skip_pubmed: bool = False,
    skip_medquad: bool = False,
    skip_kaggle: bool = False,
) -> None:
    """Run the complete dataset setup and index build pipeline.

    Args:
        skip_pubmed:  Do not run PubMed fetch step.
        skip_medquad: Do not clone/parse MedQuAD.
        skip_kaggle:  Do not validate MTSamples CSV.
    """
    start_time = time.time()
    print_banner()

    Config.ensure_dirs()

    # ── Step 1: MTSamples ──────────────────────────────────────────────
    if not skip_kaggle:
        console.rule("[bold]Step 1 / 4 — MTSamples CSV[/bold]")
        from scripts.download_kaggle import main as kaggle_main
        kaggle_main(ingest=False)  # validate only

    # ── Step 2: MedQuAD ───────────────────────────────────────────────
    if not skip_medquad:
        console.rule("[bold]Step 2 / 4 — MedQuAD Dataset[/bold]")
        from scripts.download_medquad import main as medquad_main
        medquad_main(ingest=False)

    # ── Step 3: PubMed ────────────────────────────────────────────────
    if not skip_pubmed:
        console.rule("[bold]Step 3 / 4 — PubMed Abstracts[/bold]")
        if Config.PUBMED_PATH.exists():
            console.print(
                f"[green]✓[/green] PubMed abstracts already exist at "
                f"[cyan]{Config.PUBMED_PATH}[/cyan]  (skipping re-fetch)"
            )
        else:
            from scripts.fetch_pubmed import main as pubmed_main
            pubmed_main(ingest=False)

    # ── Step 4: Full ingestion + FAISS index ───────────────────────────
    console.rule("[bold]Step 4 / 4 — Full Ingestion & Index Build[/bold]")

    console.print("[bold]Loading MTSamples…[/bold]", end=" ")
    mt_docs, mt_chunks = _load_mtsamples_safe()
    console.print(f"[green]{len(mt_docs):,} docs | {len(mt_chunks):,} chunks[/green]")

    console.print("[bold]Loading MedQuAD…[/bold]", end=" ")
    mq_docs, mq_chunks = _load_medquad_safe()
    console.print(f"[green]{len(mq_docs):,} docs | {len(mq_chunks):,} chunks[/green]")

    console.print("[bold]Loading PubMed…[/bold]", end=" ")
    pb_docs, pb_chunks = _load_pubmed_safe()
    console.print(f"[green]{len(pb_docs):,} docs | {len(pb_chunks):,} chunks[/green]")

    all_chunks = mt_chunks + mq_chunks + pb_chunks
    total_docs = len(mt_docs) + len(mq_docs) + len(pb_docs)

    if not all_chunks:
        console.print(
            "[red bold]No chunks to index![/red bold]\n"
            "Ensure at least one dataset is available before building the index."
        )
        return

    # ── Embed and build FAISS index ────────────────────────────────────
    console.print(f"\n[bold]Embedding {len(all_chunks):,} chunks…[/bold]")

    embedder = Embedder(
        model_name=Config.EMBEDDING_MODEL,
        model_dir=Config.EMBEDDING_MODEL_DIR,
        index_path=Config.FAISS_INDEX_PATH,
        metadata_path=Config.METADATA_PATH,
        batch_size=Config.EMBED_BATCH_SIZE,
        embedding_dim=Config.EMBEDDING_DIM,
    )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} chunks"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Embedding…", total=len(all_chunks))
        embedder.build_index(all_chunks, show_progress=True)
        progress.update(task, completed=len(all_chunks))

    embedder.save()

    elapsed = time.time() - start_time
    index_stats = embedder.stats()
    index_mb = index_stats.get("index_size_mb", 0.0)

    # ── Final report ───────────────────────────────────────────────────
    table = Table(
        title="[bold cyan]===== Neuro-Vault Index Built =====[/bold cyan]",
        border_style="cyan",
    )
    table.add_column("Dataset", style="bold")
    table.add_column("Docs", justify="right")
    table.add_column("Chunks", justify="right")

    table.add_row("[🏥 MTSamples]", f"{len(mt_docs):,}", f"{len(mt_chunks):,}")
    table.add_row("[❓ MedQuAD]", f"{len(mq_docs):,}", f"{len(mq_chunks):,}")
    table.add_row("[🔬 PubMed]", f"{len(pb_docs):,}", f"{len(pb_chunks):,}")
    table.add_section()
    table.add_row("[bold]Total[/bold]", f"[bold]{total_docs:,}[/bold]",
                  f"[bold]{len(all_chunks):,}[/bold]")

    console.print(table)
    console.print(f"[dim]Vector dim:[/dim]  768 (BioClinicalBERT)")
    console.print(f"[dim]Index type:[/dim]  IndexFlatL2")
    console.print(f"[dim]Index size:[/dim]  {index_mb:.1f} MB")
    console.print(f"[dim]Time taken:[/dim]  {elapsed:.1f}s")
    console.print()
    console.print(Panel(
        "[bold green]Ready![/bold green]  Run: [bold cyan]streamlit run app.py[/bold cyan]",
        border_style="green",
    ))


if __name__ == "__main__":
    args = sys.argv[1:]
    main(
        skip_pubmed="--skip-pubmed" in args,
        skip_medquad="--skip-medquad" in args,
        skip_kaggle="--skip-kaggle" in args,
    )
