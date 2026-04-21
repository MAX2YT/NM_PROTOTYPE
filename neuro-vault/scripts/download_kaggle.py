"""
scripts/download_kaggle.py — Validate MTSamples CSV and report dataset stats.

Since Kaggle requires user authentication, this script does NOT attempt to
download the file automatically.  Instead it:
  1. Checks if data/raw/mtsamples.csv already exists
  2. If not: prints clear download instructions
  3. If yes: validates columns, computes stats, and optionally loads + chunks
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config import Config
from ingest.csv_loader import load_mtsamples, validate_mtsamples
from ingest.clinical_chunker import ClinicalAwareChunker

console = Console()


def main(ingest: bool = False) -> dict:
    """Check and optionally ingest the MTSamples CSV.

    Args:
        ingest: If True, run chunking and return chunk count.

    Returns:
        dict with ``valid_rows``, ``chunk_count`` (if ingest=True).
    """
    console.rule("[bold cyan]Neuro-Vault — MTSamples Dataset[/bold cyan]")

    path = Config.MTSAMPLES_PATH

    # ── Check existence ────────────────────────────────────────────────
    if not path.exists():
        console.print(
            Panel(
                "[bold yellow]MTSamples CSV not found![/bold yellow]\n\n"
                "1. Open your browser and go to:\n"
                "   [link=https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions]"
                "https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions[/link]\n"
                "2. Click [bold]Download[/bold] → extract the ZIP\n"
                "3. Copy [bold]mtsamples.csv[/bold] to:\n"
                f"   [bold green]{path}[/bold green]\n"
                "4. Re-run this script.",
                title="[red]Action Required[/red]",
                border_style="yellow",
            )
        )
        return {"valid_rows": 0, "chunk_count": 0}

    console.print(f"[green]✓[/green] Found file: [bold]{path}[/bold]")

    # ── Validate ───────────────────────────────────────────────────────
    try:
        stats = validate_mtsamples(path)
    except Exception as exc:
        console.print(f"[red]✗ Validation error:[/red] {exc}")
        return {"valid_rows": 0, "chunk_count": 0}

    console.print(
        f"[green]✓[/green] Total rows: [bold]{stats['total_rows']:,}[/bold]  "
        f"| Valid transcriptions: [bold]{stats['valid_rows']:,}[/bold]"
    )

    # ── Specialty distribution table ───────────────────────────────────
    table = Table(title="Top-10 Medical Specialties", border_style="dim")
    table.add_column("Medical Specialty", style="cyan")
    table.add_column("Count", justify="right", style="green")

    for specialty, count in stats["specialty_distribution"]:
        table.add_row(str(specialty), str(count))

    console.print(table)

    result: dict = {"valid_rows": stats["valid_rows"], "chunk_count": 0}

    # ── Optional ingestion ─────────────────────────────────────────────
    if ingest:
        console.print("\n[bold]Loading & chunking MTSamples…[/bold]")
        docs = load_mtsamples(path)
        chunker = ClinicalAwareChunker(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )
        chunks = chunker.chunk_documents(docs)
        console.print(
            f"[green]✓[/green] MTSamples → [bold]{len(docs):,}[/bold] docs | "
            f"[bold]{len(chunks):,}[/bold] chunks"
        )
        result["chunk_count"] = len(chunks)

    return result


if __name__ == "__main__":
    main(ingest="--ingest" in sys.argv)
