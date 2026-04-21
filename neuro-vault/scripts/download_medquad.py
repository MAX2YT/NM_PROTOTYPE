"""
scripts/download_medquad.py — Clone MedQuAD repository and build train/test split.

Steps:
  1. Check if data/raw/MedQuAD/ exists
  2. If not: git clone from GitHub with live progress output
  3. Parse all XML files recursively
  4. Stratified split: 90% train / 10% test
  5. Save test set to data/eval/medquad_test.json
  6. Report statistics
  7. Optional: run chunking on train split
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from config import Config
from ingest.xml_loader import load_medquad, _collect_all_pairs, split_medquad
from ingest.clinical_chunker import ClinicalAwareChunker

console = Console()

MEDQUAD_GIT_URL = "https://github.com/abachaa/MedQuAD"


def clone_medquad() -> bool:
    """Clone the MedQuAD GitHub repository.

    Returns:
        ``True`` on success, ``False`` on failure.
    """
    target = Config.MEDQUAD_DIR
    console.print(f"[bold]Cloning MedQuAD from GitHub…[/bold]")
    console.print(f"Source: [cyan]{MEDQUAD_GIT_URL}[/cyan]")
    console.print(f"Target: [cyan]{target}[/cyan]")

    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", MEDQUAD_GIT_URL, str(target)],
            capture_output=False,
            text=True,
            check=True,
        )
        console.print("[green]✓[/green] Clone complete.")
        return True
    except subprocess.CalledProcessError as exc:
        console.print(
            f"[red]✗ Git clone failed.[/red]\n"
            f"  Error: {exc}\n\n"
            "Alternatives:\n"
            "  1. Install Git from https://git-scm.com/download/win\n"
            "  2. Or download the ZIP from:\n"
            f"     {MEDQUAD_GIT_URL}\n"
            f"  3. Extract to: {target}"
        )
        return False
    except FileNotFoundError:
        console.print(
            "[red]✗ 'git' command not found.[/red]\n"
            "Install Git from https://git-scm.com/download/win and retry."
        )
        return False


def main(ingest: bool = False) -> dict:
    """Download, parse, and optionally ingest the MedQuAD dataset.

    Args:
        ingest: If True, chunk the train split and return chunk count.

    Returns:
        dict with ``total_pairs``, ``train_pairs``, ``test_pairs``,
        ``xml_files``, ``chunk_count`` (if ingest=True).
    """
    console.rule("[bold cyan]Neuro-Vault — MedQuAD Dataset[/bold cyan]")

    medquad_dir = Config.MEDQUAD_DIR

    # ── Clone if absent ────────────────────────────────────────────────
    if not medquad_dir.exists() or not any(medquad_dir.rglob("*.xml")):
        success = clone_medquad()
        if not success:
            console.print(
                "[yellow]Skipping MedQuAD ingestion — directory not ready.[/yellow]"
            )
            return {"total_pairs": 0, "train_pairs": 0, "test_pairs": 0,
                    "xml_files": 0, "chunk_count": 0}
    else:
        xml_count = len(list(medquad_dir.rglob("*.xml")))
        console.print(
            f"[green]✓[/green] MedQuAD directory exists with "
            f"[bold]{xml_count}[/bold] XML files."
        )

    # ── Parse all pairs ────────────────────────────────────────────────
    console.print("\n[bold]Parsing XML files…[/bold]")
    Config.ensure_dirs()

    with Progress(SpinnerColumn(), TextColumn("{task.description}")) as progress:
        task = progress.add_task("Parsing MedQuAD XML files…", total=None)
        all_pairs = _collect_all_pairs(medquad_dir)
        progress.update(task, completed=True)

    xml_files = len(list(medquad_dir.rglob("*.xml")))

    # Count source folders
    sources = set(p.get("source", "") for p in all_pairs)
    console.print(
        f"[green]✓[/green] Parsed [bold]{len(all_pairs):,}[/bold] QA pairs "
        f"from [bold]{xml_files}[/bold] XML files "
        f"across [bold]{len(sources)}[/bold] source folders."
    )

    # ── Stratified split ───────────────────────────────────────────────
    train_pairs, test_pairs = split_medquad(all_pairs, test_ratio=Config.MEDQUAD_TEST_RATIO)

    # Save test set
    Config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    import json
    test_records = [
        {
            "question": p["question"],
            "answer": p["answer"],
            "source": p.get("source", ""),
            "qtype": p.get("qtype", ""),
        }
        for p in test_pairs
    ]
    with open(Config.MEDQUAD_TEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(test_records, fh, indent=2, ensure_ascii=False)

    console.print(
        f"[green]✓[/green] Test set saved: "
        f"[bold]{len(test_records):,}[/bold] pairs → "
        f"[cyan]{Config.MEDQUAD_TEST_PATH}[/cyan]"
    )

    # ── Source folder breakdown ────────────────────────────────────────
    from collections import Counter
    source_counts = Counter(p.get("source", "unknown") for p in all_pairs)
    table = Table(title="Source Folder Distribution", border_style="dim")
    table.add_column("Source", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Train", justify="right", style="green")
    table.add_column("Test", justify="right", style="yellow")

    train_counts = Counter(p.get("source", "") for p in train_pairs)
    test_counts = Counter(p.get("source", "") for p in test_pairs)

    for source, total in sorted(source_counts.items()):
        table.add_row(source, str(total), str(train_counts[source]), str(test_counts[source]))

    console.print(table)

    result = {
        "total_pairs": len(all_pairs),
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "xml_files": xml_files,
        "chunk_count": 0,
    }

    # ── Optional ingestion ─────────────────────────────────────────────
    if ingest:
        console.print("\n[bold]Chunking MedQuAD train split…[/bold]")
        # Build document dicts from train_pairs
        train_docs = [
            {
                "doc_id": f"medquad_{p.get('source', 'xx')}_{p.get('qid', i)}",
                "title": p["question"][:100],
                "text": f"Q: {p['question']}\nA: {p['answer']}",
                "question": p["question"],
                "answer": p["answer"],
                "source": p.get("source", "MedQuAD"),
                "doc_type": "medical_qa",
                "qtype": p.get("qtype", "general"),
                "dataset": "medquad",
            }
            for i, p in enumerate(train_pairs)
        ]
        chunker = ClinicalAwareChunker(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )
        chunks = chunker.chunk_documents(train_docs)
        console.print(
            f"[green]✓[/green] MedQuAD → [bold]{len(train_docs):,}[/bold] docs | "
            f"[bold]{len(chunks):,}[/bold] chunks"
        )
        result["chunk_count"] = len(chunks)

    return result


if __name__ == "__main__":
    main(ingest="--ingest" in sys.argv)
