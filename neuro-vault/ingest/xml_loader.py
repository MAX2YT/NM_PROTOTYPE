"""
ingest/xml_loader.py — Parse MedQuAD XML files into Neuro-Vault document dicts.

MedQuAD (~47,000 QA pairs) is organized in 11 source-specific subdirectories.
Each XML file contains one or more <QAPair> elements.  The loader walks the
entire tree recursively, handles malformed XML gracefully, and supports
stratified train/test splitting.
"""

from __future__ import annotations

import json
import logging
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Map subdirectory prefix → friendly source name
_SOURCE_MAP: Dict[str, str] = {
    "1_CancerGov_QA": "CancerGov",
    "2_GARD_QA": "GARD",
    "3_GHR_QA": "GHR",
    "4_MedlinePlus_QA": "MedlinePlus",
    "5_NIDDK_QA": "NIDDK",
    "6_NINDS_QA": "NINDS",
    "7_SeniorHealth_QA": "SeniorHealth",
    "8_OrthoInfo_QA": "OrthoInfo",
    "9_CDC_QA": "CDC",
    "10_MPlus_Health_Topics_QA": "MPlus_HealthTopics",
    "11_MPlusDrugs_QA": "MPlus_Drugs",
}


# ------------------------------------------------------------------ #
#  Low-level XML file parser
# ------------------------------------------------------------------ #


def parse_medquad_xml(filepath: str | Path) -> List[dict]:
    """Parse a single MedQuAD XML file and return a list of QA dicts.

    Handles files where the root element is either ``<QAPairs>`` (with
    multiple ``<QAPair>`` children) or directly a ``<QAPair>`` element.
    Empty ``<Answer>`` tags are skipped.  Any parse error causes the file
    to be skipped entirely with a warning log entry.

    Args:
        filepath: Path to a MedQuAD ``.xml`` file.

    Returns:
        List of dicts with keys:
        ``qid``, ``question``, ``answer``, ``qtype``, ``source_file``.
        Returns an empty list on any error.
    """
    filepath = Path(filepath)
    pairs: List[dict] = []

    try:
        tree = ET.parse(str(filepath))
        root = tree.getroot()
    except ET.ParseError as exc:
        logger.warning("Skipping malformed XML '%s': %s", filepath.name, exc)
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error reading '%s': %s", filepath.name, exc)
        return []

    # Collect all <QAPair> elements regardless of nesting depth
    qa_elements = root.findall(".//QAPair")
    if not qa_elements:
        # Some files are directly a single QAPair
        if root.tag == "QAPair":
            qa_elements = [root]

    for qa in qa_elements:
        question_el = qa.find("Question")
        answer_el = qa.find("Answer")

        if question_el is None or answer_el is None:
            continue

        question_text = (question_el.text or "").strip()
        answer_text = (answer_el.text or "").strip()

        if not question_text or not answer_text:
            continue

        qid = qa.get("pid", "") or question_el.get("qid", "")
        qtype = question_el.get("qtype", "general")

        pairs.append(
            {
                "qid": qid,
                "question": question_text,
                "answer": answer_text,
                "qtype": qtype,
                "source_file": filepath.name,
            }
        )

    return pairs


# ------------------------------------------------------------------ #
#  Walk entire MedQuAD directory tree
# ------------------------------------------------------------------ #


def _collect_all_pairs(base_dir: Path) -> List[dict]:
    """Recursively walk *base_dir* and collect all QA pairs.

    Args:
        base_dir: Root of the cloned MedQuAD repository.

    Returns:
        List of dicts including ``source_folder`` and ``source`` keys
        added from the subdirectory mapping.
    """
    all_pairs: List[dict] = []
    xml_files = list(base_dir.rglob("*.xml"))
    logger.info("Found %d XML files under '%s'", len(xml_files), base_dir)

    for xml_path in xml_files:
        # Determine the source folder name from the immediate child of base_dir
        try:
            rel = xml_path.relative_to(base_dir)
            top_folder = rel.parts[0] if rel.parts else "unknown"
        except ValueError:
            top_folder = "unknown"

        friendly_source = _SOURCE_MAP.get(top_folder, top_folder)
        pairs = parse_medquad_xml(xml_path)

        for pair in pairs:
            pair["source_folder"] = top_folder
            pair["source"] = friendly_source

        all_pairs.extend(pairs)

    return all_pairs


# ------------------------------------------------------------------ #
#  Stratified train/test split
# ------------------------------------------------------------------ #


def split_medquad(
    all_pairs: List[dict], test_ratio: float = 0.10, seed: int = 42
) -> Tuple[List[dict], List[dict]]:
    """Split MedQuAD pairs into train and test sets, stratified by source.

    Ensures that each source folder contributes ``test_ratio`` fraction
    of its pairs to the held-out test set.

    Args:
        all_pairs: Full list of dicts as returned by ``_collect_all_pairs``.
        test_ratio: Fraction of each source's pairs to reserve for testing.
        seed: Random seed for reproducibility.

    Returns:
        Tuple ``(train_pairs, test_pairs)``.
    """
    random.seed(seed)
    by_source: Dict[str, List[dict]] = defaultdict(list)
    for pair in all_pairs:
        by_source[pair.get("source", "unknown")].append(pair)

    train_pairs: List[dict] = []
    test_pairs: List[dict] = []

    for source, pairs in by_source.items():
        shuffled = list(pairs)
        random.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * test_ratio))
        test_pairs.extend(shuffled[:n_test])
        train_pairs.extend(shuffled[n_test:])

    logger.info(
        "MedQuAD split: %d train | %d test (%.0f%% held out)",
        len(train_pairs),
        len(test_pairs),
        test_ratio * 100,
    )
    return train_pairs, test_pairs


# ------------------------------------------------------------------ #
#  Public document loader
# ------------------------------------------------------------------ #


def load_medquad(
    base_dir: str | Path,
    split: str = "train",
    test_ratio: float = 0.10,
    test_output_path: str | Path | None = None,
) -> List[dict]:
    """Load MedQuAD XML files and return Neuro-Vault document dicts.

    Args:
        base_dir: Root directory of the cloned MedQuAD repository.
        split: ``"train"`` returns the training corpus (90%);
               ``"test"`` returns evaluation QA pairs (10%).
        test_ratio: Fraction to hold out for evaluation.
        test_output_path: If provided, the test set is written to this
                          JSON path during this call.

    Returns:
        For the ``"train"`` split: list of document dicts suitable for
        ingestion (``text = "Q: ...\nA: ..."``).
        For the ``"test"`` split: list of raw QA dicts for RAGAS eval.

    Raises:
        FileNotFoundError: If *base_dir* does not exist.
        ValueError: If *split* is not ``"train"`` or ``"test"``.
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(
            f"MedQuAD directory not found at '{base_dir}'.\n"
            "Run: python scripts/download_medquad.py"
        )
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")

    all_pairs = _collect_all_pairs(base_dir)
    train_pairs, test_pairs = split_medquad(all_pairs, test_ratio=test_ratio)

    # Persist test set if a path is given
    if test_output_path is not None:
        test_output_path = Path(test_output_path)
        test_output_path.parent.mkdir(parents=True, exist_ok=True)
        eval_records = [
            {
                "question": p["question"],
                "answer": p["answer"],
                "source": p.get("source", ""),
                "qtype": p.get("qtype", ""),
            }
            for p in test_pairs
        ]
        with open(test_output_path, "w", encoding="utf-8") as fh:
            json.dump(eval_records, fh, indent=2, ensure_ascii=False)
        logger.info(
            "Saved %d test pairs to '%s'", len(eval_records), test_output_path
        )

    # Choose the correct split
    working_pairs = train_pairs if split == "train" else test_pairs

    # ------------------------------------------------------------------ #
    #  Build document dicts
    # ------------------------------------------------------------------ #
    documents: List[dict] = []
    for pair in working_pairs:
        q = pair["question"]
        a = pair["answer"]
        doc = {
            "doc_id": f"medquad_{pair.get('source', 'xx')}_{pair.get('qid', id(pair))}",
            "title": q[:100],
            "text": f"Q: {q}\nA: {a}",
            "question": q,
            "answer": a,
            "source": pair.get("source", "MedQuAD"),
            "doc_type": "medical_qa",
            "qtype": pair.get("qtype", "general"),
            "dataset": "medquad",
        }
        documents.append(doc)

    logger.info(
        "MedQuAD loader ('%s' split) produced %d documents",
        split,
        len(documents),
    )
    return documents
