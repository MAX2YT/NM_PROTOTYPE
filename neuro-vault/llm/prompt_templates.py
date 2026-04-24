"""
llm/prompt_templates.py — Prompt construction for Neuro-Vault clinical AI.

All prompts strictly instruct the model to:
  - Answer only from provided context
  - Cite sources with dataset badges
  - Structure output as SUMMARY / EVIDENCE / SOURCES / CAUTION
  - Use ``INSUFFICIENT_CONTEXT:`` prefix when context is inadequate
"""

from __future__ import annotations

from typing import List


CLINICAL_SYSTEM_PROMPT: str = """You are Neuro-Vault, an offline clinical AI for Tamil Nadu hospitals.
Answer ONLY from the numbered context blocks below. Never use outside knowledge.

Rules:
1. You MUST use exactly this structure (with colons):
SUMMARY:
<answer>

EVIDENCE:
<points>

SOURCES:
<citations>

CAUTION:
<warnings>

2. Cite every claim: [Source: <doc_id> | <dataset>]
3. If context is insufficient, respond: INSUFFICIENT_CONTEXT: <reason>
4. For dosages/treatments always add: ⚠ Verify with prescribing guidelines.

Source authority: PubMed > MedQuAD > MTSamples

CONTEXT:
{context}
"""


TAMIL_SYSTEM_ADDENDUM: str = """
The user has asked the question in Tamil. Provide your answer in Tamil.
Maintain the same structured format (SUMMARY / EVIDENCE / SOURCES / CAUTION)
but write all prose in Tamil script.  Medical terms may be given in English
inside parentheses for clarity, e.g., "நீரிழிவு நோய் (Diabetes Mellitus)".
"""


def _format_chunk(chunk: dict, index: int) -> str:
    """Format one retrieved chunk as a numbered, labelled context block.

    Args:
        chunk: Chunk dict from the retriever.
        index: 1-based position in the context listing.

    Returns:
        Formatted string ready for prompt injection.
    """
    doc_id = chunk.get("doc_id", "unknown")
    dataset = chunk.get("dataset", "unknown")
    title = chunk.get("title", "")[:60]
    specialty = chunk.get("doc_type", "")
    # Truncate text to keep prompt compact → faster generation
    text = chunk.get("text", "")[:400]

    # Build source line
    source_parts = [doc_id, dataset]
    if title:
        source_parts.append(title)
    if specialty and specialty != dataset:
        source_parts.append(specialty)

    header = f"[{index}] " + " | ".join(source_parts)
    return f"{header}\n{text}\n"


def build_context(chunks: List[dict]) -> str:
    """Build the full context string from a list of retrieved chunks.

    Args:
        chunks: Ordered list of retrieved chunk dicts.

    Returns:
        Multi-block context string for prompt injection.
    """
    if not chunks:
        return "(No relevant context retrieved)"
    blocks = [_format_chunk(c, i + 1) for i, c in enumerate(chunks)]
    return "\n---\n".join(blocks)


def build_prompt(
    query: str,
    chunks: List[dict],
    is_tamil: bool = False,
) -> tuple[str, str]:
    """Assemble the system and user prompts for the Ollama LLM.

    Args:
        query:    The user's clinical question.
        chunks:   Retrieved context chunks.
        is_tamil: If True, append Tamil instruction addendum.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    context = build_context(chunks)
    system = CLINICAL_SYSTEM_PROMPT.format(context=context)

    if is_tamil:
        system += TAMIL_SYSTEM_ADDENDUM

    user_turn = f"Please answer the following clinical question based ONLY on the provided context.\n\nQuestion: {query}"
    return system, user_turn


def build_prompt(query: str, chunks: List[dict], is_tamil: bool = False) -> str:
    """Backward-compatible single-string prompt builder for legacy callers.

    The evaluation pipeline expects a single prompt string; concatenate the
    system and user turns in a deterministic format.
    """
    system, user_turn = build_prompts(query, chunks, is_tamil=is_tamil)
    return f"{system}\n\n{user_turn}"
