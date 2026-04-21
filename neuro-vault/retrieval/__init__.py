"""
retrieval/__init__.py — Public interface for the retrieval package.
"""

from retrieval.retriever import HybridRetriever
from retrieval.abstention import AbstentionChecker

__all__ = ["HybridRetriever", "AbstentionChecker"]
