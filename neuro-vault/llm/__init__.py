"""
llm/__init__.py — Public interface for the LLM package.
"""

from llm.ollama_client import OllamaClient
from llm.prompt_templates import CLINICAL_SYSTEM_PROMPT, build_prompt

__all__ = ["OllamaClient", "CLINICAL_SYSTEM_PROMPT", "build_prompt"]
