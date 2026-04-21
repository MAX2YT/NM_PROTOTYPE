"""
llm/ollama_client.py — Offline LLM inference via local Ollama server.

All inference happens 100% locally via the Ollama REST API running on
``localhost:11434``.  Zero external internet connectivity is required once
the model has been pulled with ``ollama pull``.

Supports:
  - Streaming generation with token-by-token yield
  - Non-streaming (full completion)
  - Health check / model availability verification
  - Graceful error handling with clear user-facing messages
"""

from __future__ import annotations

import json
import logging
from typing import Generator, Optional

import requests

from config import Config

logger = logging.getLogger(__name__)


class OllamaClient:
    """HTTP client for the local Ollama inference server.

    Args:
        base_url:    Ollama server URL (default: ``http://localhost:11434``).
        model:       Model name to use (default from ``Config.OLLAMA_MODEL``).
        timeout:     Request timeout in seconds.
        temperature: Sampling temperature for generation.
    """

    def __init__(
        self,
        base_url: str = Config.OLLAMA_BASE_URL,
        model: str = Config.OLLAMA_MODEL,
        timeout: int = Config.OLLAMA_TIMEOUT,
        temperature: float = Config.OLLAMA_TEMPERATURE,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self._generate_url = f"{self.base_url}/api/generate"
        self._tags_url = f"{self.base_url}/api/tags"

    # ------------------------------------------------------------------ #
    #  Health / availability
    # ------------------------------------------------------------------ #

    def is_available(self) -> bool:
        """Check whether the Ollama server is running and reachable.

        Returns:
            ``True`` if the server responds within 3 seconds.
        """
        try:
            resp = requests.get(self._tags_url, timeout=3)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def model_exists(self) -> bool:
        """Check whether the configured model is pulled / available.

        Returns:
            ``True`` if the model is listed by ``/api/tags``.
        """
        try:
            resp = requests.get(self._tags_url, timeout=5)
            resp.raise_for_status()
            tags = resp.json()
            models = [m["name"] for m in tags.get("models", [])]
            # Accept both "llama3:8b-instruct-q4_K_M" and "llama3"
            return any(self.model in m or m in self.model for m in models)
        except requests.RequestException:
            return False

    def list_models(self) -> list:
        """Return list of available model names.

        Returns:
            List of model name strings, or empty list on error.
        """
        try:
            resp = requests.get(self._tags_url, timeout=5)
            resp.raise_for_status()
            tags = resp.json()
            return [m["name"] for m in tags.get("models", [])]
        except requests.RequestException:
            return []

    # ------------------------------------------------------------------ #
    #  Text generation — streaming
    # ------------------------------------------------------------------ #

    def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream LLM response tokens one by one.

        Yields each token fragment as it arrives from the server.
        The caller should concatenate all yielded strings to build the
        full response.

        Args:
            prompt: Full prompt string (including system instructions if
                    no separate *system* arg is given).
            system: Optional separate system prompt.

        Yields:
            Token-level string fragments.

        Raises:
            ConnectionError: If Ollama server is unreachable.
            RuntimeError:    On unexpected server errors.
        """
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": 256,   # keep answers concise
                "num_ctx": 1024,      # small KV cache = much faster on CPU
            },
        }
        if system:
            payload["system"] = system

        try:
            with requests.post(
                self._generate_url,
                json=payload,
                stream=True,
                timeout=self.timeout,
            ) as resp:
                if resp.status_code == 404:
                    raise RuntimeError(
                        f"Model '{self.model}' not found.\n"
                        f"Pull it with: ollama pull {self.model}"
                    )
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break

        except requests.ConnectionError as exc:
            raise ConnectionError(
                f"Cannot connect to Ollama at '{self.base_url}'.\n"
                "Ensure Ollama is running: ollama serve"
            ) from exc
        except requests.Timeout as exc:
            raise TimeoutError(
                f"Ollama request timed out after {self.timeout}s."
            ) from exc

    # ------------------------------------------------------------------ #
    #  Text generation — non-streaming
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate a complete response (non-streaming).

        Args:
            prompt: Full prompt string.
            system: Optional separate system prompt.

        Returns:
            Complete response string.

        Raises:
            ConnectionError: If Ollama server is unreachable.
            RuntimeError:    On unexpected server errors.
        """
        tokens: list[str] = list(self.generate_stream(prompt, system=system))
        return "".join(tokens)

    # ------------------------------------------------------------------ #
    #  Convenience wrapper
    # ------------------------------------------------------------------ #

    def answer_clinical_query(
        self,
        prompt: str,
        stream: bool = True,
    ) -> str | Generator[str, None, None]:
        """Convenience method for clinical query answering.

        Args:
            prompt: Assembled clinical prompt (from ``build_prompt``).
            stream: If True, return a token generator; else return full string.

        Returns:
            Generator of tokens (if *stream*=True) or complete string.
        """
        if stream:
            return self.generate_stream(prompt)
        return self.generate(prompt)
