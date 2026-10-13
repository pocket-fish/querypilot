import logging
from collections.abc import Callable
from typing import Protocol

import httpx

from querypilot.config import Settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the language model call fails."""


class LLM(Protocol):
    def invoke(self, prompt: str) -> str: ...


class OllamaLLM:
    def __init__(self, model: str, base_url: str) -> None:
        from langchain_ollama import ChatOllama

        self._chat = ChatOllama(model=model, base_url=base_url, temperature=0)

    def invoke(self, prompt: str) -> str:
        try:
            return str(self._chat.invoke(prompt).content)
        except Exception as exc:
            raise LLMError(f"Ollama call failed: {exc}") from exc


class LlamaCppLLM:
    """llama.cpp `llama-server` via its OpenAI-compatible endpoint (local, free)."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout)

    def invoke(self, prompt: str) -> str:
        try:
            response = self._client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": prompt}], "temperature": 0},
            )
            response.raise_for_status()
            return str(response.json()["choices"][0]["message"]["content"])
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"llama.cpp call failed: {exc}") from exc


def create_llm(settings: Settings) -> LLM:
    if settings.llm_backend == "llamacpp":
        return LlamaCppLLM(settings.llamacpp_base_url, settings.llm_timeout)
    return OllamaLLM(settings.ollama_model, settings.ollama_base_url)


class FakeLLM:
    """Scripted LLM for tests: returns queued responses in order."""

    def __init__(self, responses: list[str | Exception] | Callable[[str], str]) -> None:
        self._responses = responses
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if callable(self._responses):
            return self._responses(prompt)
        if not self._responses:
            raise LLMError("FakeLLM has no more scripted responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
