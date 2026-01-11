import logging
from abc import ABC, abstractmethod
from typing import Optional
import httpx

from configs import get_settings

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response from prompt."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider."""

    def __init__(self):
        settings = get_settings()
        self._base_url = settings.ollama_base_url
        self._model = settings.ollama_model
        self._timeout = httpx.Timeout(120.0, connect=10.0)

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate response using Ollama.

        Parameters
        ----------
        prompt : str
            User prompt
        system_prompt : str, optional
            System prompt for context

        Returns
        -------
        str
            Generated response
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": messages,
                    "stream": False
                }
            )
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            with httpx.Client(timeout=httpx.Timeout(5.0)) as client:
                response = client.get(f"{self._base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""

    def __init__(self):
        settings = get_settings()
        self._api_key = settings.openai_api_key
        self._model = settings.openai_model
        self._base_url = "https://api.openai.com/v1"
        self._timeout = httpx.Timeout(60.0, connect=10.0)

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate response using OpenAI.

        Parameters
        ----------
        prompt : str
            User prompt
        system_prompt : str, optional
            System prompt for context

        Returns
        -------
        str
            Generated response
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self._model,
                    "messages": messages,
                    "temperature": 0.1
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

    def is_available(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self._api_key and len(self._api_key) > 10)


def get_llm_provider() -> BaseLLMProvider:
    """
    Get the appropriate LLM provider based on configuration and availability.

    Returns
    -------
    BaseLLMProvider
        Available LLM provider instance

    Raises
    ------
    RuntimeError
        If no LLM provider is available
    """
    settings = get_settings()

    if settings.llm_provider == "ollama":
        ollama = OllamaProvider()
        if ollama.is_available():
            logger.info("Using Ollama LLM provider")
            return ollama
        logger.warning("Ollama not available, trying OpenAI fallback")

    openai = OpenAIProvider()
    if openai.is_available():
        logger.info("Using OpenAI LLM provider")
        return openai

    ollama = OllamaProvider()
    if ollama.is_available():
        logger.info("Using Ollama LLM provider (fallback)")
        return ollama

    raise RuntimeError(
        "No LLM provider available. Please ensure Ollama is running or "
        "provide OPENAI_API_KEY in environment variables."
    )
