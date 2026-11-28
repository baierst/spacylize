"""LLM client wrapper for unified access to various language models.

This module provides a unified interface for interacting with different LLM
providers (OpenAI, Anthropic, Ollama, etc.) via the LiteLLM library.
"""

from typing import Optional
import dotenv
from litellm import completion, systemone

dotenv.load_dotenv()


class LLMClient:
    """Client for interacting with various LLM providers via LiteLLM.

    Supports OpenAI, Anthropic, Ollama, and other LiteLLM-compatible providers.

    Attributes:
        model: The model identifier (e.g., 'gpt-4o-mini', 'anthropic/claude-opus-4-5').
        api_key: API key for authentication.
        api_base: Optional custom API base URL for local or custom deployments.
        max_tokens: Maximum number of tokens to generate.
        think: Optional switch for the model's thinking mode.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: int = 1024,
        think: Optional[bool] = None,
    ) -> None:
        """Initialize the LLM client.

        Args:
            model: Model identifier for LiteLLM (e.g., 'gpt-4o-mini').
            api_key: API key for the LLM provider. Defaults to None.
            api_base: Custom API base URL for local or custom deployments.
            max_tokens: Maximum number of tokens to generate. Defaults to 1024.
            think: Turn the model's thinking mode on or off (e.g. False for
                Ollama models that think by default). Uses the model default if None.
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.think = think

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text using the configured LLM.

        Args:
            prompt: The user prompt to send to the LLM.
            system_prompt: Optional system prompt to set context or instructions.
            temperature: Optional sampling temperature. Uses the provider default if None.

        Returns:
            str: The generated text response from the LLM.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        completion_kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }

        if self.api_key:
            completion_kwargs["api_key"] = self.api_key

        if self.api_base:
            completion_kwargs["api_base"] = self.api_base

        if temperature is not None:
            completion_kwargs["temperature"] = temperature

        if self.think is not None:
            completion_kwargs["think"] = self.think

        response = completion(**completion_kwargs)

        return response["choices"][0]["message"]["content"]


def topic_score(
    text: str,
    domain: str,
    model: str = "ollama/nimble",
    api_base: str = "http://localhost:11434",
) -> float:
    """Probability (0-1) that a text fits the domain, from a local decision model.

    Args:
        text: The generated text.
        domain: The domain from the prompt config.
        model: Decision model (LiteLLM name), pulled with ``ollama pull nimble``.
        api_base: Ollama server URL.

    Returns:
        float: The probability that the text is an example of the domain.
    """
    question = {
        "on_topic": {
            "type": "noul",
            "instructions": f"Is this text an example of: {domain}?",
        }
    }
    response = systemone(
        model=model, state={"text": text}, questions=question, api_base=api_base
    )
    return response.answers["on_topic"].noul
